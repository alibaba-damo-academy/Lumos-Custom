import os
import os.path as osp
import torch
import pickle

from nets.third_party.wan.utils.utils import torch_checkpoint_load
import torch.nn as nn
import torch.nn.functional as F
import transformers

from collections import OrderedDict
from typing import List, Optional, Union
from dataclasses import dataclass, field
from torch.nn import Module
from transformers.training_args import TrainingArguments
from transformers import PreTrainedModel
TRAINABLE_PRECISION = torch.bfloat16

from llava import modals
from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)
from llava.mm_utils import opencv_extract_frames, process_images
from llava.model.configuration_llava import LlavaConfig
from llava.model.language_model.builder import build_llm_and_tokenizer
from llava.model.multimodal_encoder.builder import build_vision_tower
from llava.model.multimodal_projector.builder import build_mm_projector
from llava.model.utils import get_model_config
from llava.train.sequence_parallel import get_pg_manager
from llava.utils.tokenizer import infer_stop_tokens, tokenize_conversation


class VisionHead(Module):
    def __init__(self, llm_hidden_size, hidden_size=1152, learnable_query_length=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.learnable_query_length = learnable_query_length
        self.decoder_query = torch.nn.Parameter(
                                torch.randn((1, self.learnable_query_length, self.hidden_size), 
                                dtype=TRAINABLE_PRECISION), 
                                requires_grad=True
                            )

        self.visionHeadAdapter = nn.Transformer(
                                    batch_first=True, 
                                    norm_first=True, 
                                    d_model = self.hidden_size, 
                                    num_encoder_layers=4, 
                                    num_decoder_layers=4, 
                                    dim_feedforward=self.hidden_size * 4, 
                                    dropout=0.0, 
                                    dtype=TRAINABLE_PRECISION
                                )

        self.fc = nn.Sequential(
                    nn.Linear(llm_hidden_size, self.hidden_size),
                    nn.GELU(),
                    nn.Linear(self.hidden_size, self.hidden_size),
                ).to(TRAINABLE_PRECISION)

    def forward(self, vlm_last_hidden_state):
        # import pdb; pdb.set_trace();
        # print(f'vlm_last_hidden_state shape {vlm_last_hidden_state.shape}', flush=True)
        input_embeds = self.fc(vlm_last_hidden_state)
        vision_tokens = self.visionHeadAdapter(src=input_embeds, tgt=self.decoder_query.repeat(input_embeds.shape[0], 1, 1)) # B * 4 * 1152

        return vision_tokens


    def save_pretrained(self, output_dir, state_dict=None):
        """
        Save the model's state dictionary and configuration to a directory.
        
        Args:
            output_dir (str): Directory to save the model files.
            state_dict (dict, optional): State dictionary to save. If None, uses model's state_dict().
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save state dictionary
        if state_dict is None:
            state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
    
    def load_checkpoint(self, checkpoint_dir, map_location=None):
        """Load weights from ``checkpoint_dir/pytorch_model.bin`` into this module."""
        if map_location is None:
            map_location = torch.device('cpu')
        state_dict = torch_checkpoint_load(
            os.path.join(checkpoint_dir, "pytorch_model.bin"),
            map_location=map_location,
        )
        self.load_state_dict(state_dict)
        return self

class VILAWithVisionHead(PreTrainedModel):
    def __init__(self, vila, llm_hidden_size=4096):
        super().__init__(vila.config)

        self.vlm = vila
        self.config = self.vlm.config
        self.config.vh_learnable_query_length = 1
        self.config.vh_hidden_size = 1152

        self.vision_head = VisionHead(
                            llm_hidden_size=self.config.hidden_size, 
                            hidden_size=self.config.vh_hidden_size, 
                            learnable_query_length=self.config.vh_learnable_query_length
                        )
        self.config.vision_head_resume_ckpt = "/cpfs01/projects-HDD/cfff-01ff502a0784_HDD/public/qlz/projects/unified_gen/checkpoints/0422_img_gen_edit_und/checkpoint-30000"
        if hasattr(self.config, "vision_head_resume_ckpt"):
            maybe_vision_head_ckpt = os.path.join(self.config.vision_head_resume_ckpt, "vision_head")
            if os.path.exists(maybe_vision_head_ckpt):
                print("Load vision_head ckpt from:", maybe_vision_head_ckpt)
                vision_head_ckpt = os.path.join(maybe_vision_head_ckpt, "pytorch_model.bin")
                vision_head_state_dict = torch_checkpoint_load(vision_head_ckpt, map_location="cpu")
                self.vision_head.load_state_dict(vision_head_state_dict)

    def get_vision_head(self):
        vision_head = getattr(self, "vision_head", None)
        if type(vision_head) is list:
            vision_head = vision_head[0]
            
        return vision_head

    def forward(self, inputs):
        # tune the language head
        vlm_outputs = self.vlm(output_hidden_states=True, **inputs)
        vlm_last_hidden_states = vlm_outputs.hidden_states[-1]
        vision_tokens = self.vision_head(vlm_last_hidden_states) # bs * num_learnable_query * 1152
        pooled_vision_tokens = torch.mean(vision_tokens, dim=1) # bs * 1152

        return vlm_outputs, vision_tokens, pooled_vision_tokens
    

    def save_pretrained(self, output_dir, state_dict=None):
        if state_dict is None:
            # other wise fetch from deepspeed
            # state_dict = accelerator.get_state_dict(is_deepspeed_enabled)
            state_dict = self.state_dict()

        if getattr(self.vlm, "tokenizer", None):
            self.vlm.tokenizer.save_pretrained(osp.join(output_dir, "llm"))

        if self.vlm.get_llm():
            print(f"saving llm to {osp.join(output_dir, 'llm')}")
            self.vlm.llm.config._name_or_path = osp.join(output_dir, "llm")
            llm_state_dict = OrderedDict({k.split("llm.")[-1]: v for k, v in state_dict.items() if "llm" in k})
            self.vlm.llm.save_pretrained(os.path.join(output_dir, "llm"), state_dict=llm_state_dict)
            self.config.llm_cfg = self.vlm.llm.config

        if self.vlm.get_vision_tower():
            print(f"saving vision_tower to {osp.join(output_dir, 'vision_tower')}")
            self.vlm.vision_tower.config._name_or_path = osp.join(output_dir, "vision_tower")
            vision_tower_state_dict = OrderedDict(
                {k.split("vision_tower.vision_tower.")[-1]: v for k, v in state_dict.items() if "vision_tower" in k}
            )
            self.vlm.vision_tower.vision_tower.save_pretrained(
                os.path.join(output_dir, "vision_tower"),
                state_dict=vision_tower_state_dict,
            )
            self.vlm.vision_tower.image_processor.save_pretrained(os.path.join(output_dir, "vision_tower"))
            self.config.vision_tower_cfg = self.vlm.vision_tower.config
            if hasattr(self.config.vision_tower_cfg, "auto_map"):
                if "radio" not in self.get_vision_tower().__class__.__name__.lower():
                    delattr(self.config.vision_tower_cfg, "auto_map")

        if self.vlm.get_mm_projector():
            print(f"saving mm_projector to {osp.join(output_dir, 'mm_projector')}")
            self.vlm.mm_projector.config._name_or_path = osp.join(output_dir, "mm_projector")
            mm_projector_state_dict = OrderedDict(
                {k.split("mm_projector.")[-1]: v for k, v in state_dict.items() if "mm_projector" in k}
            )
            self.vlm.mm_projector.save_pretrained(
                os.path.join(output_dir, "mm_projector"),
                state_dict=mm_projector_state_dict,
            )
            self.config.mm_projector_cfg = self.vlm.mm_projector.config
        if self.get_vision_head():
            print(f"saving vision_head to {osp.join(output_dir, 'vision_head')}")
            vision_head_state_dict = OrderedDict(
                {k.split("vision_head.")[-1]: v for k, v in state_dict.items() if "vision_head" in k}
            )
            self.vision_head.save_pretrained(
                os.path.join(output_dir, "vision_head"),
                state_dict=vision_head_state_dict,
            )     

        ## update and save top-level config
        self.config._name_or_path = output_dir
        self.config.architectures = [self.vlm.__class__.__name__]
        self.config.save_pretrained(output_dir)


    @classmethod
    def load_pretrained(cls, pretrained_dir, vila_model_class=None, **kwargs):
        """
        Load the model's state dictionary and configuration from a directory.

        Args:
            pretrained_dir (str): Directory containing the saved model files.
            vila_model_class (class, optional): The class of the base VILA model. If None, assumes it is already defined.
            **kwargs: Additional arguments to pass to the model constructor.

        Returns:
            VILAWithVisionHead: An instance of the model with loaded weights.
        """
        if not os.path.exists(pretrained_dir):
            raise FileNotFoundError(f"The directory '{pretrained_dir}' does not exist.")

        # Load top-level configuration
        config_path = os.path.join(pretrained_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"The configuration file 'config.json' is missing in '{pretrained_dir}'.")
        
        config = cls.config_class.from_json_file(config_path)

        # Initialize the base VILA model if not provided
        if vila_model_class is None:
            vila_model_class = config.llm_cfg._name_or_path  # Assuming this points to the LLM class
        vila = vila_model_class.from_pretrained(os.path.join(pretrained_dir, "llm"))

        # Instantiate the current model
        model = cls(vila=vila, llm_hidden_size=config.hidden_size, **kwargs)

        # Load vision tower if it exists
        vision_tower_dir = os.path.join(pretrained_dir, "vision_tower")
        if os.path.exists(vision_tower_dir):
            vision_tower = model.vlm.get_vision_tower()
            vision_tower.load_pretrained(vision_tower_dir)
            vision_tower.image_processor = vision_tower.image_processor.from_pretrained(vision_tower_dir)

        # Load mm_projector if it exists
        mm_projector_dir = os.path.join(pretrained_dir, "mm_projector")
        if os.path.exists(mm_projector_dir):
            mm_projector = model.vlm.get_mm_projector()
            mm_projector.load_pretrained(mm_projector_dir)

        # Load vision head if it exists
        vision_head_dir = os.path.join(pretrained_dir, "vision_head")
        if os.path.exists(vision_head_dir):
            vision_head_state_dict = torch_checkpoint_load(os.path.join(vision_head_dir, "pytorch_model.bin"), map_location="cpu")
            model.vision_head.load_state_dict(vision_head_state_dict)

        # Load overall state dict if available
        overall_state_dict_path = os.path.join(pretrained_dir, "pytorch_model.bin")
        if os.path.exists(overall_state_dict_path):
            overall_state_dict = torch_checkpoint_load(overall_state_dict_path, map_location="cpu")
            model.load_state_dict(overall_state_dict, strict=False)

        return model

class VILAWithVisionHeadForAlignment(VILAWithVisionHead):
    def __init__(self, vila, siglip, llm_hidden_size=4096):
        super().__init__(vila, llm_hidden_size=llm_hidden_size)

        # initialize the siglip to be aligned
        # @qinluozheng: we dont overwrite the save_pretrained method so that the additional siglip model wont be saved
        if type(siglip) == str: 
            self.siglip = transformers.SiglipVisionModel.from_pretrained(siglip)
        elif type(siglip) == SiglipVisionModel:
            self.siglip = siglip
        else:
            raise AssertionError("Cannot load SigLipVisionModel from: ", siglip)
        
        # self.count = 0
    
    def forward(self, inputs, aligned_images):
        # tune the language head
        vlm_outputs = self.vlm(output_hidden_states=True, **inputs)
        vlm_last_hidden_states = vlm_outputs.hidden_states[-1]
        ar_loss = vlm_outputs.loss
        vila_vh_outputs = {"ar_loss":ar_loss, "vlm_outputs":vlm_outputs}
        # tune the vision head
        if aligned_images is not None:
            #import pdb; pdb.set_trace()
            vision_tokens = self.vision_head(vlm_last_hidden_states) # bs * num_learnable_query * 1152
            pooled_vision_tokens = torch.mean(vision_tokens, dim=1) # bs * 1152
            with torch.no_grad():
                feat_aligned = self.siglip(pixel_values=aligned_images).pooler_output # SiglipVisionModel
            vila_vh_outputs["siglip2_img_pooled_output"] = pooled_vision_tokens
            # saved_feat = {
            #     "siglip2_img_pooled_output":pooled_vision_tokens
            # }
            # output_path = os.path.join("/cpfs01/projects-HDD/cfff-01ff502a0784_HDD/public/qlz/projects/unified_gen/merged_vision_token_pkl", str(self.count)+".pkl")
            # with open(output_path, 'wb') as f:
            #     pickle.dump(saved_feat, f)
            # self.count += 1
            # print(output_path)

            align_loss = 1. - F.cosine_similarity(pooled_vision_tokens, feat_aligned, dim=-1).mean()
            vila_vh_outputs["align_loss"] = align_loss
            loss = ar_loss + align_loss
        else:
            loss = ar_loss
        vila_vh_outputs["loss"] = loss

        return vila_vh_outputs

