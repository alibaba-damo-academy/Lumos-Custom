import argparse
import re
from io import BytesIO
import os
import json
import glob
import numpy as np
from decord import VideoReader
from decord import cpu
import PIL

import gc
import requests
import torch
from torch.utils.data import Dataset
# import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from nets.third_party.llava.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                             DEFAULT_IMAGE_TOKEN, IMAGE_PLACEHOLDER,
                             IMAGE_TOKEN_INDEX)
from nets.third_party.llava.conversation import SeparatorStyle, conv_templates
# from nets.third_party.llava.mm_utils_inference import (KeywordsStoppingCriteria, get_model_name_from_path,
from nets.third_party.llava.mm_utils import (KeywordsStoppingCriteria, get_model_name_from_path,
                            process_images, tokenizer_image_token)
from nets.third_party.llava.model.builder import load_pretrained_model
from nets.third_party.llava.utils import disable_torch_init

disable_torch_init()

def load_video(video_path, max_frames_num):
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        fps = round(vr.get_avg_fps())
        frame_idx = np.linspace(0, total_frame_num - 2, max_frames_num, dtype=int)
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        return [Image.fromarray(img) for img in spare_frames]
    except Exception as e:
        print(f"Failed to load video {video_path} with error: {e}")
        return [Image.new("RGB", (448, 448), (0, 0, 0))] * max_frames_num

class VilaARVideoModel:
    def __init__(self, model_path, model_name=None, model_base=None, conv_mode=None,
                 num_video_frames=6, temperature=0.2, top_p=None, num_beams=1, 
                 num_return_sequences=1, max_new_tokens=512, device='cpu'):
        """
        Initialize the VilaARVideoModel with given model path and parameters.
        
        Args:
            model_path (str): Path to the pretrained model
            model_name (str, optional): Model name. If None, will be inferred from path
            model_base (str, optional): Base model path
            conv_mode (str, optional): Conversation mode
            num_video_frames (int): Number of frames to extract from video
            temperature (float): Sampling temperature
            top_p (float, optional): Top-p sampling parameter
            num_beams (int): Number of beams for beam search
            num_return_sequences (int): Number of sequences to return
            max_new_tokens (int): Maximum number of new tokens to generate
            device (str): Device to run the model on ('cpu' or 'cuda')
        """
        self.model_path = model_path
        self.model_base = model_base
        self.conv_mode = conv_mode
        self.num_video_frames = num_video_frames
        self.temperature = temperature
        self.top_p = top_p
        self.num_beams = num_beams
        self.num_return_sequences = num_return_sequences
        self.max_new_tokens = max_new_tokens
        
        # Load model
        if model_name is None:
            model_name = get_model_name_from_path(model_path)
        self.model_name = model_name

        # load_pretrained_model defaults to device_map="auto", which can leave layers on CPU.
        # Pass a concrete device so builder sets device_map={"": idx} (single GPU per rank).
        if isinstance(device, int):
            load_device = device
        elif isinstance(device, torch.device):
            load_device = "cpu" if device.type == "cpu" else (device.index if device.index is not None else 0)
        elif isinstance(device, str):
            d = device.strip().lower()
            if d == "cpu":
                load_device = "cpu"
            elif d == "cuda":
                load_device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
            elif d.startswith("cuda:"):
                load_device = int(d.split(":", 1)[1])
            else:
                load_device = device
        else:
            load_device = device

        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, model_name, model_base, device=load_device
        )
        self.model.to(device)
        if isinstance(device, int):
            self._target_device = torch.device(f"cuda:{device}")
        else:
            self._target_device = torch.device(device)
        self._offloaded = False

        # Determine conversation mode
        if conv_mode is None:
            if "llama-2" in model_name.lower():
                self.conv_mode = "llava_llama_2"
            elif "v1" in model_name.lower():
                self.conv_mode = "llava_v1"
            elif "mpt" in model_name.lower():
                self.conv_mode = "mpt"
            else:
                self.conv_mode = "llava_v0"
        else:
            self.conv_mode = conv_mode

    def _ensure_model_on_target_device(self):
        cur = next(self.model.parameters()).device
        if getattr(self, "_offloaded", False) or cur != self._target_device:
            self.model.to(self._target_device)
        self._offloaded = False

    def offload_to_cpu(self):
        """Move AR weights to CPU and release CUDA cache (e.g. after general_emb, before DiT sampling)."""
        self.model.cpu()
        self._offloaded = True
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def general_emb(self, prompt, media_path=None, task_type="vid_gen"):
        """
        Generate embeddings for various tasks (video/image generation/editing).
        
        Args:
            prompt (str): The text prompt for the task
            media_path (str, optional): Path to the input media (video/image) for editing tasks
            task_type (str): Type of task, one of:
                - "vid_gen": Video generation
                - "vid_edit": Video editing
                - "img_gen": Image generation
                - "img_edit": Image editing
                
        Returns:
            torch.Tensor: The last hidden states from the model
        """
        if 't2v' in task_type:
            task_type = "vid_gen"
        elif 'v2v' in task_type:
            task_type = "vid_edit"
        elif 't2i' in task_type:
            task_type = "img_gen"
        elif 'i2i' in task_type:
            task_type = "img_edit"
        elif 'i2v' in task_type:
            task_type = "i2v_gen"

        self._ensure_model_on_target_device()

        # Set task-specific parameters
        task_configs = {
            "vid_gen": {
                "answer": "[GEN_VID]",
                "prefix": "Generate a video about:\n",
                "needs_media": False
            },
            "vid_edit": {
                "answer": "[GEN_VID]",
                "prefix": "<video>\n",
                "needs_media": True,
                "media_loader": lambda path: load_video(path, 8)
            },
            "i2v_gen": {
                "answer": "[GEN_VID]",
                "prefix": "Generate a video from the reference image with the following prompt:\n",
                "needs_media": True,
                "media_loader": lambda path: [PIL.Image.open(path).convert("RGB")]
            },
            "img_gen": {
                "answer": "[GEN_IMG]",
                "prefix": "Generate an image about:\n",
                "needs_media": False
            },
            "img_edit": {
                "answer": "[GEN_IMG]",
                "prefix": "",
                "needs_media": True,
                "media_loader": lambda path: [PIL.Image.open(path).convert("RGB")]
            }
        }
        
        if task_type not in task_configs:
            raise ValueError(f"Invalid task_type: {task_type}. Must be one of {list(task_configs.keys())}")
            
        config = task_configs[task_type]
        
        # Load and process media if needed
        images = None
        if config["needs_media"]:
            if media_path is None:
                raise ValueError(f"media_path is required for {task_type} task")
            images = config["media_loader"](media_path)
            images = process_images(images, self.image_processor, self.model.config).half().cuda()
            images = [images]
        
        # Prepare query
        qs = config["prefix"] + prompt
        
        # Add image tokens if needed
        if images is not None:
            if self.model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
            else:
                qs = (DEFAULT_IMAGE_TOKEN + "\n") * len(images) + qs
        
        # Prepare conversation
        conv = conv_templates["llama_3"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], config["answer"])
        prompt = conv.get_prompt()
        
        # Prepare model inputs
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        attention_masks = input_ids.ne(pad_token_ids).long().cuda()
        
        inputs = {
            "input_ids": input_ids,
            "images": images,
            "labels": input_ids,
            "attention_mask": attention_masks,
        }
        
        # Generate embeddings
        with torch.inference_mode():
            vlm_outputs = self.model(output_hidden_states=True, **inputs)
            vlm_last_hidden_states = vlm_outputs.hidden_states[-1]
        
        return vlm_last_hidden_states

    def generate(self, video_path, query):
        """
        Generate caption for a given video path and query.
        
        Args:
            video_path (str): Path to the video file
            query (str): Query/prompt for generation
            
        Returns:
            tuple: (output_ids, outputs)
        """
        try:
            self._ensure_model_on_target_device()
            qs = query
            # Extract frames from video
            if video_path is not None and os.path.exists(video_path):
                from nets.third_party.llava.mm_utils import opencv_extract_frames
                if video_path.endswith(".jpg") or video_path.endswith(".png"):
                    images = [PIL.Image.open(video_path).convert("RGB")]
                else:
                    images, _ = opencv_extract_frames(video_path, self.num_video_frames)
                
                # Prepare query with image tokens
                image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                if IMAGE_PLACEHOLDER in qs:
                    if self.model.config.mm_use_im_start_end:
                        qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
                    else:
                        qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
                else:
                    if DEFAULT_IMAGE_TOKEN not in qs:
                        # Automatically append image tokens for each frame
                        if self.model.config.mm_use_im_start_end:
                            qs = (image_token_se + "\n") * len(images) + qs
                        else:
                            qs = (DEFAULT_IMAGE_TOKEN + "\n") * len(images) + qs
                # Process images and tokenize
                images_tensor = process_images(
                    images, self.image_processor, self.model.config
                ).to(self.model.device, dtype=torch.float16)
                images_tensor = [images_tensor]
            else:
                images_tensor = None
            
            # Prepare conversation
            conv = conv_templates[self.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            input_ids = tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).cuda()
            
            # Prepare stopping criteria
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
            
            # Generate outputs
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=images_tensor,
                    do_sample=True if self.temperature > 0 else False,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    num_beams=self.num_beams,
                    num_return_sequences=self.num_return_sequences,
                    max_new_tokens=self.max_new_tokens,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                )
            
            # Decode and clean output
            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            outputs = outputs.strip()
            
            if outputs.endswith(stop_str):
                outputs = outputs[: -len(stop_str)]
            outputs = outputs.strip()
            
            return output_ids, outputs
            
        except Exception as e:
            raise RuntimeError(f"Error processing video {video_path}: {str(e)}")

class VideoDataset(Dataset):
    def __init__(self, video_list_file, caption_folder):
        caption_files = list(glob.glob(os.path.join(caption_folder, '*.json')))
        processed_videos = {os.path.splitext(os.path.basename(f))[0]: 1 for f in caption_files}

        videos = []
        if not os.path.exists(caption_folder):
            os.makedirs(caption_folder)
        with open(video_list_file, 'r') as f:
            for line in f.readlines():
                v = line.strip()
                videos.append(v)

        self.videos = []
        for v in tqdm(videos):
            if os.path.splitext(os.path.basename(v))[0] not in processed_videos:
                self.videos.append(v)

        print(f"Total {len(videos)} videos, left {len(self.videos)} to process")
    
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, index):
        video_file = self.videos[index]
        return video_file
    
def collate_fn(samples):
    return {
        'videos': samples
    }

class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)



@torch.inference_mode()
def main(args):
    if args.pdb_debug:
        import pdb; pdb.set_trace()
        args.num_workers = 0
    
    if not args.pdb_debug and (torch.cuda.device_count() > 1):
        torch.distributed.init_process_group(
            backend='nccl',
            world_size=int(os.getenv('WORLD_SIZE', '1')),
            rank=int(os.getenv('RANK', '0')),
        )
        torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))
    

    # ======================================================
    # 1. read video dataset
    # ======================================================
    dataset = VideoDataset(video_list_file=args.video_list_file, caption_folder=args.caption_folder)
    if args.pdb_debug or torch.cuda.device_count() == 1:
        sampler = None 
    else:
        sampler = InferenceSampler(len(dataset))

    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=1,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn
    )

    # ======================================================
    # 2. load model using the new class
    # ======================================================
    vila_model = VilaARVideoModel(
        model_path=args.model_path,
        model_base=args.model_base,
        conv_mode=args.conv_mode,
        num_video_frames=args.num_video_frames,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        num_return_sequences=args.num_return_sequences,
        max_new_tokens=args.max_new_tokens
    )

    # ======================================================
    # 3. generate captions
    # ======================================================
    if args.error_report_file == None:
        current_directory = os.getcwd()
        args.error_report_file = os.path.join(current_directory, "failed_captioning_error.log")
    print("Setting error logging path: {}".format(args.error_report_file))
    error_logger = open(args.error_report_file, mode='a', encoding='utf-8')
    for idx, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        video_path = data['videos'][0] # the batch must be 1
        try:
            # Use the new class method for generation
            outputs = vila_model.generate(video_path, args.query)
            print(outputs)
            
            # # save results
            # caption_file = os.path.join(args.caption_folder, os.path.splitext(os.path.basename(video_path))[0] + ".json").encode('utf-8')
            # os.makedirs(os.path.dirname(caption_file), exist_ok=True)
            # with open(caption_file, 'w', encoding='utf-8') as f:
            #     json.dump({"video": video_path, "captions": outputs}, f, indent=4, ensure_ascii=False)
        except Exception as e:
            error_logger.write(video_path + '\n')
            print("Error processing : ", video_path)
            print(e)
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-list-file", type=str)
    parser.add_argument("--caption-folder", type=str, default="/zju_0038/datasets/text-to-video/webvid/llava_caption")
    parser.add_argument("--error-report-file", type=str, default=None )
    parser.add_argument('--local-rank', type=int, default=0)    
    parser.add_argument("--model-path", type=str, default="Efficient-Large-Model/VILA-2.7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--num-video-frames", type=int, default=6)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)

    parser.add_argument("--pdb_debug", action="store_true")
    args = parser.parse_args()

    main(args)