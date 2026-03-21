import os
import logging
import torch
from torch import nn
import math

from .clip_vision import clip_joint_l14, clip_joint_b16

logger = logging.getLogger(__name__)


class VisionEncoder(nn.Module):

    def __init__(self):
        super(VisionEncoder, self).__init__()

        self.vision_encoder_name = 'vit_l14'
        self.vision_encoder_pretrained = False
        self.inputs_image_res = 224
        self.vision_encoder_kernel_size = 1
        self.vision_encoder_center = True
        self.video_input_num_frames = 8
        self.vision_encoder_drop_path_rate = 0.1
        self.vision_encoder_checkpoint_num = 24

        self.vision_width = 1024
        self.embed_dim = 768 
        self.masking_prob = 0.9

        self.vision_encoder = self.build_vision_encoder()

        self.temp = nn.parameter.Parameter(torch.ones([]) * 1 / 100.0)
        self.temp_min = 1 / 100.0

    def no_weight_decay(self):
        ret = {"temp"}
        ret.update(
            {"vision_encoder." + k for k in self.vision_encoder.no_weight_decay()}
        )

        return ret


    def encode_vision(self, image, test=False):
        if image.ndim == 5:
            image = image.permute(0, 2, 1, 3, 4).contiguous()
        else:
            image = image.unsqueeze(2)

        if not test and self.masking_prob > 0.0:
            return self.vision_encoder(
                image, masking_prob=self.masking_prob
            )

        return self.vision_encoder(image)


    @torch.no_grad()
    def clip_contrastive_temperature(self, min_val=0.001, max_val=0.5):
        """Seems only used during pre-training"""
        self.temp.clamp_(min=self.temp_min)

    def build_vision_encoder(self):
        """build vision encoder
        Returns: (vision_encoder, vision_layernorm). Each is a `nn.Module`.

        """
        vision_encoder = clip_joint_l14(
            pretrained=self.vision_encoder_pretrained,
            input_resolution=self.inputs_image_res,
            kernel_size=self.vision_encoder_kernel_size,
            center=self.vision_encoder_center,
            num_frames=self.video_input_num_frames,
            drop_path=self.vision_encoder_drop_path_rate,
            checkpoint_num=self.vision_encoder_checkpoint_num,
        )

        return vision_encoder


    def get_vid_features(self, input_frames):
        clip_feat = self.encode_vision(input_frames, test=True).float()
        
        return clip_feat