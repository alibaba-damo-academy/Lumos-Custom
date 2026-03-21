import os
from typing import List

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from utils.text_encoder import text_encoder
from utils.vision_encoder import get_vision_encoder


class VideoCLIP_XL(nn.Module):
    def __init__(self):
        super(VideoCLIP_XL, self).__init__()
        self.text_model = text_encoder.load().float()
        self.vision_model = get_vision_encoder().float()