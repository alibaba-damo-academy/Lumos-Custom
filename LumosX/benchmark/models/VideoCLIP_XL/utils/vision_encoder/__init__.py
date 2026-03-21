import torch
import numpy as np
import cv2
import os

from .model_vision_encoder import VisionEncoder

def get_vision_encoder():
    vision_encoder = VisionEncoder()
    
    return vision_encoder
