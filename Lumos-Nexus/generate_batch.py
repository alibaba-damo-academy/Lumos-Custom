#!/usr/bin/env python3
"""
This script provides a unified interface for video editing tasks using V2V mode:
- Video-to-Video (V2V) editing with prompts from JSON file
- Source videos and rewrite instructions from JSON file

Usage:
    python generate_batch_edit.py --pkl_list_file data.json --task v2v --size 832*480
"""

import argparse
import gc
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union, List
import pickle as pkl
import json
import shutil

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

import torch
import torch.distributed as dist
import numpy as np
from PIL import Image
from torchvision import transforms

# Optional CV2 import for video processing
import cv2
CV2_AVAILABLE = True

# Optional imageio import for video writing fallback
import imageio.v2 as imageio
IMAGEIO_AVAILABLE = True

# Repository root (this file at repo root, or under tools/inference/)
def _repo_root() -> Path:
    p = Path(__file__).resolve().parent
    if p.name == "inference" and p.parent.name == "tools":
        return p.parent.parent
    return p


PROJECT_ROOT = _repo_root()
sys.path.insert(0, str(PROJECT_ROOT))

from video_generator import VideoGenerator
from nets.third_party.wan.utils.utils import (
    cache_video,
    cache_image,
    str2bool,
    safe_distributed_barrier,
)
from nets.third_party.wan.configs import SIZE_CONFIGS


def init_logging(rank: int = 0):
    """Initialize logging configuration."""
    log_file = f'video_generate_rank{rank}.log'
    
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_file)
            ]
        )
    else:
        logging.basicConfig(level=logging.ERROR)


def validate_args(args: argparse.Namespace) -> None:
    """
    Validate command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Raises:
        ValueError: If validation fails
    """
    # Basic validation - removed ckpt_dir requirement since paths are now fixed
    if not args.prompt:
        raise ValueError("Please provide a prompt using --prompt argument")
        
    # Set defaults
    if args.sample_steps is None:
        args.sample_steps = 40 if "i2v" in args.task else 50

    if args.sample_shift is None: 
        args.sample_shift = 5.0
        if "i2v" in args.task and args.size in ["832*480", "480*832", "352*640", "640*352"]:
            args.sample_shift = 3.0

    if args.frame_num is None:
        args.frame_num = 1 if "t2i" in args.task else 81

    # T2I validation
    if "t2i" in args.task and args.frame_num != 1:
        raise ValueError(f"Frame number must be 1 for t2i task, got {args.frame_num}")

    # Set random seed if not provided
    if args.base_seed < 0:
        args.base_seed = torch.randint(0, 2**32, (1,)).item()


def str2tuple(v: str) -> tuple:
    """
    Convert string to tuple.
    
    Examples:
        '1,2,2' -> (1, 2, 2)
        '(1,2,2)' -> (1, 2, 2)
        
    Args:
        v: String representation of a tuple
        
    Returns:
        Parsed tuple with integer values
    """
    v = v.strip()
    if v.startswith('(') and v.endswith(')'):
        v = v[1:-1]
    
    return tuple(int(x.strip()) for x in v.split(','))


def transform_image_to_tensor(image: Union[Image.Image, np.ndarray], 
                            target_size: Tuple[int, int] = (480, 832)) -> torch.Tensor:
    """
    Transform PIL Image or numpy array to tensor with resize and center crop.
    
    Args:
        image: Input image
        target_size: Target size (height, width)
        
    Returns:
        Transformed tensor
    """
    if isinstance(image, np.ndarray):
        h, w = image.shape[:2]
    else:
        w, h = image.size
        
    ratio = float(target_size[1]) / float(target_size[0])  # w/h
    
    if w < h * ratio:
        crop_size = (int(float(w) / ratio), w)
    else:
        crop_size = (h, int(float(h) * ratio))

    transform = transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
        
    return transform(image)


def extract_vae_features(image_path: str, vae, device: torch.device, 
                        target_size: Tuple[int, int]) -> Optional[torch.Tensor]:
    """
    Extract VAE features from image.
    
    Args:
        image_path: Path to image file
        vae: VAE model instance
        device: Computation device
        target_size: Target image size
        
    Returns:
        VAE encoded features or None if failed
    """
    if not os.path.exists(image_path):
        logging.warning(f"Image file not found: {image_path}")
        return None

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform_image_to_tensor(image, target_size)
    image_tensor = image_tensor.unsqueeze(1).to(device)  # [C, 1, H, W]

    with torch.no_grad():
        latent_feature = vae.encode([image_tensor])
        latent_feature = latent_feature[0]
    
    return latent_feature


def read_video_frames(video_path: str, frame_num: int, sampling_rate: int = 3, 
                     skip_num: int = 0, target_size: Tuple[int, int] = (480, 832)) -> Optional[torch.Tensor]:
    """
    Read video frames and convert to tensor.
    
    Args:
        video_path: Path to video file
        frame_num: Number of frames to extract
        sampling_rate: Frame sampling rate
        skip_num: Number of frames to skip at beginning
        target_size: Target frame size (height, width)
        
    Returns:
        Frame tensor [T, C, H, W] or None if failed
    """
    if not CV2_AVAILABLE:
        logging.error("OpenCV not available for video processing")
        return None
        
    if not os.path.exists(video_path):
        logging.warning(f"Video file not found: {video_path}")
        return None
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Cannot open video: {video_path}")
        return None
    
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logging.info(f"Video info: frames={total_frames}, fps={fps}, size={width}x{height}")

        # Adjust sampling rate if needed
        while total_frames < frame_num * sampling_rate + skip_num:
            sampling_rate -= 1
            if sampling_rate == 0:
                logging.warning(f"Cannot extract {frame_num} frames from video")
                return None
                
        logging.info(f"Using sampling rate: {sampling_rate}")

        # Check aspect ratio compatibility
        target_aspect = target_size[1] / target_size[0]  # w/h
        video_aspect = width / height
        
        if abs(target_aspect - video_aspect) > 0.5:  # Significant aspect ratio difference
            logging.warning(f"Aspect ratio mismatch: target={target_aspect:.2f}, video={video_aspect:.2f}")
        
        # Extract frames
        frames = []
        current_frame = 0
        
        while current_frame < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if current_frame < skip_num:
                current_frame += 1
                continue
            
            if (current_frame - skip_num) % sampling_rate == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                
            current_frame += 1
            
            if len(frames) >= frame_num:
                break
        
        if len(frames) != frame_num:
            logging.warning(f"Extracted {len(frames)} frames, expected {frame_num}")
            return None
        
        # Convert to tensor
        frame_tensors = []
        for frame in frames:
            frame_tensor = transform_image_to_tensor(Image.fromarray(frame), target_size)
            frame_tensors.append(frame_tensor)
        
        return torch.stack(frame_tensors)  # [T, C, H, W]
        
    finally:
        cap.release()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Video: Unified Video Generation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Basic arguments
    parser.add_argument(
        "--task", type=str, default="v2v",
        choices=["t2v", "t2i", "i2i", "v2v"],
        help="Generation task type"
    )
    parser.add_argument(
        "--prompt", type=str, required=True,
        help="Text prompt for generation"
    )
    parser.add_argument(
        "--size", type=str, default="832*480",
        help="Output size in format 'width*height'"
    )
    parser.add_argument(
        "--frame_num", type=int, default=None,
        help="Number of frames to generate (should be 4n+1 for videos)"
    )
    parser.add_argument(
        "--sample_fps", type=int, default=8,
        help="FPS of the generated video"
    )
    
    # Model paths (now fixed)
    parser.add_argument(
        "--ckpt_dir", type=str,
        help="Path to the main model checkpoint directory (now fixed)"
    )
    parser.add_argument(
        "--adapter_ckpt_dir", type=str,
        help="Path to the adapter checkpoint (now fixed)"
    )
    parser.add_argument(
        "--vision_head_ckpt_dir", type=str,
        help="Path to the vision head checkpoint (now fixed)"
    )
    parser.add_argument(
        "--new_checkpoint", type=str,
        help="Path to additional checkpoint to load (now fixed)"
    )
    parser.add_argument(
        "--ar_model_path", type=str,
        help="Path to the AR model (now fixed)"
    )
    parser.add_argument(
        "--models_dir", type=str, default=None,
        help="Absolute path to model_ckpts (base dir for wan/adapter/ar_model/...); default: <project_root>/model_ckpts"
    )
    parser.add_argument(
        "--precomputed_pkl", type=str, default=None,
        help="Path to a precomputed .pkl to bypass AR model and use prepared features"
    )
    parser.add_argument(
        "--pkl_list_file", type=str, default=None,
        help="Path to a text file containing multiple PKL file paths (one per line) or JSON file with edit data"
    )
    parser.add_argument(
        "--prompts_dir", type=str, default=None,
        help="Path to directory containing .txt prompt files (processes all .txt files in the directory)"
    )
    
    # Generation parameters
    parser.add_argument(
        "--sample_solver", type=str, default='unipc',
        choices=['unipc', 'dpm++'],
        help="Sampling solver"
    )
    parser.add_argument(
        "--sample_steps", type=int, default=None,
        help="Number of sampling steps"
    )
    parser.add_argument(
        "--sample_shift", type=float, default=None,
        help="Sampling shift factor"
    )
    parser.add_argument(
        "--sample_guide_scale", type=float, default=5.0,
        help="Classifier-free guidance scale"
    )
    parser.add_argument(
        "--base_seed", type=int, default=-1,
        help="Random seed (-1 for random)"
    )
    
    # Input files
    parser.add_argument(
        "--src_file_path", type=str, default=None,
        help="Source image/video path for editing tasks"
    )
    parser.add_argument(
        "--save_file", type=str,
        help="Output file path (auto-generated if not specified)"
    )
    
    # Output directory
    parser.add_argument(
        "--output_dir", type=str, default="./output",
        help="Output directory for generated files (default: ./output)"
    )
    
    # Advanced parameters
    parser.add_argument(
        "--adapter_in_channels", type=int, default=1152,
        help="Adapter input channels"
    )
    parser.add_argument(
        "--adapter_out_channels", type=int, default=4096,
        help="Adapter output channels"
    )
    parser.add_argument(
        "--adapter_query_length", type=int, default=256,
        help="Adapter query length"
    )
    parser.add_argument(
        "--use_visual_context_adapter", type=str2bool, default=False,
        help="Whether to use visual context adapter"
    )
    parser.add_argument(
        "--visual_context_adapter_patch_size", type=str2tuple, default=(1, 4, 4),
        help="Visual context adapter patch size (e.g., '1,4,4')"
    )
    parser.add_argument(
        "--use_visual_as_input", type=str2bool, default=False,
        help="Whether to use visual as input"
    )
    parser.add_argument(
        "--condition_mode", type=str, default="full",
        help="Conditioning mode"
    )
    parser.add_argument(
        "--max_context_len", type=int, default=1024,
        help="Maximum context length"
    )
    parser.add_argument(
        "--wan_inference_activation_checkpoint", type=str2bool, default=False,
        help="Use activation checkpointing on Wan DiT blocks during inference (slower, lower peak activation memory)."
    )
    parser.add_argument(
        "--gamma_w", type=float, default=1.4,
        help="Gamma_w parameter for temporal gating in cascade inference"
    )
    parser.add_argument(
        "--gamma_hf", type=float, default=0.7,
        help="Gamma_hf: scale on 1.3B high-frequency branch in cascade LF/HF fusion"
    )
    parser.add_argument(
        "--sigma_min", type=float, default=0.35,
        help="Cascade LF/HF Gaussian blur sigma lower bound (interpolated with w_t)"
    )
    parser.add_argument(
        "--sigma_max", type=float, default=0.7,
        help="Cascade LF/HF Gaussian blur sigma upper bound (interpolated with w_t)"
    )
    parser.add_argument(
        "--offload_model", type=str2bool, default=False,
        help="After each generate(), move DiT to CPU (saves VRAM; next sample pays full GPU transfer — disable for batch speed)."
    )
    parser.add_argument(
        "--dit_fsdp", type=str2bool, default=False,
        help="Shard 1.3B and 14B Wan DiT with FSDP (torchrun, world_size>1)."
    )
    parser.add_argument(
        "--t5_cpu", type=str2bool, default=True,
        help="Wan2.1-T2V style: keep UMT5 on CPU, move text embeddings to GPU (saves VRAM). "
        "Set false to run T5 on GPU.",
    )
    parser.add_argument(
        "--fast_cuda", type=str2bool, default=True,
        help="Enable cudnn.benchmark, TF32, matmul high precision, SDPA flash (throughput; disable for stricter repro). "
        "Override off with env VIDEO_FAST_CUDA=0."
    )
    
    # Classifier-free guidance
    parser.add_argument(
        "--classifier_free_ratio", type=float, default=0.0,
        help="Classifier-free guidance ratio"
    )
    parser.add_argument(
        "--unconditioned_context_path", type=str,
        help="Path to unconditioned context embeddings (now fixed)"
    )
    parser.add_argument(
        "--unconditioned_context_length", type=int, default=256,
        help="Unconditioned context length"
    )
    parser.add_argument(
        "--special_tokens_path", type=str,
        help="Path to special tokens file (now fixed)"
    )
    
    # AR model parameters
    parser.add_argument(
        "--ar_model_num_video_frames", type=int, default=8,
        help="Number of video frames for AR model"
    )
    parser.add_argument(
        "--ar_query", type=str,
        help="Query for AR model"
    )
    parser.add_argument(
        "--ar_conv_mode", type=str, default="llama_3",
        help="AR model conversation mode"
    )
    
    # Video processing
    parser.add_argument(
        "--sampling_rate", type=int, default=3,
        help="Video sampling rate"
    )
    parser.add_argument(
        "--skip_num", type=int, default=0,
        help="Number of frames to skip"
    )
    
    args = parser.parse_args()
    validate_args(args)
    return args


def read_prompt_list_file(file_path: str) -> list:
    """
    Read text prompts from a text file for T2V generation.
    
    Args:
        file_path: Path to the text file containing prompts (one per line)
        
    Returns:
        List of text prompts
    """
    prompts = []
    
    if not os.path.exists(file_path):
        logging.error(f"Prompt list file not found: {file_path}")
        return prompts
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # Skip empty lines and comment lines
            if not line or line.startswith('#'):
                continue
            
            # Add the prompt to the list
            prompts.append(line)
            logging.info(f"Added prompt {len(prompts)}: {line[:50]}...")
    
    logging.info(f"Loaded {len(prompts)} prompts from {file_path}")
    
    return prompts


def read_prompts_from_directory(prompts_dir: str) -> Dict[str, List[str]]:
    """
    Read prompts from all .txt files in a directory.
    
    Args:
        prompts_dir: Path to the directory containing .txt files
        
    Returns:
        Dictionary mapping filename (without .txt) to list of prompts
    """
    prompts_data = {}
    
    if not os.path.exists(prompts_dir):
        logging.error(f"Prompts directory not found: {prompts_dir}")
        return prompts_data
    
    txt_files = [f for f in os.listdir(prompts_dir) if f.endswith('.txt')]
    if not txt_files:
        logging.warning(f"No .txt files found in directory: {prompts_dir}")
        return prompts_data
    
    logging.info(f"Found {len(txt_files)} .txt files in {prompts_dir}")
    
    for txt_file in txt_files:
        txt_path = os.path.join(prompts_dir, txt_file)
        prompts = read_prompt_list_file(txt_path)
        if prompts:
            file_name = os.path.splitext(txt_file)[0]  # Remove .txt extension
            prompts_data[file_name] = prompts
            logging.info(f"Loaded {len(prompts)} prompts from {txt_file}")
    
    total_prompts = sum(len(prompts) for prompts in prompts_data.values())
    logging.info(f"Total loaded {total_prompts} prompts from {len(prompts_data)} files")
    return prompts_data


def read_pkl_list_file(file_path: str) -> list:
    """
    Read PKL file paths from a text file.
    
    Args:
        file_path: Path to the text file containing PKL paths
        
    Returns:
        List of PKL file paths
    """
    pkl_paths = []
    
    if not os.path.exists(file_path):
        logging.error(f"PKL list file not found: {file_path}")
        return pkl_paths
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # Skip empty lines and comment lines
            if not line or line.startswith('#'):
                continue
            
            # Check if the PKL file exists
            if os.path.exists(line):
                pkl_paths.append(line)
                logging.info(f"Added PKL path: {line}")
            else:
                logging.warning(f"PKL file not found (line {line_num}): {line}")
    
    logging.info(f"Loaded {len(pkl_paths)} PKL files from {file_path}")
    
    return pkl_paths


def read_edit_list_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Read video editing data from a JSON file.
    
    Args:
        file_path: Path to the JSON file containing editing data
        
    Returns:
        List of editing data dictionaries
    """
    edit_data = []
    
    if not os.path.exists(file_path):
        logging.error(f"Edit list file not found: {file_path}")
        return edit_data
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    if not isinstance(data, list):
        logging.error("JSON file should contain a list of editing data")
        return edit_data
        
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            logging.warning(f"Skipping invalid item at index {i}: not a dictionary")
            continue
            
        # Check required fields
        required_fields = ['rewrite_instruction', 'source_video_path']
        missing_fields = [field for field in required_fields if field not in item]
        
        if missing_fields:
            logging.warning(f"Skipping item {i}: missing fields {missing_fields}")
            continue
            
        # Check if source video exists
        source_path = item['source_video_path']
        
        # Handle relative paths - try multiple possible locations
        if not os.path.isabs(source_path):
            # Handle common path mappings
            mapped_path = source_path
            if source_path.startswith("source_zip/videos/"):
                # Map source_zip/videos/ to source_videos/
                mapped_path = source_path.replace("source_zip/videos/", "source_videos/")
            
            possible_paths = [
                source_path,  # Try as-is first
                mapped_path,  # Try mapped path
                os.path.join(os.path.dirname(file_path), source_path),  # Same dir as JSON
                os.path.join(os.path.dirname(file_path), mapped_path),  # Same dir as JSON with mapping
                os.path.join(os.path.dirname(file_path), "..", source_path),  # Parent dir
                os.path.join(os.path.dirname(file_path), "..", mapped_path),  # Parent dir with mapping
                os.path.join(os.path.dirname(file_path), "..", "..", source_path),  # Grandparent dir
                os.path.join(os.path.dirname(file_path), "..", "..", mapped_path),  # Grandparent dir with mapping
            ]
            
            found_path = None
            for possible_path in possible_paths:
                if os.path.exists(possible_path):
                    found_path = possible_path
                    break
            
            if found_path:
                source_path = found_path
                logging.info(f"Found source video at: {source_path}")
            else:
                logging.warning(f"Skipping item {i}: source video not found: {item['source_video_path']}")
                logging.info(f"Tried paths: {possible_paths}")
                continue
        else:
            # Absolute path
            if not os.path.exists(source_path):
                logging.warning(f"Skipping item {i}: source video not found: {source_path}")
                continue
            
        # Update the source path in the item
        item_copy = item.copy()
        item_copy['source_video_path'] = source_path
        edit_data.append(item_copy)
        logging.info(f"Added edit data {i+1}: {source_path}")

    logging.info(f"Loaded {len(edit_data)} valid editing tasks")
    return edit_data


def run_single_generation(generator, args, target_size, save_auxiliary=True, auxiliary_dir: Optional[str]=None):
    """Run single generation with specified parameters."""
    rank = int(os.getenv("RANK", 0))
    
    prompt = args.prompt.strip()
    src_file_path = args.src_file_path
    if src_file_path and not os.path.exists(src_file_path):
        logging.error(f"Source file not found: {src_file_path}")
        return False

    # Optional: bypass AR model using precomputed features from .pkl
    use_precomputed = args.precomputed_pkl is not None and os.path.exists(args.precomputed_pkl)
    pre_vlm = None
    pre_visual = None
    task = args.task
    if use_precomputed:
        logging.info(f"Loading precomputed features from: {args.precomputed_pkl}")
        with open(args.precomputed_pkl, 'rb') as f:
            pstate = pkl.load(f)
        # Try common keys
        if isinstance(pstate, dict):
            if 'vlm_last_hidden_states' in pstate:
                pre_vlm = pstate['vlm_last_hidden_states']
            if 'latent_feature' in pstate:
                pre_visual = pstate['latent_feature']
            if 'prompt' in pstate:
                prompt = str(pstate['prompt'])
            if args.frame_num is None and 'frame_num' in pstate:
                args.frame_num = int(pstate['frame_num'])
            task = 't2v'

            # Save auxiliary artifacts from pstate (only if save_auxiliary is True)
            if save_auxiliary:
                base_dir_for_aux = auxiliary_dir if auxiliary_dir else args.output_dir
                pre_dir = os.path.join(base_dir_for_aux, "precomputed")
                os.makedirs(pre_dir, exist_ok=True)

                # Save prompt
                if 'prompt' in pstate:
                    prompt_txt = os.path.join(pre_dir, "prompt.txt")
                    with open(prompt_txt, 'w', encoding='utf-8') as pf:
                        pf.write(str(pstate['prompt']))
                    logging.info(f"Saved prompt to {prompt_txt}")

                # Decode latent_feature_tgt to video if provided
                if 'latent_feature_tgt' in pstate and pstate['latent_feature_tgt'] is not None:
                    lf = pstate['latent_feature_tgt']
                    
                    # Try to decode latent_feature_tgt to video and save as mp4
                    lf_tensor: Optional[torch.Tensor] = None
                    if isinstance(lf, torch.Tensor):
                        lf_tensor = lf
                    elif isinstance(lf, np.ndarray):
                        lf_tensor = torch.tensor(lf)
                    elif isinstance(lf, (list, tuple)):
                        lf_tensor = torch.tensor(np.array(lf))

                    if lf_tensor is not None:
                        # Move to device and (optionally) dtype similar to generation path
                        lf_tensor = lf_tensor.to(generator.device)
                        with torch.no_grad():
                            videos_decoded = generator.video_x2x.vae.decode([lf_tensor])
                        if isinstance(videos_decoded, (list, tuple)) and len(videos_decoded) > 0:
                            video_tensor = videos_decoded[0]
                        else:
                            video_tensor = videos_decoded

                        decoded_out = os.path.join(pre_dir, "latent_feature_tgt_decoded.mp4")
                        cache_video(
                            tensor=video_tensor[None],
                            save_file=decoded_out,
                            fps=args.sample_fps,
                            nrow=1,
                            normalize=True,
                            value_range=(-1, 1)
                        )
                        logging.info(f"Saved decoded latent video to {decoded_out}")
                    else:
                        logging.warning("latent_feature_tgt cannot be converted to tensor for decoding.")
        else:
            logging.warning("Precomputed pkl is not a dict, skipping field extraction")

    if not use_precomputed:
        # For T2V tasks without source file, use the prompt directly
        if not src_file_path and task in ['t2v', 't2i']:
            logging.info(f"T2V/T2I task without source file, using prompt: {prompt}")
            # Use a simple AR model query for T2V/T2I tasks
            ar_query = args.ar_query or "<video>\n Generate a video based on the following description."
            # Force task type based on command line argument for T2V tasks
            if args.task == 't2v':
                task = 't2v'
                logging.info("Forcing task to T2V based on command line argument")
            elif args.task == 't2i':
                task = 't2i'
                logging.info("Forcing task to T2I based on command line argument")
        else:
            # Get AR model predictions for tasks with source files
            ar_query = args.ar_query or "<video>\n Describe this video and its style in a very detailed manner."
            ar_caption_ids, ar_caption = generator.ar_model.generate(src_file_path, prompt)
                
            logging.info(f"task id: {ar_caption_ids}")
            # Determine task based on AR model output, this is a simple version of task detection
            gen_mode_1 = torch.any(ar_caption_ids == 128003)
            gen_mode_2 = torch.any(ar_caption_ids == 128002)
                
            if not gen_mode_1 and not gen_mode_2:
                logging.info("OmniVideo model did not suggest any generation task, it is a understanding task")
                print(f"OmniVideo model output: {ar_caption}")
                return False
                
            # Set task type based on AR model output (overrides command line task)
            if (gen_mode_1 or gen_mode_2) and src_file_path:
                if src_file_path.endswith('.png') or src_file_path.endswith('.jpg'):
                    task = 'i2i'
                elif src_file_path.endswith('.mp4'):
                    task = 'v2v'
                else:
                    logging.error("Source file type not supported, currently only support png, jpg and mp4")
                    return False
            elif gen_mode_1:
                task = 't2v'
            elif gen_mode_2:
                task = 't2i'
      
    logging.info(f"Determined task: {task} (overriding command line task: {args.task})")
    
    # Update frame_num based on detected task
    if task in ['t2i', 'i2i'] and args.frame_num > 1:
        args.frame_num = 1
        logging.info("Updated frame_num to 1 for image generation task")
    elif task in ['t2v', 'v2v'] and args.frame_num == 1:
        args.frame_num = 81
        logging.info("Updated frame_num to 81 for video generation task")
    
    # Process visual input if needed
    visual_emb = None
    if task == 'i2i' and src_file_path:
        visual_emb = extract_vae_features(
            src_file_path, generator.video_x2x.vae, 
            generator.device, (target_size[1], target_size[0])
        )
        if visual_emb is not None:
            visual_emb = visual_emb[:, 0:1]
                
    elif task == 'v2v' and src_file_path:
        frames_tensor = read_video_frames(
            src_file_path, args.frame_num, args.sampling_rate, 
            args.skip_num, (target_size[1], target_size[0])
        )
        if frames_tensor is not None:
            frames_tensor = frames_tensor.to(generator.device)
            with torch.no_grad():
                visual_emb = generator.video_x2x.vae.encode(
                    frames_tensor.transpose(0,1).unsqueeze(0)
                )[0]
    
    # Generate embeddings or use precomputed
    if use_precomputed and pre_vlm is not None:
        vlm_last_hidden_states = pre_vlm.to(generator.device)
        if not isinstance(vlm_last_hidden_states, torch.Tensor):
            vlm_last_hidden_states = torch.tensor(vlm_last_hidden_states)
    if not use_precomputed:
        vlm_last_hidden_states = generator.ar_model.general_emb(
            prompt, src_file_path, task_type=task
        )

    # logging for input shape
    logging.info(f"Input shape: {vlm_last_hidden_states.shape}")
    logging.info(f"Visual emb shape: {visual_emb.shape if visual_emb is not None else None}")
    logging.info(f"Task: {task}")
    logging.info(f"Prompt: {prompt}")
    logging.info(f"Src file path: {src_file_path if src_file_path is not None else None}")
    logging.info(f"Size: {target_size}")
    logging.info(f"Frame num: {args.frame_num}")
    # Generate content
    logging.info(f"GPU {rank}: Starting generation...")
    result = generator.video_x2x.generate(
        prompt,
        visual_emb=visual_emb,
        ar_vision_input=vlm_last_hidden_states,
        size=(target_size[0], target_size[1]),  # (width, height)
        frame_num=args.frame_num,
        shift=args.sample_shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.sample_steps,
        guide_scale=args.sample_guide_scale,
        seed=args.base_seed,
        offload_model=args.offload_model,
        special_tokens=generator.special_tokens,
        classifier_free_ratio=args.classifier_free_ratio,
        unconditioned_context=generator.unconditioned_context,
        condition_mode=args.condition_mode,
        use_visual_as_input=args.use_visual_as_input,
        gamma_w=args.gamma_w,
        gamma_hf=args.gamma_hf,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
    )
    
    logging.info(f"GPU {rank}: Generation completed, result type: {type(result)}")
    if result is not None:
        logging.info(f"GPU {rank}: Result shape: {result.shape}")
    else:
        logging.warning(f"GPU {rank}: Generation failed - no output produced")
        return False
            
    # Create video-specific directory and save result
    if not args.save_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_prompt = safe_prompt.replace(' ', '_')[:50]
        
        # Extract source filename if available
        if src_file_path:
            source_filename = os.path.splitext(os.path.basename(src_file_path))[0]
            video_dir_name = f"{source_filename}_{task}_{timestamp}"
        else:
            video_dir_name = f"{task}_{safe_prompt}_{timestamp}"
        
        # Check if we have a prompt file name from batch processing
        prompt_file_name = getattr(args, 'prompt_file_name', None)
        if prompt_file_name:
            # Create prompt file directory first
            prompt_file_dir = os.path.join(args.output_dir, prompt_file_name)
            os.makedirs(prompt_file_dir, exist_ok=True)
            # Create video-specific directory under prompt file directory
            video_output_dir = os.path.join(prompt_file_dir, video_dir_name)
        else:
            # Create video-specific directory directly under output directory
            video_output_dir = os.path.join(args.output_dir, video_dir_name)
        
        os.makedirs(video_output_dir, exist_ok=True)
        logging.info(f"GPU {rank}: Created video directory: {video_output_dir}")
        
        # Save prompt to video directory
        prompt_file = os.path.join(video_output_dir, "prompt.txt")
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(prompt)
        logging.info(f"GPU {rank}: Saved prompt to: {prompt_file}")
        
        # Generate filename for video/image
        suffix = '.png' if '2i' in task else '.mp4'
        rank_suffix = f"_rank{rank}" if generator.world_size > 1 else ""
            
        filename = (f"{task}_{args.size}_{args.sample_solver}_seed{args.base_seed}_"
                   f"cfg{args.sample_guide_scale}_steps{args.sample_steps}_"
                   f"frames{args.frame_num}{rank_suffix}{suffix}")
            
        args.save_file = os.path.join(video_output_dir, filename)
    
    # Save based on task type
    if args.frame_num == 1:
        logging.info(f"GPU {rank}: Saving image to {args.save_file}")
        cache_image(
            tensor=result.squeeze(1)[None],
            save_file=args.save_file,
            nrow=1,
            normalize=True,
            value_range=(-1, 1)
        )
    else:
        logging.info(f"GPU {rank}: Saving video to {args.save_file}")
        cache_video(
            tensor=result[None],
            save_file=args.save_file,
            fps=args.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1)
        )
    
    logging.info(f"GPU {rank}: Generation completed successfully: {args.save_file}")
    return True


def run_single_edit(generator, args, edit_item, target_size, save_auxiliary=True, auxiliary_dir=None):
    """Run single video editing task."""
    rank = int(os.getenv("RANK", 0))
    
    # Extract data from edit item
    prompt = edit_item['rewrite_instruction']
    source_video_path = edit_item['source_video_path']
    
    logging.info(f"GPU {rank}: Processing edit task")
    logging.info(f"GPU {rank}: Prompt: {prompt}")
    logging.info(f"GPU {rank}: Source video: {source_video_path}")
        
    # Determine task type based on source file
    task = 'v2v'  # Default for video editing
    if source_video_path:
        if source_video_path.endswith('.png') or source_video_path.endswith('.jpg'):
            task = 'i2i'
        elif source_video_path.endswith('.mp4'):
            task = 'v2v'
        else:
            logging.error("Source file type not supported, currently only support png, jpg and mp4")
            return False
        
    logging.info(f"GPU {rank}: Determined task: {task}")
        
    # Process visual input if needed
    visual_emb = None
    if task == 'i2i' and source_video_path:
        visual_emb = extract_vae_features(
            source_video_path, generator.video_x2x.vae, 
            generator.device, (target_size[1], target_size[0])
        )
        if visual_emb is not None:
            visual_emb = visual_emb[:, 0:1]
                
    elif task == 'v2v' and source_video_path:
        frames_tensor = read_video_frames(
            source_video_path, args.frame_num, args.sampling_rate, 
            args.skip_num, (target_size[1], target_size[0])
        )
        if frames_tensor is not None:
            frames_tensor = frames_tensor.to(generator.device)
            with torch.no_grad():
                visual_emb = generator.video_x2x.vae.encode(
                    frames_tensor.transpose(0,1).unsqueeze(0)
                )[0]
        
    # Generate AR model embeddings
    vlm_last_hidden_states = generator.ar_model.general_emb(
        prompt, source_video_path, task_type=task
    )
    # Log input shapes
    logging.info(f"GPU {rank}: Input shape: {vlm_last_hidden_states.shape}")
    logging.info(f"GPU {rank}: Visual emb shape: {visual_emb.shape if visual_emb is not None else None}")
    logging.info(f"GPU {rank}: Task: {task}")
    logging.info(f"GPU {rank}: Prompt: {prompt}")
    logging.info(f"GPU {rank}: Source file path: {source_video_path}")
    logging.info(f"GPU {rank}: Size: {target_size}")
    logging.info(f"GPU {rank}: Frame num: {args.frame_num}")
    # Generate video using video_x2x
    logging.info(f"GPU {rank}: Starting generation...")
    result = generator.video_x2x.generate(
        prompt,
        visual_emb=visual_emb,
        ar_vision_input=vlm_last_hidden_states,
        size=(target_size[0], target_size[1]),  # (width, height)
        frame_num=args.frame_num,
        shift=args.sample_shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.sample_steps,
        guide_scale=args.sample_guide_scale,
        n_prompt="",
        seed=args.base_seed,
        offload_model=args.offload_model,
        special_tokens=generator.special_tokens,
        classifier_free_ratio=args.classifier_free_ratio,
        unconditioned_context=generator.unconditioned_context,
        condition_mode=args.condition_mode,
        use_visual_as_input=args.use_visual_as_input,
        gamma_w=args.gamma_w,
        gamma_hf=args.gamma_hf,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
    )
        
    if result is None:
        logging.error(f"GPU {rank}: Generation failed")
        return False
            
    # Save auxiliary artifacts if requested
    if save_auxiliary:
        base_dir_for_aux = auxiliary_dir if auxiliary_dir else args.output_dir
        pre_dir = os.path.join(base_dir_for_aux, "precomputed")
        os.makedirs(pre_dir, exist_ok=True)
        
        # Save prompt
        prompt_file = os.path.join(pre_dir, "prompt.txt")
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(f"Original instruction: {edit_item.get('instruction', 'N/A')}\n")
            f.write(f"Rewrite instruction: {prompt}\n")
            f.write(f"Source video: {source_video_path}\n")
            f.write(f"Task type: {edit_item.get('type', 'N/A')}\n")
        
        logging.info(f"GPU {rank}: Saved auxiliary files to {pre_dir}")
        
    # Create video-specific directory and generate output filename
    if not args.save_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_prompt = safe_prompt.replace(' ', '_')[:50]
        
        # Extract source video filename without extension
        source_filename = os.path.splitext(os.path.basename(source_video_path))[0]
        
        # Create video-specific directory
        video_dir_name = f"{source_filename}_{task}_{timestamp}"
        
        # Check if we have a prompt file name from batch processing
        prompt_file_name = getattr(args, 'prompt_file_name', None)
        if prompt_file_name:
            # Create prompt file directory first
            prompt_file_dir = os.path.join(args.output_dir, prompt_file_name)
            os.makedirs(prompt_file_dir, exist_ok=True)
            # Create video-specific directory under prompt file directory
            video_output_dir = os.path.join(prompt_file_dir, video_dir_name)
        else:
            # Create video-specific directory directly under output directory
            video_output_dir = os.path.join(args.output_dir, video_dir_name)
        
        os.makedirs(video_output_dir, exist_ok=True)
        logging.info(f"GPU {rank}: Created video directory: {video_output_dir}")
        
        # Save prompt to video directory
        prompt_file = os.path.join(video_output_dir, "prompt.txt")
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(f"Original instruction: {edit_item.get('instruction', 'N/A')}\n")
            f.write(f"Rewrite instruction: {prompt}\n")
            f.write(f"Source video: {source_video_path}\n")
            f.write(f"Task type: {edit_item.get('type', 'N/A')}\n")
        logging.info(f"GPU {rank}: Saved prompt to: {prompt_file}")
        
        # Generate filename for video
        suffix = '.mp4'
        rank_suffix = f"_rank{rank}" if generator.world_size > 1 else ""
        
        filename = (f"{task}_{args.size}_{args.sample_solver}_seed{args.base_seed}_"
                   f"cfg{args.sample_guide_scale}_steps{args.sample_steps}_"
                   f"frames{args.frame_num}{rank_suffix}{suffix}")
        
        args.save_file = os.path.join(video_output_dir, filename)
        
    # Save result
    logging.info(f"GPU {rank}: Saving video to {args.save_file}")
    cache_video(
        tensor=result[None],
        save_file=args.save_file,
        fps=args.sample_fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1)
    )
    
    logging.info(f"GPU {rank}: Edit completed successfully: {args.save_file}")
    return True


def run_batch_directory_generation(generator, args, prompts_data, target_size):
    """Run batch generation with prompts from multiple files in a directory."""
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    
    if rank == 0:
        logging.info(f"=== Starting Batch Directory Generation ===")
        logging.info(f"Processing {len(prompts_data)} prompt files")
        total_prompts = sum(len(prompts) for prompts in prompts_data.values())
        logging.info(f"Total prompts: {total_prompts}")
        logging.info(f"Using {world_size} GPU(s)")
        logging.info(f"Output directory: {args.output_dir}")
        logging.info(f"Task: {args.task}")
        logging.info(f"Frame num: {args.frame_num}")
        logging.info(f"Size: {target_size}")
        logging.info("=" * 50)
    
    # Create a flat list of (file_name, prompt) tuples for distribution
    all_prompt_items = []
    for file_name, prompts in prompts_data.items():
        for prompt in prompts:
            all_prompt_items.append((file_name, prompt))
    
    # Distribute prompt items across GPUs
    items_per_gpu = [[] for _ in range(world_size)]
    for i, item in enumerate(all_prompt_items):
        gpu_id = i % world_size
        items_per_gpu[gpu_id].append(item)
    
    # Each GPU processes its assigned prompt items
    my_prompt_items = items_per_gpu[rank]
    
    logging.info(f"GPU {rank}: Assigned {len(my_prompt_items)} prompt items")
    for j, (file_name, prompt) in enumerate(my_prompt_items):
        logging.info(f"  GPU {rank} Item {j+1}: [{file_name}] {prompt[:50]}...")
    
    # Check load balancing
    if rank == 0:
        for gpu_id in range(world_size):
            logging.info(f"Load balance check - GPU {gpu_id}: {len(items_per_gpu[gpu_id])} items")
    
    logging.info(f"GPU {rank}: Processing {len(my_prompt_items)} prompt items")
    
    # Calculate starting global index for each GPU
    start_global_index = sum(len(items_per_gpu[j]) for j in range(rank))
    logging.info(f"GPU {rank}: Starting global index: {start_global_index}")
    
    # Synchronize all GPUs
    if dist.is_initialized():
        logging.info(f"GPU {rank}: Synchronizing prompt distribution...")
        try:
            safe_distributed_barrier(timeout_minutes=30)
            logging.info(f"GPU {rank}: Prompt distribution synchronized")
        except Exception as e:
            logging.error(f"GPU {rank}: Synchronization failed: {e}")
            return False
    
    results = {}
    
    for i, (file_name, prompt) in enumerate(my_prompt_items):
        logging.info(f"GPU {rank}: Processing item {i+1}/{len(my_prompt_items)}: [{file_name}] {prompt[:50]}...")
        
        # Temporarily modify args for this prompt
        original_output_dir = args.output_dir
        original_prompt = args.prompt
        original_save_file = getattr(args, 'save_file', None)
        original_prompt_file_name = getattr(args, 'prompt_file_name', None)
        
        args.output_dir = args.output_dir  # Keep the base output directory
        args.prompt = prompt
        args.prompt_file_name = file_name  # Set the prompt file name
        args.save_file = None
        
        # Run single generation for this prompt (will create its own directory under prompt file directory)
        success = run_single_generation(generator, args, target_size, save_auxiliary=True)
        
        results[f"{file_name}_{prompt[:30]}"] = success
        
        status = "✅ Success" if success else "❌ Failed"
        logging.info(f"GPU {rank}: Item {i+1}/{len(my_prompt_items)} ([{file_name}] {prompt[:30]}...): {status}")
        
        # Restore original args
        args.output_dir = original_output_dir
        args.prompt = original_prompt
        args.save_file = original_save_file
        args.prompt_file_name = original_prompt_file_name

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Print summary for each GPU
    logging.info("=" * 50)
    logging.info(f"=== GPU {rank} Batch Directory Generation Summary ===")
    for i, (item_key, success) in enumerate(results.items(), 1):
        status = "✅ Success" if success else "❌ Failed"
        logging.info(f"Item {i:3d}: {status} - {item_key}")
    
    successful_items = [item for item, success in results.items() if success]
    failed_items = [item for item, success in results.items() if not success]
    
    logging.info(f"GPU {rank}: Successful: {len(successful_items)}/{len(my_prompt_items)} items")
    if successful_items:
        logging.info(f"GPU {rank}: Successful items: {len(successful_items)}")
    if failed_items:
        logging.info(f"GPU {rank}: Failed items: {len(failed_items)}")
    
    logging.info(f"GPU {rank}: Results saved in: {args.output_dir}")
    logging.info("=" * 50)
    
    # Synchronize all GPUs before returning
    if dist.is_initialized():
        logging.info(f"GPU {rank}: Waiting for all GPUs to complete...")
        dist.barrier()
        logging.info(f"GPU {rank}: All GPUs synchronized")
    
    return len(successful_items) > 0


def run_batch_prompt_generation(generator, args, prompts, target_size):
    """Run batch generation with multiple text prompts."""
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    
    if rank == 0:
        logging.info(f"=== Starting Batch Prompt Generation ===")
        logging.info(f"Processing {len(prompts)} text prompts")
        logging.info(f"Using {world_size} GPU(s)")
        logging.info(f"Output directory: {args.output_dir}")
        logging.info(f"Task: {args.task}")
        logging.info(f"Frame num: {args.frame_num}")
        logging.info(f"Size: {target_size}")
        logging.info("=" * 50)
    
    # Distribute prompts across GPUs
    prompts_per_gpu = [[] for _ in range(world_size)]
    for i, prompt in enumerate(prompts):
        gpu_id = i % world_size
        prompts_per_gpu[gpu_id].append(prompt)
    
    # Each GPU processes its assigned prompts
    my_prompts = prompts_per_gpu[rank]

    logging.info(f"GPU {rank}: Assigned {len(my_prompts)} prompts")
    for j, prompt in enumerate(my_prompts):
        logging.info(f"  GPU {rank} Prompt {j+1}: {prompt[:50]}...")
    
    logging.info(f"GPU {rank}: Processing {len(my_prompts)} prompts")

    start_global_index = sum(len(prompts_per_gpu[j]) for j in range(rank))
    logging.info(f"GPU {rank}: Starting global index: {start_global_index}")

    if dist.is_initialized():
        logging.info(f"GPU {rank}: Synchronizing prompt distribution...")
        dist.barrier()
        logging.info(f"GPU {rank}: Prompt distribution synchronized")
    
    results = {}
    
    for i, prompt in enumerate(my_prompts):
        logging.info(f"GPU {rank}: Processing prompt {i+1}/{len(my_prompts)}: {prompt[:50]}...")
        
        # Temporarily modify args for this prompt
        original_output_dir = args.output_dir
        original_prompt = args.prompt
        original_save_file = getattr(args, 'save_file', None)
        args.output_dir = args.output_dir  # Keep the base output directory
        args.prompt = prompt
        args.save_file = None
        
        # Run single generation for this prompt (will create its own directory)
        success = run_single_generation(generator, args, target_size, save_auxiliary=True)
            
        results[prompt] = success

        status = "✅ Success" if success else "❌ Failed"
        logging.info(f"GPU {rank}: Prompt {i+1}/{len(my_prompts)} ({prompt[:30]}...): {status}")
        
        # Restore original args
        args.output_dir = original_output_dir
        args.prompt = original_prompt
        args.save_file = original_save_file

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Print summary for each GPU
    logging.info("=" * 50)
    logging.info(f"=== GPU {rank} Batch Prompt Generation Summary ===")
    for i, (prompt, success) in enumerate(results.items(), 1):
        status = "✅ Success" if success else "❌ Failed"
        logging.info(f"Prompt {i:3d}: {status} - {prompt[:50]}...")
    
    successful_prompts = [p for p, success in results.items() if success]
    failed_prompts = [p for p, success in results.items() if not success]
    
    logging.info(f"GPU {rank}: Successful: {len(successful_prompts)}/{len(my_prompts)} prompts")
    if successful_prompts:
        logging.info(f"GPU {rank}: Successful prompts: {len(successful_prompts)}")
    if failed_prompts:
        logging.info(f"GPU {rank}: Failed prompts: {len(failed_prompts)}")
    
    logging.info(f"GPU {rank}: Results saved in: {args.output_dir}")
    logging.info("=" * 50)
    
    # Synchronize all GPUs before returning
    if dist.is_initialized():
        logging.info(f"GPU {rank}: Waiting for all GPUs to complete...")
        dist.barrier()
        logging.info(f"GPU {rank}: All GPUs synchronized")
    
    return len(successful_prompts) > 0


def run_batch_pkl_generation(generator, args, pkl_paths, target_size):
    """Run batch generation with multiple PKL files."""
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    
    if rank == 0:
        logging.info(f"=== Starting Batch PKL Generation ===")
        logging.info(f"Processing {len(pkl_paths)} PKL files")
        logging.info(f"Using {world_size} GPU(s)")
        logging.info(f"Output directory: {args.output_dir}")
        logging.info("=" * 50)
    
    # Distribute PKL files across GPUs
    pkl_paths_per_gpu = [[] for _ in range(world_size)]
    for i, pkl_path in enumerate(pkl_paths):
        gpu_id = i % world_size
        pkl_paths_per_gpu[gpu_id].append(pkl_path)
    
    # Each GPU processes its assigned PKL files
    my_pkl_paths = pkl_paths_per_gpu[rank]

    logging.info(f"GPU {rank}: Assigned {len(my_pkl_paths)} PKL files")
    for j, pkl_path in enumerate(my_pkl_paths):
        logging.info(f"  GPU {rank} PKL {j+1}: {os.path.basename(pkl_path)}")
    
    logging.info(f"GPU {rank}: Processing {len(my_pkl_paths)} PKL files")

    start_global_index = sum(len(pkl_paths_per_gpu[j]) for j in range(rank))
    logging.info(f"GPU {rank}: Starting global index: {start_global_index}")

    if dist.is_initialized():
        logging.info(f"GPU {rank}: Synchronizing PKL distribution...")
        try:
            safe_distributed_barrier(timeout_minutes=30)
            logging.info(f"GPU {rank}: PKL distribution synchronized")
        except Exception as e:
            logging.error(f"GPU {rank}: PKL synchronization failed: {e}")
            return False
    
    results = {}
    
    for i, pkl_path in enumerate(my_pkl_paths):
        logging.info(f"GPU {rank}: Processing PKL file {i+1}/{len(my_pkl_paths)}: {os.path.basename(pkl_path)}")
        
        # Create PKL-specific output directory
        pkl_name = os.path.splitext(os.path.basename(pkl_path))[0]
        # Use correct global index for consistent naming across GPUs
        global_index = start_global_index + i + 1
        pkl_output_dir = os.path.join(args.output_dir, pkl_name)
        os.makedirs(pkl_output_dir, exist_ok=True)
        logging.info(f"GPU {rank}: Created output directory: {pkl_output_dir}")
        
        # Temporarily modify args for this PKL file
        original_output_dir = args.output_dir
        original_precomputed_pkl = args.precomputed_pkl
        original_save_file = getattr(args, 'save_file', None)
        args.output_dir = pkl_output_dir
        args.precomputed_pkl = pkl_path
        # Ensure a fresh filename is generated under the PKL-specific directory
        args.save_file = None
        
        # Save auxiliary files for each PKL file (including prompt)
        # Each PKL file should save its own prompt and auxiliary data
        save_auxiliary = True
        
        # Run single generation for this PKL
        success = run_single_generation(generator, args, target_size, save_auxiliary)
            
        results[pkl_path] = success

        status = "✅ Success" if success else "❌ Failed"
        logging.info(f"GPU {rank}: PKL {i+1}/{len(my_pkl_paths)} ({os.path.basename(pkl_path)}): {status}")
        
        # Restore original args
        args.output_dir = original_output_dir
        args.precomputed_pkl = original_precomputed_pkl
        args.save_file = original_save_file

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Print summary for each GPU
    logging.info("=" * 50)
    logging.info(f"=== GPU {rank} Batch PKL Generation Summary ===")
    for i, (pkl_path, success) in enumerate(results.items(), 1):
        status = "✅ Success" if success else "❌ Failed"
        logging.info(f"PKL {i:3d}: {status} - {os.path.basename(pkl_path)}")
    
    successful_pkls = [p for p, success in results.items() if success]
    failed_pkls = [p for p, success in results.items() if not success]
    
    logging.info(f"GPU {rank}: Successful: {len(successful_pkls)}/{len(my_pkl_paths)} PKL files")
    if successful_pkls:
        logging.info(f"GPU {rank}: Successful PKL files: {len(successful_pkls)}")
    if failed_pkls:
        logging.info(f"GPU {rank}: Failed PKL files: {len(failed_pkls)}")
    
    logging.info(f"GPU {rank}: Results saved in: {args.output_dir}")
    logging.info("=" * 50)
    
    # Synchronize all GPUs before returning
    if dist.is_initialized():
        logging.info(f"GPU {rank}: Waiting for all GPUs to complete...")
        dist.barrier()
        logging.info(f"GPU {rank}: All GPUs synchronized")
    
    return len(successful_pkls) > 0


def run_batch_edit_generation(generator, args, edit_data, target_size):
    """Run batch video editing generation."""
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    
    if rank == 0:
        logging.info(f"=== Starting Batch Video Edit Generation ===")
        logging.info(f"Processing {len(edit_data)} edit tasks")
        logging.info(f"Using {world_size} GPU(s)")
        logging.info(f"Output directory: {args.output_dir}")
        logging.info("=" * 50)
    
    # Distribute edit tasks across GPUs
    tasks_per_gpu = len(edit_data) // world_size
    remainder = len(edit_data) % world_size
    
    start_idx = rank * tasks_per_gpu + min(rank, remainder)
    end_idx = start_idx + tasks_per_gpu + (1 if rank < remainder else 0)
    
    my_edit_tasks = edit_data[start_idx:end_idx]
    
    if rank == 0:
        logging.info(f"Distributed {len(edit_data)} tasks across {world_size} GPUs")
        for i in range(world_size):
            gpu_start = i * tasks_per_gpu + min(i, remainder)
            gpu_end = gpu_start + tasks_per_gpu + (1 if i < remainder else 0)
            gpu_tasks = edit_data[gpu_start:gpu_end]
            logging.info(f"GPU {i}: {len(gpu_tasks)} tasks (indices {gpu_start}-{gpu_end-1})")
    
    results = {}
    successful_tasks = []
    failed_tasks = []
    
    for i, edit_item in enumerate(my_edit_tasks):
        # Create task-specific output directory using source video filename
        source_video_path = edit_item['source_video_path']
        source_filename = os.path.splitext(os.path.basename(source_video_path))[0]
        task_name = f"edit_{source_filename}"
        task_output_dir = os.path.join(args.output_dir, task_name)
        os.makedirs(task_output_dir, exist_ok=True)
        logging.info(f"GPU {rank}: Created output directory: {task_output_dir}")
        
        # Temporarily modify args for this task
        original_output_dir = args.output_dir
        original_save_file = getattr(args, 'save_file', None)
        args.output_dir = task_output_dir
        args.save_file = None
        
        # Run single edit for this task
        success = run_single_edit(generator, args, edit_item, target_size, save_auxiliary=True)
        results[task_name] = success
        
        if success:
            successful_tasks.append(task_name)
        else:
            failed_tasks.append(task_name)
        
        # Restore original args
        args.output_dir = original_output_dir
        args.save_file = original_save_file

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Print summary for each GPU
    logging.info("=" * 50)
    logging.info(f"GPU {rank}: Batch edit generation completed")
    logging.info(f"GPU {rank}: Successful tasks: {len(successful_tasks)}")
    logging.info(f"GPU {rank}: Failed tasks: {len(failed_tasks)}")
    
    if successful_tasks:
        logging.info(f"GPU {rank}: Successful tasks: {successful_tasks}")
    if failed_tasks:
        logging.info(f"GPU {rank}: Failed tasks: {failed_tasks}")
    
    logging.info(f"GPU {rank}: Results saved in: {args.output_dir}")
    logging.info("=" * 50)
    
    # Synchronize all GPUs before returning
    if dist.is_initialized():
        dist.barrier()
        logging.info(f"GPU {rank}: All GPUs synchronized")
    
    return len(successful_tasks) > 0


def main():
    """Main function for video generation."""
    args = parse_args()
    
    # Initialize distributed environment
    rank = int(os.getenv("RANK", 0))
    init_logging(rank)
    
    try:
        # Create generator instance
        generator = VideoGenerator(args)
        generator.setup_distributed()
        
        # Synchronize random seed across processes
        if dist.is_initialized():
            base_seed = [args.base_seed] if rank == 0 else [None]
            dist.broadcast_object_list(base_seed, src=0)
            args.base_seed = base_seed[0] + rank * 1000
        
        # Load models and data
        generator.load_special_tokens()
        generator.load_unconditioned_context()
        generator.initialize_models()
        
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"GPU {rank}: Output directory: {output_dir}")

        if dist.is_initialized():
            dist.barrier()
            logging.info(f"GPU {rank}: Output directory synchronized")
            
        # Generate content
        logging.info(f"Starting generation with prompt: '{args.prompt}'")
        
        # Handle size configuration
        if args.size in SIZE_CONFIGS:
            target_size = SIZE_CONFIGS[args.size]
        else:
            # Parse size string like "832*480"
            w, h = args.size.split('*')
            target_size = (int(h), int(w))  # (height, width)
        logging.info(f"Target size: {target_size}")
        
        # Handle prompts directory processing
        if args.prompts_dir:
            prompts_data = read_prompts_from_directory(args.prompts_dir)
            if not prompts_data:
                logging.error("No valid prompt files found in the prompts directory")
                return False
            
            logging.info(f"Prompts directory processing enabled with {len(prompts_data)} files")
            logging.info(f"Task type: {args.task}")
            logging.info(f"Frame num: {args.frame_num}")
            logging.info(f"Size: {target_size}")
            return run_batch_directory_generation(generator, args, prompts_data, target_size)
        
        # Handle PKL list file processing
        elif args.pkl_list_file:
            # Check if it's a JSON file (edit data) or text file (PKL paths)
            if args.pkl_list_file.endswith('.json'):
                # Handle JSON edit data file
                edit_data = read_edit_list_file(args.pkl_list_file)
                if not edit_data:
                    logging.error("No valid edit tasks found in the JSON file")
                    return False
                
                logging.info(f"Edit list file processing enabled with {len(edit_data)} tasks")
                return run_batch_edit_generation(generator, args, edit_data, target_size)
            else:
                # Handle text file - check if it contains PKL paths or text prompts
                with open(args.pkl_list_file, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                
                # Check if first line looks like a file path (contains .pkl)
                if first_line.endswith('.pkl') or '/' in first_line or '\\' in first_line:
                    # Treat as PKL list file
                    pkl_paths = read_pkl_list_file(args.pkl_list_file)
                    if not pkl_paths:
                        logging.error("No valid PKL files found in the list file")
                        return False
                    
                    logging.info(f"PKL list file processing enabled with {len(pkl_paths)} files")
                    logging.info(f"Task type: {args.task}")
                    logging.info(f"Prompt: {args.prompt}")
                    logging.info(f"Frame num: {args.frame_num}")
                    logging.info(f"Size: {target_size}")
                    return run_batch_pkl_generation(generator, args, pkl_paths, target_size)
                else:
                    # Treat as text prompt list file
                    prompts = read_prompt_list_file(args.pkl_list_file)
                    if not prompts:
                        logging.error("No valid prompts found in the list file")
                        return False
                    
                    logging.info(f"Prompt list file processing enabled with {len(prompts)} prompts")
                    logging.info(f"Task type: {args.task}")
                    logging.info(f"Frame num: {args.frame_num}")
                    logging.info(f"Size: {target_size}")
                    return run_batch_prompt_generation(generator, args, prompts, target_size)
        
        # Handle edit list file processing (legacy support)
        elif hasattr(args, 'edit_list_file') and args.edit_list_file:
            edit_data = read_edit_list_file(args.edit_list_file)
            if not edit_data:
                logging.error("No valid edit tasks found in the list file")
                return False
            
            logging.info(f"Edit list file processing enabled with {len(edit_data)} tasks")
            return run_batch_edit_generation(generator, args, edit_data, target_size)
        
        # Handle single generation
        return run_single_generation(generator, args, target_size)
        
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
