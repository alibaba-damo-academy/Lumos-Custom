#!/usr/bin/env python3
"""
This script reads prompts from reasoning_prompts_with_filenames.json and generates videos.
Each prompt is saved in a category folder with filename as id.mp4.

Usage:
    python generate_reasoning_prompts.py --json_file reasoning_prompts_with_filenames.json --output_dir ./output
"""

import argparse
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union, List
import json

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
        target_size: Target image size (height, width)
        
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


def init_logging(rank: int = 0):
    """Initialize logging configuration."""
    log_file = f'omnivideo_reasoning_rank{rank}.log'
    
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


def read_reasoning_json(json_file: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Read prompts from reasoning_prompts_with_filenames.json.
    
    Args:
        json_file: Path to the JSON file
        
    Returns:
        Dictionary mapping category to list of prompt items (each with id, prompt_en, etc.)
    """
    if not os.path.exists(json_file):
        logging.error(f"JSON file not found: {json_file}")
        return {}
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract prompts by category
    prompts_by_category = {}
    for category_name, category_data in data.items():
        prompts = category_data.get("prompts", [])
        if prompts:
            prompts_by_category[category_name] = prompts
            logging.info(f"Loaded {len(prompts)} prompts from category: {category_name}")
    
    total_prompts = sum(len(prompts) for prompts in prompts_by_category.values())
    logging.info(f"Total loaded {total_prompts} prompts from {len(prompts_by_category)} categories")
    
    return prompts_by_category


def run_single_generation(generator, args, prompt_text, category_name, prompt_id, target_size):
    """Run single video generation."""
    rank = int(os.getenv("RANK", 0))
    
    # Create category directory
    category_dir = os.path.join(args.output_dir, category_name.replace(" ", "_").lower())
    os.makedirs(category_dir, exist_ok=True)
    
    # Set save file path: category_dir/id.mp4
    save_file = os.path.join(category_dir, f"{prompt_id}.mp4")
    
    # Skip if file already exists
    if os.path.exists(save_file):
        logging.info(f"GPU {rank}: File already exists, skipping: {save_file}")
        return True
    
    # Temporarily modify args
    original_prompt = args.prompt
    original_save_file = getattr(args, 'save_file', None)
    
    args.prompt = prompt_text
    args.save_file = save_file
    
    try:
        # Extract task type
        task = args.task
        
        # Prepare inputs based on task
        src_file_path = None
        if hasattr(args, 'src_file_path') and args.src_file_path:
            src_file_path = args.src_file_path
        
        # Process visual input if needed
        visual_emb = None
        if task == 'i2i' and src_file_path:
            visual_emb = extract_vae_features(
                src_file_path, generator.video_x2x.vae, 
                generator.device, (target_size[1], target_size[0])
            )
            if visual_emb is not None:
                visual_emb = visual_emb[:, 0:1]
                param_dtype = generator.video_x2x.param_dtype
                visual_emb = visual_emb.to(dtype=param_dtype)
                
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
                    param_dtype = generator.video_x2x.param_dtype
                    visual_emb = visual_emb.to(dtype=param_dtype)
        
        # Generate embeddings
        vlm_last_hidden_states = generator.ar_model.general_emb(
            prompt_text, src_file_path, task_type=task
        )
        # Log input shapes
        logging.info(f"GPU {rank}: Input shape: {vlm_last_hidden_states.shape}")
        logging.info(f"GPU {rank}: Visual emb shape: {visual_emb.shape if visual_emb is not None else None}")
        logging.info(f"GPU {rank}: Task: {task}")
        logging.info(f"GPU {rank}: Prompt: {prompt_text[:100]}...")
        logging.info(f"GPU {rank}: Size: {target_size}")
        logging.info(f"GPU {rank}: Frame num: {args.frame_num}")
        # Generate content
        logging.info(f"GPU {rank}: Starting generation...")
        # target_size is (height, width), but generate expects (width, height)
        result = generator.video_x2x.generate(
            prompt_text,
            visual_emb=visual_emb,
            ar_vision_input=vlm_last_hidden_states,
            size=(target_size[1], target_size[0]),  # Convert to (width, height)
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            special_tokens=generator.special_tokens,
            classifier_free_ratio=args.classifier_free_ratio,
            unconditioned_context=generator.unconditioned_context,
            condition_mode=args.condition_mode,
            use_visual_as_input=args.use_visual_as_input,
            gamma_w=args.gamma_w,
            gamma_hf=args.gamma_hf,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            offload_model=args.offload_model,
        )
        
        if result is None:
            logging.warning(f"GPU {rank}: Generation failed for {prompt_id}")
            return False
        
        # Save video
        logging.info(f"GPU {rank}: Saving video to {save_file}")
        cache_video(
            tensor=result[None],
            save_file=save_file,
            fps=args.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1)
        )
        
        logging.info(f"GPU {rank}: Successfully saved {save_file}")
        return True
        
    except Exception as e:
        logging.error(f"GPU {rank}: Error generating {prompt_id}: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Restore original args
        args.prompt = original_prompt
        if original_save_file is not None:
            args.save_file = original_save_file
        elif hasattr(args, 'save_file'):
            delattr(args, 'save_file')


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


def run_batch_reasoning_generation(generator, args, prompts_by_category, target_size):
    """Run batch generation with prompts from JSON file, organized by category."""
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    
    if rank == 0:
        logging.info(f"=== Starting Batch Reasoning Prompts Generation ===")
        total_prompts = sum(len(prompts) for prompts in prompts_by_category.values())
        logging.info(f"Processing {total_prompts} prompts from {len(prompts_by_category)} categories")
        logging.info(f"Using {world_size} GPU(s)")
        logging.info(f"Output directory: {args.output_dir}")
        logging.info("=" * 50)
    
    # Create a flat list of (category, prompt_item) tuples for distribution
    all_prompt_items = []
    for category_name, prompts in prompts_by_category.items():
        for prompt_item in prompts:
            all_prompt_items.append((category_name, prompt_item))
    
    # Distribute prompt items across GPUs
    items_per_gpu = [[] for _ in range(world_size)]
    for i, item in enumerate(all_prompt_items):
        gpu_id = i % world_size
        items_per_gpu[gpu_id].append(item)
    
    # Each GPU processes its assigned prompt items
    my_prompt_items = items_per_gpu[rank]
    
    logging.info(f"GPU {rank}: Assigned {len(my_prompt_items)} prompt items")
    
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
    successful_count = 0
    failed_count = 0
    
    for i, (category_name, prompt_item) in enumerate(my_prompt_items):
        prompt_id = prompt_item.get("id", f"unknown_{i}")
        prompt_text = prompt_item.get("prompt_en", "")
        
        if not prompt_text:
            logging.warning(f"GPU {rank}: Skipping {prompt_id} - no prompt_en field")
            failed_count += 1
            continue
        
        logging.info(f"GPU {rank}: Processing item {i+1}/{len(my_prompt_items)}: [{category_name}] {prompt_id}")
        
        success = run_single_generation(
            generator, args, prompt_text, category_name, prompt_id, target_size
        )
        
        results[f"{category_name}/{prompt_id}"] = success
        
        if success:
            successful_count += 1
        else:
            failed_count += 1
        
        status = "✅ Success" if success else "❌ Failed"
        logging.info(f"GPU {rank}: Item {i+1}/{len(my_prompt_items)}: {status}")
    
    # Print summary
    logging.info("=" * 50)
    logging.info(f"=== GPU {rank} Batch Reasoning Generation Summary ===")
    logging.info(f"GPU {rank}: Successful: {successful_count}, Failed: {failed_count}")
    logging.info(f"GPU {rank}: Results saved in: {args.output_dir}")
    logging.info("=" * 50)
    
    # Synchronize all GPUs before returning
    if dist.is_initialized():
        logging.info(f"GPU {rank}: Waiting for all GPUs to complete...")
        safe_distributed_barrier(timeout_minutes=30)
        logging.info(f"GPU {rank}: All GPUs synchronized")
    
    return successful_count > 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="OmniVideo: Reasoning Prompts Generation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # JSON file argument
    parser.add_argument(
        "--json_file", type=str, required=True,
        help="Path to reasoning_prompts_with_filenames.json file"
    )
    
    # Basic arguments
    parser.add_argument(
        "--prompt", type=str, default="",
        help="Text prompt for generation (not used when reading from JSON file)"
    )
    parser.add_argument(
        "--task", type=str, default="t2v",
        choices=["t2v", "t2i", "i2i", "v2v"],
        help="Generation task type"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for generated videos"
    )
    parser.add_argument(
        "--models_dir", type=str, default=None,
        help="Checkpoint root (wan, Wan2.1-T2V-14B, adapter, ar_model, ...); default <repo>/model_ckpts",
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
        "--sample_steps", type=int, default=None,
        help="Number of sampling steps"
    )
    parser.add_argument(
        "--sample_fps", type=int, default=16,
        help="FPS for output video"
    )
    parser.add_argument(
        "--sample_guide_scale", type=float, default=5.0,
        help="Guidance scale for classifier-free guidance"
    )
    parser.add_argument(
        "--base_seed", type=int, default=42,
        help="Base random seed"
    )
    parser.add_argument(
        "--seed_per_rank_stride", type=int, default=0,
        help="Per-rank seed stride. 0 keeps same seed across ranks for alignment."
    )
    parser.add_argument(
        "--sample_shift", type=float, default=None,
        help="Sampling shift factor"
    )
    
    # Model arguments
    parser.add_argument(
        "--sample_solver", type=str, default="unipc",
        help="Sampling solver"
    )
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
        "--use_visual_context_adapter", type=str2bool, default=True,
        help="Use visual context adapter"
    )
    parser.add_argument(
        "--visual_context_adapter_patch_size", type=str2tuple, default=(1, 4, 4),
        help="Visual context adapter patch size (e.g., '1,4,4')"
    )
    parser.add_argument(
        "--use_visual_as_input", type=str2bool, default=False,
        help="Use visual as input"
    )
    parser.add_argument(
        "--condition_mode", type=str, default="full",
        help="Condition mode"
    )
    parser.add_argument(
        "--max_context_len", type=int, default=2560,
        help="Maximum context length"
    )
    parser.add_argument(
        "--ar_model_num_video_frames", type=int, default=8,
        help="AR model number of video frames"
    )
    parser.add_argument(
        "--ar_conv_mode", type=str, default="llama_3",
        help="AR conversation mode"
    )
    parser.add_argument(
        "--sampling_rate", type=int, default=3,
        help="Sampling rate for video frames"
    )
    parser.add_argument(
        "--skip_num", type=int, default=1,
        help="Number of frames to skip"
    )
    parser.add_argument(
        "--unconditioned_context_length", type=int, default=2560,
        help="Unconditioned context length"
    )
    parser.add_argument(
        "--classifier_free_ratio", type=float, default=0.0,
        help="Classifier-free guidance ratio"
    )
    parser.add_argument(
        "--gamma_w", type=float, default=0.5,
        help="Gamma_w parameter for cascade inference"
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
        help="Move DiT to CPU after each generate() (VRAM vs batch speed)"
    )
    parser.add_argument(
        "--dit_fsdp", type=str2bool, default=True,
        help="FSDP shard both Wan DiTs (requires torchrun world_size>1)"
    )
    parser.add_argument(
        "--t5_cpu", type=str2bool, default=True,
        help="Wan2.1-T2V style: UMT5 on CPU, embeddings on GPU (saves VRAM). false = T5 on GPU.",
    )
    parser.add_argument(
        "--fast_cuda", type=str2bool, default=False,
        help="CUDA throughput defaults (cudnn benchmark, TF32, SDPA). Default false for reproducibility/alignment."
    )
    parser.add_argument(
        "--wan_inference_activation_checkpoint", type=str2bool, default=False,
        help="Wan DiT inference activation checkpointing (slower, lower peak activation memory)"
    )
    # V2V/I2I specific arguments
    parser.add_argument(
        "--src_file_path", type=str, default=None,
        help="Source file path for v2v or i2i tasks"
    )
    parser.add_argument(
        "--save_file", type=str, default=None,
        help="Output file path (auto-generated if not specified)"
    )
    
    args = parser.parse_args()
    
    # Validate args (skip prompt check since we read from JSON)
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
    
    return args


def main():
    """Main function for reasoning prompts generation."""
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
            args.base_seed = base_seed[0] + rank * args.seed_per_rank_stride
        
        # Load models and data
        generator.load_special_tokens()
        generator.load_unconditioned_context()
        generator.initialize_models()
        
        # Set up output directory
        os.makedirs(args.output_dir, exist_ok=True)
        logging.info(f"GPU {rank}: Output directory: {args.output_dir}")
        
        # Synchronize all GPUs
        if dist.is_initialized():
            safe_distributed_barrier(timeout_minutes=30)
            logging.info(f"GPU {rank}: Output directory synchronized")
        
        # Read JSON file
        prompts_by_category = read_reasoning_json(args.json_file)
        if not prompts_by_category:
            logging.error("No prompts found in JSON file")
            return False
        
        # Handle size configuration
        # SIZE_CONFIGS returns (width, height), but we store as (height, width) for consistency
        if args.size in SIZE_CONFIGS:
            w, h = SIZE_CONFIGS[args.size]  # (width, height)
            target_size = (h, w)  # Convert to (height, width)
        else:
            # Parse size string like "832*480"
            w, h = args.size.split('*')
            target_size = (int(h), int(w))  # (height, width)
        logging.info(f"Target size: {target_size} (height, width)")
        
        # Run batch generation
        return run_batch_reasoning_generation(generator, args, prompts_by_category, target_size)
        
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

