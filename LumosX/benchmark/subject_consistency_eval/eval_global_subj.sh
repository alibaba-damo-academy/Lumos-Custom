#!/bin/bash
# Global evaluation script for video-text and video-video consistency
# Usage: bash eval_global_subj.sh

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 eval_global_ddp.py \
  --video_dir /path/to/generated_videos_for_subject_consistency \
  --ref_image /path/to/subject_consistency_test_data \
  --videoclip_model_path ../models/VideoCLIP_XL/VideoCLIP-XL-v2.bin