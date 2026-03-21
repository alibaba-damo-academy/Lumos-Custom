#!/bin/bash
# Video quality evaluation script for dynamic degree metrics
# Usage: bash eval_quality_sub.sh

CUDA_VISIBLE_DEVICES=1 torchrun --master_port=23422 --nproc_per_node=1 ddp_eval.py \
  --video_dir /mnt/workspace/workgroup/jiazheng/workspace/LumosX/samples/lumosx_test/subject_consistency_16k6_20k \
  --ref_image /mnt/workspace/workgroup/jiazheng/workspace/LumosX/data/test_video_data_new \
  --raft_model_path /mnt/workspace/workgroup/jiazheng/workspace/LumosX/eval_code/models/raft/models/raft-things.pth