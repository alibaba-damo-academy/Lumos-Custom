#!/bin/bash
# Combined evaluation script for all metrics
# Runs: Global, Decoupled, and Quality evaluations sequentially
# Usage: bash eval_all.sh

set -e  # Exit on any error

# Common parameters (can be customized)
VIDEO_DIR="/mnt/workspace/workgroup/jiazheng/workspace/LumosX/samples/lumosx_test/subject_consistency_16k6_20k"
REF_IMAGE="/mnt/workspace/workgroup/jiazheng/workspace/LumosX/data/test_video_data_new"

# Root directory for all external evaluation models (benchmark/models)
MODELS_DIR="../models"

# Model paths
VIDEOCLIP_MODEL_PATH="./VideoCLIP_XL/VideoCLIP-XL-v2.bin"
FLORENCE_PATH="$MODELS_DIR/florence2-large-ft"
YOLO_WEIGHTS="$MODELS_DIR/yolov9/weight/best.pt"
YOLO_CONFIG="$MODELS_DIR/yolov9/data/coco.yaml"
ARCFACE_WEIGHTS="$MODELS_DIR/arcface/weight/backbone.pth"
CURRICULAR_FACE_WEIGHTS="$MODELS_DIR/face_encoder/face_encoder/glint360k_curricular_face_r101_backbone.bin"
OWLV2_PATH="$MODELS_DIR/owlv2-base-patch16-ensemble"
CLIP_PATH="$MODELS_DIR/clip-vit-large-patch14"
DINOV2_PATH="$MODELS_DIR/dinov2-base"
RAFT_MODEL_PATH="$MODELS_DIR/raft/models/raft-things.pth"

echo "=========================================="
echo "Starting Combined Evaluation"
echo "=========================================="
echo "Video Directory: $VIDEO_DIR"
echo "Reference Image Directory: $REF_IMAGE"
echo "=========================================="
echo ""

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Global Evaluation (Video-Text and Video-Video Consistency)
echo "=========================================="
echo "[1/3] Running Global Evaluation..."
echo "=========================================="
echo "Start time: $(date)"
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 eval_global_ddp.py \
  --video_dir "$VIDEO_DIR" \
  --ref_image "$REF_IMAGE" \
  --videoclip_model_path "$VIDEOCLIP_MODEL_PATH"

if [ $? -ne 0 ]; then
    echo "ERROR: Global evaluation failed!"
    exit 1
fi
echo "Global evaluation completed at: $(date)"
echo ""

# 2. Decoupled Evaluation (Subject Consistency Metrics)
echo "=========================================="
echo "[2/3] Running Decoupled Evaluation..."
echo "=========================================="
echo "Start time: $(date)"
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 eval_decouple_ddp_multi.py \
  --video_dir "$VIDEO_DIR" \
  --ref_image "$REF_IMAGE" \
  --florence_path "$FLORENCE_PATH" \
  --yolo_weights "$YOLO_WEIGHTS" \
  --yolo_config "$YOLO_CONFIG" \
  --arcface_weights "$ARCFACE_WEIGHTS" \
  --curricular_face_weights "$CURRICULAR_FACE_WEIGHTS" \
  --owlv2_path "$OWLV2_PATH" \
  --clip_path "$CLIP_PATH" \
  --dinov2_path "$DINOV2_PATH"

if [ $? -ne 0 ]; then
    echo "ERROR: Decoupled evaluation failed!"
    exit 1
fi
echo "Decoupled evaluation completed at: $(date)"
echo ""

# 3. Quality Evaluation (Dynamic Degree Metrics)
echo "=========================================="
echo "[3/3] Running Quality Evaluation..."
echo "=========================================="
echo "Start time: $(date)"
CUDA_VISIBLE_DEVICES=1 torchrun --master_port=23422 --nproc_per_node=1 ddp_eval.py \
  --video_dir "$VIDEO_DIR" \
  --ref_image "$REF_IMAGE" \
  --raft_model_path "$RAFT_MODEL_PATH"

if [ $? -ne 0 ]; then
    echo "ERROR: Quality evaluation failed!"
    exit 1
fi
echo "Quality evaluation completed at: $(date)"
echo ""

echo "=========================================="
echo "All Evaluations Completed Successfully!"
echo "=========================================="
echo "End time: $(date)"
echo ""
echo "Results saved in:"
echo "  - Global: ./results_global/"
echo "  - Decoupled: ./results_decouple/"
echo "  - Quality: ./results_quality/"
