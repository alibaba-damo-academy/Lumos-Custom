#!/bin/bash
# Decoupled evaluation script for subject consistency metrics
# Usage: bash eval_decouple_subj.sh

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 eval_decouple_ddp_multi.py \
  --video_dir /mnt/workspace/workgroup/jiazheng/workspace/LumosX/samples/lumosx_test/subject_consistency_16k6_20k \
  --ref_image /mnt/workspace/workgroup/jiazheng/workspace/LumosX/data/test_video_data_new \
  --florence_path /mnt/workspace/workgroup/jiazheng/workspace/LumosX/eval_code/models/florence2-large-ft \
  --yolo_weights /mnt/workspace/workgroup/jiazheng/workspace/LumosX/eval_code/LumosX-eval/yolov9/weight/best.pt \
  --yolo_config /mnt/workspace/workgroup/jiazheng/workspace/LumosX/eval_code/LumosX-eval/yolov9/data/coco.yaml \
  --arcface_weights /mnt/workspace/workgroup/jiazheng/workspace/LumosX/eval_code/LumosX-eval/arcface/weight/backbone.pth \
  --curricular_face_weights /mnt/workspace/workgroup/jiazheng/workspace/LumosX/eval_code/models/face_encoder/face_encoder/glint360k_curricular_face_r101_backbone.bin \
  --owlv2_path /mnt/workspace/workgroup/jiazheng/workspace/LumosX/eval_code/models/owlv2-base-patch16-ensemble \
  --clip_path /mnt/workspace/workgroup/jiazheng/workspace/LumosX/eval_code/models/clip-vit-large-patch14 \
  --dinov2_path /mnt/workspace/workgroup/jiazheng/workspace/LumosX/eval_code/models/dinov2-base