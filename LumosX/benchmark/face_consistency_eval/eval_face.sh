CUDA_VISIBLE_DEVICES=0 torchrun --master_port=23122 --nproc_per_node=1 ddp_eval_multiface.py \
  --video_dir /path/to/generated_videos_for_face_consistency \
  --ref_image /path/to/face_consistency_test_data \
  --yolo_weights ../models/yolov9/weight/best.pt \
  --yolo_config ../models/yolov9/data/coco.yaml \
  --arcface_weights ../models/arcface/weight/backbone.pth \
  --curricular_face_weights ../models/face_encoder/face_encoder/glint360k_curricular_face_r101_backbone.bin 
