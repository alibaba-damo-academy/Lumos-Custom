# LumosX Benchmark Evaluation

This directory contains comprehensive evaluation benchmarks for LumosX, including face consistency evaluation and subject-attribute consistency evaluation.

## Download Evaluation Models

The benchmark evaluation requires several pre-trained models for different metrics. Download them as follows:

**1. VideoCLIP-XL** (for Global Evaluation - Video-Text and Video-Video Consistency)
```bash
# Download from HuggingFace
cd benchmark/subject_consistency_eval/VideoCLIP_XL/
huggingface-cli download alibaba-pai/VideoCLIP-XL-v2 --local-dir . --include "VideoCLIP-XL-v2.bin"
# Or manually download from: https://huggingface.co/alibaba-pai/VideoCLIP-XL-v2/tree/main
```

**2. Florence2** (for Decoupled Evaluation - Subject Consistency)
```bash
# Download from HuggingFace into the shared benchmark/models directory
huggingface-cli download microsoft/florence2-large-ft --local-dir benchmark/models/florence2-large-ft
```

**3. YOLOv9** (for Object Detection)
```bash
cd benchmark/subject_consistency_eval/yolov9/
mkdir -p weight
cd weight
# Manually download best.pt from Google Drive
# https://drive.google.com/file/d/1wgT-iCcCTpetH_nDSkh_RAaYTr74qvkg
# Place best.pt under benchmark/subject_consistency_eval/yolov9/weight/
# Note: coco.yaml is already included in the yolov9/data/ directory
```

**4. ArcFace** (for Face Recognition)
```bash
cd benchmark/subject_consistency_eval/arcface/
mkdir -p weight
cd weight
# Manually download backbone.pth from Google Drive
# https://drive.google.com/file/d/1vN9yn_KN0DXZpKy7Kg0luM-0tNVFGm95
# Place backbone.pth under benchmark/subject_consistency_eval/arcface/weight/
```

**5. CurricularFace** (for Face Recognition)
```bash
cd benchmark/face_consistency_eval/curricularface/
# Download CurricularFace weights from Google Drive
# https://drive.google.com/open?id=1upOyrPzZ5OI3p6WkA5D5JFYCeiZuaPcp
# Place glint360k_curricular_face_r101_backbone.bin in:
#   - benchmark/face_consistency_eval/curricularface/
#   - or benchmark/subject_consistency_eval/models/face_encoder/face_encoder/
```

**6. OWLv2** (for Object Detection in Decoupled Evaluation)
```bash
# Download from HuggingFace into the shared benchmark/models directory
huggingface-cli download google/owlv2-base-patch16-ensemble --local-dir benchmark/models/owlv2-base-patch16-ensemble
```

**7. CLIP** (for Image-Text Matching)
```bash
# Download from HuggingFace into the shared benchmark/models directory
huggingface-cli download openai/clip-vit-large-patch14 --local-dir benchmark/models/clip-vit-large-patch14
```

**8. DINOv2** (for Visual Features)
```bash
# Download from HuggingFace into the shared benchmark/models directory
huggingface-cli download facebook/dinov2-base --local-dir benchmark/models/dinov2-base
```

**9. RAFT** (for Quality Evaluation - Dynamic Degree Metrics)
```bash
# First, clone the RAFT code repository
cd benchmark/models/raft

# Download the pretrained model weights
mkdir -p models
cd models
wget https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip
unzip models.zip
rm models.zip
# This will extract raft-things.pth to benchmark/models/raft/models/
```

## Directory Structure

After downloading, organize evaluation models under a shared `benchmark/models` directory as follows:

```
benchmark/
├── subject_consistency_eval/
├── face_consistency_eval/
└── models/
    ├── florence2-large-ft/
    ├── owlv2-base-patch16-ensemble/
    ├── clip-vit-large-patch14/
    ├── dinov2-base/
    ├── face_encoder/
    │   └── face_encoder/
    │       └── glint360k_curricular_face_r101_backbone.bin
    ├── raft/
    │   └── models/
    │       └── raft-things.pth
    ├── VideoCLIP_XL/
    │   └── VideoCLIP-XL-v2.bin
    ├── yolov9/
    │   ├── weight/best.pt
    │   └── data/coco.yaml
    └── arcface/
        └── weight/backbone.pth
```

**Note:** Update the model paths in `eval_all.sh` and `eval_face.sh` to match your local directory structure.

## Evaluation Scripts

### Face Consistency Evaluation

Located in `face_consistency_eval/`:
- `eval_face.sh`: Main evaluation script for face consistency
- `ddp_eval_multiface.py`: Distributed evaluation script

### Subject-Attribute Consistency Evaluation

Located in `subject_consistency_eval/`:
- `eval_subject.sh`: Combined evaluation script for all metrics (Global, Decoupled, Quality)
- `eval_global_ddp.py`: Global evaluation (Video-Text and Video-Video Consistency)
- `eval_decouple_ddp_multi.py`: Decoupled evaluation (Subject Consistency Metrics)
- `ddp_eval.py`: Quality evaluation (Dynamic Degree Metrics)

See the respective subdirectories for more details on each evaluation type.

