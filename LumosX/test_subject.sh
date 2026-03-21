torchrun --nproc_per_node=2 --master_port=23422 \
    scripts/WanX2.1/lumosx_inference_t2v_subject.py \
    configs/wanx/inference/lumosx_t2v_config.py \
    --image-size 480 832 \
    --prompt-as-path \
    --prompt-path ./test_data/subject_consistency_test_data \
    --ckpt-path "./cache/LumosX_hub/LumosX_models" \
    --t5-checkpoint-path './cache/LumosX_hub/models_t5_umt5-xxl-enc-bf16.pth' \
    --t5-tokenizer-path './cache/LumosX_hub/umt5-xxl' \
    --vae-path "./cache/LumosX_hub/vae.pth" \
    --save-dir "samples/lumosx_test/subject_consistency"