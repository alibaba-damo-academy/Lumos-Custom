"""
encode_prompt.py
"""

import torch

def encode_prompt(
    prompt,
    video_background,
    video_degradation,
    v_ref,
    text_encoder,
    vae_model,
    max_seq_len = 75600,
):
    """
    Encode the prompt and additional inputs (background and degradation) into a format suitable for the model.
    """
    neg_prompt = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'

    # Encode prompt
    context = text_encoder(prompt)

    # Encode negative prompt
    context_null = text_encoder(neg_prompt)

    # Process video background and degradation
    x_b = vae_model.encode(video_background)
    x_b = torch.stack(x_b)

    x_d = vae_model.encode(video_degradation)
    x_d = torch.stack(x_d)

    x_t = vae_model.encode(v_ref)
    x_t = torch.stack(x_t)

    return dict(
        context=context,
        context_null=context_null,
        video_background_latent = x_b,
        video_degradation_latent = x_d,
        video_ref = x_t,
        max_seq_len=max_seq_len,
    ) 