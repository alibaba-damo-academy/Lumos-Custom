import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

resolution = "480p"
aspect_ratio = "9:16"
num_frames = 81
fps = 16
frame_interval = 1
save_fps = 16

save_dir = "./samples/lumosx_output/"
seed = 0
batch_size = 1
multi_resolution = "OpenSora"
dtype = "bf16"
condition_frame_length = 5
align = None

sp_degree = 1

text_len = 512
is_cond_cfg = False
mode = "FSDP"

max_seq_len = 75600
use_fixed_seq_len = False

model = dict(
    type="LumosX_mixfp32",
    from_pretrained="/path/to/checkpoint",
    text_len=text_len,
    in_dim=16,
    dim=1536,
    ffn_dim=8960,
    freq_dim=256,
    text_dim=4096,
    out_dim=16,
    num_heads=12,
    num_layers=30,
    window_size=(-1, -1),
    qk_norm=True,
    cross_attn_norm=True,
    eps=1e-6,
    use_fixed_seq_len=use_fixed_seq_len,
    sp_degree=sp_degree,
    self_mask_bool = True,
    cross_mask_bool = True,
    pooling_type = 'mean',
    ratio = 0.5,
    RoPE_shift = True)

t5 = dict(
    name='umt5_xxl',
    text_len=text_len,
    dtype=dtype,
    checkpoint_path='/path/to/t5_checkpoint.pth',  # Override with --t5-checkpoint-path
    tokenizer_path='/path/to/umt5-xxl')  # Override with --t5-tokenizer-path

vae = dict(
    type="WanX21_VAE",
    vae_pth="/path/to/vae.pth")  # Override with --vae-path

sample_neg_prompt = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'

scheduler = dict(
    type="rflow-wanx",
    num_timesteps=1000,     # num_train_timesteps
    sample_steps=50,        # sample_steps
    sample_shift=8.0,       # sample_shift
    cfg_scale=6.0,          # sample_guide_scale
    transform_scale=1,
    use_discrete_timesteps=False,
    use_timestep_transform=False,
    use_fixed_timestep_transform=False,
)

aes = None
flow = None

prompt_as_path = True