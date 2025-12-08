import os
import argparse
from pprint import pformat
import torch

from tqdm import tqdm

from torch.utils.data import DataLoader

from src.models.text_encoder.wanx import WanX21T5Encoder
from src.models.vae.wanx_vae import WanX21_VAE
from src.models.wanx.core_model import Transformer
from src.schedulers.RFLOW_WANX21_T2V import RFLOW_WANX21_T2V

from src.datasets.datasets import VideoTextDataset

from src.utils.utils_ import set_random_seed, save_sample
from src.utils.misc import create_logger
from src.encode_prompt import encode_prompt


def main(args):

    torch.set_grad_enabled(False)

    # == device and dtype ==
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # == set device ==
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    set_random_seed(seed=0)

    # == init logger ==
    logger = create_logger(None)
    logger.info("Inference configuration:\n %s", pformat(args))
    verbose = True

    # ======================================================
    # build model & load weights
    # ======================================================

    logger.info("Building models...")
    logger.info("step 1: build text-encoder")
    text_encoder = WanX21T5Encoder(
        name='umt5_xxl',
        text_len=512,
        dtype="bf16",
        device=device,
        checkpoint_path=args.t5_path,
        tokenizer_path=args.tokenizer_path)

    logger.info("step 2: build vae")
    vae = WanX21_VAE(
        vae_pth=args.vae_path
    )
    vae.model = vae.model.to(device=device, dtype=dtype).eval()

    logger.info("step 3: build wanx core model")
    model = Transformer().to(device=device, dtype=dtype).eval()

    # load model
    ckpt = torch.load(args.unilumos_path, map_location=device)
    model.load_state_dict(ckpt)

    logger.info("Finish create wanx core model")

    # == build scheduler ==
    scheduler = RFLOW_WANX21_T2V()
    logger.info("Finish create scheduler")

    # == prepare video size ==
    image_size = args.image_size
    if isinstance(image_size, str):
        image_size = eval(image_size)
    num_frames = int(args.num_frames)

    input_size = (num_frames, *image_size)
    latent_size = [(input_size[0] - 1) // vae.model.temporal_scale_factor + 1,
                   int(input_size[1]) // vae.model.spatial_scale_factor,
                   int(input_size[2]) // vae.model.spatial_scale_factor]
    logger.info(f"input_size: {input_size}, latent_size: {latent_size}")

    # == build dataset ==
    dataset = VideoTextDataset(
        transform_name="resize_crop",
        data_path=args.data_path,
        num_frames=num_frames,
        frame_interval=1,
        image_size=image_size,
    )

    # dataloader config
    dataloader_args = dict(
        dataset=dataset,
        batch_size=1,
        num_workers=4,
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )
    dataloader = DataLoader(**dataloader_args)

    # ======================================================
    # inference
    # ======================================================

    # == prepare arguments ==
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for idx, data_i in enumerate(tqdm(dataloader, desc="Inference Progress", disable=not verbose)):

            # Extract data from dataloader
            video_ref = data_i['video'].to(device, dtype)  # Reference image as a single-frame video
            video_background = torch.zeros_like(video_ref)
            video_degradation = torch.zeros_like(video_ref)

            # get prompts
            if 'text' in data_i:
                batch_prompts = data_i['text']
            else:
                batch_prompts = [args.prompt] * len(video_ref)

            # naming for save path
            if hasattr(dataset, "data") and "path" in dataset.data.columns:
                ori_path = dataset.data.iloc[idx]["path"]
            else:
                ori_path = f"sample_{idx}"
            base_name = os.path.splitext(os.path.basename(ori_path))[0]

            # Generate random noise
            generator = torch.Generator(device=device).manual_seed(0)
            z = torch.randn(len(batch_prompts), vae.model.z_dim, *latent_size, device=device, generator=generator, dtype=dtype)

            # Encode prompt, background, and degradation
            y = encode_prompt(
                prompt=batch_prompts,
                video_background=video_background,
                video_degradation=video_degradation,
                v_ref=video_ref,
                text_encoder=text_encoder,
                vae_model=vae
            )

            x_bc = y['video_background_latent']
            x_deg = y['video_degradation_latent']
            x_ref = y['video_ref']

            # Sample using scheduler
            samples = scheduler.sample(
                model,
                y=y,
                z=z,
                prompts=batch_prompts,
                device=device,
                additional_args=None,
                progress=verbose,
                mask=None,
                generator=generator,
                mode="t2v",
                sample_steps=25,  # sample_steps
                sample_shift=8.0,  # sample_shift
            )

            samples = vae.decode(samples)
            v_ref = vae.decode(x_ref)
            v_bc = vae.decode(x_bc)
            v_deg = vae.decode(x_deg)

            video = torch.cat(samples, dim=1)
            v_ori = torch.cat(v_ref, dim=1)
            v_bc = torch.cat(v_bc, dim=1)
            v_deg = torch.cat(v_deg, dim=1)

            # save generated image/video
            save_sample(
                video,
                fps=args.fps,
                save_path=f"{save_dir}/{base_name}_gen",
                verbose=verbose,
            )

            # # save original
            # save_sample(
            #     v_ori,
            #     fps=args.fps,
            #     save_path=f"{save_dir}/{base_name}_ori",
            #     verbose=verbose,
            # )

            # # save background
            # save_sample(
            #     v_bc,
            #     fps=args.fps,
            #     save_path=f"{save_dir}/{base_name}_bc",
            #     verbose=verbose,
            # )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--t5_path", type=str, default="./weights/models_t5_umt5-xxl-enc-bf16.pth", help="path to t5 model")
    parser.add_argument("--tokenizer_path", type=str, default="./weights/umt5-xxl", help="path to tokenizer")
    parser.add_argument("--vae_path", type=str, default="./weights/vae.pth", help="path to vae model")
    parser.add_argument("--unilumos_path", type=str, default="./weights/unilumos.pt", help="path to unilumos model")

    parser.add_argument("--image_size", type=str, default=(768, 512), help="image size")
    parser.add_argument("--num_frames", type=int, default=1, help="number of frames")
    parser.add_argument("--fps", type=int, default=8, help="fps for saving results")

    parser.add_argument("--data_path", type=str, default="./examples/examples_refined.csv", help="path to data")
    parser.add_argument("--prompt", type=str, default="", help="prompt to use when the data file has no text column")

    parser.add_argument("--save_dir", type=str, default="./examples/results_image", help="path to save results")

    args = parser.parse_args()

    main(args)
