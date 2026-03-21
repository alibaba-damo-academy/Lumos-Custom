import sys
import os
sys.path.append(os.path.join(os.getcwd()))
import time
import numpy as np
from pprint import pformat
import functools
from functools import partial
import torch
import torch.distributed as dist
from source.utils.train_utils import set_random_seed
from tqdm import tqdm
from typing import Dict, List
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
# FSDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    BackwardPrefetch,
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, lambda_auto_wrap_policy
from source.acceleration.parallel_states import set_sequence_parallel_group
from source.datasets import save_sample, save_sample_imageio
from source.datasets.aspect import get_image_size, get_num_frames
from source.models.text_encoder.t5 import text_preprocessing
from source.registry import MODELS, SCHEDULERS, build_module
from source.utils.config_utils import parse_configs
from source.utils.inference_utils import (
    add_watermark,
    append_generated,
    append_score_to_prompts,
    apply_mask_strategy,
    collect_references_batch,
    dframe_to_frame,
    extract_json_from_prompts,
    extract_prompts_loop,
    get_save_path_name,
    load_prompts,
    load_prompts_alchemy_csv,
    merge_prompt,
    prepare_multi_resolution_info,
    refine_prompts_by_openai,
    split_prompt,
)
from source.utils.ckpt_utils import sharded_load, load_checkpoint
from source.utils.misc import all_exists, create_logger, is_distributed, is_main_process, to_torch_dtype, measure_gpu_memory
from source.utils.constants import PROMPT_TEMPLATE
from source.models.text_encoder import WanX21T5Encoder

from training_acc.parallelisms import parallelize
from training_acc.config import ParallelConfig
from training_acc.dist import initialize, parallel_state, log_rank, get_local_rank, get_sequence_parallel_rank, destroy_process_group
from training_acc.logger import logger as acc_logger
from training_acc.utils import get_model_numel

from source.datasets.utils import VID_EXTENSIONS, get_transforms_image, get_transforms_video, read_file, temporal_random_crop
import decord

import cv2
from PIL import ImageFile, Image

import random
import json

def encode_prompt(
    prompt,
    neg_prompt,
    text_encoder,
    max_seq_len):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    neg_prompt = [neg_prompt] if isinstance(neg_prompt, str) else neg_prompt

    context = text_encoder(prompt)
    context_null = text_encoder(neg_prompt)

    return dict(context=context, context_null=context_null, max_seq_len=max_seq_len)

def load_img_masks(masks_dirs, original_imgs,  video_class_dir, selected_frames):
    mask_imgs = list()
    for mask_dir in masks_dirs:
        mask_img_path = os.path.join(video_class_dir, mask_dir)
        mask_img = np.array(Image.open(mask_img_path))
        mask_img_normalized = mask_img.astype(np.float32) / 255.0
        
        idx = int(mask_dir.split('_')[-1][5:-4])
        position= selected_frames.index(idx)
        image = original_imgs[position]
        alpha_channel = np.full((image.shape[0], image.shape[1], 1), 255, dtype=np.uint8)
        rgba_image = np.concatenate((image, alpha_channel), axis=-1)
        rgba_image[...,-1] = rgba_image[...,-1] *  mask_img_normalized
        out = add_white_background(rgba_image)

        ######## crop mask ##########
        mask_crop = cv2.inRange(out, (255, 255, 255), (255, 255, 255))
        non_white = cv2.bitwise_not(mask_crop)
        contours, _ = cv2.findContours(non_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            out_image = out 
        else:
            all_points = np.concatenate(contours)
            x, y, w, h = cv2.boundingRect(all_points)
            out_image =out[y:y+h, x:x+w]

        mask_imgs.append(out_image)
    return mask_imgs

def add_white_background(np_image):
    background_mask = np_image[:, :, 3] == 0
    np_image[background_mask] = [255, 255, 255, 255]
    np_image = np_image[:, :, :3]
    # image = Image.fromarray(np_image, mode="RGB")
    return np_image


def strict_sequence_mask(inputs_ids: torch.Tensor, index_result: torch.Tensor) -> torch.Tensor:
    """
    Strict sequence-sensitive boolean mask generation
    Requires inputs_ids to contain a continuous subsequence identical to index_result
    
    Args:
        inputs_ids: Input tensor with arbitrary dimensions (last dimension is sequence dimension)
        index_result: Sequence to match in strict order
        
    Returns:
        bool_mask: Boolean tensor with the same shape as inputs_ids, True only at positions matching the sequence
    """
    # Ensure index_result is a 1D tensor
    index_result = index_result.view(-1)
    k = index_result.shape[0]
    
    # Get input parameter information
    orig_shape = inputs_ids.shape
    seq_dim = inputs_ids.dim() - 1  # Assume sequence dimension is the last one
    
    # Handle cases where length is insufficient
    if k > orig_shape[-1]:
        return torch.zeros_like(inputs_ids, dtype=torch.bool)
    
    # Generate sliding window view (..., num_windows, k)
    unfolded = inputs_ids.unfold(seq_dim, k, 1)
    
    # Exact match check (..., num_windows)
    match_windows = (unfolded == index_result).all(dim=-1)
    
    # Create all-False mask template
    bool_mask = torch.zeros_like(inputs_ids, dtype=torch.bool)
    
    # Map match results to original positions
    for offset in range(k):
        # Calculate slice range for each position
        start = offset
        end = inputs_ids.size(seq_dim) - k + 1 + offset
        
        # Fill for each dimension
        slices = [slice(None)] * seq_dim + [slice(start, end)]
        
        # Use logical OR to accumulate all possible matching positions
        bool_mask[slices] |= match_windows
    
    return bool_mask


def find_mask_embedding( masks_texts, text_encoder, texts):
    mask_embedding_dict = dict()
    inputs_ids = text_encoder.tokenizer(texts, add_special_tokens=True)[0]
    for mask_text in masks_texts:
        input_mask_id = text_encoder.tokenizer(mask_text,add_special_tokens=True)[0]
        index_of_end = (input_mask_id == 1).nonzero(as_tuple=True)[0]
        index_result = input_mask_id[:index_of_end[0]]
        bool_mask = strict_sequence_mask(inputs_ids,index_result)
        mask_embedding_dict[mask_text] = bool_mask
    return mask_embedding_dict

def load_face_crops( human_crops, images, selected_frames):
    face_imgs = list()
    for face_key, face_value in human_crops.items():
        face_key = int(face_key)
        position= selected_frames.index(face_key)
        np_image = images[position]
        
        x0, y0, x1, y1 = face_value
        np_image = np_image[y0:y1, x0:x1]
        face_imgs.append(np_image)
    return face_imgs

def obtain_mask_dict(mask_dir_path, image_size):
    height, width = image_size
    mask_dict = dict()

    def resize_crop_to_fill_img(pil_image, target_size):
        """
        First resize proportionally so the short edge matches target size, then center crop to target size
        Args:
            pil_image: PIL Image object
            target_size: Target size (width, height)
        Returns:
            Processed PIL Image
        """
        
        target_width, target_height = target_size
        original_width, original_height = pil_image.size
        
        # Calculate resize ratio (maintain aspect ratio, long edge does not exceed target size)
        ratio = min(target_width / original_width, target_height / original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)
        
        # Resize (maintain aspect ratio)
        resized = pil_image.resize((new_width, new_height), Image.BICUBIC)
        
        # Create white background canvas
        padded_image = Image.new("RGB", target_size, (255, 255, 255))
        
        # Paste resized image centered on white background
        offset = ((target_width - new_width) // 2, (target_height - new_height) // 2)
        padded_image.paste(resized, offset)
        
        return padded_image

    if os.path.isdir(mask_dir_path):
        import torchvision.transforms as transforms
        transform_img = transforms.Compose(
            [
                transforms.Lambda(lambda pil_image: resize_crop_to_fill_img(pil_image, (width, height))),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
        #######################background#######################
        element_keys = os.listdir(mask_dir_path)

        #########################subject#######################
        if 'subjects' in element_keys:
            subject_keys = os.listdir(os.path.join(mask_dir_path, 'subjects'))
            subjects_path = os.path.join(mask_dir_path, 'subjects')
            subject_subdirs = [entry  for entry in subject_keys if os.path.isdir(os.path.join(subjects_path, entry))]
  
            for human_i, subject_human in enumerate(subject_subdirs):
                subject_path = os.path.join(subjects_path, subject_human)
                subject_files = os.listdir(subject_path)

                is_fullbody=False
                # for subject_file in subject_files:
                #     if "fullfullbody" in subject_file:
                #         is_fullbody = True
                #         continue

                human_dict = dict()
                if not is_fullbody:
                    for subject_file in subject_files:
                        if 'fullfullbody' in subject_file:
                            continue
                        if 'face' in subject_file:
                            face_file = os.path.join(subject_path, subject_file)
                            face_name = "_".join(subject_file.split('_')[1:])
                            face_name = ".".join(face_name.split('.')[:-1])
                            face_img = [transform_img(Image.open(face_file))]
                            human_dict[face_name] = face_img
                    human_idx = 'human_x_' + str(human_i)
                    mask_dict[human_idx] = human_dict
                else:
                    for subject_file in subject_files:
                        attribute_file = os.path.join(subject_path, subject_file)
                        attribute_name = ".".join(subject_file.split('.')[:-1]).replace("fullfullbody_", "")
                        attribute_img = [transform_img(Image.open(attribute_file))]
                        human_dict[attribute_name] = attribute_img
                    
    return mask_dict


def main():
    torch.set_grad_enabled(False)
    # ======================================================
    # parse configs & runtime variables
    # ======================================================
    # == parse configs ==
    cfg = parse_configs(training=False)

    # == device and dtype ==
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_dtype = cfg.get("dtype", "fp32")
    assert cfg_dtype in ["fp16", "bf16", "fp32"], f"Unknown mixed precision {cfg_dtype}"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # == init distributed env ==
    # dist.init_process_group(backend="nccl")
    
    device_mesh = init_device_mesh("cuda", (torch.cuda.device_count(), ))
    
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    set_random_seed(seed=cfg.get("seed", 1024))
    
    device = torch.cuda.current_device()
    is_main_process_bool = dist.get_rank() == 0
    
    # parallel init
    sp_degree = cfg.get("sp_degree", 1)
    parallel_config = ParallelConfig(sp_degree = sp_degree)
    initialize(parallel_config=parallel_config)
    
    data_parallel_size = parallel_state.get_data_parallel_size()
    data_parallel_rank = parallel_state.get_data_parallel_rank()
    sp_parallel_rank = parallel_state.get_sequence_parallel_rank()

    # == init logger ==
    # exp_dir = os.path.join(cfg.model.from_pretrained, "eval")
    # os.makedirs(exp_dir, exist_ok=True)
    logger = create_logger(None)
    logger.info("Inference configuration:\n %s", pformat(cfg.to_dict()))
    verbose = cfg.get("verbose", True)

    # ======================================================
    # build model & load weights
    # ======================================================
    logger.info("Building models...")
    # == build text-encoder and vae ==    
    text_encoder = WanX21T5Encoder(
            name=cfg.t5.name,
            text_len=cfg.t5.text_len,
            dtype=cfg.t5.dtype,
            device=device,
            checkpoint_path=cfg.t5.checkpoint_path,
            tokenizer_path=cfg.t5.tokenizer_path)
    
    vae = build_module(cfg.get("vae", None), MODELS)
    vae.model = vae.model.to(device=device, dtype=dtype).eval()

    # == prepare video size ==
    image_size = cfg.get("image_size", None)
    if image_size is None:
        resolution = cfg.get("resolution", None)
        aspect_ratio = cfg.get("aspect_ratio", None)
        assert (
            resolution is not None and aspect_ratio is not None
        ), "resolution and aspect_ratio must be provided if image_size is not provided"
        image_size = get_image_size(resolution, aspect_ratio)
    num_frames = get_num_frames(cfg.num_frames)

    # == build diffusion model ==
    input_size = (num_frames, *image_size)
    logger.info(f"input_size: {input_size}")
    # vae_scale_factor_spatial = 2 ** (len(vae.config.block_out_channels) - 1)
    # vae_scale_factor_temporal = vae.config.temporal_compression_ratio
        
    # latent_size = [(input_size[0] - 1) // vae_scale_factor_temporal + 1, int(input_size[1]) // vae_scale_factor_spatial, int(input_size[2]) // vae_scale_factor_spatial]
    
    # latent_size = vae.get_latent_size(input_size)

    latent_size = [(input_size[0] - 1) // vae.model.temporal_scale_factor + 1,
                   int(input_size[1]) // vae.model.spatial_scale_factor,
                   int(input_size[2]) // vae.model.spatial_scale_factor]
    logger.info(f"input_size: {input_size}, latent_size: {latent_size}")
    # model is suggested to keep fp32 dtype with AMP and FSDP. Due to: https://github.com/huggingface/accelerate/issues/2624
    model = (
        build_module(
            cfg.model,
            MODELS,
        )
        .to(dtype).eval() # model is suggested to keep fp32 dtype with AMP and FSDP. Due to: https://github.com/huggingface/accelerate/issues/2624
    )
    # model = build_module(cfg.model, MODELS,)
    # if cfg.model.get("from_pretrained", None) is not None: #and os.path.isfile(cfg.model.from_pretrained):
    #     load_full_checkpoint(model, cfg.model.from_pretrained)
    #     print(f"load wanx model from {cfg.model.from_pretrained} at CPU")
    if cfg.model.get("from_pretrained", None) is not None and os.path.isfile(cfg.model.from_pretrained):
        load_checkpoint(model, cfg.model.from_pretrained)
    num_params, num_params_trainable = get_model_numel(model)
    logger.info(f"num_params:{num_params}B, num_params_trainable:{num_params_trainable}B")
    
    model = parallelize("wanx2_1_t2v", model)

    logger.info("Finish create wanx core model")

    mode = cfg.get('mode', 'FSDP')
    local_rank = dist.get_rank() % torch.cuda.device_count()
    if mode == 'DDP':
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)
    elif mode == 'FSDP':
        fpSixteen = MixedPrecision(param_dtype=dtype, reduce_dtype=torch.float, buffer_dtype=dtype)

        my_auto_wrap_policy = functools.partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: m in (list(model.blocks)),
        )
        
        model = FSDP(model, mixed_precision=fpSixteen, auto_wrap_policy=my_auto_wrap_policy, device_id=torch.cuda.current_device(),
            sharding_strategy=ShardingStrategy.FULL_SHARD, use_orig_params=True, device_mesh=device_mesh)
    
    model.eval()

    # Sharded load is faster for training-generated checkpoints
    if cfg.model.get("from_pretrained", None) is not None and os.path.isdir(cfg.model.from_pretrained):
        print(f"sharded load checkpoint from {cfg.model.from_pretrained}")
        sharded_load(cfg.model.from_pretrained, ema=model, mode="all") # set mode="other" to be compatible with seperate "model+ema" checkpoints
                
    # == build scheduler ==
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    # ======================================================
    # inference
    # ======================================================
    s_t = time.time()
    # == load prompts ==
    prompts = cfg.get("prompt", None)
    start_idx = cfg.get("start_index", 0)
    test_dirs = os.listdir(cfg.prompt_path)
    end_index = cfg.get("end_index", len(test_dirs))
    test_dirs = test_dirs[start_idx: end_index]

    prompts = list()
    paths = list()
    for test_dir in test_dirs:
        test_dir_path = os.path.join(cfg.prompt_path, test_dir)
        paths.append(test_dir_path)

        caption_path = os.path.join(test_dir_path, 'caption.txt')
        with open(caption_path, 'r', encoding='utf-8') as file:
            content = file.read()

        prompts.append(content)

    prompts = prompts + (prompts[:(data_parallel_size - len(prompts) % data_parallel_size)] if len(prompts) % data_parallel_size != 0 else [])
    paths = paths + (paths[:(data_parallel_size - len(paths) % data_parallel_size)] if len(paths) % data_parallel_size != 0 else [])
    # == prepare reference ==
    reference_path = cfg.get("reference_path", [""] * len(prompts))
    mask_strategy = cfg.get("mask_strategy", [""] * len(prompts))
    assert len(reference_path) == len(prompts), "Length of reference must be the same as prompts"
    assert len(mask_strategy) == len(prompts), "Length of mask_strategy must be the same as prompts"

    # == prepare arguments ==
    fps = cfg.fps
    save_fps = cfg.get("save_fps", fps // cfg.get("frame_interval", 1))
    multi_resolution = cfg.get("multi_resolution", None)
    batch_size = cfg.get("batch_size", 1)
    num_sample = cfg.get("num_sample", 1)
    loop = cfg.get("loop", 1)
    condition_frame_length = cfg.get("condition_frame_length", 5)
    condition_frame_edit = cfg.get("condition_frame_edit", 0.0)
    align = cfg.get("align", None)

    save_dir = cfg.save_dir
    # model_name = "/".join(cfg.model.from_pretrained.split("/")[-3:-1])
    # save_dir = os.path.join(save_dir, model_name)
    os.makedirs(save_dir, exist_ok=True)
    sample_name = cfg.get("sample_name", None)
    prompt_as_path = cfg.get("prompt_as_path", False)

    is_cond_cfg = cfg.get("is_cond_cfg", False)
    if is_cond_cfg:
        print("Open condition cfg durring inference")

    prompts = prompts[data_parallel_rank::data_parallel_size]
    paths = paths[data_parallel_rank::data_parallel_size]
    mask_strategy = mask_strategy[data_parallel_rank::data_parallel_size]
    reference_path = reference_path[data_parallel_rank::data_parallel_size]
    # == Iter over all samples ==
    progress_bar = tqdm(range(len(prompts)), disable=not is_main_process_bool)
    progress_bar.set_description("Steps")
    
    for i in range(0, len(prompts), batch_size):
        # == prepare batch prompts ==
        batch_prompts = prompts[i : i + batch_size]
        batch_paths = paths[i : i + batch_size]
        ms = mask_strategy[i : i + batch_size]
        refs = reference_path[i : i + batch_size]

        if ";" in batch_prompts[0]:
            batch_prompts_en = []
            for i in range(len(batch_prompts)):
                batch_prompts_en.append(batch_prompts[i].split(";")[0])
                batch_prompts[i] = ";".join(batch_prompts[i].split(";")[1:])
        else:
            batch_prompts_en = batch_prompts
            batch_paths_en = batch_paths
        
        if verbose:
            acc_logger.info(log_rank(f"idx:{i}, batch_prompts: {batch_prompts}"))
                
        # == get json from prompts ==
        batch_prompts, refs, ms = extract_json_from_prompts(batch_prompts, refs, ms)
        original_batch_prompts = batch_prompts_en
        # original_batch_paths = batch_paths_en

        # == get reference for condition ==
        refs = collect_references_batch(refs, vae, image_size)

        # == multi-resolution info ==
        model_args = prepare_multi_resolution_info(
            multi_resolution, len(batch_prompts), image_size, num_frames, fps, device, dtype
        )
        # == Iter over number of sampling for one prompt ==
        for k in range(num_sample):
            # == prepare save paths ==
            save_paths = [
                get_save_path_name(
                    save_dir,
                    sample_name=sample_name,
                    sample_idx=start_idx + idx,
                    prompt=original_batch_prompts[idx],
                    prompt_as_path=prompt_as_path,
                    num_sample=num_sample,
                    k=k,
                )
                for idx in range(len(batch_prompts))
            ]

            # NOTE: Skip if the sample already exists
            # This is useful for resuming sampling VBench
            if prompt_as_path and all_exists(save_paths):
                continue

            # == process prompts step by step ==
            # 0. split prompt
            # each element in the list is [prompt_segment_list, loop_idx_list]
            batched_prompt_segment_list = []
            batched_path_segment_list = []
            batched_loop_idx_list = []
            for prompt, path in zip(batch_prompts,batch_paths):
                prompt_segment_list, loop_idx_list = split_prompt(prompt)
                path_segment_list, loop_idx_list = split_prompt(prompt)
                batched_prompt_segment_list.append(prompt_segment_list)
                batched_path_segment_list.append(path_segment_list)
                batched_loop_idx_list.append(loop_idx_list)

            # 2. append score
            for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                batched_prompt_segment_list[idx] = append_score_to_prompts(
                    prompt_segment_list,
                    aes=cfg.get("aes", None),
                    flow=cfg.get("flow", None),
                    camera_motion=cfg.get("camera_motion", None),
                )

            # 3. merge to obtain the final prompt
            batch_prompts = []
            for prompt_segment_list, loop_idx_list in zip(batched_prompt_segment_list, batched_loop_idx_list):
                batch_prompts.append(merge_prompt(prompt_segment_list, loop_idx_list))

            # == Iter over loop generation ==
            video_clips = []
            for loop_i in range(loop):
                # == get prompt for loop i ==
                batch_prompts_loop = extract_prompts_loop(batch_prompts, loop_i)
                batch_paths_loop =  extract_prompts_loop(batch_paths, loop_i)

                # == add condition frames for loop ==
                if loop_i > 0:
                    refs, ms = append_generated(
                        vae, video_clips[-1], refs, ms, loop_i, condition_frame_length, condition_frame_edit
                    )

                # == sampling ==
                generator = torch.Generator(device=device).manual_seed(cfg.get("seed", 1024))
                z = torch.randn(len(batch_prompts), vae.model.z_dim, *latent_size, device=device, generator=generator, dtype=dtype)
                # masks = apply_mask_strategy(z, refs, ms, loop_i, align=align)
                
                y = encode_prompt(prompt=batch_prompts_loop,neg_prompt=cfg.sample_neg_prompt,
                                  text_encoder=text_encoder,
                                  max_seq_len=cfg.max_seq_len)
                
                ########################## Obtain batch img embeddings and batch text embeddings
                batch_mask_text_embeddings_list = list()
                batch_mask_img_embeddings_list = list()
                batch_mask_img_embeddings_list_null = list()
                
                for path, prompt in zip(batch_paths_loop, batch_prompts_loop):
                    mask_dir_path = path
                    mask_dict = obtain_mask_dict(mask_dir_path, image_size)
                    ##### obtain text embeddings
                    human_list = list()
                    mask_dict_keys = [human_key for human_key in mask_dict.keys()]
                    for mask_dict_key in mask_dict_keys:
                        if 'human_x_' in mask_dict_key: 
                            human_list.append({mask_dict_key: mask_dict[mask_dict_key]}) 
                            del mask_dict[mask_dict_key] 

                    masks_texts = list(map(str, mask_dict.keys()))
                    mask_text_embeddings= find_mask_embedding(masks_texts, text_encoder, prompt)

                    ##### obtain img embeddings
                    mask_img_embeddings = list()
                    mask_keys = list()
                    mask_img_embeddings_dict = dict()
                    mask_img_embeddings_null_dict = dict()

                    if len(mask_dict) != 0:
                        for mask_key, mask_value in mask_dict.items():
                            mask_value = random.choice(mask_value).unsqueeze(1).to(device, dtype)
                            mask_img_embeddings.append(mask_value)
                            mask_keys.append(mask_key)

                        mask_img_embedding_ = torch.stack(mask_img_embeddings, dim=0)
                        if is_cond_cfg:
                            mask_img_embedding_null = torch.ones_like(mask_img_embedding_)  
                            masks_value_out_null = vae.encode(mask_img_embedding_null)
                            for (mask_key, mask_value) in zip(mask_keys, masks_value_out_null):
                                mask_img_embeddings_null_dict[mask_key] = mask_value

                        masks_value_out = vae.encode(mask_img_embedding_)
                        
                        for (mask_key, mask_value) in zip(mask_keys, masks_value_out):
                            mask_img_embeddings_dict[mask_key] = mask_value

                    for human_i, mask_human_dict in enumerate(human_list):
                        human_key = 'human_x_' + str(human_i) 
                        masks_texts = list(map(str, mask_human_dict[human_key].keys()))
                        mask_text_human_embeddings= find_mask_embedding( masks_texts, text_encoder, prompt)
                        mask_text_embeddings.update({human_key:mask_text_human_embeddings})

                        mask_human_img_embeddings = list()
                        mask_human_keys = list()
                        mask_human_img_embeddings_dict = dict()
                        mask_human_img_embeddings_null_dict = dict()
                        for mask_key, mask_value in mask_human_dict[human_key].items():
                            # mask_value = torch.stack(mask_value, dim=2).to(device, dtype)
                            mask_value = random.choice(mask_value).unsqueeze(1).to(device, dtype)
                            mask_human_img_embeddings.append(mask_value)
                            mask_human_keys.append(mask_key)
                        mask_human_img_embedding_ = torch.stack(mask_human_img_embeddings, dim=0)    
                    
                        masks_human_value_out = vae.encode(mask_human_img_embedding_)
                        for (mask_key, mask_value) in zip(mask_human_keys, masks_human_value_out):
                            mask_human_img_embeddings_dict[mask_key] = mask_value

                        if is_cond_cfg:
                            masks_human_value_out_null = [torch.zeros_like(mask_value) for mask_value in masks_human_value_out]
                            for (mask_key, mask_value) in zip(mask_human_keys, masks_human_value_out_null):
                                mask_human_img_embeddings_null_dict[mask_key] = mask_value
                            mask_img_embeddings_null_dict.update({human_key:mask_human_img_embeddings_null_dict})

                        mask_img_embeddings_dict.update({human_key:mask_human_img_embeddings_dict})
                        
                    batch_mask_img_embeddings_list.append(mask_img_embeddings_dict)
                    batch_mask_text_embeddings_list.append(mask_text_embeddings)

                    if is_cond_cfg:
                        batch_mask_img_embeddings_list_null.append(mask_img_embeddings_null_dict)
                
                if is_cond_cfg:
                    model_args['img_emb_null'] = batch_mask_img_embeddings_list_null
               
                # Remove profiler to avoid OOM issues - it tracks memory and operations which consumes extra GPU memory
                # If profiling is needed, use a separate profiling script with smaller batch sizes
                samples = scheduler.sample(
                    model,
                    y,
                    z=z,
                    prompts=batch_prompts_loop,
                    device=device,
                    additional_args=model_args,
                    text_emb = batch_mask_text_embeddings_list,
                    img_emb = batch_mask_img_embeddings_list,
                    progress=verbose,
                    mask=None,
                    generator=generator,
                    cfg=cfg,
                    mode="t2v_alchemy"
                )
                
                samples = vae.decode(samples)
                video_clips.append(samples)

            if sp_parallel_rank == 0:
                # == save samples ==
                for idx, batch_prompt in enumerate(batch_prompts):
                    save_path = save_paths[idx]
                    video = [video_clips[j][idx] for j in range(loop)]
                    for j in range(1, loop):
                        video[j] = video[j][:, dframe_to_frame(condition_frame_length) :]
                    video = torch.cat(video, dim=1)
                    # fst, sed = os.path.splitext(save_path)
                    save_path = f"{save_path}_sp{sp_degree}_{image_size[0]}x{image_size[1]}"
                    # Use save_sample_imageio instead of save_sample to avoid OOM
                    # save_sample uses torchvision.write_video which may cache entire video in memory
                    # save_sample_imageio writes frame-by-frame, more memory efficient
                    save_path = save_sample_imageio(
                        video,
                        fps=save_fps,
                        save_path=save_path,
                        verbose=verbose,
                    )
                    if save_path.endswith(".mp4") and cfg.get("watermark", False):
                        time.sleep(1)  # prevent loading previous generated video
                        add_watermark(save_path)
        start_idx += len(batch_prompts)
        progress_bar.update(1)
    e_t = time.time()
    
    logger.info(f"Inference finished. Per prompt time:{(e_t-s_t)/len(prompts):.2f}")
    logger.info("Saved %s samples to %s", start_idx, save_dir)
    
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
