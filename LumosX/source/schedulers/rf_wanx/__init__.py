import torch
import inspect
from tqdm import tqdm
import numpy as np
import torch.amp as amp

from source.registry import SCHEDULERS
from source.models.wanx.utils import TensorList
import torch.distributed as dist
from .rectified_flow import FlowDPMSolverMultistepScheduler, timestep_transform

def get_sampling_sigmas(sampling_steps, shift):
    sigma = np.linspace(1, 0, sampling_steps+1)[:sampling_steps]
    sigma = (shift * sigma / (1 + (shift - 1) * sigma))

    return sigma

def retrieve_timesteps(
    scheduler,
    num_inference_steps= None,
    device= None,
    timesteps= None,
    sigmas = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    
    return timesteps, num_inference_steps

@SCHEDULERS.register_module("rflow-wanx")
class RFLOW_WANX21_T2V:
    def __init__(
        self,
        num_timesteps=1000,
        cfg_scale=4.0,
        use_discrete_timesteps=False,
        use_timestep_transform=False,
        use_fixed_timestep_transform=False,
        transform_scale=1,
        sample_method="uniform",
        **kwargs):
        self.num_timesteps = num_timesteps
        self.cfg_scale = cfg_scale
        self.use_discrete_timesteps = use_discrete_timesteps
        self.use_timestep_transform = use_timestep_transform
        self.use_fixed_timestep_transform = use_fixed_timestep_transform

        self.scheduler = FlowDPMSolverMultistepScheduler(
            num_train_timesteps=num_timesteps,
            transform_scale=transform_scale,
            use_dynamic_shifting=False,
            use_discrete_timesteps=use_discrete_timesteps,
            use_timestep_transform=use_timestep_transform,
            use_fixed_timestep_transform=use_fixed_timestep_transform,
            sample_method=sample_method)

    def create_wan_t2v_args(self, model_args):
        context = model_args["context"]
        context_null = model_args["context_null"]
        max_seq_len = model_args["max_seq_len"]
        arg_c = {'context': context, 'seq_len': max_seq_len}
        arg_null = {'context': context_null, 'seq_len': max_seq_len}
        
        return arg_c, arg_null


    def create_wan_t2v_alchemy_args(self, model_args, text_emb, img_emb):
        context = model_args["context"]
        context_null = model_args["context_null"]
        max_seq_len = model_args["max_seq_len"]
        arg_c = {'context': context, 'seq_len': max_seq_len, 'mask_text_emb': text_emb, 'mask_img_emb': img_emb}
        arg_null = {'context': context_null, 'seq_len': max_seq_len,  'mask_text_emb': text_emb, 'mask_img_emb': img_emb}
        if 'mask_clip_img_emb' in model_args:
            arg_c['mask_clip_img_emb'] = model_args['mask_clip_img_emb']
            arg_null['mask_clip_img_emb'] = model_args['mask_clip_img_emb']
        if 'mask_text_emb_null' in model_args:
            arg_null['mask_text_emb'] = model_args['mask_text_emb_null']
        if 'img_emb_null' in model_args:
            arg_null['mask_img_emb'] = model_args['img_emb_null']
        if 'mask_clip_img_emb_null' in model_args:
            arg_null['mask_clip_img_emb'] = model_args['mask_clip_img_emb_null']
        # if 'img_emb_null' in model_args:
        #     arg_null = {'context': context_null, 'seq_len': max_seq_len,  'mask_text_emb': text_emb, 'mask_img_emb': model_args['img_emb_null']}
        # else:
        #     arg_null = {'context': context_null, 'seq_len': max_seq_len,  'mask_text_emb': text_emb, 'mask_img_emb': img_emb}
        return arg_c, arg_null

    def create_wan_i2v_args(self, model_args):
        context = model_args["context"]
        context_null = model_args["context_null"]
        clip_feat = model_args["clip_feat"]
        max_seq_len = model_args["max_seq_len"]
        img_mask_latent = model_args["img_mask_latent"]
        
        arg_c = {'context': context,
                 'clip_fea': clip_feat,
                 'seq_len': max_seq_len,
                 'y': img_mask_latent}
        
        arg_null = {'context': context_null, 
                    'clip_fea': clip_feat,
                    'seq_len': max_seq_len,
                    'y': img_mask_latent}
        
        return arg_c, arg_null

    def sample(
        self,
        model,
        y,
        z,
        prompts,
        device,
        additional_args=None,
        text_emb = None,
        img_emb = None,
        mask=None,
        guidance_scale=None,
        generator=None,
        progress=True,
        cfg=None,
        mode="t2v"
    ):
        # if no specific guidance scale is provided, use the default scale when initializing the scheduler
        if guidance_scale is None:
            guidance_scale = self.cfg_scale
        
        # text encoding        
        model_args = y
        if additional_args is not None:
            model_args.update(additional_args)

        # gei sampling sigmas
        sampling_sigmas = get_sampling_sigmas(cfg.scheduler.sample_steps, cfg.scheduler.sample_shift)

        timesteps, _ = retrieve_timesteps(self.scheduler, device=device, sigmas=sampling_sigmas, shift=1)
        # assert mode in ("t2v", "i2v", "v2v", "t2v_alchemy"), f"Error: the {mode=} not in the choices ('i2v', 't2v', 'v2v', 't2v_alchemy')"

        if mode == "t2v":
            arg_c, arg_null = self.create_wan_t2v_args(model_args)

        if mode == "t2v_alchemy":
            arg_c, arg_null = self.create_wan_t2v_alchemy_args(model_args, text_emb, img_emb)
                
        if mode == "i2v":
            arg_c, arg_null = self.create_wan_i2v_args(model_args)
            
        if mode == "v2v":
            arg_c, arg_null = self.create_wan_t2v_args(model_args)
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler,
                                                                num_inference_steps=cfg.scheduler.sample_steps,
                                                                device=device,
                                                                timesteps=None)
            timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, timesteps, strength=cfg.strength, device=device)
            latent_timestep = timesteps[:1].repeat(1)
            
            ref_video_latent = self.scheduler.add_noise(model_args["ref_video_latent"], z, latent_timestep)
            
            z = ref_video_latent

        for i, t in tqdm(enumerate(timesteps), desc="Timestep", total=len(timesteps), disable=not (not dist.is_initialized() or dist.get_rank() == 0)):
            latent_model_input = z
            timestep = [t]
            timestep = torch.stack(timestep)
            with torch.no_grad():
                noise_pred_cond = model(latent_model_input, t=timestep, **arg_c)
                noise_pred_uncond = model(latent_model_input, t=timestep, **arg_null)
            
            noise_pred = noise_pred_uncond + self.cfg_scale * (noise_pred_cond - noise_pred_uncond)

            z = self.scheduler.step(noise_pred[:, :, :z.shape[2]], t, z, return_dict=False, generator=generator)[0]

        return z
    
    def get_timesteps(self, num_inference_steps, timesteps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = timesteps[t_start * self.scheduler.order :]

        return timesteps, num_inference_steps - t_start

    def training_losses(self,
                        model,
                        x_start,
                        model_kwargs=None,
                        noise=None,
                        xt_mask=None,
                        xs_mask=None,
                        weights=None,
                        t=None,
                        mae_loss_coef=None,
                        unpatchify_loss=False,
                        mode='t2v'):
        return self.scheduler.training_losses(
            model,
            x_start,
            model_kwargs,
            noise,
            xt_mask,
            xs_mask,
            weights,
            t,
            mae_loss_coef,
            unpatchify_loss,
            mode=mode)

    def training_losses_video_alchemy(self,
                        model,
                        x_start,
                        model_kwargs=None,
                        noise=None,
                        xt_mask=None,
                        xs_mask=None,
                        weights=None,
                        t=None,
                        mae_loss_coef=None,
                        unpatchify_loss=False,
                        mode='video_alchemy'):
        return self.scheduler.training_losses_video_alchemy(
            model,
            x_start,
            model_kwargs,
            noise,
            xt_mask,
            xs_mask,
            weights,
            t,
            mae_loss_coef,
            unpatchify_loss,
            mode=mode)

