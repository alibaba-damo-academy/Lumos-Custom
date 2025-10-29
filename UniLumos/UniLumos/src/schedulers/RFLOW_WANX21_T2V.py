import torch
import inspect
from tqdm import tqdm
import numpy as np
from .rectified_flow import FlowDPMSolverMultistepScheduler

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

class RFLOW_WANX21_T2V:
    def __init__(
        self,
        num_timesteps=1000,
        cfg_scale=6.0,
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

        deg_latent = model_args["video_degradation_latent"]
        bg_latent = model_args["video_background_latent"]

        arg_c = {'context': context, 'seq_len': max_seq_len, 'x_deg': deg_latent, 'x_bg': bg_latent}
        arg_null = {'context': context_null, 'seq_len': max_seq_len, 'x_deg': deg_latent, 'x_bg': bg_latent}
        
        return arg_c, arg_null

    def sample(
        self,
        model,
        y,
        z,
        prompts,
        device,
        additional_args=None,
        mask=None,
        guidance_scale=None,
        generator=None,
        progress=True,
        mode="t2v",
        sample_steps=25,
        sample_shift=8.0
    ):  

        # if no specific guidance scale is provided, use the default scale when initializing the scheduler
        if guidance_scale is None:
            guidance_scale = self.cfg_scale
        
        # text encoding        
        model_args = y
        if additional_args is not None:
            model_args.update(additional_args)

        # gei sampling sigmas
        sampling_sigmas = get_sampling_sigmas(sample_steps, sample_shift)

        timesteps, _ = retrieve_timesteps(self.scheduler, device=device, sigmas=sampling_sigmas, shift=1)

        assert mode in ("t2v", "i2v", "v2v"), f"Error: the {mode=} not in the choices ('i2v', 't2v', 'v2v')"

        if mode == "t2v":
            arg_c, arg_null = self.create_wan_t2v_args(model_args)

        for i, t in tqdm(enumerate(timesteps), desc="Timestep", total=len(timesteps)):
            latent_model_input = z

            timestep = [t]
            timestep = torch.stack(timestep)
            noise_pred_cond = model(latent_model_input, t=timestep, **arg_c)
            noise_pred_uncond = model(latent_model_input, t=timestep, **arg_null)

            noise_pred = noise_pred_uncond + self.cfg_scale * (noise_pred_cond - noise_pred_uncond)

            z = self.scheduler.step(noise_pred, t, z, return_dict=False, generator=generator)[0]

        return z
    
    def get_timesteps(self, num_inference_steps, timesteps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = timesteps[t_start * self.scheduler.order :]

        return timesteps, num_inference_steps - t_start