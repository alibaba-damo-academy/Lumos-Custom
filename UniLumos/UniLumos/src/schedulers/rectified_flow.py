# Copied from https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
# Convert dpm solver for flow matching
# Di Chen, guangpan.cd@alibaba-inc.com

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from torch.distributions import LogisticNormal
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import deprecate
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin, SchedulerOutput

from .gaussian_diffusion import _extract_into_tensor, mean_flat

def xpadding(x, patch_size):
    
    T, H, W = x.shape[2:5]
    if W % patch_size[2] != 0:
        x = F.pad(x, (0, patch_size[2] - W % patch_size[2]))
    if H % patch_size[1] != 0:
        x = F.pad(x, (0, 0, 0, patch_size[1] - H % patch_size[1]))
    if T % patch_size[0] != 0:
        x = F.pad(x, (0, 0, 0, 0, 0, patch_size[0] - T % patch_size[0]))
    return x

def patchify(x, patch_size=(1, 2, 2)):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    x = xpadding(x, patch_size)
    p, c = patch_size, x.shape[1]
    T, H, W = x.shape[2:5]
    t, h, w = T // p[0], H // p[1], W // p[2]
    x = x.reshape(shape=(x.shape[0], c, t, p[0], h, p[1], w, p[2]))
    x = torch.einsum('nctohpwq->nthwopqc', x)
    x = x.reshape(shape=(x.shape[0], t * h * w, p[0]*p[1]*p[2] * c))
    return x

def mae_loss(target, pred, mask, patch_size, norm_pix_loss=True):
    target = patchify(target, patch_size)
    pred = patchify(pred, patch_size)
    if norm_pix_loss:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.e-6)**.5

    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
    
    mask = mask.view(target.shape[0], -1)
    loss = (loss * mask).sum(dim=1) / mask.sum(dim=1)  # mean loss on removed patches, (N)
    assert loss.ndim == 1
    return loss

def timestep_transform(
    t,
    model_kwargs,
    base_resolution=512 * 512,
    base_num_frames=1,
    scale=1.0,
    num_timesteps=1,
    fixed_timestep_transform=False,
):
    if fixed_timestep_transform:
        t = t / num_timesteps
        t = scale * t / (1 + (scale - 1) * t)
        t = t * num_timesteps
        return t
    
    # Force fp16 input to fp32 to avoid nan output
    for key in ["height", "width", "num_frames"]:
        if model_kwargs[key].dtype == torch.float16:
            model_kwargs[key] = model_kwargs[key].float()
            
    t = t / num_timesteps
    resolution = model_kwargs["height"] * model_kwargs["width"]
    ratio_space = (resolution / base_resolution).sqrt()
    ratio_space[ratio_space < 1] =  1
    # NOTE: currently, we do not take fps into account
    # NOTE: temporal_reduction is hardcoded, this should be equal to the temporal reduction factor of the vae
    if model_kwargs["num_frames"][0] == 1:
        num_frames = torch.ones_like(model_kwargs["num_frames"])
    else:
        temporal_reduction_factor = model_kwargs.get("temporal_reduction_factor", 4)
        num_frames = model_kwargs["num_frames"] // temporal_reduction_factor + 1
    ratio_time = (num_frames / base_num_frames).sqrt()

    ratio = ratio_space * ratio_time * scale
    new_t = ratio * t / (1 + (ratio - 1) * t)

    new_t = new_t * num_timesteps
    return new_t


class FlowDPMSolverMultistepScheduler(SchedulerMixin, ConfigMixin):
    """
    `FlowDPMSolverMultistepScheduler` is a fast dedicated high-order solver for diffusion ODEs.
    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.
    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model. This determines the resolution of the diffusion process.
        solver_order (`int`, defaults to 2):
            The DPMSolver order which can be `1`, `2`, or `3`. It is recommended to use `solver_order=2` for guided
            sampling, and `solver_order=3` for unconditional sampling. This affects the number of model outputs stored
            and used in multistep updates.
        prediction_type (`str`, defaults to "flow_prediction"):
            Prediction type of the scheduler function; must be `flow_prediction` for this scheduler, which predicts
            the flow of the diffusion process.
        shift (`float`, *optional*, defaults to 1.0):
            A factor used to adjust the sigmas in the noise schedule. It modifies the step sizes during the sampling
            process.
        use_dynamic_shifting (`bool`, defaults to `False`):
            Whether to apply dynamic shifting to the timesteps based on image resolution. If `True`, the shifting is
            applied on the fly.
        thresholding (`bool`, defaults to `False`):
            Whether to use the "dynamic thresholding" method. This method adjusts the predicted sample to prevent
            saturation and improve photorealism.
        dynamic_thresholding_ratio (`float`, defaults to 0.995):
            The ratio for the dynamic thresholding method. Valid only when `thresholding=True`.
        sample_max_value (`float`, defaults to 1.0):
            The threshold value for dynamic thresholding. Valid only when `thresholding=True` and
            `algorithm_type="dpmsolver++"`.
        algorithm_type (`str`, defaults to `dpmsolver++`):
            Algorithm type for the solver; can be `dpmsolver`, `dpmsolver++`, `sde-dpmsolver` or `sde-dpmsolver++`. The
            `dpmsolver` type implements the algorithms in the [DPMSolver](https://huggingface.co/papers/2206.00927)
            paper, and the `dpmsolver++` type implements the algorithms in the
            [DPMSolver++](https://huggingface.co/papers/2211.01095) paper. It is recommended to use `dpmsolver++` or
            `sde-dpmsolver++` with `solver_order=2` for guided sampling like in Stable Diffusion.
        solver_type (`str`, defaults to `midpoint`):
            Solver type for the second-order solver; can be `midpoint` or `heun`. The solver type slightly affects the
            sample quality, especially for a small number of steps. It is recommended to use `midpoint` solvers.
        lower_order_final (`bool`, defaults to `True`):
            Whether to use lower-order solvers in the final steps. Only valid for < 15 inference steps. This can
            stabilize the sampling of DPMSolver for steps < 15, especially for steps <= 10.
        euler_at_final (`bool`, defaults to `False`):
            Whether to use Euler's method in the final step. It is a trade-off between numerical stability and detail
            richness. This can stabilize the sampling of the SDE variant of DPMSolver for small number of inference
            steps, but sometimes may result in blurring.
        final_sigmas_type (`str`, *optional*, defaults to "zero"):
            The final `sigma` value for the noise schedule during the sampling process. If `"sigma_min"`, the final
            sigma is the same as the last sigma in the training schedule. If `zero`, the final sigma is set to 0.
        lambda_min_clipped (`float`, defaults to `-inf`):
            Clipping threshold for the minimum value of `lambda(t)` for numerical stability. This is critical for the
            cosine (`squaredcos_cap_v2`) noise schedule.
        variance_type (`str`, *optional*):
            Set to "learned" or "learned_range" for diffusion models that predict variance. If set, the model's output
            contains the predicted Gaussian variance.
    """

    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        solver_order: int = 2,
        prediction_type: str = "flow_prediction",
        shift: Optional[float] = 1.0,
        use_dynamic_shifting=False,
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        sample_max_value: float = 1.0,
        algorithm_type: str = "dpmsolver++",
        solver_type: str = "midpoint",
        lower_order_final: bool = True,
        euler_at_final: bool = False,
        final_sigmas_type: Optional[str] = "zero",  # "zero", "sigma_min"
        lambda_min_clipped: float = -float("inf"),
        variance_type: Optional[str] = None,
        invert_sigmas: bool = False,
        num_sampling_steps=10,
        sample_method="uniform",
        loc=0.0,
        scale=1.0,
        use_discrete_timesteps=False,
        use_timestep_transform=False,
        use_fixed_timestep_transform=False,
        transform_scale=1.0,
    ):
        self.num_timesteps=num_train_timesteps
        self.num_sampling_steps=num_sampling_steps
        self.use_discrete_timesteps=use_discrete_timesteps
        self.sample_method=sample_method
        self.loc=loc
        self.scale=scale
        self.use_timestep_transform=use_timestep_transform
        self.use_fixed_timestep_transform=use_fixed_timestep_transform
        self.transform_scale=transform_scale
        if algorithm_type in ["dpmsolver", "sde-dpmsolver"]:
            deprecation_message = f"algorithm_type {algorithm_type} is deprecated and will be removed in a future version. Choose from `dpmsolver++` or `sde-dpmsolver++` instead"
            deprecate("algorithm_types dpmsolver and sde-dpmsolver", "1.0.0", deprecation_message)

        # settings for DPM-Solver
        if algorithm_type not in ["dpmsolver", "dpmsolver++", "sde-dpmsolver", "sde-dpmsolver++"]:
            if algorithm_type == "deis":
                self.register_to_config(algorithm_type="dpmsolver++")
            else:
                raise NotImplementedError(f"{algorithm_type} is not implemented for {self.__class__}")

        if solver_type not in ["midpoint", "heun"]:
            if solver_type in ["logrho", "bh1", "bh2"]:
                self.register_to_config(solver_type="midpoint")
            else:
                raise NotImplementedError(f"{solver_type} is not implemented for {self.__class__}")

        if algorithm_type not in ["dpmsolver++", "sde-dpmsolver++"] and final_sigmas_type == "zero":
            raise ValueError(
                f"`final_sigmas_type` {final_sigmas_type} is not supported for `algorithm_type` {algorithm_type}. Please choose `sigma_min` instead."
            )

        # setable values
        self.num_inference_steps = None
        alphas = np.linspace(1, 1 / num_train_timesteps, num_train_timesteps)[::-1].copy()
        sigmas = 1.0 - alphas
        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32)

        if not use_dynamic_shifting:
            # when use_dynamic_shifting is True, we apply the timestep shifting on the fly based on the image resolution
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)  # pyright: ignore

        self.sigmas = sigmas
        self.timesteps = sigmas * num_train_timesteps

        # 后续需要通过 == 寻找 index，因此需要转换为 int
        self.timesteps = torch.round(self.timesteps)
        self.timesteps = self.timesteps.to(torch.int64)
        
        self.model_outputs = [None] * solver_order
        self.lower_order_nums = 0
        self._step_index = None
        self._begin_index = None

        self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()
        
        assert sample_method in ["uniform", "logit-normal"]
        if sample_method == "logit-normal":
            self.distribution = LogisticNormal(torch.tensor([loc]), torch.tensor([scale]))
            self.sample_t = lambda x: self.distribution.sample((x.shape[0],))[:, 0].to(x.device)

    @property
    def step_index(self):
        """
        The index counter for current timestep. It will increase 1 after each scheduler step.
        """
        return self._step_index

    @property
    def begin_index(self):
        """
        The index for the first timestep. It should be set from pipeline with `set_begin_index` method.
        """
        return self._begin_index

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index
    def set_begin_index(self, begin_index: int = 0):
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.
        Args:
            begin_index (`int`):
                The begin index for the scheduler.
        """
        self._begin_index = begin_index

    # Modified from diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler.set_timesteps
    def set_timesteps(
        self,
        num_inference_steps: Union[int, None] = None,
        device: Union[str, torch.device] = None,
        sigmas: Optional[List[float]] = None,
        mu: Optional[Union[float, None]] = None,
        shift: Optional[Union[float, None]] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).
        Args:
            num_inference_steps (`int`):
                Total number of the spacing of the time steps.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """

        if self.config.use_dynamic_shifting and mu is None:
            raise ValueError(" you have to pass a value for `mu` when `use_dynamic_shifting` is set to be `True`")

        if sigmas is None:
            sigmas = np.linspace(self.sigma_max, self.sigma_min, num_inference_steps + 1).copy()[:-1]  # pyright: ignore

        if self.config.use_dynamic_shifting:
            sigmas = self.time_shift(mu, 1.0, sigmas)  # pyright: ignore
        else:
            if shift is None:
                shift = self.config.shift
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)  # pyright: ignore

        if self.config.final_sigmas_type == "sigma_min":
            sigma_last = ((1 - self.alphas_cumprod[0]) / self.alphas_cumprod[0]) ** 0.5
        elif self.config.final_sigmas_type == "zero":
            sigma_last = 0
        else:
            raise ValueError(
                f"`final_sigmas_type` must be one of 'zero', or 'sigma_min', but got {self.config.final_sigmas_type}"
            )

        timesteps = sigmas * self.config.num_train_timesteps
        sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)  # pyright: ignore

        self.sigmas = torch.from_numpy(sigmas)
        self.timesteps = torch.from_numpy(timesteps).to(device=device, dtype=torch.int64)

        self.num_inference_steps = len(timesteps)

        self.model_outputs = [
            None,
        ] * self.config.solver_order
        self.lower_order_nums = 0

        self._step_index = None
        self._begin_index = None
        self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication

    # Copied from diffusers.schedulers.scheduling_ddpm.DDPMScheduler._threshold_sample
    def _threshold_sample(self, sample: torch.Tensor) -> torch.Tensor:
        """
        "Dynamic thresholding: At each sampling step we set s to a certain percentile absolute pixel value in xt0 (the
        prediction of x_0 at timestep t), and if s > 1, then we threshold xt0 to the range [-s, s] and then divide by
        s. Dynamic thresholding pushes saturated pixels (those near -1 and 1) inwards, thereby actively preventing
        pixels from saturation at each step. We find that dynamic thresholding results in significantly better
        photorealism as well as better image-text alignment, especially when using very large guidance weights."
        https://arxiv.org/abs/2205.11487
        """
        dtype = sample.dtype
        batch_size, channels, *remaining_dims = sample.shape

        if dtype not in (torch.float32, torch.float64):
            sample = sample.float()  # upcast for quantile calculation, and clamp not implemented for cpu half

        # Flatten sample for doing quantile calculation along each image
        sample = sample.reshape(batch_size, channels * np.prod(remaining_dims))

        abs_sample = sample.abs()  # "a certain percentile absolute pixel value"

        s = torch.quantile(abs_sample, self.config.dynamic_thresholding_ratio, dim=1)
        s = torch.clamp(
            s, min=1, max=self.config.sample_max_value
        )  # When clamped to min=1, equivalent to standard clipping to [-1, 1]
        s = s.unsqueeze(1)  # (batch_size, 1) because clamp will broadcast along dim=0
        sample = torch.clamp(sample, -s, s) / s  # "we threshold xt0 to the range [-s, s] and then divide by s"

        sample = sample.reshape(batch_size, channels, *remaining_dims)
        sample = sample.to(dtype)

        return sample

    # Copied from diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler._sigma_to_t
    def _sigma_to_t(self, sigma):
        return sigma * self.config.num_train_timesteps

    def _sigma_to_alpha_sigma_t(self, sigma):
        return 1 - sigma, sigma

    # Copied from diffusers.schedulers.scheduling_flow_match_euler_discrete.set_timesteps
    def time_shift(self, mu: float, sigma: float, t: torch.Tensor):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.convert_model_output
    def convert_model_output(
        self,
        model_output: torch.Tensor,
        *args,
        sample: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Convert the model output to the corresponding type the DPMSolver/DPMSolver++ algorithm needs. DPM-Solver is
        designed to discretize an integral of the noise prediction model, and DPM-Solver++ is designed to discretize an
        integral of the data prediction model.
        <Tip>
        The algorithm and model type are decoupled. You can use either DPMSolver or DPMSolver++ for both noise
        prediction and data prediction models.
        </Tip>
        Args:
            model_output (`torch.Tensor`):
                The direct output from the learned diffusion model.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
        Returns:
            `torch.Tensor`:
                The converted model output.
        """
        timestep = args[0] if len(args) > 0 else kwargs.pop("timestep", None)
        if sample is None:
            if len(args) > 1:
                sample = args[1]
            else:
                raise ValueError("missing `sample` as a required keyward argument")
        if timestep is not None:
            deprecate(
                "timesteps",
                "1.0.0",
                "Passing `timesteps` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )

        # DPM-Solver++ needs to solve an integral of the data prediction model.
        if self.config.algorithm_type in ["dpmsolver++", "sde-dpmsolver++"]:
            if self.config.prediction_type == "flow_prediction":
                sigma_t = self.sigmas[self.step_index]
                x0_pred = sample - sigma_t * model_output
            else:
                raise ValueError(
                    f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`,"
                    " `v_prediction`, or `flow_prediction` for the FlowDPMSolverMultistepScheduler."
                )

            if self.config.thresholding:
                x0_pred = self._threshold_sample(x0_pred)

            return x0_pred

        # DPM-Solver needs to solve an integral of the noise prediction model.
        elif self.config.algorithm_type in ["dpmsolver", "sde-dpmsolver"]:
            if self.config.prediction_type == "flow_prediction":
                sigma_t = self.sigmas[self.step_index]
                epsilon = sample - (1 - sigma_t) * model_output
            else:
                raise ValueError(
                    f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`,"
                    " `v_prediction` or `flow_prediction` for the FlowDPMSolverMultistepScheduler."
                )

            if self.config.thresholding:
                sigma_t = self.sigmas[self.step_index]
                x0_pred = sample - sigma_t * model_output
                x0_pred = self._threshold_sample(x0_pred)
                epsilon = model_output + x0_pred

            return epsilon

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.dpm_solver_first_order_update
    def dpm_solver_first_order_update(
        self,
        model_output: torch.Tensor,
        *args,
        sample: torch.Tensor = None,
        noise: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        One step for the first-order DPMSolver (equivalent to DDIM).
        Args:
            model_output (`torch.Tensor`):
                The direct output from the learned diffusion model.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
        Returns:
            `torch.Tensor`:
                The sample tensor at the previous timestep.
        """
        timestep = args[0] if len(args) > 0 else kwargs.pop("timestep", None)
        prev_timestep = args[1] if len(args) > 1 else kwargs.pop("prev_timestep", None)
        if sample is None:
            if len(args) > 2:
                sample = args[2]
            else:
                raise ValueError(" missing `sample` as a required keyward argument")
        if timestep is not None:
            deprecate(
                "timesteps",
                "1.0.0",
                "Passing `timesteps` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )

        if prev_timestep is not None:
            deprecate(
                "prev_timestep",
                "1.0.0",
                "Passing `prev_timestep` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )

        sigma_t, sigma_s = self.sigmas[self.step_index + 1], self.sigmas[self.step_index]  # pyright: ignore
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s, sigma_s = self._sigma_to_alpha_sigma_t(sigma_s)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s = torch.log(alpha_s) - torch.log(sigma_s)

        h = lambda_t - lambda_s
        if self.config.algorithm_type == "dpmsolver++":
            x_t = (sigma_t / sigma_s) * sample - (alpha_t * (torch.exp(-h) - 1.0)) * model_output
        elif self.config.algorithm_type == "dpmsolver":
            x_t = (alpha_t / alpha_s) * sample - (sigma_t * (torch.exp(h) - 1.0)) * model_output
        elif self.config.algorithm_type == "sde-dpmsolver++":
            assert noise is not None
            x_t = (
                (sigma_t / sigma_s * torch.exp(-h)) * sample
                + (alpha_t * (1 - torch.exp(-2.0 * h))) * model_output
                + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
            )
        elif self.config.algorithm_type == "sde-dpmsolver":
            assert noise is not None
            x_t = (
                (alpha_t / alpha_s) * sample
                - 2.0 * (sigma_t * (torch.exp(h) - 1.0)) * model_output
                + sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0) * noise
            )
        return x_t  # pyright: ignore

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.multistep_dpm_solver_second_order_update
    def multistep_dpm_solver_second_order_update(
        self,
        model_output_list: List[torch.Tensor],
        *args,
        sample: torch.Tensor = None,
        noise: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        One step for the second-order multistep DPMSolver.
        Args:
            model_output_list (`List[torch.Tensor]`):
                The direct outputs from learned diffusion model at current and latter timesteps.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
        Returns:
            `torch.Tensor`:
                The sample tensor at the previous timestep.
        """
        timestep_list = args[0] if len(args) > 0 else kwargs.pop("timestep_list", None)
        prev_timestep = args[1] if len(args) > 1 else kwargs.pop("prev_timestep", None)
        if sample is None:
            if len(args) > 2:
                sample = args[2]
            else:
                raise ValueError(" missing `sample` as a required keyward argument")
        if timestep_list is not None:
            deprecate(
                "timestep_list",
                "1.0.0",
                "Passing `timestep_list` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )

        if prev_timestep is not None:
            deprecate(
                "prev_timestep",
                "1.0.0",
                "Passing `prev_timestep` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )

        sigma_t, sigma_s0, sigma_s1 = (
            self.sigmas[self.step_index + 1],  # pyright: ignore
            self.sigmas[self.step_index],
            self.sigmas[self.step_index - 1],  # pyright: ignore
        )

        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
        alpha_s1, sigma_s1 = self._sigma_to_alpha_sigma_t(sigma_s1)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        lambda_s1 = torch.log(alpha_s1) - torch.log(sigma_s1)

        m0, m1 = model_output_list[-1], model_output_list[-2]

        h, h_0 = lambda_t - lambda_s0, lambda_s0 - lambda_s1
        r0 = h_0 / h
        D0, D1 = m0, (1.0 / r0) * (m0 - m1)
        if self.config.algorithm_type == "dpmsolver++":
            # See https://arxiv.org/abs/2211.01095 for detailed derivations
            if self.config.solver_type == "midpoint":
                x_t = (
                    (sigma_t / sigma_s0) * sample
                    - (alpha_t * (torch.exp(-h) - 1.0)) * D0
                    - 0.5 * (alpha_t * (torch.exp(-h) - 1.0)) * D1
                )
            elif self.config.solver_type == "heun":
                x_t = (
                    (sigma_t / sigma_s0) * sample
                    - (alpha_t * (torch.exp(-h) - 1.0)) * D0
                    + (alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0)) * D1
                )
        elif self.config.algorithm_type == "dpmsolver":
            # See https://arxiv.org/abs/2206.00927 for detailed derivations
            if self.config.solver_type == "midpoint":
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - (sigma_t * (torch.exp(h) - 1.0)) * D0
                    - 0.5 * (sigma_t * (torch.exp(h) - 1.0)) * D1
                )
            elif self.config.solver_type == "heun":
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - (sigma_t * (torch.exp(h) - 1.0)) * D0
                    - (sigma_t * ((torch.exp(h) - 1.0) / h - 1.0)) * D1
                )
        elif self.config.algorithm_type == "sde-dpmsolver++":
            assert noise is not None
            if self.config.solver_type == "midpoint":
                x_t = (
                    (sigma_t / sigma_s0 * torch.exp(-h)) * sample
                    + (alpha_t * (1 - torch.exp(-2.0 * h))) * D0
                    + 0.5 * (alpha_t * (1 - torch.exp(-2.0 * h))) * D1
                    + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
                )
            elif self.config.solver_type == "heun":
                x_t = (
                    (sigma_t / sigma_s0 * torch.exp(-h)) * sample
                    + (alpha_t * (1 - torch.exp(-2.0 * h))) * D0
                    + (alpha_t * ((1.0 - torch.exp(-2.0 * h)) / (-2.0 * h) + 1.0)) * D1
                    + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
                )
        elif self.config.algorithm_type == "sde-dpmsolver":
            assert noise is not None
            if self.config.solver_type == "midpoint":
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - 2.0 * (sigma_t * (torch.exp(h) - 1.0)) * D0
                    - (sigma_t * (torch.exp(h) - 1.0)) * D1
                    + sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0) * noise
                )
            elif self.config.solver_type == "heun":
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - 2.0 * (sigma_t * (torch.exp(h) - 1.0)) * D0
                    - 2.0 * (sigma_t * ((torch.exp(h) - 1.0) / h - 1.0)) * D1
                    + sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0) * noise
                )
        return x_t  # pyright: ignore

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.multistep_dpm_solver_third_order_update
    def multistep_dpm_solver_third_order_update(
        self,
        model_output_list: List[torch.Tensor],
        *args,
        sample: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        One step for the third-order multistep DPMSolver.
        Args:
            model_output_list (`List[torch.Tensor]`):
                The direct outputs from learned diffusion model at current and latter timesteps.
            sample (`torch.Tensor`):
                A current instance of a sample created by diffusion process.
        Returns:
            `torch.Tensor`:
                The sample tensor at the previous timestep.
        """

        timestep_list = args[0] if len(args) > 0 else kwargs.pop("timestep_list", None)
        prev_timestep = args[1] if len(args) > 1 else kwargs.pop("prev_timestep", None)
        if sample is None:
            if len(args) > 2:
                sample = args[2]
            else:
                raise ValueError(" missing`sample` as a required keyward argument")
        if timestep_list is not None:
            deprecate(
                "timestep_list",
                "1.0.0",
                "Passing `timestep_list` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )

        if prev_timestep is not None:
            deprecate(
                "prev_timestep",
                "1.0.0",
                "Passing `prev_timestep` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )

        sigma_t, sigma_s0, sigma_s1, sigma_s2 = (
            self.sigmas[self.step_index + 1],  # pyright: ignore
            self.sigmas[self.step_index],
            self.sigmas[self.step_index - 1],  # pyright: ignore
            self.sigmas[self.step_index - 2],  # pyright: ignore
        )

        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
        alpha_s1, sigma_s1 = self._sigma_to_alpha_sigma_t(sigma_s1)
        alpha_s2, sigma_s2 = self._sigma_to_alpha_sigma_t(sigma_s2)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        lambda_s1 = torch.log(alpha_s1) - torch.log(sigma_s1)
        lambda_s2 = torch.log(alpha_s2) - torch.log(sigma_s2)

        m0, m1, m2 = model_output_list[-1], model_output_list[-2], model_output_list[-3]

        h, h_0, h_1 = lambda_t - lambda_s0, lambda_s0 - lambda_s1, lambda_s1 - lambda_s2
        r0, r1 = h_0 / h, h_1 / h
        D0 = m0
        D1_0, D1_1 = (1.0 / r0) * (m0 - m1), (1.0 / r1) * (m1 - m2)
        D1 = D1_0 + (r0 / (r0 + r1)) * (D1_0 - D1_1)
        D2 = (1.0 / (r0 + r1)) * (D1_0 - D1_1)
        if self.config.algorithm_type == "dpmsolver++":
            # See https://arxiv.org/abs/2206.00927 for detailed derivations
            x_t = (
                (sigma_t / sigma_s0) * sample
                - (alpha_t * (torch.exp(-h) - 1.0)) * D0
                + (alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0)) * D1
                - (alpha_t * ((torch.exp(-h) - 1.0 + h) / h**2 - 0.5)) * D2
            )
        elif self.config.algorithm_type == "dpmsolver":
            # See https://arxiv.org/abs/2206.00927 for detailed derivations
            x_t = (
                (alpha_t / alpha_s0) * sample
                - (sigma_t * (torch.exp(h) - 1.0)) * D0
                - (sigma_t * ((torch.exp(h) - 1.0) / h - 1.0)) * D1
                - (sigma_t * ((torch.exp(h) - 1.0 - h) / h**2 - 0.5)) * D2
            )
        return x_t  # pyright: ignore

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()
        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    def _init_step_index(self, timestep):
        """
        Initialize the step_index counter for the scheduler.
        """

        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    # Modified from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.step
    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor,
        generator=None,
        variance_noise: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the sample with
        the multistep DPMSolver.
        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`int`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            variance_noise (`torch.Tensor`):
                Alternative to generating noise with `generator` by directly providing the noise for the variance
                itself. Useful for methods such as [`LEdits++`].
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`.
        Returns:
            [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_utils.SchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.
        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Improve numerical stability for small number of steps
        lower_order_final = (self.step_index == len(self.timesteps) - 1) and (
            self.config.euler_at_final
            or (self.config.lower_order_final and len(self.timesteps) < 15)
            or self.config.final_sigmas_type == "zero"
        )
        lower_order_second = (
            (self.step_index == len(self.timesteps) - 2) and self.config.lower_order_final and len(self.timesteps) < 15
        )

        model_output = self.convert_model_output(model_output, sample=sample)
        for i in range(self.config.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = model_output

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)
        if self.config.algorithm_type in ["sde-dpmsolver", "sde-dpmsolver++"] and variance_noise is None:
            noise = randn_tensor(
                model_output.shape, generator=generator, device=model_output.device, dtype=torch.float32
            )
        elif self.config.algorithm_type in ["sde-dpmsolver", "sde-dpmsolver++"]:
            noise = variance_noise.to(device=model_output.device, dtype=torch.float32)  # pyright: ignore
        else:
            noise = None

        if self.config.solver_order == 1 or self.lower_order_nums < 1 or lower_order_final:
            prev_sample = self.dpm_solver_first_order_update(model_output, sample=sample, noise=noise)
        elif self.config.solver_order == 2 or self.lower_order_nums < 2 or lower_order_second:
            prev_sample = self.multistep_dpm_solver_second_order_update(self.model_outputs, sample=sample, noise=noise)
        else:
            prev_sample = self.multistep_dpm_solver_third_order_update(self.model_outputs, sample=sample)

        if self.lower_order_nums < self.config.solver_order:
            self.lower_order_nums += 1

        # Cast sample back to expected dtype
        prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one
        self._step_index += 1  # pyright: ignore

        if not return_dict:
            return (prev_sample,)

        return SchedulerOutput(prev_sample=prev_sample)

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.scale_model_input
    def scale_model_input(self, sample: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.
        Args:
            sample (`torch.Tensor`):
                The input sample.
        Returns:
            `torch.Tensor`:
                A scaled input sample.
        """
        return sample

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.scale_model_input
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
    ) -> torch.Tensor:
        
        sigma = timesteps.float() / self.num_timesteps
        # timepoints = timepoints.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)
        sigma = sigma.repeat(1, noise.shape[1], noise.shape[2], noise.shape[3], noise.shape[4])

        return (1 - sigma)  * original_samples +  sigma * noise
    
        # # Make sure sigmas and timesteps have the same device and dtype as original_samples
        # sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
        # if original_samples.device.type == "mps" and torch.is_floating_point(timesteps):
        #     # mps does not support float64
        #     schedule_timesteps = self.timesteps.to(original_samples.device, dtype=torch.float32)
        #     timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
        # else:
        #     schedule_timesteps = self.timesteps.to(original_samples.device)
        #     timesteps = timesteps.to(original_samples.device)

        # # begin_index is None when the scheduler is used for training or pipeline does not implement set_begin_index
        # if self.begin_index is None:
        #     step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timesteps]
        # elif self.step_index is not None:
        #     # add_noise is called after first denoising step (for inpainting)
        #     step_indices = [self.step_index] * timesteps.shape[0]
        # else:
        #     # add noise is called before first denoising step to create initial latent(img2img)
        #     step_indices = [self.begin_index] * timesteps.shape[0]

        # sigma = sigmas[step_indices].flatten()
        # while len(sigma.shape) < len(original_samples.shape):
        #     sigma = sigma.unsqueeze(-1)

        # alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
        # noisy_samples = alpha_t * original_samples + sigma_t * noise
        # return noisy_samples

    def __len__(self):
        return self.config.num_train_timesteps

    def create_wan_t2v_args(self, model_args):
        context = model_args["context"]
        max_seq_len = model_args["max_seq_len"]
        deg_latent = model_args["x_deg"]
        bg_latent = model_args["x_bg"]

        arg_c = {'context': context, 'seq_len': max_seq_len, 'x_deg': deg_latent, 'x_bg': bg_latent}
        
        return arg_c

    def create_wan_i2v_args(self, model_args):
        context = model_args["context"]
        clip_feat = model_args["clip_feat"]
        max_seq_len = model_args["max_seq_len"]
        img_mask_latent = model_args["img_mask_latent"]
        
        arg_c = {'context': context,
                 'clip_fea': clip_feat,
                 'seq_len': max_seq_len,
                 'y': img_mask_latent}
        return arg_c

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
                        mode="t2v"):
        """
        Compute training losses for a single timestep.
        Arguments format copied from vidgen/schedulers/iddpm/gaussian_diffusion.py/training_losses
        Note: t is int tensor and should be rescaled from [0, num_timesteps-1] to [1,0]
        """
        if t is None:
            if self.use_discrete_timesteps:
                t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=x_start.device)
            elif self.sample_method == "uniform":
                t = torch.rand((x_start.shape[0],), device=x_start.device) * self.num_timesteps
            elif self.sample_method == "logit-normal":
                t = self.sample_t(x_start) * self.num_timesteps
        
        # t = (1 - (t / self.num_timesteps) ** 3) * self.num_timesteps
        if self.use_timestep_transform:
            t = timestep_transform(t, model_kwargs, scale=self.transform_scale, num_timesteps=self.num_timesteps, fixed_timestep_transform=self.use_fixed_timestep_transform)
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        
        x_t = self.add_noise(x_start, noise, t)
        
        t = t.to(torch.int64) # compatible with inference
        
        if xt_mask is not None:
            t0 = torch.zeros_like(t)
            x_t0 = self.add_noise(x_start, noise, t0)
            x_t = torch.where(xt_mask[:, None, :, None, None], x_t, x_t0)

        terms = {}

        if mode == "t2v":
            arg_c = self.create_wan_t2v_args(model_kwargs)
        
        if mode == "i2v":
            arg_c = self.create_wan_i2v_args(model_kwargs)

        # model_output = model(x_t, t, **arg_c)
        # model_output = model(x_deg, x_bc, x_t, t, **arg_c)
        model_output = model(x_t, t, **arg_c)

        velocity_pred = model_output
        
        if weights is None:
            loss = (velocity_pred.float() - (noise.float() - x_start.float())).pow(2)
        else:
            weight = _extract_into_tensor(weights, t, x_start.shape)
            loss = weight * (velocity_pred.float() - (noise.float() - x_start.float())).pow(2)
            
        if xs_mask is None:
            loss = mean_flat(loss, mask=xt_mask)
        else:
            if unpatchify_loss:
                # unpatchify xs_mask
                T, H, W = xs_mask['shape']
                B, C, Tx, Hx, Wx = loss.shape
                pt, ph, pw = model.module.patch_size
                xs_mask = xs_mask['mask'].view(B, T*H*W, -1).repeat(1, 1, pt*ph*pw)
                xs_mask = rearrange(xs_mask, "B (T H W) (pt ph pw) -> B (T pt) (H ph) (W pw)", 
                                    B=B, T=T, H=H, W=W, pt=pt, ph=ph, pw=pw)
                xs_mask = xs_mask[:, :Tx, :Hx, :Wx]
                
                loss = loss.mean(dim=1).view(-1, Tx*Hx*Wx)
                unmask = 1 - xs_mask if xt_mask is None else (1 - xs_mask)*xt_mask.view(B, Tx, 1, 1)
                unmask = unmask.view(-1, Tx*Hx*Wx)
                rf_loss = (loss * unmask).sum(dim=1) / unmask.sum(dim=1)  # (N)
                terms["rf_loss"] = rf_loss
                
                pred_x_t = noise + velocity_pred 
                mask = xs_mask if xt_mask is None else xs_mask*xt_mask.view(B, Tx, 1, 1)
                mae_loss_value = mae_loss_unpatchify(x_t, pred_x_t, mask)
                terms["mae_loss"] = mae_loss_value
                loss = rf_loss + mae_loss_coef * mae_loss_value
            else:
                patch_size = model.module.config.patch_size
                if isinstance(patch_size, int):
                    patch_size = (1, patch_size, patch_size)
                loss = xpadding(loss, patch_size)
                rf_loss = F.avg_pool3d(loss.mean(dim=1), patch_size).flatten(1)  # (N, L)
                unmask = 1 - xs_mask['mask']
                if xt_mask is not None:
                    unmask *= xt_mask.unsqueeze(-1)
                unmask = unmask.view(loss.shape[0], -1)
                rf_loss = (rf_loss * unmask).sum(dim=1) / unmask.sum(dim=1)  # (N)
                terms["rf_loss"] = rf_loss
                pred_x_t = noise + velocity_pred
                mae_loss_value = mae_loss(x_t, pred_x_t, xs_mask["mask"], patch_size)
                terms["mae_loss"] = mae_loss_value
                loss = rf_loss + mae_loss_coef * mae_loss_value
        
        terms["loss"] = loss

        return terms