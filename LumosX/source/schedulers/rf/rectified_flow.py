import torch
from torch.distributions import LogisticNormal
import torch.nn.functional as F
from einops import rearrange
from ..iddpm.gaussian_diffusion import _extract_into_tensor, mean_flat

# some code are inspired by https://github.com/magic-research/piecewise-rectified-flow/blob/main/scripts/train_perflow.py
# and https://github.com/magic-research/piecewise-rectified-flow/blob/main/src/scheduler_perflow.py


def timestep_transform(
    t,
    model_kwargs,
    base_resolution=512 * 512,
    base_num_frames=1,
    scale=1.0,
    num_timesteps=1,
):
    # Force fp16 input to fp32 to avoid nan output
    for key in ["height", "width", "num_frames"]:
        if model_kwargs[key].dtype == torch.float16:
            model_kwargs[key] = model_kwargs[key].float()
            
    t = t / num_timesteps
    resolution = model_kwargs["height"] * model_kwargs["width"]
    ratio_space = (resolution / base_resolution).sqrt()
    # NOTE: currently, we do not take fps into account
    # NOTE: temporal_reduction is hardcoded, this should be equal to the temporal reduction factor of the vae
    if model_kwargs["num_frames"][0] == 1:
        num_frames = torch.ones_like(model_kwargs["num_frames"])
    else:
        num_frames = model_kwargs["num_frames"] // 17 * 5
    ratio_time = (num_frames / base_num_frames).sqrt()

    ratio = ratio_space * ratio_time * scale
    new_t = ratio * t / (1 + (ratio - 1) * t)

    new_t = new_t * num_timesteps
    return new_t

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


def mae_loss_unpatchify(target, pred, mask, norm_pix_loss=True):
    B, C = target.shape[:2]
    target = target.view(B, C, -1)
    pred = pred.view(B, C, -1)
    if norm_pix_loss:
        mean = target.mean(dim=1, keepdim=True)
        var = target.var(dim=1, keepdim=True)
        target = (target - mean) / (var + 1.e-6)**.5

    loss = (pred - target) ** 2
    loss = loss.mean(dim=1)  # [N, L], mean loss per patch
    
    mask = mask.reshape(B, -1)
    loss = (loss * mask).sum(dim=1) / mask.sum(dim=1)  # mean loss on removed patches, (N)
    assert loss.ndim == 1
    return loss


class RFlowScheduler:
    def __init__(
        self,
        num_timesteps=1000,
        num_sampling_steps=10,
        use_discrete_timesteps=False,
        sample_method="uniform",
        loc=0.0,
        scale=1.0,
        use_timestep_transform=False,
        transform_scale=1.0,
    ):
        self.num_timesteps = num_timesteps
        self.num_sampling_steps = num_sampling_steps
        self.use_discrete_timesteps = use_discrete_timesteps

        # sample method
        assert sample_method in ["uniform", "logit-normal"]
        assert (
            sample_method == "uniform" or not use_discrete_timesteps
        ), "Only uniform sampling is supported for discrete timesteps"
        self.sample_method = sample_method
        if sample_method == "logit-normal":
            self.distribution = LogisticNormal(torch.tensor([loc]), torch.tensor([scale]))
            self.sample_t = lambda x: self.distribution.sample((x.shape[0],))[:, 0].to(x.device)

        # timestep transform
        self.use_timestep_transform = use_timestep_transform
        self.transform_scale = transform_scale

    def training_losses(self, model, x_start, model_kwargs=None, noise=None, xt_mask=None, xs_mask=None, weights=None, t=None, mae_loss_coef=None, unpatchify_loss=False):
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

        if self.use_timestep_transform:
            t = timestep_transform(t, model_kwargs, scale=self.transform_scale, num_timesteps=self.num_timesteps)

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape

        x_t = self.add_noise(x_start, noise, t)
        if xt_mask is not None:
            t0 = torch.zeros_like(t)
            x_t0 = self.add_noise(x_start, noise, t0)
            x_t = torch.where(xt_mask[:, None, :, None, None], x_t, x_t0)

        terms = {}
        model_output = model(x_t, t, **model_kwargs)
        velocity_pred = model_output.chunk(2, dim=1)[0]
        if weights is None:
            loss = (velocity_pred - (x_start - noise)).pow(2)
        else:
            weight = _extract_into_tensor(weights, t, x_start.shape)
            loss = weight * (velocity_pred - (x_start - noise)).pow(2)
            
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
                
                pred_x_t = velocity_pred + noise
                mask = xs_mask if xt_mask is None else xs_mask*xt_mask.view(B, Tx, 1, 1)
                mae_loss_value = mae_loss_unpatchify(x_t, pred_x_t, mask)
                terms["mae_loss"] = mae_loss_value
                loss = rf_loss + mae_loss_coef * mae_loss_value
            else:
                patch_size = model.module.patch_size
                loss = xpadding(loss, patch_size)
                rf_loss = F.avg_pool3d(loss.mean(dim=1), patch_size).flatten(1)  # (N, L)
                unmask = 1 - xs_mask['mask']
                if xt_mask is not None:
                    unmask *= xt_mask.unsqueeze(-1)
                unmask = unmask.view(loss.shape[0], -1)
                rf_loss = (rf_loss * unmask).sum(dim=1) / unmask.sum(dim=1)  # (N)
                terms["rf_loss"] = rf_loss
                pred_x_t = velocity_pred + noise
                mae_loss_value = mae_loss(x_t, pred_x_t, xs_mask["mask"], patch_size)
                terms["mae_loss"] = mae_loss_value
                loss = rf_loss + mae_loss_coef * mae_loss_value
        terms["loss"] = loss

        return terms

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        """
        compatible with diffusers add_noise()
        """
        timepoints = timesteps.float() / self.num_timesteps
        timepoints = 1 - timepoints  # [1,1/1000]

        # timepoint  (bsz) noise: (bsz, 4, frame, w ,h)
        # expand timepoint to noise shape
        timepoints = timepoints.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        timepoints = timepoints.repeat(1, noise.shape[1], noise.shape[2], noise.shape[3], noise.shape[4])

        return timepoints * original_samples + (1 - timepoints) * noise
