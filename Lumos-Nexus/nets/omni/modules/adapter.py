import os
import logging
import torch
import torch.nn as nn
import torch.distributed as dist

from nets.third_party.wan.utils.utils import torch_checkpoint_load


class DM_Adapter(nn.Module):
    def __init__(
        self,
        in_channels=1152,
        out_channels=4096,
        learnable_query_length=256, 
        TRAINABLE_PRECISION=torch.float32, # torch.float16, torch.bfloat16
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,       # Controls whether to use FSDP sharding
        use_usp=False,
        t5_cpu=False,
        save_ckpt_dir=None,   # Directory to save checkpoints
        load_ckpt_dir=''    # Directory to load checkpoints from
    ):
        r"""
        Args:
            in_channels (int): Number of input feature channels (e.g., siglip2 img encoder output channels)
            out_channels (int): Number of output feature channels (e.g., Diffusion Model condition channels)
            learnable_query_length (int): Length of learnable query, default 256
            TRAINABLE_PRECISION: Data type used for model computation, default torch.bfloat16
            device_id (int): GPU device id to use
            rank (int): Current process rank
            t5_fsdp, dit_fsdp, use_usp, t5_cpu: Other related settings (extend as needed)
            save_ckpt_dir (str): Directory to save checkpoints
            load_ckpt_dir (str): Directory to load checkpoints from
        """
        super().__init__()
        self.device = torch.device(f"cuda:{device_id}")
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.dit_fsdp = dit_fsdp  # Use the passed value instead of hardcoding internally
        self.learnable_query_length = learnable_query_length

        # Checkpoint related directories
        self.save_ckpt_dir = save_ckpt_dir
        self.load_ckpt_dir = load_ckpt_dir

        # Learnable query
        self.decoder_query = nn.Parameter(
            torch.randn((1, self.learnable_query_length, self.out_channels), dtype=TRAINABLE_PRECISION),
            requires_grad=True
        )

        # Channel mapping: map siglip2 img encoder channels to Diffusion Model condition channels
        self.fc = nn.Sequential(
            nn.Linear(self.in_channels, self.out_channels),
            nn.GELU(),
            nn.Linear(self.out_channels, self.out_channels),
        ).to(TRAINABLE_PRECISION)
        
        # Adapter: implements information interaction based on Transformer
        self.adapter = nn.Transformer(
            batch_first=True, 
            norm_first=True, 
            d_model=self.out_channels, 
            num_encoder_layers=4, 
            num_decoder_layers=4, 
            dim_feedforward=self.out_channels * 4, 
            dropout=0.0, 
            dtype=TRAINABLE_PRECISION
        )
        
        self.init_weights() 

    def init_weights(self):
        # Initialize learnable query
        nn.init.normal_(self.decoder_query, mean=0.0, std=0.02)

        # Initialize weights for all modules
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def load_checkpoint(self, checkpoint_path):
        # logging.info(f"Loading checkpoint from {checkpoint_path} ")
        map_loc = self.device if self.device.type == "cuda" else torch.device("cpu")
        checkpoint = torch_checkpoint_load(checkpoint_path, map_location=map_loc)
        state_dict = checkpoint.get("model_state_dict", {})
        # Handle "module." prefix saved by DDP
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key[len("module."):] if key.startswith("module.") else key
            new_state_dict[new_key] = value
        self.load_state_dict(new_state_dict)
        epoch = checkpoint.get("epoch", 0)
        step = checkpoint.get("step", 0)
        # logging.info(f"Checkpoint loaded: epoch {epoch}, step {step}")
        return checkpoint

    def load_ckpt(self):
        ckpt_path = os.path.join(self.load_ckpt_dir, "adapter_pytorch_model.bin")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint file {ckpt_path} does not exist.")

        # Map checkpoint to current device when loading
        ckpt = torch_checkpoint_load(ckpt_path, map_location=self.device)
        # If checkpoint contains 'state_dict' key, take its value, otherwise consider the entire checkpoint as state_dict
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

        # Strip potentially existing "module." prefix
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key[len("module."):] if key.startswith("module.") else key
            new_state_dict[new_key] = value

        # Use non-strict loading and print missing and unexpected keys for debugging
        missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)
        logging.info(f"Loaded checkpoint from {ckpt_path}. Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}")

    def save_pretrained(self, state_dict=None):
        if self.save_ckpt_dir is None:
            raise ValueError("save_ckpt_dir is not set in initialization.")
        os.makedirs(self.save_ckpt_dir, exist_ok=True)
        if state_dict is None:
            state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(self.save_ckpt_dir, "adapter_pytorch_model.bin"))

    def forward(self, x):
        # Channel mapping
        x = self.fc(x)
        # Expand learnable query to current batch size
        decoder_query_expanded = self.decoder_query.repeat(x.shape[0], 1, 1)
        # Use Transformer adapter for information interaction
        x = self.adapter(x, tgt=decoder_query_expanded)
        return x

if __name__ == '__main__':
    import torch.optim as optim
    import torch.nn.functional as F
    # Set test parameters
    batch_size = 4
    seq_len = 10             # Input sequence length (for fc use)
    in_channels = 128        # Input feature dimension
    out_channels = 256       # Output feature dimension
    learnable_query_length = 64  # Learnable query length
    epochs = 100             # Training epochs
    learning_rate = 1e-3

    # Select device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Construct model input: shape [B, N_in, in_channels]
    input_tensor = torch.randn(batch_size, seq_len, in_channels, device=device)

    # Construct externally given target tensor: shape [B, learnable_query_length, out_channels]
    target_tensor = torch.randn(batch_size, learnable_query_length, out_channels, device=device)

    # Instantiate DM_Adapter model, no checkpoint loading (pass load_ckpt_dir=None), and move to device
    model = DM_Adapter(
        in_channels=in_channels, 
        out_channels=out_channels,
        learnable_query_length=learnable_query_length,
        TRAINABLE_PRECISION=torch.float32,  # Can use float32 for unit testing, easier for debugging
        device_id=0,
        save_ckpt_dir="./save_ckpt",
        load_ckpt_dir=None  # No need to load checkpoint
    ).to(device)