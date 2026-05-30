# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from functools import partial

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy

__all__ = ["shard_model", "shard_wan_dit_for_fsdp"]


def shard_model(
    model,
    device_id,
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float32,
    process_group=None,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    sync_module_states=True,
):
    model = FSDP(
        module=model,
        process_group=process_group,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=partial(
            lambda_auto_wrap_policy, lambda_fn=lambda m: m in model.blocks),
        mixed_precision=MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype),
        device_id=device_id,
        sync_module_states=sync_module_states)
    return model


def shard_wan_dit_for_fsdp(
    wan_model,
    device_id,
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float32,
    process_group=None,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    sync_module_states=True,
    full_precision: bool = False,
):
    """
    Wrap a single WanModel (has .blocks) with FSDP for multi-GPU inference.
    VideoMixedConditionModel is not flat; shard each Wan subtree separately.

    If ``full_precision`` is True, ``mixed_precision`` is disabled (fp32), and
    per-device memory is reduced only via parameter sharding (FULL_SHARD).
    """
    if full_precision:
        mp = None
    else:
        mp = MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype,
        )
    return FSDP(
        module=wan_model,
        process_group=process_group,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: m in wan_model.blocks,
        ),
        mixed_precision=mp,
        device_id=device_id,
        sync_module_states=sync_module_states,
    )
