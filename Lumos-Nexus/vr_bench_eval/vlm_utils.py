"""Resolve Qwen3-VL weights for VR-Bench old evaluation scripts."""

import os


def ensure_qwen_vl_snapshot(local_dir: str, repo_id: str = "Qwen/Qwen3-VL-30B-A3B-Instruct") -> str:
    """Download the Qwen3-VL snapshot if an absolute local directory is empty."""
    local_dir = os.path.abspath(local_dir)
    if os.path.isfile(os.path.join(local_dir, "config.json")):
        return local_dir
    os.makedirs(local_dir, exist_ok=True)
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False)
    return local_dir


def prepare_vlm_path(vlm_path: str) -> str:
    """Use absolute paths as local snapshots; pass relative/HF repo ids through."""
    raw = os.path.expanduser(vlm_path.strip())
    if os.path.isabs(raw):
        return ensure_qwen_vl_snapshot(raw)
    return raw
