export ENABLE_COMPILE=true
export HF_ENDPOINT=https://hf-mirror.com
export PATH=/mnt/workspace/workgroup/pengwei.lpw/conda-envs/wanx/bin:$PATH
export PYTHONPATH=$PWD
export TORCH_NCCL_ENABLE_MONITORING=0
export TOKENIZERS_PARALLELISM=false
export NCCL_NET_PLUGIN=none
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export CUDA_VISIBLE_DEVICES=0

torchrun --nproc_per_node=1 unilumos_infer_abc.py