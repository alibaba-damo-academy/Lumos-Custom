#!/usr/bin/env bash
# Batch text-to-video via generate_batch_edit.py at repository root (Wan 14B path).
#
# Required:
#   export PROMPT_LIST_FILE=/path/to/prompts.txt   # one prompt per line, or PKL paths (see generate_batch_edit.py)
#
# Optional:
#   OMNI_MODELS_DIR   - default: <repo>/model_ckpts (local disk >> /mnt/ft NFS for load speed)
#   OUTPUT_DIR        - default: ./outputs/t2v_batch_14
#   FRAME_NUM         - default 81 (must be 4n+1)
#   SAMPLE_STEPS, SIZE, MASTER_PORT, NGPUS_PER_NODE, NNODES, NODE_RANK
#   CUDA_VISIBLE_DEVICES - default 0,1 (only first two GPUs; set before run to override)
#   WAN_LOAD_STAGGER_SEC=8  - export before run: rank r sleeps r*sec before heavy I/O (eases NFS when using torchrun multi-GPU)
#   --offload_model true    - only if OOM: moves DiT to CPU after each video (slow between batch items; default is false)
#   --dit_fsdp true         - experimental: FSDP shard both Wan DiTs (torchrun NGPUS>=2)
#   --fast_cuda false       - disable cudnn benchmark / TF32 / SDPA tweaks (default on for speed)
#   Speed tips: wan_inference_activation_checkpoint false; local SSD for ckpt
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Resolve repo root: directory that contains generate_batch_edit.py (works if script is at root or under tools/...)
_PROJECT_ROOT="$SCRIPT_DIR"
while [[ "$_PROJECT_ROOT" != "/" ]]; do
  if [[ -f "${_PROJECT_ROOT}/generate_batch_edit.py" ]]; then
    PROJECT_ROOT="$_PROJECT_ROOT"
    break
  fi
  _PROJECT_ROOT="$(dirname "$_PROJECT_ROOT")"
done
if [[ -z "${PROJECT_ROOT:-}" ]]; then
  echo "error: could not find generate_batch_edit.py above ${SCRIPT_DIR}" >&2
  exit 1
fi
unset _PROJECT_ROOT

export PYTHONPATH="${PROJECT_ROOT}/nets/third_party:${PYTHONPATH:-}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:+$PYTORCH_CUDA_ALLOC_CONF,}expandable_segments:True"

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
# Default: first two physical GPUs only (override by exporting CUDA_VISIBLE_DEVICES before run).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
# nvidia-smi --list-gpus counts *all* physical GPUs and ignores CUDA_VISIBLE_DEVICES; derive count from CVD.
_cvd="${CUDA_VISIBLE_DEVICES// /}"
VISIBLE_GPU_COUNT=0
if [[ -n "$_cvd" ]]; then
  IFS=',' read -ra _cvd_parts <<< "$_cvd"
  for _part in "${_cvd_parts[@]}"; do
    [[ -z "$_part" ]] && continue
    if [[ "$_part" =~ ^[0-9]+-[0-9]+$ ]]; then
      _a="${_part%-*}"
      _b="${_part#*-}"
      VISIBLE_GPU_COUNT=$((VISIBLE_GPU_COUNT + _b - _a + 1))
    else
      VISIBLE_GPU_COUNT=$((VISIBLE_GPU_COUNT + 1))
    fi
  done
else
  VISIBLE_GPU_COUNT="$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')"
fi
unset _cvd _cvd_parts _part _a _b
[[ -z "$VISIBLE_GPU_COUNT" || "$VISIBLE_GPU_COUNT" -lt 1 ]] && VISIBLE_GPU_COUNT=1
if [[ -z "${NGPUS_PER_NODE:-}" ]]; then
  NGPUS_PER_NODE="$VISIBLE_GPU_COUNT"
elif [[ "$NGPUS_PER_NODE" -gt "$VISIBLE_GPU_COUNT" ]]; then
  echo "warning: NGPUS_PER_NODE=$NGPUS_PER_NODE > visible GPUs ($VISIBLE_GPU_COUNT, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES); using $VISIBLE_GPU_COUNT" >&2
  NGPUS_PER_NODE="$VISIBLE_GPU_COUNT"
fi

export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-1000}"

# Relative to current working directory (typically repo root); override via OMNI_MODELS_DIR.
MODELS_DIR="${OMNI_MODELS_DIR:-./model_ckpts}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/t2v_batch/Lumos_Nexus}"
LIST_FILE="${PROMPT_LIST_FILE:-./vbench_prompt.txt}"


FRAME_NUM="${FRAME_NUM:-81}"
SAMPLE_STEPS="${SAMPLE_STEPS:-50}"
SIZE="${SIZE:-832*480}"

if [[ ! -d "$MODELS_DIR" ]]; then
  echo "error: models directory not found: $MODELS_DIR (set OMNI_MODELS_DIR)" >&2
  exit 1
fi
MODELS_DIR="$(cd "$MODELS_DIR" && pwd)"

if [[ -z "$LIST_FILE" ]]; then
  echo "error: set PROMPT_LIST_FILE or PKL_LIST_FILE" >&2
  exit 1
fi
if [[ ! -f "$LIST_FILE" ]]; then
  echo "error: list file not found: $LIST_FILE" >&2
  exit 1
fi

CMD=(
  torchrun
  --nproc_per_node="$NGPUS_PER_NODE"
  --master_addr="$MASTER_ADDR"
  --master_port="$MASTER_PORT"
  --nnodes="$NNODES"
  --node_rank="$NODE_RANK"
  "${PROJECT_ROOT}/generate_batch.py"
  --models_dir "$MODELS_DIR"
  --output_dir "$OUTPUT_DIR"
  --sample_solver unipc
  --adapter_in_channels 1152
  --adapter_out_channels 4096
  --adapter_query_length 256
  --use_visual_context_adapter true
  --visual_context_adapter_patch_size "1,4,4"
  --use_visual_as_input false
  --condition_mode full
  --max_context_len 2560
  --ar_model_num_video_frames 8
  --ar_conv_mode llama_3
  --sampling_rate 3
  --skip_num 1
  --unconditioned_context_length 2560
  --classifier_free_ratio 0.0
)

CMD+=(
  --task t2v
  --size "$SIZE"
  --frame_num "$FRAME_NUM"
  --sample_steps "$SAMPLE_STEPS"
  --pkl_list_file "$LIST_FILE"
  --sample_fps 16
  --sample_guide_scale 5.0
  --gamma_w 0.3
  --gamma_hf 0.7
  --sigma_min 0.35
  --sigma_max 0.7
  --base_seed 42
)

echo "Running: ${#CMD[@]} args, output -> $OUTPUT_DIR"
"${CMD[@]}"
echo "Done. Outputs: $OUTPUT_DIR"
