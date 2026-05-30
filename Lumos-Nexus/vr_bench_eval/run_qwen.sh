#!/bin/bash
# Run evaluation scripts in parallel on a fixed set of physical GPUs (batch size = number of GPUs).

set -e

# ==================== config ====================
# Comma-separated physical GPU ids to use, e.g. "0,1" -> 2-way parallel, 8 scripts = 4 batches.
# Override: export RUN_GPU_IDS="0,1" ./run_qwen.sh
RUN_GPU_IDS="${RUN_GPU_IDS:-4,5,6,7}"
IFS=',' read -ra GPU_ARR <<< "${RUN_GPU_IDS// /}"
if [ "${#GPU_ARR[@]}" -lt 1 ] || [ -z "${GPU_ARR[0]}" ]; then
  echo "error: RUN_GPU_IDS is empty or invalid: ${RUN_GPU_IDS:-}" >&2
  exit 1
fi
ACTUAL_NUM_GPUS="${#GPU_ARR[@]}"


VR_BENCH_VIDEO_ROOT="${VR_BENCH_VIDEO_ROOT:-/ssd/jzxing/nexus_vid/outputs/}"
VR_BENCH_MODELS="${VR_BENCH_MODELS:-Lumos_Nexus}"

SCRIPTS=(
    "DRF_evaluation.py"
    "ETV_evaluation.py"
    "MMC_evaluation.py"
    "PCR_evaluation.py"
    "CCR_evaluation.py"
    "CAC_evaluation.py"
    "BBR_evaluation.py"
    "CAR_evaluation.py"
)

ALL_SCRIPTS=("${SCRIPTS[@]}")

if [ "${#ALL_SCRIPTS[@]}" -gt "$ACTUAL_NUM_GPUS" ]; then
    echo "info: ${#ALL_SCRIPTS[@]} scripts > ${ACTUAL_NUM_GPUS} GPU(s); batching"
    echo "      parallel per batch: ${ACTUAL_NUM_GPUS}, batches: $(( (${#ALL_SCRIPTS[@]} + ACTUAL_NUM_GPUS - 1) / ACTUAL_NUM_GPUS ))"
fi

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVALUATIONS_DIR="$BASE_DIR/evaluations"
LOG_DIR="$BASE_DIR/evaluation_logs"
mkdir -p "$LOG_DIR"
export PYTHONPATH="${BASE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# Qwen ~5GB: Hugging Face cache defaults under $HOME; "No space left on device" -> put cache on a large disk, e.g.:
#   export VR_BENCH_HF_HOME=/ssd/yourname/.cache/huggingface
#   ./run_qwen.sh
# Or point to an already-downloaded local model dir (no hub download):
#   export VR_BENCH_VLM_PATH=/ssd/yourname/models/Qwen3-VL-30B-A3B-Instruct
if [[ -n "${VR_BENCH_HF_HOME:-}" ]]; then
  export HF_HOME="$VR_BENCH_HF_HOME"
  mkdir -p "$HF_HOME"
fi

echo "============================================================"
echo "vr_bench_eval / parallel runs"
echo "============================================================"
echo "BASE_DIR:        $BASE_DIR"
echo "EVALUATIONS_DIR: $EVALUATIONS_DIR"
echo "LOG_DIR:         $LOG_DIR"
echo "RUN_GPU_IDS:     $RUN_GPU_IDS  (parallel slots: $ACTUAL_NUM_GPUS)"
echo "VIDEO_ROOT:      $VR_BENCH_VIDEO_ROOT"
echo "MODELS:          $VR_BENCH_MODELS"
echo "tasks:           ${#ALL_SCRIPTS[@]}"
echo "============================================================"
echo ""

for script in "${ALL_SCRIPTS[@]}"; do
    if [ ! -f "$EVALUATIONS_DIR/$script" ]; then
        echo "error: missing script: $EVALUATIONS_DIR/$script" >&2
        exit 1
    fi
done

START_TIME=$(date +%s)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

SUCCESS_COUNT=0
FAIL_COUNT=0
RESULTS=()
TOTAL_TASKS=${#ALL_SCRIPTS[@]}

BATCH_NUM=0
TASK_INDEX=0

while [ $TASK_INDEX -lt $TOTAL_TASKS ]; do
    BATCH_NUM=$((BATCH_NUM + 1))
    REMAINING_TASKS=$((TOTAL_TASKS - TASK_INDEX))
    BATCH_SIZE=$((REMAINING_TASKS < ACTUAL_NUM_GPUS ? REMAINING_TASKS : ACTUAL_NUM_GPUS))
    
    echo ""
    echo "============================================================"
    echo "batch $BATCH_NUM: starting $BATCH_SIZE task(s) ($REMAINING_TASKS remaining)"
    echo "============================================================"

    PIDS=()
    BATCH_SCRIPTS=()
    BATCH_GPU_IDS=()

    for ((j=0; j<BATCH_SIZE; j++)); do
        SCRIPT="${ALL_SCRIPTS[$TASK_INDEX]}"
        PHYSICAL_GPU="${GPU_ARR[j]}"
        LOG_FILE="$LOG_DIR/batch${BATCH_NUM}_gpu${PHYSICAL_GPU}_${SCRIPT%.py}_${TIMESTAMP}.log"

        echo "start [batch $BATCH_NUM, CUDA device $PHYSICAL_GPU]: $SCRIPT"
        echo "  log: $(basename "$LOG_FILE")"

        VR_BENCH_VIDEO_ROOT="$VR_BENCH_VIDEO_ROOT" \
        VR_BENCH_MODELS="$VR_BENCH_MODELS" \
        CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" \
        python "$EVALUATIONS_DIR/$SCRIPT" > "$LOG_FILE" 2>&1 &

        PIDS+=($!)
        BATCH_SCRIPTS+=("$SCRIPT")
        BATCH_GPU_IDS+=("$PHYSICAL_GPU")
        TASK_INDEX=$((TASK_INDEX + 1))
    done

    echo ""
    echo "waiting for batch $BATCH_NUM..."

    for i in "${!PIDS[@]}"; do
        PID="${PIDS[$i]}"
        SCRIPT="${BATCH_SCRIPTS[$i]}"
        GPU_ID="${BATCH_GPU_IDS[$i]}"

        wait $PID
        EXIT_CODE=$?
        
        if [ $EXIT_CODE -eq 0 ]; then
            echo "ok   [batch $BATCH_NUM, GPU $GPU_ID] $SCRIPT"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            RESULTS+=("ok [batch $BATCH_NUM GPU $GPU_ID] $SCRIPT")
        else
            echo "fail [batch $BATCH_NUM, GPU $GPU_ID] $SCRIPT (exit $EXIT_CODE)" >&2
            FAIL_COUNT=$((FAIL_COUNT + 1))
            RESULTS+=("fail [batch $BATCH_NUM GPU $GPU_ID] $SCRIPT exit=$EXIT_CODE")
        fi
    done

    echo "batch $BATCH_NUM done (cumulative ok=$SUCCESS_COUNT fail=$FAIL_COUNT)"
done

# elapsed
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "============================================================"
echo "summary"
echo "============================================================"
echo "tasks:   $TOTAL_TASKS"
echo "ok:      $SUCCESS_COUNT"
echo "fail:    $FAIL_COUNT"
echo "elapsed: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "details:"
for result in "${RESULTS[@]}"; do
    echo "  $result"
done
echo ""
echo "logs: $LOG_DIR"
echo "============================================================"

if [ $FAIL_COUNT -gt 0 ]; then
    exit 1
fi

exit 0

