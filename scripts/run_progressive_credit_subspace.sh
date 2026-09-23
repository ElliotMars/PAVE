#!/usr/bin/env bash
set -u

if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_BIN="$PYTHON"
elif [[ -x "/opt/data/private/envs/Online/bin/python" ]]; then
    PYTHON_BIN="/opt/data/private/envs/Online/bin/python"
elif [[ -x "/root/miniconda3/envs/online/bin/python" ]]; then
    PYTHON_BIN="/root/miniconda3/envs/online/bin/python"
else
    PYTHON_BIN="python"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

ROOT_PATH="$PROJECT_ROOT/data/"
CHECKPOINT_ROOT="$PROJECT_ROOT/checkpoints"
LOG_DIR="$PROJECT_ROOT/log/$(date +%Y%m%d_%H%M%S)_progressive_subspace"
mkdir -p "$LOG_DIR"

dataset_text="${DATASETS:-ETTh2 ETTm1 WTH ECL}"
length_text="${LENS:-1 24 48}"
dataset_text="${dataset_text//,/ }"
length_text="${length_text//,/ }"
read -r -a datasets <<< "$dataset_text"
read -r -a lens <<< "$length_text"

if [[ ${#datasets[@]} -eq 0 || ${#lens[@]} -eq 0 ]]; then
    echo "DATASETS and LENS must not be empty" >&2
    exit 1
fi
for data in "${datasets[@]}"; do
    if [[ ! -f "${ROOT_PATH}${data}.csv" ]]; then
        echo "Missing data file: ${ROOT_PATH}${data}.csv" >&2
        exit 1
    fi
done

GPU_IDS_STR="${GPU_IDS:-0}"
MAX_PER_GPU="${MAX_PER_GPU:-2}"
IFS=',' read -r -a GPU_LIST <<< "$GPU_IDS_STR"
if [[ ${#GPU_LIST[@]} -eq 0 || "$MAX_PER_GPU" -le 0 ]]; then
    echo "GPU_IDS must not be empty and MAX_PER_GPU must be positive" >&2
    exit 1
fi

PRETRAIN_MODE="${PRETRAIN_MODE:-retrain}"
ROUTER_GRANULARITY="${ROUTER_GRANULARITY:-horizon_channel}"
CORRECTION_LR="${CORRECTION_LR:-0.1}"
LOCAL_CREDIT_WEIGHT="${LOCAL_CREDIT_WEIGHT:-0.1}"
STABLE_BUFFER_SIZE="${STABLE_BUFFER_SIZE:-32}"
RECOVERY_BUFFER_SIZE="${RECOVERY_BUFFER_SIZE:-32}"
SUBSPACE_RANK="${SUBSPACE_RANK:-16}"
SUBSPACE_LAMBDA="${SUBSPACE_LAMBDA:-10000}"
EXPERT_UPDATE_STRATEGY="${EXPERT_UPDATE_STRATEGY:-subspace}"

NUM_EXPERTS="${NUM_EXPERTS:-4}"
TOP_K="${TOP_K:-$NUM_EXPERTS}"
RECOVERY_BATCH_SIZE="${RECOVERY_BATCH_SIZE:-2}"
RECOVERY_LOSS_WEIGHT="${RECOVERY_LOSS_WEIGHT:-0.1}"
RECOVERY_SKETCH_WEIGHT="${RECOVERY_SKETCH_WEIGHT:-1.0}"
RECOVERY_DEGRADATION_MARGIN="${RECOVERY_DEGRADATION_MARGIN:-0.0}"
DISABLE_DIRECTIONAL_RECOVERY="${DISABLE_DIRECTIONAL_RECOVERY:-0}"
SUBSPACE_MAX_RANK="${SUBSPACE_MAX_RANK:-32}"
SUBSPACE_ENERGY_THRESHOLD="${SUBSPACE_ENERGY_THRESHOLD:-0.95}"
SUBSPACE_REFRESH_INTERVAL="${SUBSPACE_REFRESH_INTERVAL:-100}"
SUBSPACE_MIN_SAMPLES="${SUBSPACE_MIN_SAMPLES:-4}"
SUBSPACE_GAMMA_MIN="${SUBSPACE_GAMMA_MIN:-0.0}"
SUBSPACE_GAMMA_MAX="${SUBSPACE_GAMMA_MAX:-1.0}"
MEMORY_REFRESH_INTERVAL="${MEMORY_REFRESH_INTERVAL:-100}"
ONLINE_LOG_INTERVAL="${ONLINE_LOG_INTERVAL:-500}"
MAX_ONLINE_STEPS="${MAX_ONLINE_STEPS:--1}"
STRICT_ONLINE_CHECKS="${STRICT_ONLINE_CHECKS:-0}"
ITR="${ITR:-1}"
SEED="${SEED:-0}"
ONLINE_LEARNING="${ONLINE_LEARNING:-full}"

declare -a RUN_PIDS=()
declare -A PID_GPU=()
next_gpu=0

cleanup_finished() {
    local alive=()
    local pid
    for pid in "${RUN_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            alive+=("$pid")
        else
            unset 'PID_GPU[$pid]'
        fi
    done
    RUN_PIDS=("${alive[@]}")
}

running_on_gpu() {
    local gpu="$1"
    local count=0
    local pid
    for pid in "${RUN_PIDS[@]}"; do
        if [[ "${PID_GPU[$pid]:-}" == "$gpu" ]] && kill -0 "$pid" 2>/dev/null; then
            count=$((count + 1))
        fi
    done
    echo "$count"
}

wait_for_slot() {
    local gpu="$1"
    while true; do
        cleanup_finished
        if [[ "$(running_on_gpu "$gpu")" -lt "$MAX_PER_GPU" ]]; then
            return
        fi
        sleep 2
    done
}

append_disable_flags() {
    EXTRA_FLAGS=()
    [[ "${DISABLE_VERSION_AWARENESS:-0}" == "1" ]] &&
        EXTRA_FLAGS+=(--disable_version_awareness)
    [[ "${DISABLE_RECOVERY:-0}" == "1" ]] &&
        EXTRA_FLAGS+=(--disable_recovery)
    [[ "$DISABLE_DIRECTIONAL_RECOVERY" == "1" ]] &&
        EXTRA_FLAGS+=(--disable_directional_recovery)
    [[ "${DISABLE_CREDIT_WEIGHTED_SUBSPACE:-0}" == "1" ]] &&
        EXTRA_FLAGS+=(--disable_credit_weighted_subspace)
    [[ "${DISABLE_ONLINE_CORRECTION:-0}" == "1" ]] &&
        EXTRA_FLAGS+=(--disable_online_correction)
    [[ "${DISABLE_TSB:-0}" == "1" ]] &&
        EXTRA_FLAGS+=(--disable_tsb)
    [[ "${DISABLE_EXPERT_ONLINE_UPDATE:-0}" == "1" ]] &&
        EXTRA_FLAGS+=(--disable_expert_online_update)
    [[ "$STRICT_ONLINE_CHECKS" == "1" ]] &&
        EXTRA_FLAGS+=(--strict_online_checks)
}

find_latest_checkpoint() {
    local method_tag="$1"
    local data="$2"
    local pred_len="$3"
    local pattern="$CHECKPOINT_ROOT/${method_tag}_${data}_pl${pred_len}_olfull_optadam_tb1_"
    local latest=""
    local checkpoint
    shopt -s nullglob
    for checkpoint in "${pattern}"*/checkpoint.pth; do
        [[ -z "$latest" || "$checkpoint" > "$latest" ]] && latest="$checkpoint"
    done
    shopt -u nullglob
    echo "$latest"
}

append_disable_flags
checkpoint_tag="${CHECKPOINT_TAG:-pc_subspace_${ROUTER_GRANULARITY}_${EXPERT_UPDATE_STRATEGY}_ne${NUM_EXPERTS}_tk${TOP_K}}"
echo "[CONFIG] datasets=${datasets[*]} lens=${lens[*]} gpus=${GPU_LIST[*]}"
echo "[CONFIG] strategy=$EXPERT_UPDATE_STRATEGY router=$ROUTER_GRANULARITY pretrain=$PRETRAIN_MODE"

for pred_len in "${lens[@]}"; do
for data in "${datasets[@]}"; do
    gpu="${GPU_LIST[$next_gpu]}"
    next_gpu=$(((next_gpu + 1) % ${#GPU_LIST[@]}))
    wait_for_slot "$gpu"

    offline_lr="1e-3"
    online_expert_lr="1e-4"
    online_router_lr="1e-5"
    [[ "$data" == "WTH" ]] && online_expert_lr="5e-5"
    if [[ "$data" == "ECL" ]]; then
        offline_lr="3e-3"
        online_expert_lr="1e-5"
    fi

    pretrain_args=(--pretrain_mode "$PRETRAIN_MODE" --checkpoint_tag "$checkpoint_tag")
    if [[ "$PRETRAIN_MODE" == "load" ]]; then
        checkpoint="$(find_latest_checkpoint "multi_expert_${checkpoint_tag}" "$data" "$pred_len")"
        if [[ -z "$checkpoint" ]]; then
            echo "No compatible checkpoint for data=$data pred_len=$pred_len" >&2
            exit 1
        fi
        pretrain_args+=(--pretrained_checkpoint "$checkpoint")
    fi

    log="$LOG_DIR/progressive_subspace_${data}_${pred_len}_full.out"
    echo "[RUN] data=$data pred_len=$pred_len gpu=$gpu log=$log"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u main.py \
        --method multi_expert \
        --root_path "$ROOT_PATH" \
        --data "$data" \
        --features M \
        --seq_len 60 \
        --label_len 0 \
        --pred_len "$pred_len" \
        --test_bsz 1 \
        --batch_size 32 \
        --itr "$ITR" \
        --train_epochs 15 \
        --patience 3 \
        --learning_rate "$offline_lr" \
        --learning_rate_expert "$offline_lr" \
        --learning_rate_router "$offline_lr" \
        --online_lr_expert "$online_expert_lr" \
        --online_lr_router "$online_router_lr" \
        --online_learning "$ONLINE_LEARNING"\
        --delay_fb \
        --progressive_fb \
        --router_granularity "$ROUTER_GRANULARITY" \
        --correction_lr "$CORRECTION_LR" \
        --local_credit_weight "$LOCAL_CREDIT_WEIGHT" \
        --stable_buffer_size "$STABLE_BUFFER_SIZE" \
        --recovery_buffer_size "$RECOVERY_BUFFER_SIZE" \
        --recovery_batch_size "$RECOVERY_BATCH_SIZE" \
        --recovery_loss_weight "$RECOVERY_LOSS_WEIGHT" \
        --recovery_sketch_weight "$RECOVERY_SKETCH_WEIGHT" \
        --recovery_degradation_margin "$RECOVERY_DEGRADATION_MARGIN" \
        --memory_refresh_interval "$MEMORY_REFRESH_INTERVAL" \
        --subspace_scope regressor \
        --subspace_rank "$SUBSPACE_RANK" \
        --subspace_max_rank "$SUBSPACE_MAX_RANK" \
        --subspace_energy_threshold "$SUBSPACE_ENERGY_THRESHOLD" \
        --subspace_refresh_interval "$SUBSPACE_REFRESH_INTERVAL" \
        --subspace_min_samples "$SUBSPACE_MIN_SAMPLES" \
        --subspace_lambda "$SUBSPACE_LAMBDA" \
        --subspace_gamma_min "$SUBSPACE_GAMMA_MIN" \
        --subspace_gamma_max "$SUBSPACE_GAMMA_MAX" \
        --expert_update_strategy "$EXPERT_UPDATE_STRATEGY" \
        --num_experts "$NUM_EXPERTS" \
        --top_k "$TOP_K" \
        --online_log_interval "$ONLINE_LOG_INTERVAL" \
        --max_online_steps "$MAX_ONLINE_STEPS" \
        --seed "$SEED" \
        "${EXTRA_FLAGS[@]}" \
        "${pretrain_args[@]}" > "$log" 2>&1 &
    pid=$!
    RUN_PIDS+=("$pid")
    PID_GPU[$pid]="$gpu"
done
done

wait
echo "All progressive credit/subspace runs finished. log_dir=$LOG_DIR"
