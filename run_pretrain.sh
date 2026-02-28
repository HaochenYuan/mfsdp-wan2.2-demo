#!/bin/bash
# =============================================================================
# Megatron-FSDP Pretrain: Wan2.2-TI2V-5B
# =============================================================================
# Demonstrates Megatron-FSDP and PyTorch FSDP2 training with optional
# TE Linear replacement and fuse_wgrad_accumulation.
#
# Usage:
#   ./run_pretrain.sh baseline   # Megatron-FSDP: TE Linear, no fuse
#   ./run_pretrain.sh fuse       # Megatron-FSDP: TE Linear + fuse_wgrad
#   ./run_pretrain.sh compare    # Run baseline + fuse sequentially
#   ./run_pretrain.sh pytorch    # PyTorch FSDP2: nn.Linear (no TE)
# =============================================================================

set -e

# ==================== Megatron-LM Path ====================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MEGATRON_PATH="${MEGATRON_PATH:-${SCRIPT_DIR}/Megatron-LM-dev-demo}"

# cuDNN paths (for flash-attn / TE compilation if needed)
export CUDNN_INCLUDE_DIR=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/include
export CUDNN_LIB_DIR=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib
export CPLUS_INCLUDE_PATH=$CUDNN_INCLUDE_DIR:$CPLUS_INCLUDE_PATH
export C_INCLUDE_PATH=$CUDNN_INCLUDE_DIR:$C_INCLUDE_PATH
export LIBRARY_PATH=$CUDNN_LIB_DIR:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDNN_LIB_DIR:$LD_LIBRARY_PATH

if [ -d "${MEGATRON_PATH}" ]; then
    export PYTHONPATH="${MEGATRON_PATH}:${PYTHONPATH}"
    echo "Megatron-LM: ${MEGATRON_PATH}"
else
    echo "ERROR: Megatron-LM not found at ${MEGATRON_PATH}"
    exit 1
fi

# ==================== Install Dependencies ====================
if [ "${SKIP_DEPS_INSTALL:-false}" != "true" ]; then
    export PIP_BREAK_SYSTEM_PACKAGES=1
    echo "Installing dependencies..."
    pip install flash-attn transformer_engine[pytorch] easydict diffusers \
        ftfy tqdm regex sentencepiece transformers accelerate \
        decord imageio peft opencv-python-headless\
        --root-user-action=ignore --no-build-isolation 2>&1 | tail -1
fi

# ==================== Configuration ====================
# Model config: "ti2v-5B", "t2v-A14B", "i2v-A14B", etc.
WAN_MODEL_PATH="${WAN_MODEL_PATH:-/path/to/Wan2.2-TI2V-5B}"
MODEL_CONFIG="${MODEL_CONFIG:-ti2v-5B}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${WAN_MODEL_PATH}}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/wan22_pretrain}"

# Random init (skip T5/VAE/checkpoint loading, for perf tuning)
RANDOM_INIT="${RANDOM_INIT:-true}"

# Architecture overrides (only with RANDOM_INIT=true)
OVERRIDE_DIM="${OVERRIDE_DIM:-}"
OVERRIDE_FFN_DIM="${OVERRIDE_FFN_DIM:-}"
OVERRIDE_NUM_LAYERS="${OVERRIDE_NUM_LAYERS:-}"
OVERRIDE_NUM_HEADS="${OVERRIDE_NUM_HEADS:-}"

# Training config
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
NUM_ITERATIONS="${NUM_ITERATIONS:-20}"

# Distributed
NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NUM_GPUS=$((NNODES * NPROC_PER_NODE))

# Auto-calculate gradient accumulation
GRADIENT_ACCUMULATION=$((GLOBAL_BATCH_SIZE / (MICRO_BATCH_SIZE * NUM_GPUS)))
NUM_SAMPLES=$((NUM_ITERATIONS * GLOBAL_BATCH_SIZE * 2))  # 2x buffer

# Validate
EXPECTED_GLOBAL=$((MICRO_BATCH_SIZE * NUM_GPUS * GRADIENT_ACCUMULATION))
if [ ${EXPECTED_GLOBAL} -ne ${GLOBAL_BATCH_SIZE} ]; then
    echo "ERROR: global_batch_size (${GLOBAL_BATCH_SIZE}) must = micro_batch_size × num_gpus × gradient_accumulation"
    echo "  micro_batch_size=${MICRO_BATCH_SIZE}, num_gpus=${NUM_GPUS}"
    exit 1
fi

# Video config
FRAME_NUM="${FRAME_NUM:-35}"
RESOLUTION="${RESOLUTION:-480 832}"

# Network
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"
NODE_RANK="${NODE_RANK:-0}"

# Megatron-FSDP
ZERO_DP_STRATEGY="${ZERO_DP_STRATEGY:-optim_grads_params}"
PARAM_DTYPE="${PARAM_DTYPE:-bf16}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"

GRAD_REDUCE_IN_FP32="${GRAD_REDUCE_IN_FP32:-true}"

# FSDP backend: "megatron" or "pytorch"
FSDP_BACKEND="${FSDP_BACKEND:-megatron}"

OVERLAP_GRAD_REDUCE="${OVERLAP_GRAD_REDUCE:-true}"
OVERLAP_PARAM_GATHER="${OVERLAP_PARAM_GATHER:-true}"
FSDP_DOUBLE_BUFFER="${FSDP_DOUBLE_BUFFER:-false}"
NCCL_UB="${NCCL_UB:-false}"
DISABLE_SYMMETRIC_REG="${DISABLE_SYMMETRIC_REG:-false}"

# Profiler
ENABLE_PROFILER="${ENABLE_PROFILER:-true}"
PROFILER_START="${PROFILER_START:-18}"
PROFILER_END="${PROFILER_END:-20}"

# ==================== Build base command ====================
build_cmd() {
    local USE_TE="$1"
    local FUSE_WGRAD="$2"
    local SUFFIX="$3"

    CMD="torchrun \
        --nnodes=${NNODES} \
        --nproc_per_node=${NPROC_PER_NODE} \
        --node_rank=${NODE_RANK} \
        --master_addr=${MASTER_ADDR} \
        --master_port=${MASTER_PORT} \
        pretrain_t2v.py \
        --model-config ${MODEL_CONFIG} \
        --output-dir ${OUTPUT_DIR}/${SUFFIX} \
        --fsdp-backend ${FSDP_BACKEND} \
        --max-steps ${NUM_ITERATIONS} \
        --batch-size ${MICRO_BATCH_SIZE} \
        --gradient-accumulation-steps ${GRADIENT_ACCUMULATION} \
        --learning-rate ${LEARNING_RATE} \
        --warmup-steps ${WARMUP_STEPS} \
        --num-samples ${NUM_SAMPLES} \
        --frame-num ${FRAME_NUM} \
        --resolution ${RESOLUTION} \
        --zero-dp-strategy ${ZERO_DP_STRATEGY} \
        --param-dtype ${PARAM_DTYPE} \
        --seed 42 \
        --num-workers 4 \
        --log-steps 1"

    # Optional flags
    [ "${OVERLAP_GRAD_REDUCE}" = "true" ] && CMD="${CMD} --overlap-grad-reduce"
    [ "${OVERLAP_PARAM_GATHER}" = "true" ] && CMD="${CMD} --overlap-param-gather"
    [ "${GRAD_REDUCE_IN_FP32}" = "true" ] && CMD="${CMD} --grad-reduce-in-fp32"
    [ "${USE_TE}" = "true" ] && CMD="${CMD} --use-te-linear"
    [ "${FUSE_WGRAD}" = "true" ] && CMD="${CMD} --fuse-wgrad-accumulation"
    [ "${FSDP_DOUBLE_BUFFER}" = "true" ] && CMD="${CMD} --fsdp-double-buffer"
    [ "${NCCL_UB}" = "true" ] && CMD="${CMD} --nccl-ub"
    [ "${DISABLE_SYMMETRIC_REG}" = "true" ] && CMD="${CMD} --disable-symmetric-registration"
    [ "${RANDOM_INIT}" = "true" ] && CMD="${CMD} --random-init"
    [ "${RANDOM_INIT}" != "true" ] && CMD="${CMD} --checkpoint-dir ${CHECKPOINT_DIR}"

    # Architecture overrides (only with RANDOM_INIT=true)
    [ -n "${OVERRIDE_DIM}" ] && CMD="${CMD} --override-dim ${OVERRIDE_DIM}"
    [ -n "${OVERRIDE_FFN_DIM}" ] && CMD="${CMD} --override-ffn-dim ${OVERRIDE_FFN_DIM}"
    [ -n "${OVERRIDE_NUM_LAYERS}" ] && CMD="${CMD} --override-num-layers ${OVERRIDE_NUM_LAYERS}"
    [ -n "${OVERRIDE_NUM_HEADS}" ] && CMD="${CMD} --override-num-heads ${OVERRIDE_NUM_HEADS}"

    # Profiler
    if [ "${ENABLE_PROFILER}" = "true" ]; then
        CMD="${CMD} --enable-profiler"
        CMD="${CMD} --profiler-start-step ${PROFILER_START}"
        CMD="${CMD} --profiler-end-step ${PROFILER_END}"
        CMD="${CMD} --profiler-output-dir ${OUTPUT_DIR}/${SUFFIX}/profiler"
    fi

    echo "${CMD}"
}

MODE="${1:-compare}"

case "${MODE}" in
    pytorch)
        FSDP_BACKEND="pytorch"
        ;;
esac

echo "=============================================="
echo "Wan2.2 Megatron-FSDP Pretrain"
echo "=============================================="
echo "Model config:      ${MODEL_CONFIG}"
[ "${RANDOM_INIT}" = "true" ] && echo "Random init:       YES (no checkpoint)"
[ -n "${OVERRIDE_DIM}" ] && echo "Override dim:      ${OVERRIDE_DIM}"
[ -n "${OVERRIDE_FFN_DIM}" ] && echo "Override ffn_dim:  ${OVERRIDE_FFN_DIM}"
[ -n "${OVERRIDE_NUM_LAYERS}" ] && echo "Override layers:   ${OVERRIDE_NUM_LAYERS}"
[ -n "${OVERRIDE_NUM_HEADS}" ] && echo "Override heads:    ${OVERRIDE_NUM_HEADS}"
echo "Mode:              ${MODE}"
echo "micro_batch_size:  ${MICRO_BATCH_SIZE}"
echo "global_batch_size: ${GLOBAL_BATCH_SIZE}"
echo "gradient_accum:    ${GRADIENT_ACCUMULATION}"
echo "num_iterations:    ${NUM_ITERATIONS}"
echo "GPUs:              ${NNODES}×${NPROC_PER_NODE} = ${NUM_GPUS}"
echo "Frame:             ${FRAME_NUM}, Resolution: ${RESOLUTION}"
echo "FSDP Backend:      ${FSDP_BACKEND}"
echo "ZeRO Strategy:     ${ZERO_DP_STRATEGY}"
echo "Grad reduce FP32:  ${GRAD_REDUCE_IN_FP32}"
echo "FSDP Double Buffer: ${FSDP_DOUBLE_BUFFER}"
echo "NCCL UB:            ${NCCL_UB}"
echo "=============================================="

cd "$(dirname "$0")"

case "${MODE}" in
    baseline)
        echo ""
        echo ">>> Running BASELINE (TE Linear, no fuse) <<<"
        CMD=$(build_cmd "true" "false" "mfsdp_baseline")
        echo "Command: ${CMD}"
        echo ""
        eval ${CMD}
        ;;

    fuse)
        echo ""
        echo ">>> Running FUSE (TE Linear + fuse_wgrad_accumulation) <<<"
        CMD=$(build_cmd "true" "true" "mfsdp_fuse")
        echo "Command: ${CMD}"
        echo ""
        eval ${CMD}
        ;;

    pytorch)
        FSDP_BACKEND="pytorch"
        echo ""
        echo ">>> Running PyTorch FSDP2 (nn.Linear, no TE) <<<"
        CMD=$(build_cmd "false" "false" "pytorch_fsdp2")
        echo "Command: ${CMD}"
        echo ""
        eval ${CMD}
        ;;
    
    compare)
        echo ""
        echo "========== [1/2] BASELINE (TE Linear, no fuse) =========="
        CMD=$(build_cmd "true" "false" "baseline")
        echo "Command: ${CMD}"
        echo ""
        eval ${CMD}

        echo ""
        echo "========== [2/2] FUSE (TE Linear + fuse_wgrad_accumulation) =========="
        CMD=$(build_cmd "true" "true" "fuse")
        echo "Command: ${CMD}"
        echo ""
        eval ${CMD}

        echo ""
        echo "========== Comparison complete =========="
        echo "Baseline profiler: ${OUTPUT_DIR}/baseline/profiler/"
        echo "Fuse profiler:     ${OUTPUT_DIR}/fuse/profiler/"
        ;;

    *)
    echo "Usage: $0 [baseline|fuse|compare|pytorch]"
    exit 1
    ;;
esac
