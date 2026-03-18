#!/bin/bash
# =============================================================================
# Megatron-FSDP Pretrain: Wan2.2-TI2V-5B with Interleaved FSDP Units
# =============================================================================
# FSDP units: [attn_0] [ffn_0+attn_1] ... [ffn_{N-2}+attn_{N-1}] [ffn_{N-1}]
#
# Usage:
#   ./run_pretrain_interleaved.sh fuse       # Megatron-FSDP: TE Linear + fuse_wgrad
#   ./run_pretrain_interleaved.sh baseline   # Megatron-FSDP: TE Linear, no fuse
# =============================================================================

set -e

# ==================== Megatron-LM Path ====================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MEGATRON_PATH="${MEGATRON_PATH:-${SCRIPT_DIR}/Megatron-LM-dev-demo}"

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
WAN_MODEL_PATH="${WAN_MODEL_PATH:-/lustre/fsw/coreai_devtech_all/haocheny/Wan2.7_onsite_support/Wan2.2-TI2V-5B}"
MODEL_CONFIG="${MODEL_CONFIG:-ti2v-5B}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/wan22_ti2v_5B_interleaved}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${WAN_MODEL_PATH}}"

# Auto-detect Diffusers pipeline format
if [ -f "${WAN_MODEL_PATH}/model_index.json" ] && [ ! -f "${WAN_MODEL_PATH}/config.json" ]; then
    DIFFUSERS_FORMAT="${DIFFUSERS_FORMAT:-true}"
    echo "Auto-detected Diffusers pipeline format for ${WAN_MODEL_PATH}"
fi

RANDOM_INIT_T5="${RANDOM_INIT_T5:-true}"
RANDOM_INIT_VAE="${RANDOM_INIT_VAE:-true}"
RANDOM_INIT_DIT="${RANDOM_INIT_DIT:-true}"

# Training config
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
NUM_ITERATIONS="${NUM_ITERATIONS:-20}"

# Distributed
NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NUM_GPUS=$((NNODES * NPROC_PER_NODE))

GRADIENT_ACCUMULATION=$((GLOBAL_BATCH_SIZE / (MICRO_BATCH_SIZE * NUM_GPUS)))
NUM_SAMPLES=$((NUM_ITERATIONS * GLOBAL_BATCH_SIZE * 2))

EXPECTED_GLOBAL=$((MICRO_BATCH_SIZE * NUM_GPUS * GRADIENT_ACCUMULATION))
if [ ${EXPECTED_GLOBAL} -ne ${GLOBAL_BATCH_SIZE} ]; then
    echo "ERROR: global_batch_size (${GLOBAL_BATCH_SIZE}) must = micro_batch_size × num_gpus × gradient_accumulation"
    exit 1
fi

FRAME_NUM="${FRAME_NUM:-35}"
RESOLUTION="${RESOLUTION:-480 832}"

MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"
NODE_RANK="${NODE_RANK:-0}"

ZERO_DP_STRATEGY="${ZERO_DP_STRATEGY:-optim_grads_params}"
PARAM_DTYPE="${PARAM_DTYPE:-bf16}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"

GRAD_REDUCE_IN_FP32="${GRAD_REDUCE_IN_FP32:-true}"
FSDP_BACKEND="${FSDP_BACKEND:-megatron}"

OVERLAP_GRAD_REDUCE="${OVERLAP_GRAD_REDUCE:-true}"
OVERLAP_PARAM_GATHER="${OVERLAP_PARAM_GATHER:-true}"
FSDP_DOUBLE_BUFFER="${FSDP_DOUBLE_BUFFER:-false}"
NCCL_UB="${NCCL_UB:-false}"
DISABLE_SYMMETRIC_REG="${DISABLE_SYMMETRIC_REG:-false}"

DIFFUSERS_FORMAT="${DIFFUSERS_FORMAT:-false}"
TRANSFORMER_SUBDIR="${TRANSFORMER_SUBDIR:-transformer}"

ENABLE_PROFILER="${ENABLE_PROFILER:-true}"
PROFILER_START="${PROFILER_START:-18}"
PROFILER_END="${PROFILER_END:-20}"

# ==================== Build command ====================
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
        pretrain_t2v_interleaved.py \
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

    [ "${OVERLAP_GRAD_REDUCE}" = "true" ] && CMD="${CMD} --overlap-grad-reduce"
    [ "${OVERLAP_PARAM_GATHER}" = "true" ] && CMD="${CMD} --overlap-param-gather"
    [ "${GRAD_REDUCE_IN_FP32}" = "true" ] && CMD="${CMD} --grad-reduce-in-fp32"
    [ "${USE_TE}" = "true" ] && CMD="${CMD} --use-te-linear"
    [ "${FUSE_WGRAD}" = "true" ] && CMD="${CMD} --fuse-wgrad-accumulation"
    [ "${FSDP_DOUBLE_BUFFER}" = "true" ] && CMD="${CMD} --fsdp-double-buffer"
    [ "${NCCL_UB}" = "true" ] && CMD="${CMD} --nccl-ub"
    [ "${DISABLE_SYMMETRIC_REG}" = "true" ] && CMD="${CMD} --disable-symmetric-registration"
    [ "${RANDOM_INIT_T5}" = "true" ] && CMD="${CMD} --random-init-t5"
    [ "${RANDOM_INIT_VAE}" = "true" ] && CMD="${CMD} --random-init-vae"
    [ "${RANDOM_INIT_DIT}" = "true" ] && CMD="${CMD} --random-init-dit"
    if [ "${RANDOM_INIT_T5}" != "true" ] || [ "${RANDOM_INIT_VAE}" != "true" ] || [ "${RANDOM_INIT_DIT}" != "true" ]; then
        CMD="${CMD} --checkpoint-dir ${CHECKPOINT_DIR}"
    fi
    if [ "${DIFFUSERS_FORMAT}" = "true" ]; then
        if [ "${RANDOM_INIT_T5}" != "true" ] || [ "${RANDOM_INIT_VAE}" != "true" ] || [ "${RANDOM_INIT_DIT}" != "true" ]; then
            CMD="${CMD} --diffusers-format --transformer-subdir ${TRANSFORMER_SUBDIR}"
        fi
    fi

    if [ "${ENABLE_PROFILER}" = "true" ]; then
        CMD="${CMD} --enable-profiler"
        CMD="${CMD} --profiler-start-step ${PROFILER_START}"
        CMD="${CMD} --profiler-end-step ${PROFILER_END}"
        CMD="${CMD} --profiler-output-dir ${OUTPUT_DIR}/${SUFFIX}/profiler"
    fi

    echo "${CMD}"
}

MODE="${1:-fuse}"

echo "=============================================="
echo "Wan2.2 Megatron-FSDP Pretrain (INTERLEAVED)"
echo "=============================================="
echo "Model config:      ${MODEL_CONFIG}"
echo "Random init T5:    ${RANDOM_INIT_T5}"
echo "Random init VAE:   ${RANDOM_INIT_VAE}"
echo "Random init DiT:   ${RANDOM_INIT_DIT}"
echo "Mode:              ${MODE}"
echo "micro_batch_size:  ${MICRO_BATCH_SIZE}"
echo "global_batch_size: ${GLOBAL_BATCH_SIZE}"
echo "gradient_accum:    ${GRADIENT_ACCUMULATION}"
echo "num_iterations:    ${NUM_ITERATIONS}"
echo "GPUs:              ${NNODES}×${NPROC_PER_NODE} = ${NUM_GPUS}"
echo "Frame:             ${FRAME_NUM}, Resolution: ${RESOLUTION}"
echo "FSDP Backend:      ${FSDP_BACKEND}"
echo "ZeRO Strategy:     ${ZERO_DP_STRATEGY}"
echo "FSDP Units:        Interleaved (FirstUnit + StaggeredUnits + LastUnit)"
echo "=============================================="

cd "$(dirname "$0")"

case "${MODE}" in
    baseline)
        echo ""
        echo ">>> Running INTERLEAVED BASELINE (TE Linear, no fuse) <<<"
        CMD=$(build_cmd "true" "false" "mfsdp_interleaved_baseline")
        echo "Command: ${CMD}"
        echo ""
        eval ${CMD}
        ;;

    fuse)
        echo ""
        echo ">>> Running INTERLEAVED FUSE (TE Linear + fuse_wgrad_accumulation) <<<"
        CMD=$(build_cmd "true" "true" "mfsdp_interleaved_fuse")
        echo "Command: ${CMD}"
        echo ""
        eval ${CMD}
        ;;

    note)
        echo ""
        echo ">>> Running INTERLEAVED without TE <<<"
        CMD=$(build_cmd "false" "false" "mfsdp_interleaved_note")
        echo "Command: ${CMD}"
        echo ""
        eval ${CMD}
        ;;

    *)
        echo "Usage: $0 [baseline|fuse|note]"
        exit 1
        ;;
esac
