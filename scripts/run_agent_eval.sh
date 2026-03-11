#!/usr/bin/env bash
# 4卡并行运行 ESC Agent 评估
# 用法: bash scripts/run_agent_eval.sh [test|valid] [model_path]
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate RLforESC

SPLIT="${1:-test}"
MODEL="${2:-models/Qwen2.5-7B-Instruct}"
NUM_GPUS=4
OUT_DIR="results/agent_eval"
LOG_DIR="logs"

mkdir -p "$OUT_DIR" "$LOG_DIR"

echo "========================================"
echo " ESC Agent 评估 (4卡并行)"
echo " split     : $SPLIT"
echo " model     : $MODEL"
echo " output    : $OUT_DIR"
echo "========================================"

PIDS=()
for GPU_ID in 0 1 2 3; do
    python scripts/esc_agent.py \
        --mode evaluate \
        --model_path "$MODEL" \
        --split "$SPLIT" \
        --gpu_id $GPU_ID \
        --num_gpus $NUM_GPUS \
        --output_dir "$OUT_DIR" \
        --n_examples 2 \
        --temperature 0 \
        --max_new_tokens 384 \
        > "$LOG_DIR/agent_eval_gpu${GPU_ID}.log" 2>&1 &
    PIDS+=($!)
    echo "  GPU $GPU_ID 启动 (PID ${PIDS[-1]})"
done

echo ""
echo "  等待所有进程完成..."
FAILED=0
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    if wait $PID; then
        echo "  GPU $i (PID $PID) 完成 ✓"
    else
        echo "  GPU $i (PID $PID) 失败 ✗"
        FAILED=1
    fi
done

if [ $FAILED -eq 0 ]; then
    echo ""
    echo "  最终合并结果:"
    tail -30 "$LOG_DIR/agent_eval_gpu3.log"
    echo ""
    echo "  完整报告 → $OUT_DIR/eval_${SPLIT}_merged.json"
else
    echo "  有进程失败，请检查 logs/agent_eval_gpu*.log"
    exit 1
fi
