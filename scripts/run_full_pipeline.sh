#!/usr/bin/env bash
# Full GRPO pipeline: boundary delineation → standard GRPO → dual GRPO
# Uses all 4 × A100 80GB GPUs.
#
# Usage:
#   bash scripts/run_full_pipeline.sh
#
# Logs written to logs/  (one file per stage per GPU)

set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate RLforESC

mkdir -p logs data/knowledge_boundaries

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Knowledge Boundary Delineation  (4-GPU parallel inference)
# ─────────────────────────────────────────────────────────────────────────────
echo "=========================================="
echo " Stage 1: Knowledge Boundary Delineation"
echo "=========================================="

PIDS=()
for GPU_ID in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/delineate_knowledge_boundary.py \
        --model_path checkpoints/qwen25-esc \
        --K 8 \
        --temperature 0.4 \
        --max_new_tokens 25 \
        --output_dir data/knowledge_boundaries \
        --gpu_id $GPU_ID \
        --num_gpus 4 \
        > logs/delineate_gpu${GPU_ID}.log 2>&1 &
    PIDS+=($!)
    echo "  GPU $GPU_ID started (PID ${PIDS[-1]})"
done

echo "Waiting for all 4 delineation processes..."
for PID in "${PIDS[@]}"; do
    wait $PID && echo "  PID $PID finished OK" || echo "  PID $PID FAILED"
done
echo "Stage 1 complete. Boundary file: data/knowledge_boundaries/train_turns_with_boundaries.json"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Standard GRPO  (4-GPU DDP via accelerate)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo " Stage 2: Standard GRPO (4-GPU DDP)"
echo "=========================================="

accelerate launch \
    --num_processes 4 \
    --mixed_precision bf16 \
    scripts/train_grpo.py \
        --mode standard \
        --model_path checkpoints/qwen25-esc \
        --output_dir checkpoints/grpo-standard \
        --num_train_epochs 1 \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 4 \
        --num_generations 8 \
        --max_completion_length 256 \
        --max_prompt_length 1536 \
        --learning_rate 5e-7 \
        --beta 0.001 \
        --temperature 0.9 \
        --save_steps 100 \
        --logging_steps 10 \
    2>&1 | tee logs/grpo_standard.log

echo "Stage 2 complete. Checkpoint: checkpoints/grpo-standard"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: Dual-reward GRPO  (4-GPU DDP via accelerate)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo " Stage 3: Dual-reward GRPO (4-GPU DDP)"
echo "=========================================="

accelerate launch \
    --num_processes 4 \
    --mixed_precision bf16 \
    scripts/train_grpo.py \
        --mode dual \
        --model_path checkpoints/qwen25-esc \
        --boundaries_path data/knowledge_boundaries/train_turns_with_boundaries.json \
        --output_dir checkpoints/grpo-dual \
        --num_train_epochs 1 \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 4 \
        --num_generations 8 \
        --max_completion_length 256 \
        --max_prompt_length 1536 \
        --learning_rate 5e-7 \
        --beta 0.001 \
        --temperature 0.9 \
        --save_steps 100 \
        --logging_steps 10 \
    2>&1 | tee logs/grpo_dual.log

echo ""
echo "=========================================="
echo " All stages complete!"
echo " checkpoints/grpo-standard"
echo " checkpoints/grpo-dual"
echo "=========================================="
