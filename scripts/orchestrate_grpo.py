#!/usr/bin/env python3
"""
Orchestrator: KB v4 完成后自动启动 GRPO 训练
=============================================
后台运行方式：
  nohup /home/aiscuser/.conda/envs/verl/bin/python -u scripts/orchestrate_grpo.py \
      > logs/orchestrator.log 2>&1 &
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ── 配置 ──────────────────────────────────────────────────────────────────────
VERL_PYTHON   = "/home/aiscuser/.conda/envs/verl/bin/python"
MERGED_PATH   = Path("data/knowledge_boundaries_v4/train_turns_with_boundaries.json")
SHARD_PATHS   = [Path(f"data/knowledge_boundaries_v4/train_turns_shard{i}.json") for i in range(4)]
MIN_TURNS     = 13_000          # 少于此数视为未完成
POLL_INTERVAL = 60              # 秒
GRPO_LOG      = Path("logs/grpo_dual_agent.log")

# ── GRPO 训练参数（严格对应论文 arXiv:2509.12661）────────────────────────────
#   batch_size = 4 GPUs × 1 per_device × 64 grad_accum = 256  ← 论文值
#   max_steps  = 300                                            ← 论文 "300 episodes"
#   lr         = 1e-6, beta = 0.001                            ← 论文值
#   G (num_generations) = 8                                    ← 论文值
GRPO_CMD = [
    "torchrun",
    "--nproc_per_node=4",
    "--master_port=29501",
    "scripts/train_grpo.py",
    "--mode",                        "dual",
    "--model_path",                  "/mnt/teamdrive/models/Qwen2.5-7B-Instruct",
    "--boundaries_path",             str(MERGED_PATH),
    "--output_dir",                  "checkpoints/grpo-dual-agent",
    "--max_steps",                   "300",
    "--per_device_train_batch_size", "1",
    "--gradient_accumulation_steps", "64",
    "--num_generations",             "8",
    "--learning_rate",               "1e-6",
    "--beta",                        "0.001",
    "--temperature",                 "0.9",
    "--max_prompt_length",           "1536",
    "--max_completion_length",       "30",
    "--save_steps",                  "50",
    "--logging_steps",               "5",
    "--bf16",
    "--gradient_checkpointing",
]


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)


def kb_is_ready() -> tuple[bool, int]:
    """检查 merged 文件是否存在且样本数足够。"""
    if not MERGED_PATH.exists():
        return False, 0
    try:
        data = json.loads(MERGED_PATH.read_text())
        return len(data) >= MIN_TURNS, len(data)
    except Exception:
        return False, 0


def wait_for_kb():
    log("=" * 60)
    log("Orchestrator 启动 — 等待 KB v4 划分完成...")
    log(f"监控文件: {MERGED_PATH}")
    log(f"最低样本要求: {MIN_TURNS}")
    log("=" * 60)

    while True:
        ready, n = kb_is_ready()
        if ready:
            log(f"✓ KB v4 完成！共 {n} 条样本。")
            break

        # 检查各 shard 进度（作为参考）
        shard_sizes = []
        for p in SHARD_PATHS:
            if p.exists():
                try:
                    d = json.loads(p.read_text())
                    shard_sizes.append(len(d))
                except Exception:
                    shard_sizes.append(0)
            else:
                shard_sizes.append(0)

        total_done = sum(shard_sizes)
        log(f"等待中... Shards: {shard_sizes} | 合计已完成: {total_done}/13952 turns "
            f"({100*total_done/13952:.1f}%)")
        time.sleep(POLL_INTERVAL)


def launch_grpo():
    log("")
    log("=" * 60)
    log("启动 GRPO Dual-Reward 训练")
    log(f"命令: {' '.join(GRPO_CMD)}")
    log(f"输出日志: {GRPO_LOG}")
    log("=" * 60)

    GRPO_LOG.parent.mkdir(parents=True, exist_ok=True)
    Path("checkpoints/grpo-dual-agent").mkdir(parents=True, exist_ok=True)

    with open(GRPO_LOG, "w") as f:
        f.write(f"# GRPO 训练日志 — 启动时间: {datetime.now()}\n")
        f.write(f"# 命令: {' '.join(GRPO_CMD)}\n\n")
        f.flush()

        proc = subprocess.Popen(
            GRPO_CMD,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )

    log(f"GRPO 进程 PID: {proc.pid}")
    log("等待训练完成（可能需要数小时）...")

    # 每 10 分钟打印一次状态
    while True:
        try:
            ret = proc.wait(timeout=600)
            break
        except subprocess.TimeoutExpired:
            # 读最新 log 行
            try:
                lines = GRPO_LOG.read_text().splitlines()
                last = next((l for l in reversed(lines) if l.strip()), "")
                log(f"训练中... 最新日志: {last[-120:]}")
            except Exception:
                log("训练中...")

    if ret == 0:
        log("")
        log("=" * 60)
        log("✓ GRPO 训练完成！")
        log(f"  Checkpoint 保存在: checkpoints/grpo-dual-agent/")
        log(f"  训练日志: {GRPO_LOG}")
        log("=" * 60)
    else:
        log("")
        log(f"✗ GRPO 训练异常退出，return code: {ret}")
        log(f"  请检查日志: {GRPO_LOG}")
        sys.exit(ret)


# ── 入口 ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        wait_for_kb()
        time.sleep(10)   # 等待文件完整写入
        launch_grpo()
    except KeyboardInterrupt:
        log("Orchestrator 被手动中断。")
        sys.exit(0)
