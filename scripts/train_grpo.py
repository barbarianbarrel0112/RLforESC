"""
GRPO Fine-tuning for Emotional Support Conversation
====================================================
Based on arXiv:2509.12661v1

Two modes:
  --mode standard  : Standard GRPO with binary accuracy reward
                     r_i = 1.0 if completion_i has correct strategy, else 0.0

  --mode dual      : Paper's novel dual-reward GRPO
                     r_i = w_acc(region) * [1 if correct else 0]
                           + w_ent(region) * ei
                     where ci, ei, region come from pre-computed knowledge boundaries

Region-specific weights (from paper):
  HK (Highly Known, ci=1):  w_acc=1.0, w_ent=0.0
  WK (Weakly Known, 0<ci<1): w_acc=0.5, w_ent=0.5
  UK (Unknown, ci=0):        w_acc=0.0, w_ent=1.0

KL penalty β = 0.001 (applied inside GRPOConfig)

Usage:
  # Standard GRPO
  python scripts/train_grpo.py --mode standard \\
      --model_path checkpoints/qwen25-esc \\
      --output_dir checkpoints/grpo-standard

  # Dual-reward GRPO (requires pre-computed boundaries)
  python scripts/train_grpo.py --mode dual \\
      --model_path checkpoints/qwen25-esc \\
      --boundaries_path data/knowledge_boundaries/train_turns_with_boundaries.json \\
      --output_dir checkpoints/grpo-dual
"""

import argparse
import json
import math
import re
import os
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# ── Strategy set ─────────────────────────────────────────────────────────────
STRATEGIES = [
    "Question", "Restatement or Paraphrasing", "Reflection of feelings",
    "Self-disclosure", "Affirmation and Reassurance", "Providing Suggestions",
    "Information", "Others",
]
STRATEGY_PATTERN = re.compile(
    r"^\s*\[(" + "|".join(re.escape(s) for s in STRATEGIES) + r")\]",
    re.IGNORECASE,
)

SYSTEM_PROMPT = """\
You are an emotional support counselor. Your ONLY task is to output a strategy tag.

⚠️ STRICT FORMAT: Your entire response must be exactly one line:
[Strategy Name]

Choose from these 8 strategies:

1. **Question** – Ask open-ended questions to understand the user's situation.

2. **Restatement or Paraphrasing** – Restate what the user said to show understanding.

3. **Reflection of feelings** – Identify and reflect back the user's emotions.

4. **Self-disclosure** – Share a brief relevant personal experience to build connection.

5. **Affirmation and Reassurance** – Validate feelings and offer encouragement.

6. **Providing Suggestions** – Offer practical advice when the user is ready.

7. **Information** – Provide factual information relevant to the user's situation.

8. **Others** – For conversation openers, closers, or brief transitions only.

Output ONLY [Strategy Name] — nothing else. Example: [Reflection of feelings]
"""

# log(|S|) = ln(8) ≈ 2.079，用于论文 reward 公式的熵归一化
LOG_S = math.log(len(STRATEGIES))


def extract_strategy(text: str) -> str | None:
    m = STRATEGY_PATTERN.match(text)
    if not m:
        return None
    raw = m.group(1).strip()
    for s in STRATEGIES:
        if s.lower() == raw.lower():
            return s
    return None


# ── Dataset builders ──────────────────────────────────────────────────────────

def load_standard_dataset(esconv_dir: str, tokenizer) -> Dataset:
    """
    Load ESConv_cleaned/train.json → HuggingFace Dataset
    Each sample:  prompt (tokenized chat) + target_strategy (for reward)
    """
    train_path = Path(esconv_dir) / "train.json"
    with open(train_path) as f:
        dialogs = json.load(f)

    samples = []
    for dlg in dialogs:
        turns = dlg["dialog"]
        clean_turns = []
        for t in turns:
            role = t.get("role", t.get("speaker", ""))
            content = t.get("content", t.get("text", "")).strip()
            strategy = t.get("strategy", None)
            clean_turns.append({"role": role, "content": content, "strategy": strategy})

        for i, t in enumerate(clean_turns):
            if t["role"] != "assistant" or not t["strategy"]:
                continue

            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            for prev in clean_turns[:i]:
                messages.append({"role": prev["role"], "content": prev["content"]})

            prompt_text = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            samples.append({
                "prompt": prompt_text,
                "target_strategy": t["strategy"],
                # these two are unused by standard reward but kept for consistency
                "ci": 1.0,
                "ei": 0.0,
                "region": "WK",
            })

    return Dataset.from_list(samples)


def load_dual_dataset(boundaries_path: str, tokenizer) -> Dataset:
    """
    Load pre-computed knowledge-boundary file → Dataset
    prompt_messages already in chat format (list of dicts)
    """
    with open(boundaries_path) as f:
        records = json.load(f)

    samples = []
    for rec in records:
        prompt_text = tokenizer.apply_chat_template(
            rec["prompt_messages"],
            add_generation_prompt=True,
            tokenize=False,
        )
        samples.append({
            "prompt": prompt_text,
            "target_strategy": rec["target_strategy"],
            "ci": rec["ci"],
            "ei": rec["ei"],
            "region": rec["region"],
        })

    return Dataset.from_list(samples)


# ── Reward functions ──────────────────────────────────────────────────────────

def make_standard_reward_fn():
    """标准 GRPO baseline: r = 1 if correct else 0"""
    def reward_fn(prompts, completions, target_strategy, **kwargs):
        rewards = []
        for comp, tgt in zip(completions, target_strategy):
            text = comp if isinstance(comp, str) else comp[0]["content"]
            pred = extract_strategy(text)
            rewards.append(1.0 if pred == tgt else 0.0)
        return rewards
    return reward_fn


def make_dual_reward_fn():
    """
    论文 dual reward（arXiv:2509.12661）完整还原：

        r(hi, ỹi) = ci + r_region(hi, ỹi)

    其中：
      ci        = 本次 completion ỹi 的准确率（1 if correct else 0）
      r_region  = 基于知识边界区域和 KB 熵 ei 的区域奖励：
                  HK/WK: r_region = 1 - ei / log(|S|)   ← 降低熵，强化已知策略
                  UK:    r_region = ei / log(|S|)        ← 提高熵，鼓励未知探索
      ei        = 预计算的 KB 多分类熵（来自 knowledge_boundaries 文件）
                  ei = -Σ_{s∈S} p(s|hi) log p(s|hi)，单位 nats
      log(|S|)  = ln(8) ≈ 2.079，用于归一化使 r_region ∈ [0, 1]

    KL penalty 由 GRPOConfig(beta=0.001) 在 trainer 内部处理，等价于论文的
    -β log π(ỹi|hi) / π_ref(ỹi|hi) 项。

    奖励范围：
      HK (ci_KB=1, ei_KB≈0):  correct → 1+1=2.0, wrong → 0+1=1.0
      UK (ci_KB=0, ei_KB≈ln8): correct → 1+1=2.0, wrong → 0+1=1.0
      WK (0<ci_KB<1, 中间ei): 介于 HK 和 UK 之间
    GRPO 通过组内 advantage = ri - mean(r_group) 提取学习信号。
    """
    def reward_fn(prompts, completions, target_strategy, ei, region, **kwargs):
        rewards = []
        for comp, tgt, _ei, reg in zip(completions, target_strategy, ei, region):
            text = comp if isinstance(comp, str) else comp[0]["content"]

            # ci = 本次 completion 的准确率（逐 completion 变化，是 GRPO 的学习信号）
            ci = 1.0 if extract_strategy(text) == tgt else 0.0

            # r_region = 基于 KB 预计算的静态区域权重（逐 prompt 固定）
            _ei = float(_ei)
            if reg in ("HK", "WK"):
                r_region = 1.0 - _ei / LOG_S   # 已知区域：鼓励低熵（自信正确）
            else:  # UK
                r_region = _ei / LOG_S          # 未知区域：鼓励高熵（多样探索）

            rewards.append(ci + r_region)
        return rewards
    return reward_fn


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GRPO training for ESC")
    parser.add_argument("--mode", choices=["standard", "dual"], required=True,
                        help="standard = binary accuracy reward; dual = paper's dual reward")
    parser.add_argument("--model_path", default="/mnt/teamdrive/models/Qwen2.5-7B-Instruct",
                        help="初始策略模型（Agent方案用base model，SFT方案用checkpoints/qwen25-esc-v2）")
    parser.add_argument("--esconv_dir", default="data/ESConv_cleaned",
                        help="ESConv cleaned 目录（standard mode 使用）")
    parser.add_argument("--boundaries_path",
                        default="data/knowledge_boundaries_v4/train_turns_with_boundaries.json",
                        help="知识边界文件（dual mode 使用，v4为Agent+多分类熵修正版）")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for GRPO checkpoint")

    # Training hyperparameters
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-7,
                        help="RL learning rate (much lower than SFT)")
    parser.add_argument("--num_generations", type=int, default=8,
                        help="G: number of completions per prompt (GRPO group size)")
    parser.add_argument("--max_prompt_length", type=int, default=1536)
    parser.add_argument("--max_completion_length", type=int, default=256)
    parser.add_argument("--beta", type=float, default=0.001,
                        help="β KL penalty coefficient")
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="Sampling temperature during GRPO rollout")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Override num_train_epochs if > 0")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # ── Load tokenizer & model ─────────────────────────────────────────────
    print(f"Loading model from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        jinja_path = Path(args.model_path) / "chat_template.jinja"
        if jinja_path.exists():
            tokenizer.chat_template = jinja_path.read_text()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        # Don't use device_map="auto" with DDP — accelerate handles placement
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    print("Model loaded.")

    # ── Build dataset ──────────────────────────────────────────────────────
    if args.mode == "standard":
        print("Mode: STANDARD — building dataset from ESConv_cleaned ...")
        dataset = load_standard_dataset(args.esconv_dir, tokenizer)
        reward_fn = make_standard_reward_fn()
    else:
        print(f"Mode: DUAL — loading knowledge boundaries from {args.boundaries_path} ...")
        dataset = load_dual_dataset(args.boundaries_path, tokenizer)
        reward_fn = make_dual_reward_fn()

    print(f"Dataset size: {len(dataset)} turns")

    # ── GRPO config ────────────────────────────────────────────────────────
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_dir=f"logs/grpo_{args.mode}",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        # GRPO-specific
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        beta=args.beta,
        temperature=args.temperature,
        # misc
        report_to="tensorboard",
        dataloader_num_workers=2,
        remove_unused_columns=False,   # we need target_strategy, ci, ei, region
        **({"max_steps": args.max_steps} if args.max_steps > 0 else {}),
    )

    # ── Trainer ────────────────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        processing_class=tokenizer,
    )

    print(f"\n{'='*60}")
    print(f"  GRPO Training — mode: {args.mode.upper()}")
    print(f"  Samples : {len(dataset)}")
    print(f"  Epochs  : {args.num_train_epochs}")
    print(f"  G (gens): {args.num_generations}")
    print(f"  LR      : {args.learning_rate}")
    print(f"  KL β    : {args.beta}")
    print(f"  Output  : {args.output_dir}")
    print(f"{'='*60}\n")

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"\nDone. Checkpoint saved → {args.output_dir}")


if __name__ == "__main__":
    main()
