"""
Knowledge Boundary Delineation — Agent CoT Version
====================================================
替代 delineate_knowledge_boundary.py，改用：
  - 模型：Qwen2.5-7B-Instruct（原始基座，无 SFT）
  - System Prompt：包含 8 种策略定义（Agent 风格），要求以 [Strategy] 开头输出
  - K=10, temperature=0.4, max_new_tokens=30
  - 4 GPU 并行，输出到 data/knowledge_boundaries_v3/

运行方式（4卡并行）：
  for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i python scripts/delineate_kb_agent.py \
      --gpu_id $i --num_gpus 4 \
      > logs/kb_agent_gpu$i.log 2>&1 &
  done
"""

import argparse
import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ── Strategy set ──────────────────────────────────────────────────────────────
STRATEGIES = [
    "Question",
    "Restatement or Paraphrasing",
    "Reflection of feelings",
    "Self-disclosure",
    "Affirmation and Reassurance",
    "Providing Suggestions",
    "Information",
    "Others",
]

STRATEGY_PATTERN = re.compile(
    r"^\s*\[(" + "|".join(re.escape(s) for s in STRATEGIES) + r")\]",
    re.IGNORECASE,
)

# ── Agent-style system prompt（含8策略定义，但不要求CoT以节省token）────────────
AGENT_SYSTEM_PROMPT = """\
You are an emotional support counselor. Your ONLY task is to output a strategy tag.

⚠️ STRICT FORMAT: Your entire response must be exactly one line:
[Strategy Name]

Choose from these 8 strategies:

## The 8 Support Strategies

1. **Question** – Ask open-ended questions to better understand the user's \
situation, feelings, or needs. Use at conversation start or when more \
information is needed.

2. **Restatement or Paraphrasing** – Restate or paraphrase what the user said \
to show understanding and help them feel heard.

3. **Reflection of feelings** – Identify and reflect back the emotions the \
user is experiencing (e.g., "It sounds like you're feeling...").

4. **Self-disclosure** – Share a brief relevant personal experience or feeling \
to normalize the user's experience and build connection.

5. **Affirmation and Reassurance** – Validate the user's feelings and efforts; \
offer encouragement and reassurance that things can improve.

6. **Providing Suggestions** – Offer practical advice, coping strategies, or \
action steps when the user is ready for solutions.

7. **Information** – Provide factual information, explain concepts, or clarify \
misunderstandings relevant to the user's situation.

8. **Others** – Use for conversation openers, closers, brief acknowledgments, \
or transitions that don't fit the above categories. Keep these short.

## Important
- Output ONLY [Strategy Name] — one line, nothing else.
- Match the strategy to the user's CURRENT emotional need, not just the topic.
- Emotional validation (Reflection, Restatement, Affirmation) almost always \
comes BEFORE advice (Providing Suggestions).
- "Others" is only for brief openers, closers, or transitions.

Example valid outputs:
[Question]
[Reflection of feelings]
[Providing Suggestions]
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def multiclass_entropy(strategy_preds: list, K: int) -> float:
    """
    论文公式: ei = -Σ_{s∈S} p(s|hi) log p(s|hi)
    p(s|hi) = K次采样中预测为策略s的次数 / K
    使用自然对数(nats)，与分母 log(|S|)=ln(8) 保持一致。
    """
    counts = Counter(strategy_preds)
    ei = 0.0
    for count in counts.values():
        if count > 0:
            p = count / K
            ei -= p * math.log(p)   # 自然对数
    return ei


def binary_entropy(ci: float) -> float:
    """保留用于对比，论文实际不使用此公式。"""
    if ci <= 0.0 or ci >= 1.0:
        return 0.0
    return -ci * math.log2(ci) - (1 - ci) * math.log2(1 - ci)


def classify_region(ci: float) -> str:
    if ci >= 1.0:
        return "HK"
    if ci <= 0.0:
        return "UK"
    return "WK"


def extract_strategy(text: str) -> str | None:
    m = STRATEGY_PATTERN.match(text)
    if not m:
        return None
    raw = m.group(1).strip()
    for s in STRATEGIES:
        if s.lower() == raw.lower():
            return s
    return None


def build_prompt_messages(clean_turns: list, turn_idx: int) -> list:
    """Build chat messages for all turns UP TO (not including) turn_idx."""
    messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
    for t in clean_turns[:turn_idx]:
        messages.append({"role": t["role"], "content": t["content"]})
    return messages


def load_training_turns(esconv_dir: str) -> list[dict]:
    train_path = Path(esconv_dir) / "train.json"
    dialogs = json.loads(train_path.read_text())

    records = []
    for dlg in dialogs:
        dialog_id = dlg.get("dialog_id", dlg.get("conv_id", str(len(records))))
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
            prompt_msgs = build_prompt_messages(clean_turns, i)
            records.append({
                "dialog_id": dialog_id,
                "turn_idx": i,
                "prompt_messages": prompt_msgs,
                "target_strategy": t["strategy"],
                "target_response": t["content"],
            })

    return records


@torch.inference_mode()
def sample_k_completions(
    model, tokenizer, messages: list, K: int, temperature: float, max_new_tokens: int
) -> list[str]:
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    # apply_chat_template may return a tensor or a BatchEncoding
    if hasattr(encoded, "input_ids"):
        input_ids = encoded.input_ids.to(model.device)
    else:
        input_ids = encoded.to(model.device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        num_return_sequences=K,
        pad_token_id=tokenizer.eos_token_id,
    )

    prompt_len = input_ids.shape[-1]
    completions = []
    for out in outputs:
        text = tokenizer.decode(out[prompt_len:], skip_special_tokens=True).strip()
        completions.append(text)
    return completions


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/mnt/teamdrive/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--esconv_dir", default="data/ESConv_cleaned")
    parser.add_argument("--output_dir", default="data/knowledge_boundaries_v3")
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max_new_tokens", type=int, default=20,
                        help="Only need [Strategy Name] tag (~8 tokens). 20 is safe.")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.num_gpus > 1:
        output_path = Path(args.output_dir) / f"train_turns_shard{args.gpu_id}.json"
    else:
        output_path = Path(args.output_dir) / "train_turns_with_boundaries.json"

    # Resume
    done_keys: set = set()
    results: list = []
    if args.resume and output_path.exists():
        results = json.loads(output_path.read_text())
        done_keys = {(r["dialog_id"], r["turn_idx"]) for r in results}
        print(f"[GPU {args.gpu_id}] Resuming: {len(results)} turns already done.")

    # Load model
    device = f"cuda:{args.gpu_id}"
    print(f"[GPU {args.gpu_id}] Loading {args.model_path} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.eval()
    print(f"[GPU {args.gpu_id}] Model loaded.")

    # Load & shard data
    records = load_training_turns(args.esconv_dir)
    print(f"[GPU {args.gpu_id}] Total assistant turns: {len(records)}")
    if args.num_gpus > 1:
        records = records[args.gpu_id::args.num_gpus]
        print(f"[GPU {args.gpu_id}] Shard: {len(records)} turns")

    # Benchmark first turn for ETA
    t0 = time.time()
    first_done = False

    # Delineation loop
    for idx, rec in enumerate(tqdm(records, desc=f"GPU{args.gpu_id}")):
        key = (rec["dialog_id"], rec["turn_idx"])
        if key in done_keys:
            continue

        completions = sample_k_completions(
            model, tokenizer,
            rec["prompt_messages"],
            K=args.K,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )

        # 提取每次采样的策略预测
        strategy_preds = [extract_strategy(c) or "__none__" for c in completions]

        correct = sum(1 for p in strategy_preds if p == rec["target_strategy"])
        ci = correct / args.K

        # 论文多分类熵: ei = -Σ p(s) ln p(s)，包含 __none__（解析失败）
        ei = multiclass_entropy(strategy_preds, args.K)

        results.append({
            "dialog_id": rec["dialog_id"],
            "turn_idx": rec["turn_idx"],
            "prompt_messages": rec["prompt_messages"],
            "target_strategy": rec["target_strategy"],
            "target_response": rec["target_response"],
            "ci": ci,
            "ei": ei,
            "region": classify_region(ci),
            "strategy_counts": dict(Counter(strategy_preds)),  # 完整分布，供调试
        })

        # ETA after first turn
        if not first_done:
            elapsed = time.time() - t0
            total_todo = len(records) - len(done_keys)
            eta_min = elapsed * total_todo / 60
            print(f"\n[GPU {args.gpu_id}] First turn: {elapsed:.1f}s → ETA ≈ {eta_min:.0f} min ({eta_min/60:.1f} hr)")
            first_done = True

        # Checkpoint every 100 turns
        if len(results) % 100 == 0:
            output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))

    # Final save
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))

    # Stats
    hk = sum(1 for r in results if r["region"] == "HK")
    wk = sum(1 for r in results if r["region"] == "WK")
    uk = sum(1 for r in results if r["region"] == "UK")
    total = len(results)
    avg_ci = sum(r["ci"] for r in results) / total if total else 0.0

    print(f"\n[GPU {args.gpu_id}] === Knowledge Boundary Stats ===")
    print(f"Total : {total}")
    print(f"HK    : {hk:5d} ({100*hk/total:.1f}%)")
    print(f"WK    : {wk:5d} ({100*wk/total:.1f}%)")
    print(f"UK    : {uk:5d} ({100*uk/total:.1f}%)")
    print(f"Avg ci: {avg_ci:.4f}")
    print(f"Saved → {output_path}")

    # Merge if last GPU
    if args.num_gpus > 1 and args.gpu_id == args.num_gpus - 1:
        print("\nMerging shards ...")
        time.sleep(5)
        merged = []
        for sid in range(args.num_gpus):
            shard = Path(args.output_dir) / f"train_turns_shard{sid}.json"
            if shard.exists():
                merged.extend(json.loads(shard.read_text()))
        merged.sort(key=lambda r: (str(r["dialog_id"]), r["turn_idx"]))
        final = Path(args.output_dir) / "train_turns_with_boundaries.json"
        final.write_text(json.dumps(merged, ensure_ascii=False, indent=2))
        print(f"Merged {len(merged)} turns → {final}")


if __name__ == "__main__":
    main()
