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

Choose the single most appropriate strategy from the 8 defined below.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## The 8 Support Strategies + Examples
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**1. Question** – Ask open-ended questions to understand the user's situation,
feelings, or needs. Use when you need more information or to invite sharing.
  User: "I've been feeling really overwhelmed lately."
  → [Question]
  User: "My relationship with my partner has been really difficult."
  → [Question]
  User: "I'm anxious about my job but I don't know how to explain it."
  → [Question]

**2. Restatement or Paraphrasing** – Reflect back what the user said in your
own words to show you've understood and help them feel heard.
  User: "I haven't been sleeping and I'm falling behind at work."
  → [Restatement or Paraphrasing]
  User: "My mom criticizes everything I do and it's exhausting."
  → [Restatement or Paraphrasing]
  User: "I moved to a new city for work but I don't know anyone here."
  → [Restatement or Paraphrasing]

**3. Reflection of feelings** – Name and validate the specific emotion the user
is experiencing. Use "It sounds like you're feeling..." or "That must be...".
  User: "I worked toward that promotion for two years and didn't get it."
  → [Reflection of feelings]
  User: "My best friend stopped talking to me without any explanation."
  → [Reflection of feelings]
  User: "Everything feels pointless. I don't see the purpose in anything."
  → [Reflection of feelings]

**4. Self-disclosure** – Share a brief, relevant personal experience or feeling
to normalize the user's situation and build genuine connection.
  User: "I feel like I'm the only one who struggles this much."
  → [Self-disclosure]
  User: "I can't stop overthinking every decision I make."
  → [Self-disclosure]
  User: "Working from home has made me feel so disconnected from everyone."
  → [Self-disclosure]

**5. Affirmation and Reassurance** – Validate the user's feelings and efforts;
offer encouragement and express confidence that things can improve.
  User: "I've been trying so hard but nothing seems to be working."
  → [Affirmation and Reassurance]
  User: "I'm scared I'm going to completely fall apart."
  → [Affirmation and Reassurance]
  User: "I don't know if I'm making the right decision about any of this."
  → [Affirmation and Reassurance]

**6. Providing Suggestions** – Offer practical advice, coping strategies, or
concrete action steps. Use only when the user is ready for solutions.
  User: "I can't stop thinking about work even when I'm home trying to relax."
  → [Providing Suggestions]
  User: "I want to reconnect with my friends but don't know how to start."
  → [Providing Suggestions]
  User: "I've been feeling anxious all the time and I'm not sure what to do."
  → [Providing Suggestions]

**7. Information** – Provide factual information, explain relevant concepts, or
clarify misunderstandings to help the user understand their situation.
  User: "Is it normal to feel this sad for so long after a breakup?"
  → [Information]
  User: "I keep hearing about therapy but I'm not sure what it actually is."
  → [Information]
  User: "I'm not sure whether my anxiety is serious enough to see a doctor."
  → [Information]

**8. Others** – For conversation openers, closers, brief acknowledgments, or
transitions that don't fit the above. Keep these short and natural.
  User: "Hi, I'm not sure where to start or what to say."
  → [Others]
  User: "Thank you, this conversation really helped me feel better."
  → [Others]
  User: "Actually, I think I feel a bit better now than when we started."
  → [Others]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## Key Rules
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Output ONLY [Strategy Name] — nothing else, no explanation, no response text.
- Emotional validation (Reflection, Restatement, Affirmation) almost always
  comes BEFORE advice (Providing Suggestions) or Information.
- Match the strategy to the user's CURRENT emotional need, not just the topic.
- "Others" is ONLY for openers, closers, and very brief acknowledgments.
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
