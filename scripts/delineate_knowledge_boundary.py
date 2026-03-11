"""
Knowledge Boundary Delineation for ESC
=======================================
Based on arXiv:2509.12661v1

For each assistant turn in the training set:
  1. Build conversation history up to (but NOT including) the assistant response
  2. Sample K completions from the SFT model at low temperature
  3. Check how many start with the correct strategy tag
  4. ci = correct_count / K
  5. Classify:
       ci == 1.0  → Highly Known   (HK)
       ci == 0.0  → Unknown        (UK)
       0 < ci < 1 → Weakly Known   (WK)
  6. Also compute binary entropy: ei = H(ci) = -ci*log2(ci) - (1-ci)*log2(1-ci)

Output: data/knowledge_boundaries/train_turns_with_boundaries.json
  [
    {
      "dialog_id": <str>,
      "turn_idx": <int>,          # index of this assistant turn within the dialogue
      "prompt_messages": [...],   # chat messages up to (not including) this turn
      "target_strategy": <str>,
      "target_response": <str>,
      "ci": <float>,
      "ei": <float>,
      "region": "HK" | "WK" | "UK"
    },
    ...
  ]
"""

import argparse
import json
import math
import os
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

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

SYSTEM_PROMPT = (
    "You are a compassionate emotional support counselor. "
    "Listen empathetically to the user's concerns and provide helpful, "
    "emotionally supportive responses. Begin each response with a strategy "
    "tag like [Question] or [Reflection of feelings] to guide your approach."
)


def binary_entropy(ci: float) -> float:
    """H(ci) = -ci*log2(ci) - (1-ci)*log2(1-ci), defined as 0 for ci in {0, 1}."""
    if ci <= 0.0 or ci >= 1.0:
        return 0.0
    return -ci * math.log2(ci) - (1 - ci) * math.log2(1 - ci)


def classify_region(ci: float) -> str:
    if ci >= 1.0:
        return "HK"
    if ci <= 0.0:
        return "UK"
    return "WK"


def build_prompt_messages(dialog: list, turn_idx: int) -> list:
    """Return chat messages for all turns UP TO (not including) turn_idx."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for t in dialog[:turn_idx]:
        messages.append({"role": t["role"], "content": t["content"]})
    return messages


def extract_strategy(text: str) -> str | None:
    m = STRATEGY_PATTERN.match(text)
    if not m:
        return None
    raw = m.group(1).strip()
    # case-insensitive match back to canonical name
    for s in STRATEGIES:
        if s.lower() == raw.lower():
            return s
    return None


@torch.inference_mode()
def sample_k_completions(
    model, tokenizer, messages: list, K: int, temperature: float, max_new_tokens: int
) -> list[str]:
    """Tokenize messages and generate K independent completions."""
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    # Generate K completions in one forward pass (num_return_sequences)
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
        tokens = out[prompt_len:]
        text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        completions.append(text)
    return completions


def load_training_turns(esconv_dir: str) -> list[dict]:
    """Parse train.json → flat list of assistant-turn records."""
    train_path = Path(esconv_dir) / "train.json"
    with open(train_path) as f:
        dialogs = json.load(f)

    records = []
    for dlg in dialogs:
        dialog_id = dlg.get("dialog_id", dlg.get("conv_id", str(len(records))))
        turns = dlg["dialog"]

        # Build a clean turn list (role + content + strategy)
        clean_turns = []
        for t in turns:
            role = t.get("role", t.get("speaker", ""))
            content = t.get("content", t.get("text", "")).strip()
            strategy = t.get("strategy", None)
            clean_turns.append({"role": role, "content": content, "strategy": strategy})

        # Iterate over assistant turns that have a strategy label
        for i, t in enumerate(clean_turns):
            if t["role"] != "assistant":
                continue
            if not t["strategy"]:
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


def main():
    parser = argparse.ArgumentParser(description="Delineate knowledge boundaries for ESConv training turns")
    parser.add_argument("--model_path", default="checkpoints/qwen25-esc",
                        help="Path to the SFT checkpoint")
    parser.add_argument("--esconv_dir", default="data/ESConv_cleaned",
                        help="Path to cleaned ESConv directory")
    parser.add_argument("--output_dir", default="data/knowledge_boundaries",
                        help="Output directory for boundary JSON")
    parser.add_argument("--K", type=int, default=8,
                        help="Number of completions to sample per turn")
    parser.add_argument("--temperature", type=float, default=0.4,
                        help="Sampling temperature (low for diversity/accuracy balance)")
    parser.add_argument("--max_new_tokens", type=int, default=25,
                        help="Max tokens per sampled completion — only the strategy tag matters, ~5–10 tokens")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of turns to process per GPU call (1 is safest for K*prompt context)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip turns already in output file (resume from checkpoint)")
    # ── Multi-GPU sharding (run one process per GPU) ────────────────────────
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="Which GPU to use (0-3). Each process handles its own shard.")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Total number of parallel processes. Data is split into num_gpus shards.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    # Each shard writes its own partial file; a merge step combines them
    if args.num_gpus > 1:
        output_path = Path(args.output_dir) / f"train_turns_shard{args.gpu_id}.json"
    else:
        output_path = Path(args.output_dir) / "train_turns_with_boundaries.json"

    # ── Load already-processed turns if resuming ───────────────────────────
    done_keys: set[tuple] = set()
    results: list[dict] = []
    if args.resume and output_path.exists():
        with open(output_path) as f:
            results = json.load(f)
        done_keys = {(r["dialog_id"], r["turn_idx"]) for r in results}
        print(f"[GPU {args.gpu_id}] Resuming: {len(results)} turns already processed.")

    # ── Load model on the designated GPU ───────────────────────────────────
    device = f"cuda:{args.gpu_id}"
    print(f"[GPU {args.gpu_id}] Loading model from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    # Restore chat template if it wasn't persisted in tokenizer_config.json
    if tokenizer.chat_template is None:
        jinja_path = Path(args.model_path) / "chat_template.jinja"
        if jinja_path.exists():
            tokenizer.chat_template = jinja_path.read_text()
            print(f"[GPU {args.gpu_id}] Loaded chat_template from {jinja_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.eval()
    print("Model loaded.")

    # ── Load training turns & shard ─────────────────────────────────────────
    records = load_training_turns(args.esconv_dir)
    print(f"[GPU {args.gpu_id}] Total assistant turns: {len(records)}")

    if args.num_gpus > 1:
        # Slice this process's share
        records = records[args.gpu_id::args.num_gpus]
        print(f"[GPU {args.gpu_id}] Shard {args.gpu_id}/{args.num_gpus}: {len(records)} turns")

    # ── Delineation loop ────────────────────────────────────────────────────
    for rec in tqdm(records, desc=f"GPU {args.gpu_id}"):
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

        correct = sum(
            1 for c in completions
            if extract_strategy(c) == rec["target_strategy"]
        )
        ci = correct / args.K
        ei = binary_entropy(ci)

        results.append({
            "dialog_id": rec["dialog_id"],
            "turn_idx": rec["turn_idx"],
            "prompt_messages": rec["prompt_messages"],
            "target_strategy": rec["target_strategy"],
            "target_response": rec["target_response"],
            "ci": ci,
            "ei": ei,
            "region": classify_region(ci),
        })

        # Save checkpoint every 200 turns
        if len(results) % 200 == 0:
            with open(output_path, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    # ── Final save ──────────────────────────────────────────────────────────
    with open(output_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # ── Statistics ──────────────────────────────────────────────────────────
    hk = sum(1 for r in results if r["region"] == "HK")
    wk = sum(1 for r in results if r["region"] == "WK")
    uk = sum(1 for r in results if r["region"] == "UK")
    total = len(results)
    avg_ci = sum(r["ci"] for r in results) / total if total else 0.0
    avg_ei = sum(r["ei"] for r in results) / total if total else 0.0

    print(f"\n[GPU {args.gpu_id}] === Knowledge Boundary Statistics ===")
    print(f"Turns processed : {total}")
    print(f"Highly Known    : {hk:5d}  ({100*hk/total:.1f}%)")
    print(f"Weakly Known    : {wk:5d}  ({100*wk/total:.1f}%)")
    print(f"Unknown         : {uk:5d}  ({100*uk/total:.1f}%)")
    print(f"Mean ci         : {avg_ci:.4f}")
    print(f"Mean ei         : {avg_ei:.4f}")
    print(f"Saved → {output_path}")

    # ── Merge shards if this is the last GPU ───────────────────────────────
    if args.num_gpus > 1 and args.gpu_id == args.num_gpus - 1:
        print("\nAll shards done — waiting a moment then merging ...")
        import time
        time.sleep(5)  # allow other processes to finish writing
        merged = []
        for shard_id in range(args.num_gpus):
            shard_path = Path(args.output_dir) / f"train_turns_shard{shard_id}.json"
            if shard_path.exists():
                with open(shard_path) as f:
                    merged.extend(json.load(f))
        # Sort back to original order (by dialog_id, turn_idx) for reproducibility
        merged.sort(key=lambda r: (r["dialog_id"], r["turn_idx"]))
        final_path = Path(args.output_dir) / "train_turns_with_boundaries.json"
        with open(final_path, "w") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        print(f"Merged {len(merged)} turns → {final_path}")


if __name__ == "__main__":
    main()
