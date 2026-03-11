"""
Knowledge Boundary Analysis
============================
After running delineate_knowledge_boundary.py, analyze the resulting JSON
to understand WHERE and WHY knowledge boundaries form, and generate data-driven
insights for improvement.

Usage:
    python scripts/analyze_knowledge_boundary.py \
        --boundaries_path data/knowledge_boundaries/train_turns_with_boundaries.json \
        --output_dir results/boundary_analysis

Produces:
    boundary_analysis.json   — full statistics
    (prints a rich text report to stdout)
"""

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


# ── Helpers ──────────────────────────────────────────────────────────────────

STRATEGIES = [
    "Question", "Restatement or Paraphrasing", "Reflection of feelings",
    "Self-disclosure", "Affirmation and Reassurance", "Providing Suggestions",
    "Information", "Others",
]

def pct(n, total):
    return f"{100*n/total:.1f}%" if total else "N/A"

def avg(lst):
    return sum(lst) / len(lst) if lst else 0.0

def std(lst):
    if len(lst) < 2:
        return 0.0
    m = avg(lst)
    return math.sqrt(sum((x - m) ** 2 for x in lst) / len(lst))

def turn_depth(rec):
    """How deep in the conversation is this assistant turn? (# of messages before it)"""
    return len(rec["prompt_messages"]) - 1   # minus system message

def prompt_token_estimate(rec):
    """Rough token count via word split × 1.3."""
    text = " ".join(
        m["content"] for m in rec["prompt_messages"] if m["role"] != "system"
    )
    return int(len(text.split()) * 1.3)


# ── Main analysis ─────────────────────────────────────────────────────────────

def analyze(records: list[dict]) -> dict:
    total = len(records)
    regions = [r["region"] for r in records]
    ci_vals = [r["ci"] for r in records]
    ei_vals = [r["ei"] for r in records]

    counts = Counter(regions)
    hk, wk, uk = counts["HK"], counts["WK"], counts["UK"]

    # ── 1. Overall distribution ───────────────────────────────────────────────
    overall = {
        "total_turns": total,
        "HK_count": hk, "HK_pct": round(100*hk/total, 2),
        "WK_count": wk, "WK_pct": round(100*wk/total, 2),
        "UK_count": uk, "UK_pct": round(100*uk/total, 2),
        "mean_ci": round(avg(ci_vals), 4),
        "std_ci":  round(std(ci_vals), 4),
        "mean_ei": round(avg(ei_vals), 4),
    }

    # ── 2. Per-strategy breakdown ─────────────────────────────────────────────
    by_strategy = defaultdict(lambda: {"ci_vals": [], "regions": []})
    for r in records:
        s = r["target_strategy"]
        by_strategy[s]["ci_vals"].append(r["ci"])
        by_strategy[s]["regions"].append(r["region"])

    strategy_stats = {}
    for s, data in by_strategy.items():
        n = len(data["ci_vals"])
        rc = Counter(data["regions"])
        strategy_stats[s] = {
            "count": n,
            "mean_ci": round(avg(data["ci_vals"]), 4),
            "std_ci":  round(std(data["ci_vals"]), 4),
            "HK_pct": round(100*rc["HK"]/n, 1),
            "WK_pct": round(100*rc["WK"]/n, 1),
            "UK_pct": round(100*rc["UK"]/n, 1),
        }
    # Sort by mean_ci ascending (hardest first)
    strategy_stats = dict(sorted(strategy_stats.items(), key=lambda x: x[1]["mean_ci"]))

    # ── 3. Boundary vs. turn depth ────────────────────────────────────────────
    depth_buckets = defaultdict(list)  # bucket → list of ci
    for r in records:
        d = turn_depth(r)
        bucket = (d // 4) * 4   # groups: 0-3, 4-7, 8-11, …
        depth_buckets[bucket].append(r["ci"])

    depth_stats = {}
    for bucket in sorted(depth_buckets.keys()):
        cis = depth_buckets[bucket]
        depth_stats[f"depth_{bucket}-{bucket+3}"] = {
            "count": len(cis),
            "mean_ci": round(avg(cis), 4),
        }

    # ── 4. Boundary vs. context length (tokens) ───────────────────────────────
    length_buckets = defaultdict(list)
    for r in records:
        toks = prompt_token_estimate(r)
        bucket = (toks // 200) * 200
        length_buckets[bucket].append(r["ci"])

    length_stats = {}
    for bucket in sorted(length_buckets.keys()):
        cis = length_buckets[bucket]
        length_stats[f"len_{bucket}-{bucket+199}"] = {
            "count": len(cis),
            "mean_ci": round(avg(cis), 4),
        }

    # ── 5. Per-dialogue CI variance (how consistent is boundary across a convo?) ──
    by_dialog = defaultdict(list)
    for r in records:
        by_dialog[r["dialog_id"]].append(r["ci"])

    dialog_ci_stds = [std(v) for v in by_dialog.values() if len(v) > 1]
    dialog_stats = {
        "num_dialogs": len(by_dialog),
        "mean_within_dialog_ci_std": round(avg(dialog_ci_stds), 4),
        "pct_dialogs_mixed_regions": round(
            100 * sum(1 for v in by_dialog.values()
                      if len(set(
                          "HK" if c==1 else "UK" if c==0 else "WK" for c in v
                      )) > 1) / len(by_dialog), 1
        ),
    }

    # ── 6. Strategy transition patterns for WK turns ─────────────────────────
    # For each WK turn, what strategy came JUST before it?
    # (proxy for: does the difficulty depend on context transition?)
    prev_strategy_ci = defaultdict(list)  # (prev_strat, curr_strat) → list[ci]
    for r in records:
        msgs = r["prompt_messages"]
        # Find the last assistant message in context
        prev_asst = None
        for m in reversed(msgs):
            if m["role"] == "assistant":
                # Try to extract strategy from its content
                m_strat = re.match(
                    r"^\s*\[(" + "|".join(re.escape(s) for s in STRATEGIES) + r")\]",
                    m["content"], re.IGNORECASE
                )
                if m_strat:
                    prev_asst = m_strat.group(1)
                break
        if prev_asst:
            key = (prev_asst, r["target_strategy"])
            prev_strategy_ci[key].append(r["ci"])

    # Top 10 hardest transitions (lowest mean ci)
    transition_stats = {
        f"{k[0]} → {k[1]}": {
            "count": len(v),
            "mean_ci": round(avg(v), 4),
        }
        for k, v in sorted(prev_strategy_ci.items(), key=lambda x: avg(x[1]))
        if len(v) >= 5
    }
    # Keep top 10 hardest and top 5 easiest
    items = list(transition_stats.items())
    transition_summary = {
        "hardest_10": dict(items[:10]),
        "easiest_5":  dict(items[-5:]),
    }

    # ── 7. UK turns — what would the model do instead? ───────────────────────
    # (We don't have the sampled completions, so just report their strategy distribution)
    uk_strategies = Counter(r["target_strategy"] for r in records if r["region"] == "UK")
    uk_strategy_dist = dict(uk_strategies.most_common())

    return {
        "overall": overall,
        "by_strategy": strategy_stats,
        "by_turn_depth": depth_stats,
        "by_context_length": length_stats,
        "dialog_level": dialog_stats,
        "strategy_transitions": transition_summary,
        "uk_strategy_distribution": uk_strategy_dist,
    }


# ── Report printer ────────────────────────────────────────────────────────────

def print_report(stats: dict):
    o = stats["overall"]
    print("=" * 62)
    print("  KNOWLEDGE BOUNDARY ANALYSIS REPORT")
    print("=" * 62)

    print(f"\n{'─'*40}")
    print("  1. OVERALL DISTRIBUTION")
    print(f"{'─'*40}")
    print(f"  Total turns  : {o['total_turns']}")
    print(f"  Highly Known : {o['HK_count']:5d}  ({o['HK_pct']:.1f}%)")
    print(f"  Weakly Known : {o['WK_count']:5d}  ({o['WK_pct']:.1f}%)")
    print(f"  Unknown      : {o['UK_count']:5d}  ({o['UK_pct']:.1f}%)")
    print(f"  Mean ci      : {o['mean_ci']:.4f}  ± {o['std_ci']:.4f}")
    print(f"  Mean entropy : {o['mean_ei']:.4f}")

    print(f"\n{'─'*40}")
    print("  2. PER-STRATEGY (sorted hardest → easiest)")
    print(f"{'─'*40}")
    print(f"  {'Strategy':<36} {'N':>5} {'mean_ci':>8} {'HK%':>6} {'WK%':>6} {'UK%':>6}")
    print(f"  {'-'*36} {'-'*5} {'-'*8} {'-'*6} {'-'*6} {'-'*6}")
    for s, d in stats["by_strategy"].items():
        print(f"  {s:<36} {d['count']:>5} {d['mean_ci']:>8.4f} "
              f"{d['HK_pct']:>5.1f}% {d['WK_pct']:>5.1f}% {d['UK_pct']:>5.1f}%")

    print(f"\n{'─'*40}")
    print("  3. CI BY TURN DEPTH (position in dialogue)")
    print(f"{'─'*40}")
    print(f"  {'Depth bucket':<20} {'Count':>6} {'mean_ci':>8}")
    for k, v in stats["by_turn_depth"].items():
        bar = "█" * int(v["mean_ci"] * 20)
        print(f"  {k:<20} {v['count']:>6} {v['mean_ci']:>8.4f}  {bar}")

    print(f"\n{'─'*40}")
    print("  4. CI BY CONTEXT LENGTH (tokens)")
    print(f"{'─'*40}")
    print(f"  {'Token bucket':<20} {'Count':>6} {'mean_ci':>8}")
    for k, v in list(stats["by_context_length"].items())[:12]:
        bar = "█" * int(v["mean_ci"] * 20)
        print(f"  {k:<20} {v['count']:>6} {v['mean_ci']:>8.4f}  {bar}")

    print(f"\n{'─'*40}")
    print("  5. DIALOGUE-LEVEL CONSISTENCY")
    print(f"{'─'*40}")
    dl = stats["dialog_level"]
    print(f"  Dialogues                    : {dl['num_dialogs']}")
    print(f"  Mean within-dialog CI std    : {dl['mean_within_dialog_ci_std']:.4f}")
    print(f"  Dialogues with mixed regions : {dl['pct_dialogs_mixed_regions']:.1f}%")

    print(f"\n{'─'*40}")
    print("  6. HARDEST STRATEGY TRANSITIONS (prev→curr, mean ci)")
    print(f"{'─'*40}")
    for trans, d in stats["strategy_transitions"]["hardest_10"].items():
        print(f"  {trans:<50} n={d['count']:>4}  ci={d['mean_ci']:.4f}")

    print(f"\n{'─'*40}")
    print("  7. UNKNOWN TURNS — target strategy distribution")
    print(f"{'─'*40}")
    uk_total = sum(stats["uk_strategy_distribution"].values())
    for s, n in sorted(stats["uk_strategy_distribution"].items(),
                        key=lambda x: -x[1]):
        bar = "█" * int(20 * n / max(stats["uk_strategy_distribution"].values()))
        print(f"  {s:<36} {n:>5} ({100*n/uk_total:.1f}%)  {bar}")

    print("\n" + "=" * 62)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boundaries_path",
                        default="data/knowledge_boundaries/train_turns_with_boundaries.json")
    parser.add_argument("--output_dir", default="results/boundary_analysis")
    args = parser.parse_args()

    print(f"Loading {args.boundaries_path} ...")
    with open(args.boundaries_path) as f:
        records = json.load(f)
    print(f"Loaded {len(records)} turns.\n")

    stats = analyze(records)
    print_report(stats)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    out = Path(args.output_dir) / "boundary_analysis.json"
    with open(out, "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\nFull stats saved → {out}")


if __name__ == "__main__":
    main()
