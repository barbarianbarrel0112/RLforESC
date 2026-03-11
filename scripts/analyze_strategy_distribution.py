"""
ESConv Strategy Distribution & "Others" Deep Analysis
======================================================
Analyzes the cleaned ESConv dataset across all splits, with special focus on
the "Others" strategy: when it appears, what context triggers it, and what
the model actually says when using it.

Outputs (all in results/strategy_analysis/):
  strategy_distribution.csv       — per-split strategy counts & percentages
  others_context.csv              — what comes before/after Others
  others_content_samples.txt      — 30 random Others responses (manual read)
  others_by_emotion.csv           — Others rate per emotion type
  others_by_problem.csv           — Others rate per problem type
  strategy_transitions.csv        — full strategy → strategy transition matrix
  strategy_cooccurrence.csv       — per-dialogue strategy usage patterns
  analysis_summary.json           — machine-readable full stats
"""

import json
import csv
import re
import random
from collections import Counter, defaultdict
from pathlib import Path

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
SHORT = {
    "Question": "Ques",
    "Restatement or Paraphrasing": "Restate",
    "Reflection of feelings": "Reflect",
    "Self-disclosure": "Self-disc",
    "Affirmation and Reassurance": "Affirm",
    "Providing Suggestions": "Suggest",
    "Information": "Info",
    "Others": "Others",
}

DATA_DIR  = Path("data/ESConv_cleaned")
OUT_DIR   = Path("results/strategy_analysis")
SPLITS    = ["train", "valid", "test"]
SEED      = 42


# ─── Loaders ─────────────────────────────────────────────────────────────────

def load_split(split: str) -> list[dict]:
    with open(DATA_DIR / f"{split}.json") as f:
        return json.load(f)


def iter_assistant_turns(dialogs):
    """Yield (dialog_meta, turn_idx, prev_strat, curr_strat, next_strat, content)."""
    for dlg in dialogs:
        meta = {k: dlg[k] for k in ("emotion_type", "problem_type",
                                     "experience_type", "survey_score")
                if k in dlg}
        turns = dlg["dialog"]
        asst_turns = [(i, t) for i, t in enumerate(turns)
                      if t["speaker"] == "assistant" and t.get("strategy")]
        for j, (i, t) in enumerate(asst_turns):
            prev_s = asst_turns[j-1][1]["strategy"] if j > 0 else None
            next_s = asst_turns[j+1][1]["strategy"] if j < len(asst_turns)-1 else None
            yield meta, i, prev_s, t["strategy"], next_s, t["content"]


# ─── 1. Strategy distribution ─────────────────────────────────────────────────

def strategy_distribution(dialogs_by_split: dict) -> dict:
    counts = {}
    for split, dialogs in dialogs_by_split.items():
        c = Counter()
        for dlg in dialogs:
            for t in dlg["dialog"]:
                if t["speaker"] == "assistant" and t.get("strategy"):
                    c[t["strategy"]] += 1
        counts[split] = c
    return counts


def write_distribution_csv(counts: dict, out_path: Path):
    all_splits = list(counts.keys())
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["Strategy"] + [f"{s}_count" for s in all_splits] + \
                               [f"{s}_pct" for s in all_splits]
        w.writerow(header)
        totals = {s: sum(counts[s].values()) for s in all_splits}
        for strat in STRATEGIES:
            row = [strat]
            for s in all_splits:
                row.append(counts[s].get(strat, 0))
            for s in all_splits:
                n = counts[s].get(strat, 0)
                row.append(f"{100*n/totals[s]:.2f}%")
            w.writerow(row)
        # Total row
        row = ["TOTAL"]
        for s in all_splits:
            row.append(totals[s])
        for s in all_splits:
            row.append("100.00%")
        w.writerow(row)
    print(f"  Saved → {out_path}")


# ─── 2. Others: context (before/after) ────────────────────────────────────────

def others_context(dialogs_by_split: dict) -> dict:
    before_counter = Counter()   # what strategy comes BEFORE Others
    after_counter  = Counter()   # what strategy comes AFTER Others
    pos_in_convo   = []          # relative position (0=first asst turn, etc.)
    total_others   = 0

    for split, dialogs in dialogs_by_split.items():
        for meta, turn_idx, prev_s, curr_s, next_s, content in iter_assistant_turns(dialogs):
            if curr_s != "Others":
                continue
            total_others += 1
            before_counter[prev_s or "<start>"] += 1
            after_counter[next_s or "<end>"] += 1

    return {
        "total": total_others,
        "before": dict(before_counter.most_common()),
        "after":  dict(after_counter.most_common()),
    }


def write_others_context_csv(ctx: dict, out_path: Path):
    rows = []
    all_keys = sorted(set(list(ctx["before"].keys()) + list(ctx["after"].keys())))
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Strategy", "Count_as_BEFORE_Others",
                    "Pct_as_BEFORE", "Count_as_AFTER_Others", "Pct_as_AFTER"])
        tot_b = sum(ctx["before"].values())
        tot_a = sum(ctx["after"].values())
        for k in all_keys:
            b = ctx["before"].get(k, 0)
            a = ctx["after"].get(k, 0)
            w.writerow([k, b, f"{100*b/tot_b:.1f}%", a, f"{100*a/tot_a:.1f}%"])
    print(f"  Saved → {out_path}")


# ─── 3. Others: by emotion type & problem type ────────────────────────────────

def others_by_meta(dialogs_by_split: dict, meta_key: str) -> dict:
    """For each meta value, compute: total_turns, others_turns, others_rate."""
    total  = defaultdict(int)
    others = defaultdict(int)

    for split, dialogs in dialogs_by_split.items():
        for dlg in dialogs:
            val = dlg.get(meta_key, "unknown")
            for t in dlg["dialog"]:
                if t["speaker"] == "assistant" and t.get("strategy"):
                    total[val] += 1
                    if t["strategy"] == "Others":
                        others[val] += 1

    result = {}
    for val in sorted(total.keys()):
        n, o = total[val], others[val]
        result[val] = {"total": n, "others": o,
                       "others_rate": round(100*o/n, 2) if n else 0}
    return result


def write_meta_csv(data: dict, meta_key: str, out_path: Path):
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([meta_key, "total_asst_turns", "others_count", "others_rate%"])
        for val, d in sorted(data.items(), key=lambda x: -x[1]["others_rate"]):
            w.writerow([val, d["total"], d["others"], d["others_rate"]])
    print(f"  Saved → {out_path}")


# ─── 4. Transition matrix ─────────────────────────────────────────────────────

def transition_matrix(dialogs_by_split: dict) -> dict:
    """trans[from][to] = count of consecutive assistant-turn strategy pairs."""
    trans = defaultdict(Counter)
    for split, dialogs in dialogs_by_split.items():
        for meta, turn_idx, prev_s, curr_s, next_s, content in iter_assistant_turns(dialogs):
            if prev_s:
                trans[prev_s][curr_s] += 1
    return trans


def write_transition_csv(trans: dict, out_path: Path):
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["From \\ To"] + STRATEGIES)
        for from_s in STRATEGIES:
            row_total = sum(trans[from_s].values()) or 1
            row = [from_s]
            for to_s in STRATEGIES:
                n = trans[from_s].get(to_s, 0)
                row.append(f"{n} ({100*n/row_total:.0f}%)")
            w.writerow(row)
    print(f"  Saved → {out_path}")


# ─── 5. Others content samples ────────────────────────────────────────────────

def sample_others_content(dialogs_by_split: dict, n: int = 40) -> list[dict]:
    samples = []
    for split, dialogs in dialogs_by_split.items():
        for meta, turn_idx, prev_s, curr_s, next_s, content in iter_assistant_turns(dialogs):
            if curr_s == "Others":
                samples.append({
                    "split": split,
                    "emotion": meta.get("emotion_type", ""),
                    "problem": meta.get("problem_type", ""),
                    "prev_strategy": prev_s or "<start>",
                    "next_strategy": next_s or "<end>",
                    "content": content,
                })
    random.seed(SEED)
    random.shuffle(samples)
    return samples[:n]


def write_others_samples(samples: list, out_path: Path):
    with open(out_path, "w") as f:
        f.write("=" * 72 + "\n")
        f.write(f"  SAMPLE OF 'Others' STRATEGY RESPONSES  (n={len(samples)})\n")
        f.write("=" * 72 + "\n\n")
        for i, s in enumerate(samples, 1):
            f.write(f"[{i:02d}] Split: {s['split']}  |  "
                    f"Emotion: {s['emotion']}  |  Problem: {s['problem']}\n")
            f.write(f"     PREV: {s['prev_strategy']:<36} NEXT: {s['next_strategy']}\n")
            f.write(f"     CONTENT: {s['content'][:200]}\n")
            f.write("-" * 72 + "\n")
    print(f"  Saved → {out_path}")


# ─── 6. Per-dialogue strategy diversity ──────────────────────────────────────

def dialogue_strategy_diversity(dialogs_by_split: dict) -> dict:
    """For each dialogue: how many unique strategies used, Others presence."""
    has_others = 0
    unique_counts = []
    others_position_ratios = []   # at what relative position does Others appear?
    total_dlgs = 0

    for split, dialogs in dialogs_by_split.items():
        for dlg in dialogs:
            total_dlgs += 1
            asst = [t for t in dlg["dialog"]
                    if t["speaker"] == "assistant" and t.get("strategy")]
            strats = [t["strategy"] for t in asst]
            unique_counts.append(len(set(strats)))
            if "Others" in strats:
                has_others += 1
                # relative position of first Others
                idx = strats.index("Others")
                others_position_ratios.append(idx / max(len(strats)-1, 1))

    avg_unique = sum(unique_counts) / len(unique_counts) if unique_counts else 0
    avg_pos    = sum(others_position_ratios) / len(others_position_ratios) \
                 if others_position_ratios else 0
    return {
        "total_dialogues": total_dlgs,
        "dialogues_with_Others": has_others,
        "pct_dialogues_with_Others": round(100*has_others/total_dlgs, 1),
        "avg_unique_strategies_per_dialogue": round(avg_unique, 2),
        "avg_relative_position_of_Others": round(avg_pos, 3),
        "note": "0.0=first turn, 1.0=last turn"
    }


# ─── 7. Print summary to stdout ──────────────────────────────────────────────

def print_summary(dist, ctx, by_emotion, by_problem, diversity):
    print("\n" + "=" * 64)
    print("  STRATEGY DISTRIBUTION REPORT")
    print("=" * 64)

    # Distribution table
    all_splits = list(dist.keys())
    totals = {s: sum(dist[s].values()) for s in all_splits}
    print(f"\n  {'Strategy':<36}", end="")
    for s in all_splits:
        print(f"  {s:>8}", end="")
    print()
    print(f"  {'-'*36}", end="")
    for s in all_splits:
        print(f"  {'--------':>8}", end="")
    print()
    for strat in STRATEGIES:
        print(f"  {strat:<36}", end="")
        for s in all_splits:
            n = dist[s].get(strat, 0)
            pct = 100*n/totals[s]
            marker = " ◀" if strat == "Others" else ""
            print(f"  {n:4d}({pct:4.1f}%){marker}", end="")
        print()
    print(f"  {'TOTAL':<36}", end="")
    for s in all_splits:
        print(f"  {totals[s]:>8}", end="")
    print()

    print(f"\n{'─'*64}")
    print("  'Others' DEEP DIVE")
    print(f"{'─'*64}")
    print(f"  Total 'Others' turns (all splits): {ctx['total']}")
    print(f"  Dialogues containing Others: "
          f"{diversity['dialogues_with_Others']} / "
          f"{diversity['total_dialogues']} "
          f"({diversity['pct_dialogues_with_Others']:.1f}%)")
    print(f"  Avg position in dialogue (0=start,1=end): "
          f"{diversity['avg_relative_position_of_Others']:.3f}")
    print(f"  Avg unique strategies per dialogue: "
          f"{diversity['avg_unique_strategies_per_dialogue']:.2f}")

    print(f"\n  WHAT COMES BEFORE 'Others':")
    for k, v in sorted(ctx["before"].items(), key=lambda x: -x[1]):
        bar = "█" * int(20 * v / max(ctx["before"].values()))
        print(f"    {k:<36} {v:4d}  {bar}")

    print(f"\n  WHAT COMES AFTER 'Others':")
    for k, v in sorted(ctx["after"].items(), key=lambda x: -x[1]):
        bar = "█" * int(20 * v / max(ctx["after"].values()))
        print(f"    {k:<36} {v:4d}  {bar}")

    print(f"\n  'Others' RATE BY EMOTION TYPE (highest first):")
    for val, d in sorted(by_emotion.items(), key=lambda x: -x[1]["others_rate"])[:10]:
        print(f"    {val:<30} {d['others']:3d}/{d['total']:4d}  "
              f"({d['others_rate']:.1f}%)")

    print(f"\n  'Others' RATE BY PROBLEM TYPE (highest first):")
    for val, d in sorted(by_problem.items(), key=lambda x: -x[1]["others_rate"])[:10]:
        print(f"    {val:<30} {d['others']:3d}/{d['total']:4d}  "
              f"({d['others_rate']:.1f}%)")

    print("\n" + "=" * 64)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data ...")
    dialogs_by_split = {s: load_split(s) for s in SPLITS}
    all_dialogs = [dlg for ds in dialogs_by_split.values() for dlg in ds]
    all_split   = {"all": all_dialogs, **dialogs_by_split}

    # 1. Distribution
    print("1. Computing strategy distribution ...")
    dist = strategy_distribution(all_split)
    write_distribution_csv(dist, OUT_DIR / "strategy_distribution.csv")

    # 2. Others context
    print("2. Analyzing 'Others' context ...")
    ctx = others_context(all_split)
    write_others_context_csv(ctx, OUT_DIR / "others_context.csv")

    # 3. Others by emotion / problem
    print("3. Others rate by emotion & problem type ...")
    by_emotion = others_by_meta(all_split, "emotion_type")
    by_problem = others_by_meta(all_split, "problem_type")
    write_meta_csv(by_emotion, "emotion_type", OUT_DIR / "others_by_emotion.csv")
    write_meta_csv(by_problem, "problem_type", OUT_DIR / "others_by_problem.csv")

    # 4. Transition matrix
    print("4. Computing strategy transition matrix ...")
    trans = transition_matrix(all_split)
    write_transition_csv(trans, OUT_DIR / "strategy_transitions.csv")

    # 5. Others content samples
    print("5. Sampling Others content ...")
    samples = sample_others_content(all_split, n=40)
    write_others_samples(samples, OUT_DIR / "others_content_samples.txt")

    # 6. Dialogue diversity
    print("6. Dialogue-level diversity stats ...")
    diversity = dialogue_strategy_diversity(all_split)

    # 7. Print summary
    print_summary(dist, ctx, by_emotion, by_problem, diversity)

    # 8. Save full JSON
    summary = {
        "distribution": {
            split: {k: int(v) for k, v in dist[split].items()}
            for split in dist
        },
        "others_context": ctx,
        "others_by_emotion": by_emotion,
        "others_by_problem": by_problem,
        "dialogue_diversity": diversity,
    }
    with open(OUT_DIR / "analysis_summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  Saved → {OUT_DIR / 'analysis_summary.json'}")
    print(f"\nAll files in: {OUT_DIR}/")


if __name__ == "__main__":
    main()
