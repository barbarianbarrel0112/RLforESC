"""
ESConv 数据清洗脚本

清洗步骤:
  1. 过滤超长对话 (>60 轮)
  2. 去除首轮无意义寒暄 (Hello / Hi 等)
  3. 基于 situation 字段去重
  4. 验证 dialog 格式完整性
  5. 生成详细清洗报告
  6. 保存清洗后的 train/valid/test 分割到 data/ESConv_cleaned/

运行方式: python clean_esconv.py
"""

import json
import re
import hashlib
import statistics
from pathlib import Path
from collections import Counter, defaultdict

# ── 路径 ──────────────────────────────────────────────────────────────────────
SRC_DIR  = Path("data/ESConv")
DEST_DIR = Path("data/ESConv_cleaned")
DEST_DIR.mkdir(parents=True, exist_ok=True)

# ── 清洗参数 ───────────────────────────────────────────────────────────────────
MAX_TURNS       = 60       # 超过此轮次的对话丢弃
MIN_TURNS       = 4        # 少于此轮次的对话丢弃（原数据最少16轮，保险起见保留）
TRIVIAL_PATTERN = re.compile(
    r"^\s*(hello|hi|hey|greetings|good (morning|afternoon|evening)|howdy"
    r"|how are you|how r u|sup|what'?s up|yo)\s*[!.,?]*\s*$",
    re.IGNORECASE,
)
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
# TEST_RATIO  = 0.1  (剩余部分)


# ── 工具函数 ───────────────────────────────────────────────────────────────────

def situation_hash(conv: dict) -> str:
    """对 situation 字段做 MD5，用于去重。"""
    situation = conv.get("situation", "").strip().lower()
    return hashlib.md5(situation.encode()).hexdigest()


def is_trivial_turn(content: str) -> bool:
    """判断是否是无意义寒暄。"""
    return bool(TRIVIAL_PATTERN.match(content.strip()))


def count_tokens_approx(conv: dict) -> int:
    """粗略估算 token 数（英文：字符数 / 4）。"""
    text = " ".join(t.get("content", "") for t in conv.get("dialog", []))
    return len(text) // 4


# ── 清洗主逻辑 ─────────────────────────────────────────────────────────────────

def clean_dataset(data: list) -> tuple[list, dict]:
    """
    对数据集应用所有清洗步骤。

    Returns:
        cleaned: 清洗后的对话列表
        report:  每步骤的统计字典
    """
    report = {
        "original_count": len(data),
        "removed_too_long": [],
        "removed_too_short": [],
        "removed_duplicate": [],
        "removed_invalid": [],
        "trivial_turns_removed": 0,
        "total_turns_before": 0,
        "total_turns_after": 0,
    }

    cleaned = []
    seen_hashes = set()

    for idx, conv in enumerate(data):
        dialog = conv.get("dialog", [])
        report["total_turns_before"] += len(dialog)

        # ── 步骤 1: 过滤无效格式 ──────────────────────────────────────────
        if not dialog or not isinstance(dialog, list):
            report["removed_invalid"].append(idx)
            continue

        # ── 步骤 2: 去除首尾的寒暄轮次 ───────────────────────────────────
        # 删除开头连续的无意义轮次
        trivial_removed = 0
        while dialog and is_trivial_turn(dialog[0].get("content", "")):
            dialog = dialog[1:]
            trivial_removed += 1

        # 删除结尾连续的无意义轮次
        while dialog and is_trivial_turn(dialog[-1].get("content", "")):
            dialog = dialog[:-1]
            trivial_removed += 1

        report["trivial_turns_removed"] += trivial_removed

        # ── 步骤 3: 过滤超长/过短对话 ────────────────────────────────────
        n_turns = len(dialog)
        if n_turns > MAX_TURNS:
            report["removed_too_long"].append({
                "idx": idx,
                "turns": n_turns,
                "emotion": conv.get("emotion_type", ""),
                "problem": conv.get("problem_type", ""),
            })
            continue
        if n_turns < MIN_TURNS:
            report["removed_too_short"].append({
                "idx": idx,
                "turns": n_turns,
            })
            continue

        # ── 步骤 4: 去重（基于 situation）────────────────────────────────
        h = situation_hash(conv)
        if h in seen_hashes:
            report["removed_duplicate"].append({
                "idx": idx,
                "situation_preview": conv.get("situation", "")[:80],
            })
            continue
        seen_hashes.add(h)

        # ── 步骤 5: 规范化对话字段 ────────────────────────────────────────
        normalized_dialog = []
        for turn in dialog:
            speaker  = turn.get("speaker", "").strip()
            content  = turn.get("content", "").strip()
            ann      = turn.get("annotation", {})
            strategy = ann.get("strategy", "") if isinstance(ann, dict) else ""

            # 统一 speaker 名称
            if speaker in ("seeker", "usr"):
                speaker = "user"
            elif speaker in ("supporter", "sys"):
                speaker = "assistant"

            if not content:
                continue

            normalized_dialog.append({
                "speaker":  speaker,
                "content":  content,
                "strategy": strategy,
            })

        if len(normalized_dialog) < MIN_TURNS:
            report["removed_invalid"].append(idx)
            continue

        report["total_turns_after"] += len(normalized_dialog)

        cleaned.append({
            "emotion_type":   conv.get("emotion_type", ""),
            "problem_type":   conv.get("problem_type", ""),
            "situation":      conv.get("situation", "").strip(),
            "survey_score":   conv.get("survey_score", {}),
            "experience_type": conv.get("experience_type", ""),
            "dialog":         normalized_dialog,
        })

    report["final_count"] = len(cleaned)
    return cleaned, report


# ── 统计分析 ───────────────────────────────────────────────────────────────────

def compute_stats(data: list) -> dict:
    """计算清洗后数据集的统计信息。"""
    turns_per_conv   = []
    user_lens        = []
    assist_lens      = []
    strategy_counter = Counter()
    emotion_counter  = Counter()
    problem_counter  = Counter()
    token_estimates  = []

    for conv in data:
        dialog = conv["dialog"]
        turns_per_conv.append(len(dialog))
        token_estimates.append(count_tokens_approx(conv))
        emotion_counter[conv.get("emotion_type", "unknown")] += 1
        problem_counter[conv.get("problem_type", "unknown")] += 1

        for t in dialog:
            n = len(t["content"])
            if t["speaker"] == "user":
                user_lens.append(n)
            else:
                assist_lens.append(n)
                if t.get("strategy"):
                    strategy_counter[t["strategy"]] += 1

    def s(lst):
        if not lst:
            return {}
        return {
            "count":  len(lst),
            "min":    min(lst),
            "max":    max(lst),
            "mean":   round(statistics.mean(lst), 2),
            "median": statistics.median(lst),
            "stdev":  round(statistics.stdev(lst), 2) if len(lst) > 1 else 0,
        }

    return {
        "turns_per_conv":   s(turns_per_conv),
        "user_char_len":    s(user_lens),
        "assist_char_len":  s(assist_lens),
        "token_estimate":   s(token_estimates),
        "emotion_dist":     dict(emotion_counter.most_common()),
        "problem_dist":     dict(problem_counter.most_common()),
        "strategy_dist":    dict(strategy_counter.most_common()),
    }


# ── 报告打印 ───────────────────────────────────────────────────────────────────

def print_report(report: dict, before_stats: dict, after_stats: dict,
                 split_counts: dict):
    sep = "─" * 62

    print("\n" + "=" * 62)
    print("  ESConv 数据清洗报告")
    print("=" * 62)

    # ── 总体变化 ──────────────────────────────────────────────────
    orig  = report["original_count"]
    final = report["final_count"]
    removed = orig - final
    print(f"""
【总体变化】
  清洗前: {orig} 条对话
  清洗后: {final} 条对话
  移除数: {removed} 条 ({removed/orig*100:.1f}%)
""")

    # ── 各步骤明细 ────────────────────────────────────────────────
    print("【各步骤移除明细】")
    steps = [
        ("过滤超长对话 (>60轮)",   report["removed_too_long"]),
        ("过滤过短对话 (<4轮)",    report["removed_too_short"]),
        ("去重 (situation重复)",    report["removed_duplicate"]),
        ("无效格式",                report["removed_invalid"]),
    ]
    for label, lst in steps:
        n = len(lst)
        print(f"  {label:<28s}: {n} 条")

    print(f"\n  寒暄轮次（去除）        : {report['trivial_turns_removed']} 轮")
    print(f"  总轮次: {report['total_turns_before']} -> {report['total_turns_after']}")

    # ── 超长对话详情 ──────────────────────────────────────────────
    if report["removed_too_long"]:
        print(f"\n  被过滤的超长对话:")
        for item in report["removed_too_long"]:
            print(f"    idx={item['idx']:4d}  turns={item['turns']:3d}"
                  f"  {item['emotion']:<15s}  {item['problem']}")

    # ── 去重详情 ──────────────────────────────────────────────────
    if report["removed_duplicate"]:
        print(f"\n  被去重的对话 (situation 重复):")
        for item in report["removed_duplicate"][:5]:
            print(f"    idx={item['idx']:4d}  \"{item['situation_preview']}...\"")
        if len(report["removed_duplicate"]) > 5:
            print(f"    ... 共 {len(report['removed_duplicate'])} 条")

    # ── 清洗前后对比 ─────────────────────────────────────────────
    print(f"\n{sep}")
    print("【清洗前后基本统计对比】")
    print(f"  {'指标':<22s}  {'清洗前':>12s}  {'清洗后':>12s}")
    print(f"  {sep}")

    def row(label, key, sub):
        b = before_stats.get(key, {}).get(sub, "N/A")
        a = after_stats.get(key, {}).get(sub, "N/A")
        bv = f"{b:.2f}" if isinstance(b, float) else str(b)
        av = f"{a:.2f}" if isinstance(a, float) else str(a)
        print(f"  {label:<22s}  {bv:>12s}  {av:>12s}")

    row("对话数",               "turns_per_conv", "count")
    row("每对话平均轮次",        "turns_per_conv", "mean")
    row("每对话最大轮次",        "turns_per_conv", "max")
    row("用户发言均长(字符)",    "user_char_len",  "mean")
    row("支持者发言均长(字符)",  "assist_char_len","mean")
    row("估算Token均值",         "token_estimate", "mean")
    row("估算Token最大",         "token_estimate", "max")

    # ── 分布对比 ──────────────────────────────────────────────────
    print(f"\n{sep}")
    print("【清洗后情感类型分布】")
    total = sum(after_stats["emotion_dist"].values())
    for emo, cnt in after_stats["emotion_dist"].items():
        bar = "█" * int(cnt / total * 30)
        print(f"  {emo:<22s} {cnt:>5d} ({cnt/total*100:5.1f}%) {bar}")

    print(f"\n【清洗后策略分布】")
    total_s = sum(after_stats["strategy_dist"].values())
    for strat, cnt in after_stats["strategy_dist"].items():
        bar = "█" * int(cnt / total_s * 30)
        print(f"  {strat:<35s} {cnt:>5d} ({cnt/total_s*100:5.1f}%) {bar}")

    # ── 分割统计 ──────────────────────────────────────────────────
    print(f"\n{sep}")
    print("【train / valid / test 分割】")
    for split, n in split_counts.items():
        print(f"  {split:<10s}: {n} 条")

    print("\n" + "=" * 62)
    print("  清洗完成，数据已保存至 data/ESConv_cleaned/")
    print("=" * 62 + "\n")


# ── 分割与保存 ─────────────────────────────────────────────────────────────────

def split_and_save(cleaned: list) -> dict:
    """按 8:1:1 分割并保存。"""
    n = len(cleaned)
    train_end = int(n * TRAIN_RATIO)
    valid_end = int(n * (TRAIN_RATIO + VALID_RATIO))

    splits = {
        "train": cleaned[:train_end],
        "valid": cleaned[train_end:valid_end],
        "test":  cleaned[valid_end:],
    }

    for split_name, split_data in splits.items():
        out_path = DEST_DIR / f"{split_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        print(f"  [{split_name}] {len(split_data)} 条 -> {out_path}")

    # 保存完整清洗后数据集
    all_path = DEST_DIR / "ESConv_cleaned.json"
    with open(all_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
    print(f"  [all]   {len(cleaned)} 条 -> {all_path}")

    return {k: len(v) for k, v in splits.items()}


# ── 主入口 ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 62)
    print("  ESConv 数据清洗工具")
    print("=" * 62)

    # 加载原始数据
    src = SRC_DIR / "ESConv.json"
    if not src.exists():
        print(f"[错误] 找不到 {src}，请先运行 download_datasets.py")
        exit(1)

    with open(src, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    print(f"\n已加载原始数据: {len(raw_data)} 条对话")

    # 计算清洗前统计
    before_stats = compute_stats(raw_data)

    # 执行清洗
    print("\n正在清洗...")
    cleaned, report = clean_dataset(raw_data)

    # 计算清洗后统计
    after_stats = compute_stats(cleaned)

    # 保存
    print("\n正在保存分割文件...")
    split_counts = split_and_save(cleaned)

    # 打印报告
    print_report(report, before_stats, after_stats, split_counts)
