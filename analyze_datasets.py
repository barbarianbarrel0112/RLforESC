"""
ESConv + ESTES 数据集初步分析脚本

分析内容：
  1. 基本统计（样本数、对话轮次、发言长度）
  2. 情感类别分布
  3. 策略标注分布
  4. 数据格式说明
  5. 预处理建议

运行方式: python analyze_datasets.py
"""

import json
import os
import re
from pathlib import Path
from collections import Counter, defaultdict
import statistics

DATA_DIR   = Path("data")
ESCONV_DIR = DATA_DIR / "ESConv"
ESTES_DIR  = DATA_DIR / "ESTES"

# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_jsonl(path: Path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def sep(title=""):
    if title:
        print(f"\n{'─'*20} {title} {'─'*20}")
    else:
        print("─" * 60)

def stat_lengths(lengths: list, label: str):
    if not lengths:
        print(f"  {label}: 无数据")
        return
    print(f"  {label}:")
    print(f"    数量: {len(lengths)}")
    print(f"    最小: {min(lengths)}")
    print(f"    最大: {max(lengths)}")
    print(f"    平均: {statistics.mean(lengths):.2f}")
    print(f"    中位数: {statistics.median(lengths):.1f}")
    print(f"    标准差: {statistics.stdev(lengths):.2f}" if len(lengths) > 1 else "")

def print_distribution(counter: Counter, label: str, top_n: int = 20):
    print(f"\n  {label} 分布 (共 {len(counter)} 类):")
    total = sum(counter.values())
    for item, cnt in counter.most_common(top_n):
        bar = "█" * int(cnt / total * 40)
        print(f"    {str(item):<40s} {cnt:>5d} ({cnt/total*100:5.1f}%) {bar}")

# ─────────────────────────────────────────────────────────────────────────────
# ESConv 分析
# ─────────────────────────────────────────────────────────────────────────────

def analyze_esconv():
    print("\n" + "=" * 60)
    print("  ESConv 数据集分析")
    print("=" * 60)

    # 加载主文件
    main_path = ESCONV_DIR / "ESConv.json"
    if not main_path.exists():
        print("  [错误] ESConv.json 不存在，请先运行 download_datasets.py")
        return None

    data = load_json(main_path)
    print(f"\n  总对话数: {len(data)}")

    # ── 预览一条样本的结构 ────────────────────────────────────────────
    sep("样本结构预览")
    sample = data[0]
    print(f"  顶层 key: {list(sample.keys())}")

    # ESConv 结构: emotion_type, problem_type, situation, survey_score, dialog
    # dialog 是列表，每个元素: {speaker, annotation, content, strategy (可能)}
    if "dialog" in sample:
        d0 = sample["dialog"][0]
        print(f"  dialog[0] key: {list(d0.keys())}")
        print(f"  dialog[0] 示例: {json.dumps(d0, ensure_ascii=False)[:200]}")

    # ── 基本统计 ──────────────────────────────────────────────────────
    sep("基本统计")
    n_turns_list     = []   # 每条对话的轮次数
    n_tokens_user    = []   # 用户发言字数
    n_tokens_support = []   # 支持者发言字数
    emotion_counter  = Counter()
    problem_counter  = Counter()
    strategy_counter = Counter()

    for conv in data:
        dialog = conv.get("dialog", conv.get("conversation", []))
        n_turns_list.append(len(dialog))

        emotion_counter[conv.get("emotion_type", "unknown")] += 1
        problem_counter[conv.get("problem_type", "unknown")] += 1

        for turn in dialog:
            speaker  = turn.get("speaker", turn.get("role", ""))
            content  = turn.get("content", turn.get("text", ""))
            strategy = turn.get("strategy", turn.get("annotation", {}).get("strategy", None))
            # 有些版本 annotation 是字典
            if isinstance(turn.get("annotation"), dict):
                strategy = turn["annotation"].get("strategy", strategy)

            n_chars = len(content)
            if speaker in ("seeker", "usr", "user"):
                n_tokens_user.append(n_chars)
            elif speaker in ("supporter", "sys", "system"):
                n_tokens_support.append(n_chars)
                if strategy:
                    if isinstance(strategy, list):
                        strategy_counter.update(strategy)
                    else:
                        strategy_counter[strategy] += 1

    stat_lengths(n_turns_list,     "每条对话的轮次数")
    stat_lengths(n_tokens_user,    "用户发言字数")
    stat_lengths(n_tokens_support, "支持者发言字数")

    # ── 分布 ──────────────────────────────────────────────────────────
    sep("类别分布")
    print_distribution(emotion_counter, "情感类型 (emotion_type)")
    print_distribution(problem_counter, "问题类型 (problem_type)")
    print_distribution(strategy_counter, "支持策略 (strategy)")

    # ── 分割统计 ──────────────────────────────────────────────────────
    sep("train/valid/test 分割统计")
    for split in ["train", "valid", "test"]:
        for ext in [".json", ".txt"]:
            p = ESCONV_DIR / f"{split}{ext}"
            if p.exists():
                if ext == ".json":
                    d = load_json(p)
                    print(f"  {split}.json: {len(d)} 条对话")
                else:
                    with open(p, "r", encoding="utf-8") as f:
                        lines = [l.strip() for l in f if l.strip()]
                    print(f"  {split}.txt:  {len(lines)} 行")
                break

    # ── 格式说明 ──────────────────────────────────────────────────────
    sep("ESConv 数据格式说明")
    print("""
  {
    "emotion_type": "anxiety",         # 情感类型标签
    "problem_type": "job crisis",      # 问题领域
    "situation": "...",                # 求助者描述的情境
    "survey_score": {...},             # 问卷评分（对话前后情绪变化）
    "dialog": [
      {
        "speaker": "usr",              # 发言者: usr / sys
        "content": "...",              # 发言内容
        "annotation": {
          "strategy": "Question"       # 支持策略（仅 sys 有）
        }
      },
      ...
    ]
  }
    """)

    return {
        "n_dialogs": len(data),
        "emotion_types": list(emotion_counter.keys()),
        "strategies": list(strategy_counter.keys()),
        "avg_turns": statistics.mean(n_turns_list) if n_turns_list else 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ESTES 分析
# ─────────────────────────────────────────────────────────────────────────────

def analyze_estes():
    print("\n" + "=" * 60)
    print("  ESTES 数据集分析")
    print("=" * 60)

    # 找到 ESTES 文件
    splits_data = {}
    for split in ["train", "validation", "test"]:
        p = ESTES_DIR / f"{split}.json"
        if p.exists():
            try:
                raw = load_json(p)
                # HF datasets 导出可能是 {"data": [...]} 或直接列表
                if isinstance(raw, dict) and "data" in raw:
                    splits_data[split] = raw["data"]
                elif isinstance(raw, list):
                    splits_data[split] = raw
                else:
                    splits_data[split] = [raw]
            except Exception:
                # 尝试 JSONL
                try:
                    splits_data[split] = load_jsonl(p)
                except Exception as e:
                    print(f"  [错误] 加载 {split}.json 失败: {e}")

    if not splits_data:
        print("  [错误] 未找到 ESTES 数据文件，请先运行 download_datasets.py")
        return None

    # ── 预览结构 ──────────────────────────────────────────────────────
    sep("样本结构预览")
    all_data = []
    for split, rows in splits_data.items():
        all_data.extend(rows)
        print(f"  {split}: {len(rows)} 条")

    if all_data:
        print(f"\n  顶层 key: {list(all_data[0].keys())}")
        print(f"  第一条样本预览:\n  {json.dumps(all_data[0], ensure_ascii=False, indent=2)[:500]}")

    # ── 基本统计 ──────────────────────────────────────────────────────
    sep("基本统计")
    n_turns_list     = []
    n_tokens_all     = []
    emotion_counter  = Counter()
    strategy_counter = Counter()
    intensity_counter = Counter()

    for item in all_data:
        # ESTES 的结构可能是:
        # conversation (list of turns) + 情感强度/策略标注
        dialog = (item.get("dialog")
                  or item.get("conversation")
                  or item.get("turns")
                  or [])
        if isinstance(dialog, list):
            n_turns_list.append(len(dialog))
            for turn in dialog:
                content = (turn.get("content")
                           or turn.get("text")
                           or turn.get("utterance", ""))
                n_tokens_all.append(len(content))
                strat = turn.get("strategy") or turn.get("strategy_label")
                if strat:
                    if isinstance(strat, list):
                        strategy_counter.update(strat)
                    else:
                        strategy_counter[strat] += 1

        # 情感
        emo = (item.get("emotion")
               or item.get("emotion_type")
               or item.get("emotion_label"))
        if emo:
            emotion_counter[emo] += 1

        # 情感强度（ESTES 特有）
        intensity = item.get("emotion_intensity") or item.get("intensity")
        if intensity is not None:
            intensity_counter[str(intensity)] += 1

    stat_lengths(n_turns_list, "每条对话的轮次数")
    stat_lengths(n_tokens_all, "发言字数（所有角色）")

    sep("类别分布")
    if emotion_counter:
        print_distribution(emotion_counter, "情感标签")
    if strategy_counter:
        print_distribution(strategy_counter, "支持策略")
    if intensity_counter:
        print_distribution(intensity_counter, "情感强度")

    sep("ESTES 数据格式说明")
    print("""
  ESTES (thu-coai/ESTES) 在 ESConv 的基础上增加了：
  - 细粒度情感强度标注 (1-5)
  - 更详细的策略标注（每个支持者发言均有）
  - 情感变化轨迹

  基本结构:
  {
    "emotion_type": "...",
    "emotion_intensity": 3,        # 情感强度 1-5
    "problem_type": "...",
    "situation": "...",
    "dialog": [
      {
        "speaker": "usr",
        "content": "...",
        "emotion": "...",          # 逐句情感
        "strategy": "...",         # 支持策略 (sys)
      }
    ]
  }
    """)

    return {
        "n_dialogs": len(all_data),
        "emotion_types": list(emotion_counter.keys()),
        "strategies": list(strategy_counter.keys()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 综合建议
# ─────────────────────────────────────────────────────────────────────────────

def print_recommendations(esconv_info, estes_info):
    print("\n" + "=" * 60)
    print("  数据预处理建议")
    print("=" * 60)
    print("""
1. 数据格式统一
   ├── 将 ESConv 和 ESTES 转换为统一的 JSON 格式
   ├── 统一 speaker 字段: "usr" -> "user", "sys" -> "assistant"
   └── 统一 strategy 字段命名

2. 训练格式构建（适合 Llama 3.1 Instruct）
   使用 ChatML / Llama-3.1 格式:
   ┌─────────────────────────────────────────────────────────┐
   │ <|begin_of_text|>                                       │
   │ <|start_header_id|>system<|end_header_id|>              │
   │ 你是一个专业的情感支持助手...                            │
   │ <|eot_id|>                                              │
   │ <|start_header_id|>user<|end_header_id|>                │
   │ [用户发言]<|eot_id|>                                    │
   │ <|start_header_id|>assistant<|end_header_id|>           │
   │ [支持者发言]<|eot_id|>                                  │
   └─────────────────────────────────────────────────────────┘

3. 策略注入（可选）
   方案A: 策略作为 system prompt 的一部分
   方案B: 策略作为 [STRATEGY: xxx] 标签插入 assistant 发言前
   方案C: 独立的策略预测头（需要修改模型结构，不推荐全量微调时用）

4. 数据清洗建议
   ├── 过滤极短对话（< 4 轮）
   ├── 过滤极长对话（> 截断长度，建议 2048 tokens）
   ├── 检查并去除重复对话
   └── 验证策略标注的完整性

5. 训练样本构建策略
   ├── 多轮对话拼接（一整条对话作为一个样本）
   ├── 或: 滑动窗口（每个 assistant 发言作为一个样本，包含历史上下文）
   └── 推荐: 多轮对话拼接，只对 assistant 部分计算 loss

6. 数据增强（可选）
   └── 用更大模型（如 GPT-4o）对 ESConv 扩充 ESTES 风格的情感强度标注

7. 注意事项
   ├── 中文/英文: ESConv 和 ESTES 均为英文数据集
   ├── Token 数: Llama 3.1 8B 支持 128K 上下文，但微调建议用 ≤2048
   └── 不平衡问题: 情感类别和策略类别可能不均衡，考虑加权 loss 或过采样
""")


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  情感支持对话数据集分析")
    print("  ESConv + ESTES")
    print("=" * 60)

    esconv_info = analyze_esconv()
    estes_info  = analyze_estes()
    print_recommendations(esconv_info, estes_info)

    print("\n分析完成！")
    print("下一步: 运行 train_llama.py 开始微调（请先配置好 GPU 环境）")
