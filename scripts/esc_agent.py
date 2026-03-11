#!/usr/bin/env python3
"""
ESC Agent — 基于 Qwen2.5 的情感支持对话 Agent
==============================================

【方法论】
  SFT 的本质局限：模型学的是 [策略标签] 前缀的模式匹配，
  而非"何时该用哪个策略"的序列决策。知识分布测试显示 81% 的
  训练样本处于 Unknown-Knowledge 区域，SFT 无论如何调参都无法
  有效教会模型策略选择。

  本 Agent 方案改为：
  ① 系统提示中给出 8 种策略的理论定义（基于 ESC 框架 + 自定义 Others）
  ② 从训练集自动精选每种策略的代表性示例（few-shot in-context learning）
  ③ 链式思考（Chain-of-Thought）：模型先分析用户情绪状态和对话阶段，
     再选策略，再生成回应 —— 充分调用模型预训练阶段积累的知识

  对比 SFT 的优势：
  ✓ 不受训练集策略分布偏斜的影响
  ✓ 策略选择可解释（思考链可读、可审计）
  ✓ 灵活可调整（改提示即改行为，无需重训）
  ✓ 充分利用 Qwen2.5 预训练的情感理解能力

【使用方式】
  # 交互式（手动输入测试）
  python scripts/esc_agent.py --mode interactive

  # 批量评估（在测试集上计算策略准确率）
  python scripts/esc_agent.py --mode evaluate --split test

  # 单样本调试（查看思考链）
  python scripts/esc_agent.py --mode debug --dialog_id 5
"""

import json
import re
import argparse
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ─────────────────────────────────────────────────────────────────────────────
# 1. 策略定义（基于 ESC 理论，Others 使用自定义定义）
# ─────────────────────────────────────────────────────────────────────────────

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

# 每条定义包含：zh_name（中文名）、definition（定义）、when_to_use（适用时机）
STRATEGY_DEFS: Dict[str, Dict] = {
    "Question": {
        "zh_name": "提问",
        "definition": (
            "通过开放式问题，引导用户主动探索和表达自己的感受、想法或处境。"
            "好的提问创造安全的倾诉空间，而非审讯。"
            "优先使用 what/how 类问题，避免让人感到被质疑的 why。"
        ),
        "when_to_use": (
            "• 对话初期，需要了解用户的背景和处境\n"
            "• 用户的描述较模糊，需要更多细节\n"
            "• 希望引导用户从新的角度自我反思\n"
            "• 感觉用户还有话没说完"
        ),
    },
    "Restatement or Paraphrasing": {
        "zh_name": "重述/释义",
        "definition": (
            "用自己的话把用户刚才说的内容复述出来。"
            "核心目的：①让用户感到被听见；"
            "②帮用户整理和澄清思路；"
            "③确认自己的理解是否准确。"
            "不要逐字重复，而是抓取核心意思重新表达。"
        ),
        "when_to_use": (
            "• 用户说了一大段复杂的情况，需要确认理解\n"
            "• 用户情绪较乱，需要帮其梳理\n"
            "• 想表达深度倾听，而不只是简单点头"
        ),
    },
    "Reflection of feelings": {
        "zh_name": "情感反射",
        "definition": (
            "识别并准确命名用户话语背后的情绪（如孤独、愤怒、委屈、无助、羞耻），"
            "让用户感到自己的情绪被看见和接纳。"
            "重点在情绪层面，而非事件层面。"
            "典型句式：「听起来你感到...」「那种...的感觉一定很难受」"
        ),
        "when_to_use": (
            "• 用户描述了情绪性事件，但未明确说出自己的感受\n"
            "• 用户的情绪强烈，需要优先被接纳，而非直接进入解决方案\n"
            "• 用户似乎在压抑某种情感，需要温和地点破"
        ),
    },
    "Self-disclosure": {
        "zh_name": "自我披露",
        "definition": (
            "适度分享顾问自己的相关经历或感受，"
            "让用户感到自己的处境并不孤单，建立真实的人与人之间的连结。"
            "注意：要适度，不要喧宾夺主；分享后应快速引回用户自身。"
            "典型句式：「我曾经也经历过...」「很多人在类似情况下都会...」"
        ),
        "when_to_use": (
            "• 用户感到自己的问题很奇怪或很少见，需要被正常化\n"
            "• 双方已建立一定信任，分享可加深连结\n"
            "• 用户询问顾问是否真正理解自己的处境"
        ),
    },
    "Affirmation and Reassurance": {
        "zh_name": "肯定与安抚",
        "definition": (
            "肯定用户的努力、勇气和内在力量，对其表示信任；"
            "同时提供安慰，降低焦虑或自我怀疑。"
            "区别于空洞的鼓励，好的肯定应指向用户具体的行为或品质。"
            "典型句式：「你能做到...这本身就需要很大的勇气」「你已经比你意识到的更坚强」"
        ),
        "when_to_use": (
            "• 用户在自我批评或自责\n"
            "• 用户对未来感到无望或绝望\n"
            "• 用户在面对困难时表现出了值得被看见的品质\n"
            "• 用户需要情感上的稳定和安全感"
        ),
    },
    "Providing Suggestions": {
        "zh_name": "提供建议",
        "definition": (
            "在用户情绪得到充分接纳之后，提供具体可行的行动建议或应对策略。"
            "重要：时机比内容更关键——在情绪高峰期直接给建议往往适得其反。"
            "建议应尊重用户自主性：用「可以考虑...」而非「你应该...」。"
        ),
        "when_to_use": (
            "• 用户明确要求建议或解决方法\n"
            "• 情绪处理阶段已完成，用户开始转向问题解决\n"
            "• 用户陷入思维死循环，需要外部视角打破"
        ),
    },
    "Information": {
        "zh_name": "提供信息",
        "definition": (
            "提供与用户处境相关的客观知识、心理健康教育或资源，"
            "如：压力的生理机制、沟通技巧、专业帮助渠道等。"
            "目的是增加用户的理解和掌控感，而非说教。"
            "注意：信息要准确、简洁，与用户当前的需求直接相关。"
        ),
        "when_to_use": (
            "• 用户对某个心理/社会现象感到困惑，想要了解\n"
            "• 用户需要了解某种具体技能（如非暴力沟通、睡眠卫生）\n"
            "• 用户询问资源或专业帮助途径"
        ),
    },
    "Others": {
        "zh_name": "其他（对话维持）",
        "definition": (
            "用于维持对话流畅性的非实质性支持内容，包含四种子类型：\n"
            "  • 开场型：对话最开始的问候、建立关系（「你好，我在这里陪你」）\n"
            "  • 收尾型：告别、祝愿（「保重，随时可以来找我聊」）\n"
            "  • 回应/过渡型：简短确认和承接（「我明白」「请继续说」「听起来真的很不容易」）\n"
            "  • 衔接型：话题转换或总结过渡语\n"
            "注意：此类回应不含实质性支持策略，应简短，为下一个有效策略做铺垫。"
        ),
        "when_to_use": (
            "• 对话的最开始（问候，建立安全感）\n"
            "• 对话接近尾声（告别，积极收尾）\n"
            "• 需要简短确认后再推进，不适合立即给出实质内容\n"
            "• 话题自然过渡时的衔接"
        ),
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. 兜底示例（当数据集示例选取失败时使用）
#    每种策略各一条，格式：{user, context, response}
# ─────────────────────────────────────────────────────────────────────────────

FALLBACK_EXAMPLES: Dict[str, List[Dict]] = {
    "Question": [
        {
            "context": "对话刚开始，用户尚未说明来意",
            "user": "I've just been feeling really off lately and I don't know why.",
            "response": "I'm sorry to hear that. Can you tell me a bit more about what's been going on? When did you start feeling this way?",
        }
    ],
    "Restatement or Paraphrasing": [
        {
            "context": "用户描述了复杂的工作压力处境",
            "user": "My boss keeps adding more to my plate even after I told him I'm overwhelmed, and my coworkers just laugh it off like it's normal.",
            "response": "So if I'm understanding correctly—you're already stretched too thin, you've tried to speak up about it, but instead of getting support you're being dismissed, both by your boss and your colleagues.",
        }
    ],
    "Reflection of feelings": [
        {
            "context": "用户刚得知好友背后说坏话",
            "user": "I just found out my best friend has been talking behind my back. I don't even know what to say.",
            "response": "That sounds really painful. Finding out someone you trusted has been doing that—it's not just hurtful, it probably feels like a betrayal. It makes sense that you wouldn't even know where to start.",
        }
    ],
    "Self-disclosure": [
        {
            "context": "用户感到自己是唯一承担家庭责任的人",
            "user": "I feel like no one understands what it's like to be the only one holding everything together. It's exhausting.",
            "response": "A lot of people in that caretaker position describe it as carrying something invisible that no one else seems to see. You're definitely not alone in feeling that weight—and the exhaustion that comes with it is very real.",
        }
    ],
    "Affirmation and Reassurance": [
        {
            "context": "用户多次求职失败，开始自我怀疑",
            "user": "I've been applying for months and keep getting rejected. Maybe I'm just not good enough.",
            "response": "The fact that you've kept going for months through all those rejections says something real about your persistence. That's not easy. You're not failing—you're dealing with something genuinely hard, and you're still showing up for it.",
        }
    ],
    "Providing Suggestions": [
        {
            "context": "用户说明显焦虑导致失眠，希望得到帮助",
            "user": "I can't stop the anxious thoughts at night. I just lie there for hours.",
            "response": "One thing that helps some people is writing down the worries before bed—getting them out of your head and onto paper so your brain can let go. You might also try a slow exhale technique: breathe in for 4 counts, out for 6. The longer exhale activates your nervous system's calming response.",
        }
    ],
    "Information": [
        {
            "context": "用户对自己无故出现的焦虑感到困惑",
            "user": "I've been feeling really anxious for no reason at all. Is something wrong with me?",
            "response": "Nothing is 'wrong' with you—anxiety without an obvious trigger is actually quite common. It can build up from accumulated stress, disrupted sleep, or physical factors even when life seems calm on the surface. What you're describing has a name, and it responds well to the right kind of support.",
        }
    ],
    "Others": [
        {
            "context": "对话刚开始，用户有些犹豫",
            "user": "Hi... I'm not sure where to start.",
            "response": "Hi, it's completely okay to take your time. I'm here and I'm listening. Just start wherever feels right for you.",
        }
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# 3. 从数据集自动精选 few-shot 示例
# ─────────────────────────────────────────────────────────────────────────────

def _survey_score_avg(dialog: Dict) -> float:
    """计算对话的平均评分（supporter + seeker 综合）。"""
    try:
        scores = dialog["survey_score"]
        vals = []
        for role_scores in scores.values():
            for v in role_scores.values():
                try:
                    vals.append(float(v))
                except (ValueError, TypeError):
                    pass
        return sum(vals) / len(vals) if vals else 0.0
    except Exception:
        return 0.0


def select_examples(
    train_dialogs: List[Dict],
    n_per_strategy: int = 2,
    seed: int = 42,
) -> Dict[str, List[Dict]]:
    """
    从训练集中为每种策略各挑选 n_per_strategy 条代表性示例。

    筛选标准（按优先级）：
    1. 对话整体质量高（survey_score 均值 >= 4.0）
    2. 该 assistant 回应不为空，且长度合理（非 Others: ≥30字符；Others: ≤120字符）
    3. 上下文轮数不超过 6 轮（保持示例简洁，避免 prompt 过长）
    4. 随机打散后取前 n 条（保证多样性）

    返回：{strategy_name: [{context_summary, user, response}, ...]}
    """
    rng = random.Random(seed)

    # 候选池：{strategy -> list of (score, context_turns, user_msg, asst_msg)}
    pool: Dict[str, List] = defaultdict(list)

    for dialog in train_dialogs:
        score = _survey_score_avg(dialog)
        turns = dialog.get("dialog", [])

        for i, turn in enumerate(turns):
            if turn["speaker"] != "assistant":
                continue
            strategy = turn.get("strategy", "").strip()
            if not strategy or strategy not in STRATEGIES:
                continue

            content = turn["content"].strip()
            if not content:
                continue

            # 长度过滤
            if strategy == "Others":
                if len(content) > 150:
                    continue
            else:
                if len(content) < 25:
                    continue

            # 找到这条 assistant 回应之前最近的 user 消息
            user_msg = ""
            for j in range(i - 1, -1, -1):
                if turns[j]["speaker"] == "user":
                    user_msg = turns[j]["content"].strip()
                    break
            if not user_msg:
                continue

            # 上下文（最多 3 轮：user-asst-user）
            context_turns = turns[max(0, i - 6) : i]
            if len(context_turns) > 6:
                continue  # 跳过上下文太深的

            # 用对话情景生成简短的 context_summary
            emotion = dialog.get("emotion_type", "")
            problem = dialog.get("problem_type", "")
            ctx_summary = f"用户情绪：{emotion}；问题类型：{problem}" if emotion else ""

            pool[strategy].append((score, ctx_summary, user_msg, content))

    # 从候选池中挑选
    result: Dict[str, List[Dict]] = {}
    for strategy in STRATEGIES:
        candidates = pool.get(strategy, [])

        if not candidates:
            # 没有候选，使用兜底示例
            result[strategy] = FALLBACK_EXAMPLES.get(strategy, [])
            continue

        # 按 score 降序排列，取前 20 条后随机选 n 条（保证质量同时引入多样性）
        candidates.sort(key=lambda x: -x[0])
        top_pool = candidates[: min(20, len(candidates))]
        rng.shuffle(top_pool)
        selected = top_pool[:n_per_strategy]

        result[strategy] = [
            {
                "context": ctx,
                "user": user,
                "response": resp,
            }
            for _, ctx, user, resp in selected
        ]

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4. 系统提示构建
# ─────────────────────────────────────────────────────────────────────────────

def build_system_prompt(examples: Dict[str, List[Dict]]) -> str:
    """
    构建包含策略定义 + few-shot 示例的系统提示。

    结构：
    [角色设定]
    [输出格式说明]
    [8种策略：定义 + 适用时机 + 示例]
    [注意事项]
    """
    lines = []

    # ── 角色与任务 ──────────────────────────────────────────────────────────
    lines.append("""\
You are a compassionate and skilled emotional support counselor.
Your task: given the conversation history and the user's latest message,
(1) analyze the user's emotional state and the current stage of the conversation,
(2) select the single most appropriate support strategy from the 8 defined below,
(3) generate a warm, natural response using that strategy.

## Output Format
Always respond in exactly this structure:
<thinking>
[Analyze: what is the user feeling? what stage is the conversation at? which strategy fits best and why?]
</thinking>
<strategy>[Exact strategy name from the list below]</strategy>
<response>
[Your actual response to the user]
</response>

Do not add any text outside these tags.
""")

    # ── 策略定义 + 示例 ──────────────────────────────────────────────────────
    lines.append("## The 8 Emotional Support Strategies\n")

    for idx, strategy in enumerate(STRATEGIES, 1):
        info = STRATEGY_DEFS[strategy]
        zh = info["zh_name"]
        defn = info["definition"]
        when = info["when_to_use"]

        lines.append(f"### {idx}. {strategy}（{zh}）")
        lines.append(f"**Definition**: {defn}")
        lines.append(f"**When to use**:\n{when}")

        # 示例
        exs = examples.get(strategy, [])
        if exs:
            lines.append("**Examples**:")
            for ex in exs:
                ctx = ex.get("context", "")
                user_msg = ex["user"]
                resp = ex["response"]
                if ctx:
                    lines.append(f"  > *[Context: {ctx}]*")
                lines.append(f"  > User: {user_msg}")
                lines.append(f"  > Counselor [{strategy}]: {resp}")
        lines.append("")

    # ── 注意事项 ────────────────────────────────────────────────────────────
    lines.append("""\
## Important Guidelines
- Match the strategy to the user's CURRENT emotional need, not just the topic.
- Emotional processing (Reflection, Restatement, Affirmation) almost always comes BEFORE advice.
- "Others" is for conversation openers, closers, and brief acknowledgments only — keep these short.
- Your response should feel natural and human, not clinical or formulaic.
- Do NOT label the strategy in your response text; the tag goes only in <strategy>.
""")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 5. 对话历史格式化
# ─────────────────────────────────────────────────────────────────────────────

def format_history(history: List[Dict], max_turns: int = 10) -> str:
    """
    将对话历史格式化为纯文本，供用户消息的上下文部分使用。

    history: [{"role": "user"/"assistant", "content": str}, ...]
    最多保留最近 max_turns 轮，防止 prompt 过长。
    """
    if not history:
        return "(This is the start of the conversation.)"

    # 只保留最近 max_turns 条
    recent = history[-max_turns:]
    lines = ["[Conversation so far]"]
    for turn in recent:
        role = "User" if turn["role"] == "user" else "Counselor"
        lines.append(f"{role}: {turn['content']}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 6. 输出解析
# ─────────────────────────────────────────────────────────────────────────────

def parse_output(text: str) -> Tuple[str, str, str]:
    """
    从模型输出中解析 thinking、strategy、response 三个部分。

    返回：(thinking, strategy, response)
    如果解析失败，strategy 返回 "Unknown"，response 返回原始文本。
    """
    # 提取 <thinking>...</thinking>
    thinking = ""
    m = re.search(r"<thinking>(.*?)</thinking>", text, re.DOTALL | re.IGNORECASE)
    if m:
        thinking = m.group(1).strip()

    # 提取 <strategy>...</strategy>
    strategy = "Unknown"
    m = re.search(r"<strategy>(.*?)</strategy>", text, re.DOTALL | re.IGNORECASE)
    if m:
        raw = m.group(1).strip()
        # 宽容匹配：找到最接近的合法策略名
        for s in STRATEGIES:
            if s.lower() in raw.lower():
                strategy = s
                break
        if strategy == "Unknown":
            strategy = raw  # 保留原始内容供调试

    # 提取 <response>...</response>
    response = text.strip()
    m = re.search(r"<response>(.*?)</response>", text, re.DOTALL | re.IGNORECASE)
    if m:
        response = m.group(1).strip()

    return thinking, strategy, response


# ─────────────────────────────────────────────────────────────────────────────
# 7. ESCAgent 主类
# ─────────────────────────────────────────────────────────────────────────────

class ESCAgent:
    """
    情感支持对话 Agent。

    工作流程：
    用户输入 + 对话历史
        ↓
    build_system_prompt（策略定义 + few-shot 示例）
        ↓
    Qwen2.5 推理（Chain-of-Thought）
        ↓
    解析输出 → (thinking, predicted_strategy, response)
    """

    def __init__(
        self,
        model_path: str = "models/Qwen2.5-7B-Instruct",
        data_dir: str = "data/ESConv_cleaned",
        n_examples_per_strategy: int = 2,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        device: str = "auto",
        seed: int = 42,
    ):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.seed = seed

        # ── 加载 few-shot 示例 ────────────────────────────────────────────
        print("  [Agent] 从训练集选取 few-shot 示例 ...")
        train_path = Path(data_dir) / "train.json"
        if train_path.exists():
            train_dialogs = json.loads(train_path.read_text(encoding="utf-8"))
            self.examples = select_examples(
                train_dialogs,
                n_per_strategy=n_examples_per_strategy,
                seed=seed,
            )
            total = sum(len(v) for v in self.examples.values())
            print(f"  [Agent] 共选取 {total} 条示例（每策略最多 {n_examples_per_strategy} 条）")
        else:
            print("  [Agent] 找不到训练集，使用内置兜底示例")
            self.examples = FALLBACK_EXAMPLES

        # ── 构建系统提示 ──────────────────────────────────────────────────
        self.system_prompt = build_system_prompt(self.examples)

        # ── 加载模型 ──────────────────────────────────────────────────────
        print(f"  [Agent] 加载 tokenizer: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # 生成任务用 left padding

        print(f"  [Agent] 加载模型: {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()
        print("  [Agent] 模型加载完成。")

    # ── 核心推理 ─────────────────────────────────────────────────────────────

    def predict(
        self,
        user_input: str,
        history: Optional[List[Dict]] = None,
        return_thinking: bool = False,
    ) -> Dict:
        """
        给定用户输入和对话历史，返回策略选择 + 回应。

        参数：
            user_input:      用户当前的输入文本
            history:         之前的对话历史，格式为
                             [{"role": "user"/"assistant", "content": str}, ...]
            return_thinking: 是否在返回结果中包含思考链

        返回：
            {
                "strategy":  str,   # 选择的策略名
                "response":  str,   # 生成的回应
                "thinking":  str,   # 思考链（可选）
                "raw_output": str,  # 模型原始输出（用于调试）
            }
        """
        if history is None:
            history = []

        # 构建 messages（用 Qwen chat template）
        # 系统消息：策略定义 + 示例
        # 用户消息：对话历史（文本） + 当前用户输入
        history_text = format_history(history)
        user_content = f"{history_text}\n\n[Current user message]\n{user_input}"

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": user_content},
        ]

        # 应用 chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=4096,  # 系统提示较长，给足空间
        ).to(self.model.device)

        # 生成
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # 只取新生成的部分
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        raw_output = self.tokenizer.decode(new_ids, skip_special_tokens=True)

        # 解析输出
        thinking, strategy, response = parse_output(raw_output)

        result = {
            "strategy":   strategy,
            "response":   response,
            "raw_output": raw_output,
        }
        if return_thinking:
            result["thinking"] = thinking

        return result

    # ── 批量评估 ─────────────────────────────────────────────────────────────

    def evaluate(
        self,
        dialogs: List[Dict],
        max_dialogs: int = 50,
        verbose: bool = False,
    ) -> Dict:
        """
        在数据集上评估 Agent 的策略选择准确率。

        对每条对话，模拟逐轮推进：将前 i 轮作为历史，第 i+1 轮 user 消息作为输入，
        预测策略，与 ground truth 比较。

        返回：
            {
                "accuracy":        float,          # 整体准确率
                "per_strategy":    {str: float},   # 每种策略的准确率
                "confusion":       {str: {str: int}}, # 混淆矩阵
                "n_total":         int,
                "n_correct":       int,
            }
        """
        random.seed(self.seed)
        selected = random.sample(dialogs, min(max_dialogs, len(dialogs)))

        n_total = 0
        n_correct = 0
        per_strategy_correct: Dict[str, int] = defaultdict(int)
        per_strategy_total:   Dict[str, int] = defaultdict(int)
        confusion: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for d_idx, dialog in enumerate(selected):
            turns = dialog.get("dialog", [])
            history: List[Dict] = []

            for i, turn in enumerate(turns):
                if turn["speaker"] == "assistant":
                    gt_strategy = turn.get("strategy", "").strip()
                    if not gt_strategy or gt_strategy not in STRATEGIES:
                        # 追加到历史后继续
                        history.append({"role": "assistant", "content": turn["content"]})
                        continue

                    # 找到最近的 user 消息
                    user_msg = ""
                    for j in range(i - 1, -1, -1):
                        if turns[j]["speaker"] == "user":
                            user_msg = turns[j]["content"].strip()
                            break
                    if not user_msg:
                        history.append({"role": "assistant", "content": turn["content"]})
                        continue

                    # 预测
                    result = self.predict(user_msg, history=history)
                    pred_strategy = result["strategy"]

                    correct = (pred_strategy == gt_strategy)
                    n_total += 1
                    n_correct += int(correct)
                    per_strategy_total[gt_strategy] += 1
                    per_strategy_correct[gt_strategy] += int(correct)
                    confusion[gt_strategy][pred_strategy] += 1

                    if verbose:
                        mark = "✓" if correct else "✗"
                        print(f"  [{d_idx}:{i}] {mark} GT={gt_strategy:<36} PRED={pred_strategy}")

                    history.append({"role": "assistant", "content": turn["content"]})

                elif turn["speaker"] == "user":
                    history.append({"role": "user", "content": turn["content"]})

            if (d_idx + 1) % 5 == 0:
                running_acc = n_correct / n_total if n_total else 0
                print(f"  进度: {d_idx+1}/{len(selected)} 对话 | "
                      f"累计准确率: {running_acc:.1%} ({n_correct}/{n_total})")

        accuracy = n_correct / n_total if n_total else 0.0
        per_strategy_acc = {
            s: per_strategy_correct[s] / per_strategy_total[s]
            for s in per_strategy_total
        }

        return {
            "accuracy":     round(accuracy, 4),
            "per_strategy": {k: round(v, 4) for k, v in sorted(
                per_strategy_acc.items(), key=lambda x: x[1])},
            "confusion":    {k: dict(v) for k, v in confusion.items()},
            "n_total":      n_total,
            "n_correct":    n_correct,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 8. 交互式模式
# ─────────────────────────────────────────────────────────────────────────────

def run_interactive(agent: ESCAgent):
    """命令行交互式对话，输入 'quit' 退出，'reset' 重置对话历史。"""
    print("\n" + "=" * 60)
    print("  ESC Agent — 交互式对话")
    print("  输入 'quit' 退出 | 'reset' 重置对话 | 'debug' 显示思考链")
    print("=" * 60 + "\n")

    history: List[Dict] = []
    show_thinking = False

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("再见！")
            break
        if user_input.lower() == "reset":
            history = []
            print("  [对话历史已重置]\n")
            continue
        if user_input.lower() == "debug":
            show_thinking = not show_thinking
            print(f"  [思考链显示: {'开启' if show_thinking else '关闭'}]\n")
            continue

        result = agent.predict(user_input, history=history, return_thinking=True)

        if show_thinking:
            print(f"\n  [思考]\n  {result['thinking']}\n")

        print(f"\n  [策略: {result['strategy']}]")
        print(f"Counselor: {result['response']}\n")

        history.append({"role": "user",      "content": user_input})
        history.append({"role": "assistant", "content": result["response"]})


# ─────────────────────────────────────────────────────────────────────────────
# 9. 单样本调试模式
# ─────────────────────────────────────────────────────────────────────────────

def run_debug(agent: ESCAgent, data_dir: str, split: str, dialog_id: int):
    """对数据集中指定对话的每个 assistant turn 进行预测，打印完整思考链。"""
    path = Path(data_dir) / f"{split}.json"
    dialogs = json.loads(path.read_text(encoding="utf-8"))
    dialog = dialogs[dialog_id]

    print(f"\n{'='*64}")
    print(f"  调试对话 #{dialog_id} — {split} split")
    print(f"  情绪类型: {dialog.get('emotion_type', '?')} | "
          f"问题类型: {dialog.get('problem_type', '?')}")
    print(f"{'='*64}\n")

    turns = dialog.get("dialog", [])
    history: List[Dict] = []

    for i, turn in enumerate(turns):
        if turn["speaker"] == "user":
            print(f"  User [{i}]: {turn['content']}")
            history.append({"role": "user", "content": turn["content"]})

        elif turn["speaker"] == "assistant":
            gt = turn.get("strategy", "?")
            user_msg = ""
            for j in range(i - 1, -1, -1):
                if turns[j]["speaker"] == "user":
                    user_msg = turns[j]["content"].strip()
                    break

            if user_msg:
                result = agent.predict(user_msg, history=history[:-0] if history else [],
                                       return_thinking=True)
                match = "✓" if result["strategy"] == gt else "✗"
                print(f"\n  Counselor [{i}]")
                print(f"    GT strategy  : {gt}")
                print(f"  {match} Pred strategy: {result['strategy']}")
                print(f"    Thinking     : {result['thinking'][:200]}...")
                print(f"    Response     : {result['response'][:150]}")
                print(f"    GT response  : {turn['content'][:150]}")
                print()

            history.append({"role": "assistant", "content": turn["content"]})


# ─────────────────────────────────────────────────────────────────────────────
# 10. 报告打印
# ─────────────────────────────────────────────────────────────────────────────

def print_eval_report(stats: Dict, split: str):
    print(f"\n{'='*60}")
    print(f"  ESC Agent 评估报告 — {split} split")
    print(f"{'='*60}")
    print(f"  总体准确率: {stats['accuracy']:.1%}  "
          f"({stats['n_correct']}/{stats['n_total']})\n")

    print(f"  {'策略':<36} {'准确率':>8} {'样本数':>6}")
    print(f"  {'-'*36} {'-'*8} {'-'*6}")
    for s, acc in sorted(stats["per_strategy"].items(), key=lambda x: -x[1]):
        n = sum(stats["confusion"].get(s, {}).values())
        bar = "█" * int(acc * 20)
        print(f"  {s:<36} {acc:>7.1%}  {n:>5}  {bar}")

    print(f"\n  混淆矩阵（行=真实策略，列=预测策略，只显示前5错误）:")
    for gt_s, preds in sorted(stats["confusion"].items()):
        mistakes = {k: v for k, v in preds.items() if k != gt_s}
        if not mistakes:
            continue
        top_mistakes = sorted(mistakes.items(), key=lambda x: -x[1])[:3]
        mistake_str = ", ".join(f"{k}×{v}" for k, v in top_mistakes)
        print(f"    {gt_s:<36} → 误判为: {mistake_str}")

    print(f"\n{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
# 11. 入口
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ESC Agent — 情感支持对话 Agent")
    parser.add_argument(
        "--mode",
        choices=["interactive", "evaluate", "debug"],
        default="interactive",
        help="运行模式",
    )
    parser.add_argument(
        "--model_path",
        default="models/Qwen2.5-7B-Instruct",
        help="模型路径（默认使用 base 模型，也可传入 SFT checkpoint）",
    )
    parser.add_argument(
        "--data_dir",
        default="data/ESConv_cleaned",
        help="ESConv 数据目录",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "valid", "test"],
        help="评估用的数据集 split",
    )
    parser.add_argument(
        "--dialog_id",
        type=int,
        default=0,
        help="debug 模式下使用的对话编号",
    )
    parser.add_argument(
        "--max_dialogs",
        type=int,
        default=50,
        help="评估模式下最多评估多少条对话",
    )
    parser.add_argument(
        "--n_examples",
        type=int,
        default=2,
        help="每种策略在系统提示中展示几条 few-shot 示例",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="生成温度（0 = greedy）",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--output_dir",
        default="results/agent_eval",
        help="评估结果保存目录",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="评估时打印每条预测",
    )
    # 多卡数据并行参数（由 launch 脚本自动传入）
    parser.add_argument("--gpu_id",   type=int, default=0)
    parser.add_argument("--num_gpus", type=int, default=1)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  ESC Agent 启动")
    print(f"  模式: {args.mode} | 模型: {args.model_path}")
    print(f"{'='*60}")

    agent = ESCAgent(
        model_path=args.model_path,
        data_dir=args.data_dir,
        n_examples_per_strategy=args.n_examples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=f"cuda:{args.gpu_id}",  # 每个进程绑定到指定 GPU
    )

    if args.mode == "interactive":
        run_interactive(agent)

    elif args.mode == "evaluate":
        path = Path(args.data_dir) / f"{args.split}.json"
        dialogs = json.loads(path.read_text(encoding="utf-8"))

        # 数据并行分片：GPU i 处理 dialogs[i::num_gpus]
        shard = dialogs[args.gpu_id::args.num_gpus]
        print(f"\n  [GPU {args.gpu_id}] 负责 {len(shard)}/{len(dialogs)} 条对话 "
              f"(共 {args.num_gpus} 卡并行)\n")

        stats = agent.evaluate(shard, max_dialogs=len(shard), verbose=args.verbose)
        print_eval_report(stats, f"{args.split} [GPU {args.gpu_id}]")

        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        # 每个 GPU 写自己的 shard 结果
        shard_path = Path(args.output_dir) / f"eval_{args.split}_shard{args.gpu_id}.json"
        with open(shard_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"\n  Shard 结果已保存 → {shard_path}")

        # 如果是最后一个 GPU（或单卡），尝试合并所有 shard
        if args.gpu_id == args.num_gpus - 1 or args.num_gpus == 1:
            import time, glob
            # 等待其他 shard 文件写入（最多等 60s）
            for _ in range(12):
                shards_done = glob.glob(
                    str(Path(args.output_dir) / f"eval_{args.split}_shard*.json"))
                if len(shards_done) >= args.num_gpus:
                    break
                print(f"  等待其他 GPU shard... ({len(shards_done)}/{args.num_gpus})")
                time.sleep(5)

            # 合并
            all_confusion: Dict = defaultdict(lambda: defaultdict(int))
            all_per_strat_correct: Dict[str, int] = defaultdict(int)
            all_per_strat_total:   Dict[str, int] = defaultdict(int)
            total_n, total_correct = 0, 0

            for sp in sorted(glob.glob(
                    str(Path(args.output_dir) / f"eval_{args.split}_shard*.json"))):
                s = json.loads(Path(sp).read_text())
                total_n       += s["n_total"]
                total_correct += s["n_correct"]
                for strat, acc in s["per_strategy"].items():
                    n = sum(s["confusion"].get(strat, {}).values())
                    all_per_strat_total[strat]   += n
                    all_per_strat_correct[strat] += round(acc * n)
                for gt, preds in s["confusion"].items():
                    for pred, cnt in preds.items():
                        all_confusion[gt][pred] += cnt

            merged = {
                "accuracy":     round(total_correct / total_n, 4) if total_n else 0,
                "per_strategy": {
                    k: round(all_per_strat_correct[k] / all_per_strat_total[k], 4)
                    for k in all_per_strat_total if all_per_strat_total[k] > 0
                },
                "confusion":    {k: dict(v) for k, v in all_confusion.items()},
                "n_total":      total_n,
                "n_correct":    total_correct,
            }
            out_path = Path(args.output_dir) / f"eval_{args.split}_merged.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(merged, f, ensure_ascii=False, indent=2)
            print_eval_report(merged, f"{args.split} [MERGED]")
            print(f"\n  合并结果已保存 → {out_path}")

    elif args.mode == "debug":
        run_debug(agent, args.data_dir, args.split, args.dialog_id)


if __name__ == "__main__":
    main()
