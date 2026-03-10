"""
快速测试脚本：对比微调前后的情感支持对话效果

用法:
  python test_model.py                         # 默认跑预设测试用例
  python test_model.py --interactive           # 交互式对话模式
  python test_model.py --compare               # 同时加载基础模型做对比
"""

import argparse
import json
import textwrap
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─────────────────────────────────────────────────────────────────────────────
# 路径配置
# ─────────────────────────────────────────────────────────────────────────────

BASE_MODEL_PATH  = "models/Qwen2.5-7B-Instruct"
FT_MODEL_PATH    = "checkpoints/qwen25-esc"

SYSTEM_PROMPT = """You are a compassionate and professional emotional support counselor.
Your role is to listen empathetically, understand the user's emotional state, and provide appropriate support.
Use evidence-based emotional support strategies such as:
- Emotional Validation: Acknowledge and validate the user's feelings
- Self-disclosure: Share relevant personal experiences when appropriate
- Providing Suggestions: Offer practical advice when the user is ready
- Affirmation and Reassurance: Encourage and reassure the user
- Restatement or Paraphrasing: Reflect back what the user said to show understanding
- Question: Ask open-ended questions to explore feelings further
- Information: Provide relevant information or psychoeducation
- Others: Use general supportive communication

Always maintain a warm, non-judgmental, and supportive tone."""

# ─────────────────────────────────────────────────────────────────────────────
# 预设测试用例（涵盖不同情感类型）
# ─────────────────────────────────────────────────────────────────────────────

TEST_CASES = [
    {
        "label": "Case 1 · 焦虑 / 职业危机",
        "emotion": "anxiety",
        "situation": "I hate my job but I am scared to quit and seek a new career.",
        "history": [],
        "user_input": "I've been feeling so anxious lately. I hate my current job — it's incredibly stressful and pays well, but I'm terrified of leaving to find something better. I don't know what to do.",
    },
    {
        "label": "Case 2 · 抑郁 / 分手",
        "emotion": "depression",
        "situation": "My partner of 3 years just broke up with me out of nowhere.",
        "history": [
            {"role": "user",      "content": "My girlfriend broke up with me last week and I can't stop thinking about her."},
            {"role": "assistant", "content": "[Reflection of feelings] It sounds like you're really hurting right now. Breakups can be incredibly painful, especially when they feel unexpected. How are you holding up day to day?"},
            {"role": "user",      "content": "Honestly not great. I can't sleep, I barely eat. I just feel empty inside."},
        ],
        "user_input": "I keep wondering if I did something wrong. Like maybe if I had been different, she would have stayed.",
    },
    {
        "label": "Case 3 · 悲伤 / 学业压力",
        "emotion": "sadness",
        "situation": "I failed my midterms and feel like a complete failure.",
        "history": [],
        "user_input": "I failed two midterms this week and I'm devastated. I've always been a good student and now I feel like a total failure. My parents are going to be so disappointed.",
    },
    {
        "label": "Case 4 · 愤怒 / 人际冲突",
        "emotion": "anger",
        "situation": "My best friend betrayed my trust by telling others my secret.",
        "history": [
            {"role": "user",      "content": "I told my best friend something really personal and she told everyone."},
            {"role": "assistant", "content": "[Affirmation and Reassurance] That's a serious breach of trust, and your feelings of anger and hurt make complete sense. You confided in someone you trusted deeply."},
        ],
        "user_input": "I'm so angry I can barely talk to her. But we've been friends for 10 years and I don't know if I should just cut her off or try to work it out.",
    },
    {
        "label": "Case 5 · 恐惧 / 健康焦虑",
        "emotion": "fear",
        "situation": "I've been having unexplained physical symptoms and I'm scared it might be something serious.",
        "history": [],
        "user_input": "I've had headaches and fatigue for weeks. I looked online and now I'm convinced I have something terrible. I'm scared to go to the doctor because I don't want to hear bad news.",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# 模型加载
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_path: str, label: str = ""):
    print(f"\n加载模型: {label or model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.eval()
    print(f"  已加载（设备: {next(model.parameters()).device}）")
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# 推理
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def generate(
    model,
    tokenizer,
    history: list,
    user_input: str,
    max_new_tokens: int = 300,
    min_new_tokens: int = 0,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_input})

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens if min_new_tokens > 0 else None,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_ids = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ─────────────────────────────────────────────────────────────────────────────
# 格式化输出
# ─────────────────────────────────────────────────────────────────────────────

W = 70  # 输出宽度

def hr(char="─"):
    print(char * W)

def wrap(text: str, prefix: str = "  ") -> str:
    return textwrap.fill(text, width=W - len(prefix), initial_indent=prefix, subsequent_indent=prefix)

def print_case(case: dict, ft_response: str, base_response: str = None):
    hr("═")
    print(f"  {case['label']}")
    hr("═")
    print(f"  情感: {case['emotion']}")
    print(wrap(f"背景: {case['situation']}", "  "))

    if case["history"]:
        print(f"\n  【对话历史】")
        for turn in case["history"]:
            role = "用户" if turn["role"] == "user" else "模型"
            print(wrap(f"[{role}] {turn['content']}", "  "))

    print(f"\n  【用户输入】")
    print(wrap(case["user_input"], "  "))

    hr()
    print("  【微调后模型回复】")
    print(wrap(ft_response, "  "))

    if base_response is not None:
        hr()
        print("  【基础模型回复（对比）】")
        print(wrap(base_response, "  "))

    hr()


def print_metrics(responses: list):
    """简单统计回复长度分布。"""
    print("\n" + "═" * W)
    print("  测试统计")
    hr()
    print(f"  {'用例':<30s}  {'字符数':>8s}  {'词数':>8s}")
    hr("─")
    for label, resp in responses:
        words = len(resp.split())
        chars = len(resp)
        print(f"  {label:<30s}  {chars:>8d}  {words:>8d}")
    print("═" * W + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 交互模式
# ─────────────────────────────────────────────────────────────────────────────

def interactive_mode(model, tokenizer):
    print("\n" + "═" * W)
    print("  交互式情感支持对话（输入 'quit' 退出，'reset' 清空历史）")
    print("═" * W)
    history = []
    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break
        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "reset":
            history = []
            print("  [对话历史已清空]")
            continue

        response = generate(model, tokenizer, history, user_input)
        print(f"\n模型: {response}")

        history.append({"role": "user",      "content": user_input})
        history.append({"role": "assistant", "content": response})


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ft_model",    default=FT_MODEL_PATH,   help="微调后模型路径")
    parser.add_argument("--base_model",  default=BASE_MODEL_PATH,  help="基础模型路径")
    parser.add_argument("--interactive", action="store_true",      help="进入交互对话模式")
    parser.add_argument("--compare",     action="store_true",      help="同时加载基础模型做对比")
    parser.add_argument("--cases",       type=int, nargs="+",      help="只运行指定编号的用例（1-5）")
    parser.add_argument("--max_new_tokens",  type=int,   default=300, help="最大生成 token 数")
    parser.add_argument("--min_new_tokens",  type=int,   default=60,  help="最小生成 token 数（强制最低长度）")
    parser.add_argument("--temperature",     type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.15)
    parser.add_argument("--save_output",    type=str, default=None, help="将结果保存为 JSON")
    args = parser.parse_args()

    # 加载微调模型
    ft_model, ft_tokenizer = load_model(args.ft_model, "微调后 Qwen2.5-7B-Instruct")

    # 交互模式
    if args.interactive:
        interactive_mode(ft_model, ft_tokenizer)
        return

    # 可选：加载基础模型对比
    base_model = base_tokenizer = None
    if args.compare:
        base_model, base_tokenizer = load_model(args.base_model, "基础 Qwen2.5-7B-Instruct")

    # 选择用例
    cases = TEST_CASES
    if args.cases:
        cases = [TEST_CASES[i - 1] for i in args.cases if 1 <= i <= len(TEST_CASES)]

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
    )

    print(f"\n运行 {len(cases)} 个测试用例 ...")
    all_results = []
    metric_rows = []

    for case in cases:
        ft_resp = generate(ft_model, ft_tokenizer, case["history"], case["user_input"], **gen_kwargs)
        base_resp = None
        if base_model:
            base_resp = generate(base_model, base_tokenizer, case["history"], case["user_input"], **gen_kwargs)

        print_case(case, ft_resp, base_resp)
        metric_rows.append((case["label"][:30], ft_resp))
        all_results.append({
            "label":         case["label"],
            "emotion":       case["emotion"],
            "user_input":    case["user_input"],
            "ft_response":   ft_resp,
            "base_response": base_resp,
        })

    print_metrics(metric_rows)

    if args.save_output:
        Path(args.save_output).write_text(
            json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"结果已保存至: {args.save_output}")


if __name__ == "__main__":
    main()
