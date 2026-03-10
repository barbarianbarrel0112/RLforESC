"""
Llama 3.1 8B 全量微调 (Full Fine-Tuning / SFT)
用于情感支持对话 (ESConv + ESTES)

运行前准备:
  1. 安装依赖: pip install -r requirements.txt
  2. 登录 HuggingFace: huggingface-cli login  (需要 Llama 3.1 访问权限)
  3. 下载数据集: python download_datasets.py
  4. 准备至少 2x80GB VRAM (A100/H100) 或 多卡 (推荐 4x40GB A100)

运行方式:
  # 单卡
  python train_llama.py --config configs/train_config.yaml

  # 多卡 (torchrun / deepspeed)
  torchrun --nproc_per_node=4 train_llama.py --config configs/train_config.yaml
  deepspeed train_llama.py --deepspeed configs/ds_config.json --config configs/train_config.yaml

注意: 全量微调 8B 模型，BF16 下约需 64GB VRAM；
      配合 gradient_checkpointing + ZeRO-3 可降至 4x24GB。
"""

import os
import json
import logging
import argparse
import math
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    get_cosine_schedule_with_warmup,
    set_seed,
    BitsAndBytesConfig,
)
from transformers.trainer_utils import get_last_checkpoint
import transformers

# ─────────────────────────────────────────────────────────────────────────────
# 日志
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 超参数配置
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        metadata={"help": "HuggingFace 模型名称或本地路径"},
    )
    use_flash_attention: bool = field(
        default=True,
        metadata={"help": "使用 Flash Attention 2（需要安装 flash-attn）"},
    )
    torch_dtype: str = field(
        default="bfloat16",
        metadata={"help": "模型精度: bfloat16 / float16 / float32"},
    )


@dataclass
class DataArguments:
    esconv_dir: str = field(
        default="data/ESConv",
        metadata={"help": "ESConv 数据目录"},
    )
    estes_dir: str = field(
        default="data/ESTES",
        metadata={"help": "ESTES 数据目录"},
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "最大 token 长度（超出截断）"},
    )
    use_strategy_prompt: bool = field(
        default=True,
        metadata={"help": "将策略标注注入到 system prompt"},
    )
    data_seed: int = field(
        default=42,
        metadata={"help": "数据划分随机种子"},
    )
    train_on_inputs: bool = field(
        default=False,
        metadata={"help": "是否对用户输入部分计算 loss（False 表示只对 assistant 计算）"},
    )


@dataclass
class TrainArguments(TrainingArguments):
    # 覆盖或新增一些默认值
    output_dir: str = field(default="checkpoints/llama3-esc")
    num_train_epochs: int = field(default=3)
    per_device_train_batch_size: int = field(default=2)
    per_device_eval_batch_size: int = field(default=2)
    gradient_accumulation_steps: int = field(default=8)  # 等效 batch_size=16
    learning_rate: float = field(default=2e-5)
    weight_decay: float = field(default=0.01)
    warmup_ratio: float = field(default=0.03)
    lr_scheduler_type: str = field(default="cosine")
    logging_steps: int = field(default=10)
    save_strategy: str = field(default="epoch")
    evaluation_strategy: str = field(default="epoch")
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default="eval_loss")
    greater_is_better: bool = field(default=False)
    fp16: bool = field(default=False)
    bf16: bool = field(default=True)          # A100/H100 推荐 bf16
    gradient_checkpointing: bool = field(default=True)
    dataloader_num_workers: int = field(default=4)
    report_to: str = field(default="tensorboard")
    run_name: str = field(default="llama3-esc-sft")
    remove_unused_columns: bool = field(default=False)
    # DeepSpeed 配置（可选，命令行 --deepspeed 指定）
    deepspeed: Optional[str] = field(default=None)


# ─────────────────────────────────────────────────────────────────────────────
# 对话模板 (Llama 3.1 Instruct 格式)
# ─────────────────────────────────────────────────────────────────────────────

LLAMA3_SYSTEM_PROMPT = """You are a compassionate and professional emotional support counselor.
Your role is to listen empathetically, understand the user's emotional state, and provide appropriate support.
Use evidence-based emotional support strategies such as:
- Emotional Validation: Acknowledge and validate the user's feelings
- Self-disclosure: Share relevant personal experiences when appropriate
- Providing Suggestions: Offer practical advice when the user is ready
- Affirmation and Reassurance: Encourage and reassure the user
- Restatement: Reflect back what the user said to show understanding
- Question: Ask open-ended questions to explore feelings further
- Information: Provide relevant information or psychoeducation

Always maintain a warm, non-judgmental, and supportive tone."""

STRATEGY_SYSTEM_PROMPT_TEMPLATE = """You are a compassionate and professional emotional support counselor.
Your role is to listen empathetically and provide appropriate support using the following strategy: [{strategy}].

Strategy descriptions:
- Emotional Validation: Acknowledge and validate the user's feelings without judgment
- Self-disclosure: Share relevant personal experiences to build connection
- Providing Suggestions: Offer practical, actionable advice when appropriate
- Affirmation and Reassurance: Provide encouragement and reassurance
- Restatement: Paraphrase what the user said to show active listening
- Question: Ask thoughtful questions to better understand the situation
- Information: Share relevant information or psychoeducation
- Others: Use general supportive communication

Always maintain a warm, non-judgmental, and supportive tone."""


def format_conversation_llama3(
    dialog: List[Dict],
    tokenizer,
    system_prompt: str = LLAMA3_SYSTEM_PROMPT,
    use_strategy: bool = False,
) -> Tuple[str, List[Tuple[int, int]]]:
    """
    将对话格式化为 Llama 3.1 Instruct 格式。

    Returns:
        formatted_text: 完整的格式化字符串
        assistant_spans: assistant 回复在字符串中的 (start, end) 位置列表
    """
    messages = [{"role": "system", "content": system_prompt}]

    for turn in dialog:
        speaker = turn.get("speaker", turn.get("role", ""))
        content = turn.get("content", turn.get("text", "")).strip()
        strategy = (turn.get("strategy")
                    or turn.get("annotation", {}).get("strategy", "")
                    if isinstance(turn.get("annotation"), dict) else "")

        if not content:
            continue

        if speaker in ("seeker", "usr", "user"):
            messages.append({"role": "user", "content": content})
        elif speaker in ("supporter", "sys", "system", "assistant"):
            if use_strategy and strategy:
                # 将策略作为前缀嵌入 assistant 回复
                content = f"[{strategy}] {content}"
            messages.append({"role": "assistant", "content": content})

    # 用 tokenizer 的 apply_chat_template 构建最终格式
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return formatted


# ─────────────────────────────────────────────────────────────────────────────
# 数据集加载与预处理
# ─────────────────────────────────────────────────────────────────────────────

def load_esconv(data_dir: str) -> Dict[str, List]:
    """加载 ESConv 数据，返回 {train, valid, test} 字典。"""
    data_dir = Path(data_dir)
    result = {}
    for split in ["train", "valid", "test"]:
        for ext in [".json", ".txt"]:
            p = data_dir / f"{split}{ext}"
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    result[split] = json.load(f)
                logger.info(f"ESConv [{split}]: {len(result[split])} 条对话")
                break
        else:
            # 尝试从主文件加载
            main = data_dir / "ESConv.json"
            if main.exists() and split not in result:
                with open(main, "r", encoding="utf-8") as f:
                    all_data = json.load(f)
                n = len(all_data)
                result["train"] = all_data[:int(n * 0.8)]
                result["valid"] = all_data[int(n * 0.8):int(n * 0.9)]
                result["test"]  = all_data[int(n * 0.9):]
                logger.info(f"ESConv: 从主文件生成分割 {[f'{k}:{len(v)}' for k,v in result.items()]}")
                break
    return result


def load_estes(data_dir: str) -> Dict[str, List]:
    """加载 ESTES 数据，返回 {train, validation, test} 字典。"""
    data_dir = Path(data_dir)
    result = {}
    for split in ["train", "validation", "test"]:
        p = data_dir / f"{split}.json"
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict) and "data" in raw:
                raw = raw["data"]
            elif isinstance(raw, dict):
                raw = list(raw.values())
            result[split] = raw if isinstance(raw, list) else [raw]
            logger.info(f"ESTES [{split}]: {len(result[split])} 条对话")
    return result


def normalize_dialog(item: Dict) -> Optional[Dict]:
    """将不同数据集的格式统一为标准格式。"""
    dialog = (item.get("dialog")
              or item.get("conversation")
              or item.get("turns"))
    if not dialog or not isinstance(dialog, list):
        return None

    normalized_turns = []
    for turn in dialog:
        speaker = turn.get("speaker", turn.get("role", "")).lower()
        content = turn.get("content", turn.get("text", turn.get("utterance", ""))).strip()
        # 统一 speaker 名称
        if speaker in ("seeker", "usr", "user"):
            speaker = "user"
        elif speaker in ("supporter", "sys", "system", "assistant"):
            speaker = "assistant"
        else:
            continue  # 跳过未知角色

        strategy = ""
        if "strategy" in turn:
            strategy = turn["strategy"] or ""
        elif "annotation" in turn and isinstance(turn["annotation"], dict):
            strategy = turn["annotation"].get("strategy", "")

        if content:
            normalized_turns.append({
                "speaker": speaker,
                "content": content,
                "strategy": strategy,
            })

    if len(normalized_turns) < 2:
        return None

    return {
        "dialog": normalized_turns,
        "emotion_type": item.get("emotion_type", item.get("emotion", "")),
        "problem_type": item.get("problem_type", ""),
        "situation": item.get("situation", ""),
    }


class EmotionalSupportDataset(Dataset):
    """
    情感支持对话数据集。

    每个样本是一条完整的多轮对话，格式化为 Llama 3.1 Instruct 格式。
    只对 assistant 回复部分计算 loss（即 train_on_inputs=False）。
    """

    def __init__(
        self,
        dialogs: List[Dict],
        tokenizer,
        max_seq_length: int = 2048,
        use_strategy: bool = True,
        train_on_inputs: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.use_strategy = use_strategy
        self.train_on_inputs = train_on_inputs

        # 过滤并标准化
        self.samples = []
        for item in dialogs:
            normalized = normalize_dialog(item)
            if normalized is not None:
                self.samples.append(normalized)

        logger.info(f"有效样本数: {len(self.samples)} / {len(dialogs)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        dialog = item["dialog"]

        # 动态系统提示：根据第一个策略决定 system prompt
        if self.use_strategy:
            strategies = [t["strategy"] for t in dialog if t.get("strategy")]
            first_strategy = strategies[0] if strategies else "Others"
            system_prompt = STRATEGY_SYSTEM_PROMPT_TEMPLATE.format(strategy=first_strategy)
        else:
            system_prompt = LLAMA3_SYSTEM_PROMPT

        # 逐步构建消息列表并标记 assistant 部分的 token 位置
        # 方法: 先 tokenize 完整对话，再找 assistant 部分的位置
        formatted = format_conversation_llama3(
            dialog,
            self.tokenizer,
            system_prompt=system_prompt,
            use_strategy=self.use_strategy,
        )

        # Tokenize
        tokenized = self.tokenizer(
            formatted,
            max_length=self.max_seq_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        input_ids = tokenized["input_ids"]

        # 构建 labels: 只对 assistant 部分计算 loss
        labels = self._build_labels(dialog, input_ids, system_prompt)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(tokenized["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def _build_labels(
        self,
        dialog: List[Dict],
        input_ids: List[int],
        system_prompt: str,
    ) -> List[int]:
        """
        构建 labels，非 assistant 部分设为 -100（不参与 loss 计算）。
        """
        if self.train_on_inputs:
            # 全部参与 loss
            return list(input_ids)

        labels = [-100] * len(input_ids)

        # 找到每个 assistant turn 在 token 序列中的位置
        # Llama 3.1 assistant turn 的结束标记: <|eot_id|>
        # assistant 开始标记: <|start_header_id|>assistant<|end_header_id|>\n\n
        assistant_header = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        eot_token = "<|eot_id|>"

        # 逐个 assistant turn 找其在完整 token 序列中的位置
        # 策略: 重新 tokenize 前缀，用前缀长度定位
        messages = [{"role": "system", "content": system_prompt}]
        for turn in dialog:
            speaker = turn["speaker"]
            content = turn["content"]
            if self.use_strategy and turn.get("strategy") and speaker == "assistant":
                content = f"[{turn['strategy']}] {content}"

            if speaker == "user":
                messages.append({"role": "user", "content": content})
            elif speaker == "assistant":
                # tokenize 到当前 assistant turn 开始之前的前缀
                prefix_messages = messages.copy()
                prefix = self.tokenizer.apply_chat_template(
                    prefix_messages,
                    tokenize=False,
                    add_generation_prompt=True,  # 添加 assistant 开始标记
                )
                prefix_ids = self.tokenizer(
                    prefix,
                    add_special_tokens=False,
                    return_tensors=None,
                )["input_ids"]

                # assistant 回复的结束: prefix + content + eot
                full_turn = prefix + content + eot_token
                full_ids = self.tokenizer(
                    full_turn,
                    add_special_tokens=False,
                    return_tensors=None,
                )["input_ids"]

                start = len(prefix_ids)
                end   = len(full_ids)

                # 在截断后的范围内赋值
                for i in range(start, min(end, len(labels))):
                    labels[i] = input_ids[i]

                # 更新消息列表，加入该 assistant turn
                messages.append({"role": "assistant", "content": turn["content"]})

        return labels


# ─────────────────────────────────────────────────────────────────────────────
# 数据整合与拆分
# ─────────────────────────────────────────────────────────────────────────────

def build_datasets(
    data_args: DataArguments,
    tokenizer,
) -> Tuple[EmotionalSupportDataset, EmotionalSupportDataset, EmotionalSupportDataset]:
    """加载并整合 ESConv + ESTES，返回 train/valid/test Dataset。"""

    esconv = load_esconv(data_args.esconv_dir)
    estes  = load_estes(data_args.estes_dir)

    # 合并 train/valid/test
    def merge(split_name: str, estes_split: str) -> List[Dict]:
        result = []
        if split_name in esconv:
            result.extend(esconv[split_name])
        if estes_split in estes:
            result.extend(estes[estes_split])
        return result

    train_dialogs = merge("train", "train")
    valid_dialogs = merge("valid", "validation")
    test_dialogs  = merge("test",  "test")

    # 打乱训练集
    rng = random.Random(data_args.data_seed)
    rng.shuffle(train_dialogs)

    logger.info(f"合并后: train={len(train_dialogs)}, valid={len(valid_dialogs)}, test={len(test_dialogs)}")

    common_kwargs = dict(
        tokenizer=tokenizer,
        max_seq_length=data_args.max_seq_length,
        use_strategy=data_args.use_strategy_prompt,
        train_on_inputs=data_args.train_on_inputs,
    )

    train_dataset = EmotionalSupportDataset(train_dialogs, **common_kwargs)
    valid_dataset = EmotionalSupportDataset(valid_dialogs, **common_kwargs)
    test_dataset  = EmotionalSupportDataset(test_dialogs,  **common_kwargs)

    return train_dataset, valid_dataset, test_dataset


# ─────────────────────────────────────────────────────────────────────────────
# 自定义 Collator（处理变长序列）
# ─────────────────────────────────────────────────────────────────────────────

class ESCDataCollator:
    """动态 padding，并确保 labels 中的 pad 位置为 -100。"""

    def __init__(self, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        max_len = max(f["input_ids"].shape[0] for f in features)
        max_len = min(max_len, self.max_length)

        batch_input_ids   = []
        batch_attn_mask   = []
        batch_labels      = []

        for f in features:
            seq_len = f["input_ids"].shape[0]
            pad_len = max_len - seq_len

            input_ids   = f["input_ids"][:max_len]
            attn_mask   = f["attention_mask"][:max_len]
            labels      = f["labels"][:max_len]

            if pad_len > 0:
                input_ids = torch.cat([input_ids,
                                       torch.full((pad_len,), self.pad_id, dtype=torch.long)])
                attn_mask = torch.cat([attn_mask,
                                       torch.zeros(pad_len, dtype=torch.long)])
                labels    = torch.cat([labels,
                                       torch.full((pad_len,), -100, dtype=torch.long)])

            batch_input_ids.append(input_ids)
            batch_attn_mask.append(attn_mask)
            batch_labels.append(labels)

        return {
            "input_ids":      torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_attn_mask),
            "labels":         torch.stack(batch_labels),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 模型加载
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_args: ModelArguments, train_args: TrainArguments):
    """加载 Llama 3.1 8B 模型和 tokenizer。"""

    logger.info(f"加载 tokenizer: {model_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
    )

    # Llama 3.1 没有 pad_token，用 eos_token 代替
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 设置 padding 方向（Decoder-only 模型用 left padding 推理，但训练用 right）
    tokenizer.padding_side = "right"

    # 模型精度
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }
    torch_dtype = dtype_map.get(model_args.torch_dtype, torch.bfloat16)

    logger.info(f"加载模型: {model_args.model_name_or_path} (dtype={model_args.torch_dtype})")

    model_kwargs = dict(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    # Flash Attention 2
    if model_args.use_flash_attention:
        try:
            import flash_attn  # noqa: F401
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("启用 Flash Attention 2")
        except ImportError:
            logger.warning("flash_attn 未安装，使用默认 attention")

    # DeepSpeed ZeRO-3 下不能 device_map=auto
    if train_args.deepspeed:
        model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            **model_kwargs,
            device_map="auto",
        )

    # Gradient checkpointing（节省显存）
    if train_args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数: 总计 {total_params/1e9:.2f}B, 可训练 {trainable_params/1e9:.2f}B")

    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# 评估指标
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    """计算 perplexity 等指标（Trainer 回调用）。"""
    logits, labels = eval_pred
    # 只计算非 -100 位置的 loss
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")
    loss = loss_fct(
        torch.tensor(shift_logits).view(-1, shift_logits.shape[-1]),
        torch.tensor(shift_labels).view(-1),
    )
    perplexity = math.exp(loss.item()) if loss.item() < 100 else float("inf")
    return {"perplexity": perplexity}


# ─────────────────────────────────────────────────────────────────────────────
# 主训练流程
# ─────────────────────────────────────────────────────────────────────────────

def train(
    model_args: ModelArguments,
    data_args: DataArguments,
    train_args: TrainArguments,
):
    # 设置随机种子
    set_seed(train_args.seed)

    # 加载模型和 tokenizer
    model, tokenizer = load_model_and_tokenizer(model_args, train_args)

    # 构建数据集
    train_dataset, valid_dataset, test_dataset = build_datasets(data_args, tokenizer)

    # Data collator
    data_collator = ESCDataCollator(tokenizer, max_length=data_args.max_seq_length)

    # Trainer
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        # compute_metrics=compute_metrics,  # 全量微调时 logits 太大，建议关闭
    )

    # 断点续训
    last_checkpoint = None
    if os.path.isdir(train_args.output_dir):
        last_checkpoint = get_last_checkpoint(train_args.output_dir)
        if last_checkpoint:
            logger.info(f"从断点继续训练: {last_checkpoint}")

    # 开始训练
    logger.info("=" * 60)
    logger.info("开始全量微调 Llama 3.1 8B")
    logger.info(f"  训练集: {len(train_dataset)} 条")
    logger.info(f"  验证集: {len(valid_dataset)} 条")
    logger.info(f"  批次大小(等效): {train_args.per_device_train_batch_size * train_args.gradient_accumulation_steps}")
    logger.info(f"  学习率: {train_args.learning_rate}")
    logger.info(f"  Epochs: {train_args.num_train_epochs}")
    logger.info("=" * 60)

    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    # 保存最终模型
    trainer.save_model(train_args.output_dir)
    tokenizer.save_pretrained(train_args.output_dir)

    # 记录训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # 测试集评估
    logger.info("在测试集上评估 ...")
    test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    trainer.log_metrics("test", test_metrics)
    trainer.save_metrics("test", test_metrics)

    logger.info(f"训练完成！模型保存至: {train_args.output_dir}")
    return trainer


# ─────────────────────────────────────────────────────────────────────────────
# 推理/生成示例
# ─────────────────────────────────────────────────────────────────────────────

def generate_response(
    model,
    tokenizer,
    user_message: str,
    history: Optional[List[Dict]] = None,
    strategy: Optional[str] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """
    单次推理示例（训练完成后调用）。

    Args:
        model: 微调后的模型
        tokenizer: 对应的 tokenizer
        user_message: 当前用户输入
        history: 历史对话列表 [{"role": "user"/"assistant", "content": "..."}]
        strategy: 可选的策略标注
        max_new_tokens: 最大生成 token 数
    """
    if history is None:
        history = []

    system_prompt = (
        STRATEGY_SYSTEM_PROMPT_TEMPLATE.format(strategy=strategy)
        if strategy
        else LLAMA3_SYSTEM_PROMPT
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 只返回新生成的 token
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response.strip()


# ─────────────────────────────────────────────────────────────────────────────
# 命令行入口
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainArguments)
    )
    model_args, data_args, train_args = parser.parse_args_into_dataclasses()
    return model_args, data_args, train_args


if __name__ == "__main__":
    model_args, data_args, train_args = parse_args()
    train(model_args, data_args, train_args)
