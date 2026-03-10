"""
Qwen2.5-7B-Instruct 全量微调 (Full Fine-Tuning / SFT)
用于情感支持对话 (ESConv_cleaned)

运行方式:
  python train_qwen.py
  torchrun --nproc_per_node=N train_qwen.py
"""

import os
import json
import logging
import math
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    set_seed,
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
        default="models/Qwen2.5-7B-Instruct",
        metadata={"help": "本地模型路径或 HuggingFace 模型名称"},
    )
    use_flash_attention: bool = field(
        default=True,
        metadata={"help": "使用 Flash Attention 2"},
    )
    torch_dtype: str = field(
        default="bfloat16",
        metadata={"help": "模型精度: bfloat16 / float16 / float32"},
    )


@dataclass
class DataArguments:
    esconv_dir: str = field(
        default="data/ESConv_cleaned",
        metadata={"help": "清洗后的 ESConv 数据目录"},
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "最大 token 长度（超出截断）"},
    )
    use_strategy_prompt: bool = field(
        default=True,
        metadata={"help": "将策略标注注入到 assistant 回复前缀"},
    )
    shuffle_seed: int = field(
        default=42,
        metadata={"help": "训练集打乱随机种子"},
    )
    train_on_inputs: bool = field(
        default=False,
        metadata={"help": "False = 只对 assistant 部分计算 loss"},
    )


@dataclass
class TrainArguments(TrainingArguments):
    output_dir: str = field(default="checkpoints/qwen25-esc")
    num_train_epochs: int = field(default=3)
    per_device_train_batch_size: int = field(default=2)
    per_device_eval_batch_size: int = field(default=2)
    gradient_accumulation_steps: int = field(default=8)   # 等效 batch=16
    learning_rate: float = field(default=2e-5)
    weight_decay: float = field(default=0.01)
    warmup_ratio: float = field(default=0.03)
    lr_scheduler_type: str = field(default="cosine")
    logging_steps: int = field(default=10)
    save_strategy: str = field(default="epoch")
    eval_strategy: str = field(default="epoch")
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default="eval_loss")
    greater_is_better: bool = field(default=False)
    fp16: bool = field(default=False)
    bf16: bool = field(default=True)
    gradient_checkpointing: bool = field(default=True)
    dataloader_num_workers: int = field(default=4)
    report_to: str = field(default="tensorboard")
    run_name: str = field(default="qwen25-esc-sft")
    remove_unused_columns: bool = field(default=False)
    deepspeed: Optional[str] = field(default=None)


# ─────────────────────────────────────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────────────────────────────────────

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
# 数据集加载
# ─────────────────────────────────────────────────────────────────────────────

def load_esconv(data_dir: str) -> Dict[str, List]:
    data_dir = Path(data_dir)
    result = {}
    for split in ["train", "valid", "test"]:
        p = data_dir / f"{split}.json"
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                result[split] = json.load(f)
            logger.info(f"ESConv_cleaned [{split}]: {len(result[split])} 条对话")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Dataset 类
# ─────────────────────────────────────────────────────────────────────────────

class EmotionalSupportDataset(Dataset):
    """
    每个样本是一条完整多轮对话，格式化为 Qwen2.5 chat template。
    只对 assistant 回复部分计算 loss。
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
        self.samples = [d for d in dialogs if self._is_valid(d)]
        logger.info(f"有效样本: {len(self.samples)} / {len(dialogs)}")

    def _is_valid(self, item: Dict) -> bool:
        dialog = item.get("dialog", [])
        return isinstance(dialog, list) and len(dialog) >= 2

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        dialog = item["dialog"]

        # 构建 messages 列表
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for turn in dialog:
            speaker = turn["speaker"]
            content = turn["content"].strip()
            strategy = turn.get("strategy", "")
            if not content:
                continue
            if speaker == "user":
                messages.append({"role": "user", "content": content})
            elif speaker == "assistant":
                if self.use_strategy and strategy:
                    content = f"[{strategy}] {content}"
                messages.append({"role": "assistant", "content": content})

        # 格式化完整对话
        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        tokenized = self.tokenizer(
            full_text,
            max_length=self.max_seq_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        input_ids = tokenized["input_ids"]

        if self.train_on_inputs:
            labels = list(input_ids)
        else:
            labels = self._build_labels(messages, input_ids)

        return {
            "input_ids":      torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(tokenized["attention_mask"], dtype=torch.long),
            "labels":         torch.tensor(labels, dtype=torch.long),
        }

    def _build_labels(self, messages: List[Dict], input_ids: List[int]) -> List[int]:
        """
        标记 assistant 部分的 token 位置，其余设为 -100。
        使用 apply_chat_template 定位，不依赖任何模型特定的特殊 token 字符串。
        """
        labels = [-100] * len(input_ids)

        # 逐个 assistant turn 定位
        prefix_messages = []
        for msg in messages:
            if msg["role"] == "assistant":
                # prefix: 到本 turn 开始之前（带 generation prompt）
                prefix_text = self.tokenizer.apply_chat_template(
                    prefix_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prefix_ids = self.tokenizer(
                    prefix_text,
                    add_special_tokens=False,
                    return_tensors=None,
                )["input_ids"]

                # full: 到本 turn 结束（apply_chat_template 自动加 eot）
                full_text = self.tokenizer.apply_chat_template(
                    prefix_messages + [msg],
                    tokenize=False,
                    add_generation_prompt=False,
                )
                full_ids = self.tokenizer(
                    full_text,
                    add_special_tokens=False,
                    return_tensors=None,
                )["input_ids"]

                start = len(prefix_ids)
                end = len(full_ids)
                for i in range(start, min(end, len(labels))):
                    labels[i] = input_ids[i]

            prefix_messages.append(msg)

        return labels


# ─────────────────────────────────────────────────────────────────────────────
# Data Collator
# ─────────────────────────────────────────────────────────────────────────────

class ESCDataCollator:
    def __init__(self, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        max_len = min(max(f["input_ids"].shape[0] for f in features), self.max_length)

        batch_input_ids, batch_attn_mask, batch_labels = [], [], []
        for f in features:
            seq_len = f["input_ids"].shape[0]
            pad_len = max_len - seq_len

            ids    = f["input_ids"][:max_len]
            mask   = f["attention_mask"][:max_len]
            lbs    = f["labels"][:max_len]

            if pad_len > 0:
                ids  = torch.cat([ids,  torch.full((pad_len,), self.pad_id, dtype=torch.long)])
                mask = torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)])
                lbs  = torch.cat([lbs,  torch.full((pad_len,), -100, dtype=torch.long)])

            batch_input_ids.append(ids)
            batch_attn_mask.append(mask)
            batch_labels.append(lbs)

        return {
            "input_ids":      torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_attn_mask),
            "labels":         torch.stack(batch_labels),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 模型加载
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_args: ModelArguments, train_args: TrainArguments):
    logger.info(f"加载 tokenizer: {model_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
    )

    # Qwen2.5 有 pad_token，但若无则 fallback 到 eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(model_args.torch_dtype, torch.bfloat16)

    logger.info(f"加载模型: {model_args.model_name_or_path} (dtype={model_args.torch_dtype})")

    model_kwargs = dict(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    if model_args.use_flash_attention:
        try:
            import flash_attn  # noqa: F401
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("启用 Flash Attention 2")
        except ImportError:
            logger.warning("flash_attn 未安装，使用默认 attention")

    if train_args.deepspeed:
        model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(**model_kwargs, device_map="auto")

    if train_args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"参数量: 总计 {total/1e9:.2f}B, 可训练 {trainable/1e9:.2f}B")

    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# 主训练流程
# ─────────────────────────────────────────────────────────────────────────────

def train(model_args: ModelArguments, data_args: DataArguments, train_args: TrainArguments):
    set_seed(train_args.seed)

    model, tokenizer = load_model_and_tokenizer(model_args, train_args)

    # 加载数据
    esconv = load_esconv(data_args.esconv_dir)
    rng = random.Random(data_args.shuffle_seed)

    train_dialogs = esconv.get("train", [])
    valid_dialogs = esconv.get("valid", [])
    test_dialogs  = esconv.get("test",  [])
    rng.shuffle(train_dialogs)

    logger.info(f"数据量: train={len(train_dialogs)}, valid={len(valid_dialogs)}, test={len(test_dialogs)}")

    common_kwargs = dict(
        tokenizer=tokenizer,
        max_seq_length=data_args.max_seq_length,
        use_strategy=data_args.use_strategy_prompt,
        train_on_inputs=data_args.train_on_inputs,
    )
    train_dataset = EmotionalSupportDataset(train_dialogs, **common_kwargs)
    valid_dataset = EmotionalSupportDataset(valid_dialogs, **common_kwargs)
    test_dataset  = EmotionalSupportDataset(test_dialogs,  **common_kwargs)

    data_collator = ESCDataCollator(tokenizer, max_length=data_args.max_seq_length)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    # 断点续训
    last_checkpoint = None
    if os.path.isdir(train_args.output_dir):
        last_checkpoint = get_last_checkpoint(train_args.output_dir)
        if last_checkpoint:
            logger.info(f"从断点继续: {last_checkpoint}")

    logger.info("=" * 60)
    logger.info("开始全量微调 Qwen2.5-7B-Instruct")
    logger.info(f"  训练集: {len(train_dataset)} 条")
    logger.info(f"  验证集: {len(valid_dataset)} 条")
    logger.info(f"  等效 batch: {train_args.per_device_train_batch_size * train_args.gradient_accumulation_steps}")
    logger.info(f"  学习率: {train_args.learning_rate}")
    logger.info(f"  Epochs: {train_args.num_train_epochs}")
    logger.info("=" * 60)

    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    trainer.save_model(train_args.output_dir)
    tokenizer.save_pretrained(train_args.output_dir)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("在测试集上评估 ...")
    test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    trainer.log_metrics("test", test_metrics)
    trainer.save_metrics("test", test_metrics)

    logger.info(f"训练完成！模型保存至: {train_args.output_dir}")
    return trainer


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainArguments))
    model_args, data_args, train_args = parser.parse_args_into_dataclasses()
    train(model_args, data_args, train_args)
