# RLforESC — Fine-tuning for Emotional Support Conversation

SFT baseline for emotional support conversation (ESConv).
Trained on **Qwen2.5-7B-Instruct** with the cleaned ESConv dataset.
Planned next step: GRPO/PPO reinforcement learning with strategy-level reward.

---

## Project Structure

```
RLforESC/
├── scripts/
│   ├── download_datasets.py   # Download ESConv & ESTES from GitHub / HuggingFace
│   ├── analyze_datasets.py    # Dataset statistics & distribution plots
│   ├── clean_esconv.py        # Data cleaning pipeline (see below)
│   ├── train_llama.py         # Full fine-tuning script (Llama 3.1 variant)
│   ├── train_qwen.py          # Full fine-tuning script (Qwen2.5, used in v1)
│   └── test_model.py          # Inference & evaluation on preset test cases
├── results/                   # Test output JSONs (tracked in git)
├── data/                      # Datasets — NOT tracked, auto-generated
├── models/                    # Base model weights — NOT tracked
├── checkpoints/               # Fine-tuned checkpoints — NOT tracked
├── logs/                      # Training logs — NOT tracked
└── requirements.txt
```

---

## Environment Setup

```bash
# Create conda environment
conda create -n RLforESC python=3.10 -y
conda activate RLforESC

# Install PyTorch (CUDA 12.4)
pip install torch==2.4.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install remaining dependencies
pip install transformers>=4.43.0 datasets>=2.20.0 accelerate>=0.31.0 \
            huggingface-hub>=0.24.0 tensorboard deepspeed tqdm pyyaml \
            scipy scikit-learn requests

# Install Flash Attention 2 (recommended for A100/H100)
pip install flash-attn --no-build-isolation
```

---

## Data Pipeline

### 1. Download

```bash
python scripts/download_datasets.py
```

- **ESConv** (ACL 2021, Liu et al.): downloaded from [thu-coai/Emotional-Support-Conversation](https://github.com/thu-coai/Emotional-Support-Conversation), 1,300 English counseling dialogues with 8 annotated support strategies.
- **ESTES** (thu-coai/ESTES on HuggingFace): requires `huggingface-cli login` with a token that has been granted access.

### 2. Analyze

```bash
python scripts/analyze_datasets.py
```

Prints per-split statistics, emotion/strategy distributions, and token-length histograms.

### 3. Clean — `scripts/clean_esconv.py`

The raw ESConv data is already high quality (no missing strategy labels, no invalid records).
The cleaning pipeline applies four targeted steps:

| Step | What it does | Removed |
|------|-------------|---------|
| **Format validation** | Drop records with no `dialog` list or fewer than 2 valid turns | 0 |
| **Trivial-turn stripping** | Remove leading/trailing turns matching `Hello / Hi / Hey / How are you / …` (regex, case-insensitive) | 524 turns |
| **Long-dialogue filter** | Drop entire conversations with **> 60 turns** — these are statistical outliers (top ~2%) and would require excessive truncation during training | 24 dialogues |
| **Deduplication** | MD5 hash on the `situation` field; keep only the first occurrence of each unique situation description | 4 dialogues |

**Before → After:**

| Metric | Before | After |
|--------|--------|-------|
| Dialogues | 1,300 | **1,272** (−2.2%) |
| Total turns | 38,365 | **35,946** |
| Max turns / dialogue | 120 | **60** |
| Avg turns / dialogue | 29.5 | **28.3** |
| Est. token max | 2,409 | **1,701** |
| Est. token mean | 625 | **609** |

All 1,272 remaining dialogues have complete strategy annotations across all supporter turns (8 strategy classes, no missing values).

**Normalization applied at the same time:**

- `speaker` unified: `seeker/usr → user`, `supporter/sys → assistant`
- `strategy` extracted from nested `annotation.strategy`
- Whitespace stripped from all `content` fields

**Split (80 / 10 / 10):**

| Split | Dialogues |
|-------|-----------|
| train | 1,017 |
| valid | 127 |
| test  | 128 |

```bash
python scripts/clean_esconv.py
# Output: data/ESConv_cleaned/{train,valid,test,ESConv_cleaned}.json
```

---

## Training

### Qwen2.5-7B-Instruct (v1 — SFT baseline)

Single A100 80 GB, BF16, Flash Attention 2.

```bash
# Download base model
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-7B-Instruct',
                  local_dir='models/Qwen2.5-7B-Instruct',
                  local_dir_use_symlinks=False)
"

# Start training
python scripts/train_qwen.py \
  --model_name_or_path models/Qwen2.5-7B-Instruct \
  --esconv_dir data/ESConv_cleaned \
  --output_dir checkpoints/qwen25-esc \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --bf16 True \
  --gradient_checkpointing True
```

**Key hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Model | Qwen2.5-7B-Instruct |
| Epochs | 3 |
| Per-device batch | 2 |
| Gradient accumulation | 8 (effective batch = 16) |
| Learning rate | 2e-5 |
| LR schedule | Cosine + 3% warmup |
| Precision | BF16 |
| Attention | Flash Attention 2 |
| Gradient checkpointing | ✓ |
| Loss mask | Assistant turns only |
| Strategy injection | `[StrategyName]` prefix in assistant output |
| Max seq length | 2048 |

**Training results (v1):**

| Metric | Value |
|--------|-------|
| Train loss (final) | 1.688 |
| Valid loss (best, epoch 1) | 1.771 |
| Test loss | 1.769 |
| Training time | 27 min 11 s (1× A100 80 GB) |

---

## Inference / Testing

```bash
# Run 5 preset test cases
python scripts/test_model.py \
  --ft_model checkpoints/qwen25-esc \
  --min_new_tokens 60 \
  --temperature 0.85 \
  --save_output results/my_results.json

# Compare fine-tuned vs base model
python scripts/test_model.py --compare

# Interactive chat mode
python scripts/test_model.py --interactive

# Run only specific cases (1–5)
python scripts/test_model.py --cases 1 3
```

**Recommended generation parameters:**

| Parameter | Default | Recommended |
|-----------|---------|-------------|
| `max_new_tokens` | 300 | 400 |
| `min_new_tokens` | 0 | 60 |
| `temperature` | 0.7 | 0.85 |
| `repetition_penalty` | 1.1 | 1.15 |

Test result JSONs are saved in `results/` and tracked in git for reproducibility.

---

## Datasets

| Dataset | Source | License |
|---------|--------|---------|
| ESConv | [thu-coai/Emotional-Support-Conversation](https://github.com/thu-coai/Emotional-Support-Conversation) | CC BY-NC 4.0 |
| ESTES | [thu-coai/ESTES](https://huggingface.co/datasets/thu-coai/ESTES) | CC BY-NC 4.0 |

---

## Roadmap

- [x] SFT baseline on ESConv (Qwen2.5-7B-Instruct)
- [ ] Download & integrate ESTES dataset
- [ ] GRPO / PPO reinforcement learning with strategy-level reward
- [ ] Automatic evaluation (BLEU, ROUGE, empathy metrics)
- [ ] Multi-turn inference demo

---

## Citation

```bibtex
@inproceedings{liu2021esconv,
  title     = {Towards Emotional Support Dialog Systems},
  author    = {Liu, Siyang and Zheng, Chujie and Demasi, Orianna and
               Sabour, Sahand and Li, Yu and Yu, Zhou and Jiang, Yong
               and Huang, Minlie},
  booktitle = {ACL 2021},
  year      = {2021}
}
```
