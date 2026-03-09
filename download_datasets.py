"""
下载 ESConv 和 ESTES 数据集用于情感支持对话研究。

ESConv:  Emotional Support Conversation (ACL 2021, Liu et al.)
         来源: https://github.com/thu-coai/Emotional-Support-Conversation
ESTES:   Empathetic and Emotional Support with Strategy (从 Hugging Face 下载)

运行方式: python download_datasets.py
"""

import os
import json
import requests
import zipfile
import shutil
from pathlib import Path

# ── 输出目录 ────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
ESCONV_DIR = DATA_DIR / "ESConv"
ESTES_DIR  = DATA_DIR / "ESTES"

DATA_DIR.mkdir(exist_ok=True)
ESCONV_DIR.mkdir(exist_ok=True)
ESTES_DIR.mkdir(exist_ok=True)

# ── ESConv 原始文件直链 (GitHub raw) ────────────────────────────────────────
ESCONV_FILES = {
    "ESConv.json": (
        "https://raw.githubusercontent.com/thu-coai/Emotional-Support-Conversation"
        "/main/ESConv.json"
    ),
    "train.txt": (
        "https://raw.githubusercontent.com/thu-coai/Emotional-Support-Conversation"
        "/main/data/train.txt"
    ),
    "valid.txt": (
        "https://raw.githubusercontent.com/thu-coai/Emotional-Support-Conversation"
        "/main/data/valid.txt"
    ),
    "test.txt": (
        "https://raw.githubusercontent.com/thu-coai/Emotional-Support-Conversation"
        "/main/data/test.txt"
    ),
}

# ── ESTES Hugging Face 数据集 API ────────────────────────────────────────────
# ESTES 在 HF Hub 上: thu-coai/ESTES
ESTES_HF_SPLITS = {
    "train": (
        "https://huggingface.co/datasets/thu-coai/ESTES/resolve/main/data/train.json"
    ),
    "validation": (
        "https://huggingface.co/datasets/thu-coai/ESTES/resolve/main/data/validation.json"
    ),
    "test": (
        "https://huggingface.co/datasets/thu-coai/ESTES/resolve/main/data/test.json"
    ),
}

# ── 工具函数 ─────────────────────────────────────────────────────────────────

def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """带进度条的文件下载。"""
    print(f"  下载 {desc or dest.name} ...")
    try:
        resp = requests.get(url, stream=True, timeout=60,
                            headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r    进度: {pct:.1f}%  ({downloaded}/{total} bytes)", end="")
        print()
        print(f"    已保存到: {dest}")
        return True
    except Exception as e:
        print(f"    [错误] 下载失败: {e}")
        return False


def try_hf_datasets(dataset_name: str, save_dir: Path):
    """尝试用 datasets 库从 Hugging Face 下载。"""
    try:
        from datasets import load_dataset
        print(f"  使用 datasets 库加载 {dataset_name} ...")
        ds = load_dataset(dataset_name)
        for split_name, split_data in ds.items():
            out_path = save_dir / f"{split_name}.json"
            split_data.to_json(str(out_path), force_ascii=False)
            print(f"    [{split_name}] {len(split_data)} 条 -> {out_path}")
        return True
    except Exception as e:
        print(f"  datasets 库加载失败: {e}")
        return False


# ── 下载 ESConv ───────────────────────────────────────────────────────────────

def download_esconv():
    print("\n" + "=" * 60)
    print("  下载 ESConv 数据集")
    print("=" * 60)

    # 方案1: 直接下载原始 JSON
    main_file = ESCONV_DIR / "ESConv.json"
    if main_file.exists():
        print(f"  ESConv.json 已存在，跳过下载。")
    else:
        ok = download_file(
            ESCONV_FILES["ESConv.json"],
            main_file,
            "ESConv.json (完整数据)"
        )
        if not ok:
            # 方案2: 用 datasets 库
            print("  尝试通过 Hugging Face datasets 库下载 ...")
            try_hf_datasets("thu-coai/ESConv", ESCONV_DIR)
            return

    # 下载分割文件（train/valid/test）
    for fname, url in list(ESCONV_FILES.items())[1:]:
        dest = ESCONV_DIR / fname
        if dest.exists():
            print(f"  {fname} 已存在，跳过。")
        else:
            ok = download_file(url, dest, fname)
            if not ok:
                # 分割文件不影响主数据，给出提示即可
                print(f"    分割文件 {fname} 下载失败，将在分析时从主文件生成。")

    # 如果 split 文件不存在，从主文件生成
    _generate_esconv_splits()
    print("  ESConv 下载完成！")


def _generate_esconv_splits():
    """从 ESConv.json 生成 train/valid/test 分割（若 txt 文件不存在）。"""
    main_file = ESCONV_DIR / "ESConv.json"
    if not main_file.exists():
        return

    split_files_exist = all(
        (ESCONV_DIR / f).exists() for f in ["train.txt", "valid.txt", "test.txt"]
    )
    if split_files_exist:
        return

    print("  从 ESConv.json 生成 train/valid/test 分割 ...")
    with open(main_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ESConv 官方 split: 约 1040/130/130
    n = len(data)
    train_end = int(n * 0.8)
    valid_end = int(n * 0.9)
    splits = {
        "train": data[:train_end],
        "valid": data[train_end:valid_end],
        "test":  data[valid_end:],
    }
    for split_name, split_data in splits.items():
        out_path = ESCONV_DIR / f"{split_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        print(f"    [{split_name}] {len(split_data)} 条 -> {out_path}")


# ── 下载 ESTES ────────────────────────────────────────────────────────────────

def download_estes():
    print("\n" + "=" * 60)
    print("  下载 ESTES 数据集")
    print("=" * 60)

    # 方案1: datasets 库
    if try_hf_datasets("thu-coai/ESTES", ESTES_DIR):
        print("  ESTES 下载完成！")
        return

    # 方案2: 直接从 HF raw URL 下载
    print("  尝试直接从 HF URL 下载 ...")
    any_ok = False
    for split_name, url in ESTES_HF_SPLITS.items():
        dest = ESTES_DIR / f"{split_name}.json"
        if dest.exists():
            print(f"  {split_name}.json 已存在，跳过。")
            any_ok = True
        else:
            ok = download_file(url, dest, f"ESTES {split_name}")
            any_ok = any_ok or ok

    if any_ok:
        print("  ESTES 下载完成（部分）！")
    else:
        print("  [警告] ESTES 下载失败，请手动下载:")
        print("    https://huggingface.co/datasets/thu-coai/ESTES")
        print("  或运行: pip install datasets && python -c \"from datasets import load_dataset; load_dataset('thu-coai/ESTES')\"")


# ── 验证下载结果 ──────────────────────────────────────────────────────────────

def verify_downloads():
    print("\n" + "=" * 60)
    print("  验证下载结果")
    print("=" * 60)

    def check(path: Path, label: str):
        if path.exists():
            size = path.stat().st_size
            print(f"  [OK] {label}: {path} ({size / 1024:.1f} KB)")
        else:
            print(f"  [MISS] {label}: {path} 不存在")

    # ESConv
    check(ESCONV_DIR / "ESConv.json",  "ESConv 主文件")
    for split in ["train", "valid", "test"]:
        # 兼容 .json 和 .txt 两种格式
        json_f = ESCONV_DIR / f"{split}.json"
        txt_f  = ESCONV_DIR / f"{split}.txt"
        if json_f.exists():
            check(json_f, f"ESConv {split}")
        elif txt_f.exists():
            check(txt_f, f"ESConv {split}")
        else:
            print(f"  [MISS] ESConv {split}: 不存在")

    # ESTES
    for split in ["train", "validation", "test"]:
        check(ESTES_DIR / f"{split}.json", f"ESTES {split}")

    print()


# ── 主入口 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  情感支持对话数据集下载工具")
    print("  ESConv + ESTES")
    print("=" * 60)

    download_esconv()
    download_estes()
    verify_downloads()

    print("下载流程结束。请运行 analyze_datasets.py 进行数据分析。")
