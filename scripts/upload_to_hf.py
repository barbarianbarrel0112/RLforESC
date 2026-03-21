import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
from huggingface_hub import HfApi, login

login(token="YOUR_HF_TOKEN_HERE")  # 替换为你的 HuggingFace write token

api = HfApi()

# 创建仓库
api.create_repo("barbarianbarrel0112/qwen25-esc-sft-v2", repo_type="model", exist_ok=True)
print("仓库已创建/确认: barbarianbarrel0112/qwen25-esc-sft-v2")

# 上传最终模型（排除中间 checkpoint 和训练工具文件）
api.upload_folder(
    folder_path="checkpoints/qwen25-esc-v2",
    repo_id="barbarianbarrel0112/qwen25-esc-sft-v2",
    repo_type="model",
    ignore_patterns=[
        "checkpoint-*/**",
        "runs/**",
        "*.pth",
        "zero_to_fp32.py",
        "training_args.bin",
    ],
)
print("上传完成！")
print("模型地址: https://huggingface.co/barbarianbarrel0112/qwen25-esc-sft-v2")
