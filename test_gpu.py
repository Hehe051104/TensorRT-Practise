import torch

print("=== 边缘实验室 2.0 终极探针 ===")
print(f"PyTorch 版本: {torch.__version__}")
print(f"GPU 是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"成功捕获猛兽: {torch.cuda.get_device_name(0)}")
else:
    print("翻车了，显卡没连上。")