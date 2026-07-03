import onnxruntime as ort
import numpy as np
import time

print("=== 边缘实验室：4060 ONNX 推理引擎启动 ===")
onnx_model_path = "resnet18_edge_v1.onnx"

# 1. 召唤硬件提供商 (Execution Providers)
# 极其重要：把 CUDA 放前面，如果没配好显卡，它会按顺序降级去用 CPU
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession(onnx_model_path, providers=providers)

# 严谨性校验：看看到底有没有成功挂载 4060
actual_provider = session.get_providers()[0]
print(f"当前接管算力的硬件引擎: {actual_provider}")
if actual_provider != 'CUDAExecutionProvider':
    print("⚠️ 警告：4060 猛兽未唤醒，正在使用 CPU 龟速推车！")

# 2. 构造测试弹药
# ONNX Runtime 只认纯净的 NumPy 数组，不需要 PyTorch 的 Tensor
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# 伪造一张 224x224 的 RGB 图片数据
dummy_image = np.random.randn(1, 3, 224, 224).astype(np.float32)

print("\n=== 正在预热发动机 (Warm-up) ===")
# 显卡刚启动时有延迟，先跑几遍空跑，让显存热起来
for _ in range(10):
    session.run([output_name], {input_name: dummy_image})

print("=== 开始极限性能测试 (连续轰击 100 次) ===")
start_time = time.time()

for _ in range(100):
    # 这就是核心推理代码，极其精简
    result = session.run([output_name], {input_name: dummy_image})

end_time = time.time()

# 3. 计算战报
avg_time = (end_time - start_time) / 100 * 1000
fps = 1000 / avg_time
print(f"\n🎯 测试完成！")
print(f"⚡ 单张图片平均耗时: {avg_time:.2f} 毫秒")
print(f"🚀 吞吐量 (FPS): 每秒可处理 {fps:.0f} 张图片")