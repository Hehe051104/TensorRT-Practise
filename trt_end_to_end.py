# pycuda.autoinit 自动创建的是 CUDA 上下文 (CUcontext)，它管理底层 GPU 资源（如设备、内存分配、流等）。

# engine.create_execution_context() 创建的是 TensorRT 执行上下文 (IExecutionContext)，
# 它负责管理一次推理的状态（如中间激活值、绑定信息、临时显存等）。

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import requests
import time

print("\n=== 边缘视觉：TensorRT 10.x 端到端实弹射击 ===")

# ==========================================
# 战前准备：获取字典与目标图像
# ==========================================
print("📡 正在连接 ImageNet 数据库获取标签...")
response = requests.get("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt")
labels = response.text.split("\n")

print("📸 正在锁定测试目标 (高清金毛犬)...")
image_url = "https://images.unsplash.com/photo-1552053831-71594a27632d?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&q=80"
img_data = requests.get(image_url).content
img_array = np.frombuffer(img_data, np.uint8)
img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

# ==========================================
# 阶段一：极其严谨的预处理 (防范隐式类型背刺)
# ==========================================
start_preprocess = time.perf_counter()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (224, 224))
# 强制转换为 float32，避免变 double
img_normalized = img_resized.astype(np.float32) / 255.0

# 显式指定 float32 的均值和标准差
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
img_normalized = (img_normalized - mean) / std

img_transposed = np.transpose(img_normalized, (2, 0, 1))
input_tensor = np.expand_dims(img_transposed, axis=0)

# 【核心防线】必须确保内存是连续排布的 (C-contiguous)，否则 GPU 指针会读取乱码！
h_input = np.ascontiguousarray(input_tensor)

time_preprocess = time.perf_counter() - start_preprocess

# ==========================================
# 阶段二：加载 TensorRT 核武引擎 (v10.x)
# ==========================================
engine_file_path = "resnet18_fp16.engine"
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()
input_name = engine.get_tensor_name(0)
output_name = engine.get_tensor_name(1)

# 显存分配
h_output = np.ascontiguousarray(np.empty((1, 1000), dtype=np.float32))
d_input = cuda.mem_alloc(h_input.nbytes)
d_output = cuda.mem_alloc(h_output.nbytes)
stream = cuda.Stream()

context.set_tensor_address(input_name, int(d_input))
context.set_tensor_address(output_name, int(d_output))

# ==========================================
# 阶段三：雷霆一击 (纯血推理)
# ==========================================
start_infer = time.perf_counter()

# 数据搬运 (Host -> Device)
cuda.memcpy_htod_async(d_input, h_input, stream)
# 点火计算
context.execute_async_v3(stream_handle=stream.handle)
# 结果捞回 (Device -> Host)
cuda.memcpy_dtoh_async(h_output, d_output, stream)
# 同步等待
stream.synchronize()

time_infer = time.perf_counter() - start_infer

# ==========================================
# 阶段四：后处理与结果宣告
# ==========================================
predicted_idx = np.argmax(h_output[0])
scores = np.exp(h_output[0] - np.max(h_output[0]))
probabilities = scores / np.sum(scores)
confidence = probabilities[predicted_idx] * 100

print("\n" + "="*45)
print(f"🎯 最终判定目标物种: 【 {labels[predicted_idx].upper()} 】")
print(f"📊 模型自信程度: {confidence:.2f} %")
print("="*45)

print(f"\n⏱️ 端到端耗时拆解:")
print(f"  - 图像预处理: {time_preprocess*1000:.3f} 毫秒")
print(f"  - TRT 纯推理: {time_infer*1000:.3f} 毫秒 (含数据搬运)")
print("=== 实弹射击任务圆满结束 ===")