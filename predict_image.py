import onnxruntime as ort
import numpy as np
import cv2
import requests
import time

print("\n=== 边缘视觉：4060 旗舰 ONNX 识别系统启动 ===")

# ==========================================
# 战前准备：获取模型与知识库
# ==========================================
model_path = "resnet18_edge_v1.onnx"

print("📡 正在连接 ImageNet 数据库，下载中文识别字典...")
# 临时从 GitHub 获取 1000 类别的字典 (这里依然用英文原版，后续可以替换为中文翻译字典)
response = requests.get("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt")
labels = response.text.split("\n")

print("📸 正在锁定测试目标 (一张高清金毛犬的照片)...")
# 这是一张网图，你可以随时换成本地的图片路径，比如 img = cv2.imread('test.jpg')
image_url = "https://images.unsplash.com/photo-1552053831-71594a27632d?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&q=80"
img_data = requests.get(image_url).content
img_array = np.frombuffer(img_data, np.uint8)
img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

# ==========================================
# 阶段一：极其严谨的图像预处理 (核心难点)
# ==========================================
print("🔄 正在进行图像张量化转换...")
start_preprocess = time.perf_counter()

# 1. 色彩空间校正：OpenCV 默认是 BGR，但 PyTorch 模型训练时吃的是 RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 2. 物理尺寸压缩：把大图强行压缩到模型要求的 224x224 像素
img_resized = cv2.resize(img_rgb, (224, 224))

# 3. 归一化：把像素值从 0-255 压到 0-1 的浮点数
img_normalized = img_resized.astype(np.float32) / 255.0

# 4. ImageNet 祖传魔法：减去均值，除以标准差 (必须做，否则准确率暴跌)
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
img_normalized = (img_normalized - mean) / std

# 5. 维度大挪移：
# 图像原本是 (高度 H, 宽度 W, 通道 C) -> (224, 224, 3)
# 模型需要的是 (通道 C, 高度 H, 宽度 W) -> (3, 224, 224)
img_transposed = np.transpose(img_normalized, (2, 0, 1))

# 6. 批量化伪装：模型只能按批次处理，我们强行给单张图片套一层批次外壳
# 变成 (BatchSize N, 通道 C, 高度 H, 宽度 W) -> (1, 3, 224, 224)
input_tensor = np.expand_dims(img_transposed, axis=0)

time_preprocess = time.perf_counter() - start_preprocess

# ==========================================
# 阶段二：唤醒 4060 进行降维打击
# ==========================================
print("🚀 引擎点火，4060 算力接管推理流程...")
# 设置双保险：优先调用 CUDA，挂了就回退 CPU
session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

start_infer = time.perf_counter()
# 这一句，就是几毫秒内发生几百亿次运算的魔法瞬间
outputs = session.run(None, {input_name: input_tensor})
time_infer = time.perf_counter() - start_infer

# ==========================================
# 阶段三：后处理与结果宣告
# ==========================================
# outputs[0] 是一个包含 1000 个小数的数组 (代表 1000 个类别的得分)
# 1. 找出得分最高的那个类别的索引
predicted_idx = np.argmax(outputs[0])

# 2. 计算置信度 (将原始得分通过 Softmax 转换为 0~1 的概率)
# 为了避免手写 Softmax 出错，这里用一个简易且稳妥的实现
scores = np.exp(outputs[0] - np.max(outputs[0]))
probabilities = scores / np.sum(scores)
confidence = probabilities[0][predicted_idx] * 100

print("\n" + "="*40)
print(f"🎯 最终判定目标物种: 【 {labels[predicted_idx].upper()} 】")
print(f"📊 模型自信程度: {confidence:.2f} %")
print("="*40)

print(f"\n⏱️ 耗时报告:")
print(f"  - 预处理耗时: {time_preprocess*1000:.2f} 毫秒")
print(f"  - 纯推理耗时: {time_infer*1000:.2f} 毫秒")
print("=== 任务圆满结束 ===")