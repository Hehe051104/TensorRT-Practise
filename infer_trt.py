import tensorrt as trt  # trt.Runtime 是 TensorRT 的运行时环境。它负责反序列化（deserialize）引擎文件，并能在 GPU 上执行推理
import pycuda.driver as cuda
import pycuda.autoinit  # 自动处理上下文
import numpy as np
import time

print("=== 边缘核武：TensorRT 10.x 极速引擎点火 ===")

engine_file_path = "resnet18_fp16.engine"
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# 1. 加载引擎
with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

# 2. 通过引擎创建上下文  真正的运行环境
context = engine.create_execution_context()

# 3. 获取输入输出节点的名称 (TRT 10.x 强制要求按名称操作)
# 之前的脚本我们锁定了输入输出，通常 ResNet18 的名字是 'input' 和 'output'
input_name = engine.get_tensor_name(0)
output_name = engine.get_tensor_name(1)
print(f"📡 检测到节点名称: {input_name} -> {output_name}")

# 4. 显存分配   要是连续分配的数据  PyCUDA 的 cuda.memcpy_htod_async 函数要求源数据是连续内存块
h_input = np.ascontiguousarray(np.random.randn(1, 3, 224, 224).astype(np.float32))
h_output = np.ascontiguousarray(np.empty((1, 1000), dtype=np.float32))

d_input = cuda.mem_alloc(h_input.nbytes)   # 拿到的是指针
d_output = cuda.mem_alloc(h_output.nbytes)

stream = cuda.Stream()
# 所有操作都使用 同一个 stream，所以 GPU 会严格按照 A → B → C 的顺序执行。
# 如果没有 stream（或使用默认流），顺序仍然按 CPU 发出的顺序在默认流中执行，但默认流会阻塞其他流，且无法实现多流并行

# ==========================================
# 关键修正：TensorRT 10.x 的地址绑定魔法
# ==========================================
# 不再使用 bindings=[int(d_input)...]，而是直接告诉上下文每个名字对应的显存地址
context.set_tensor_address(input_name, int(d_input))
context.set_tensor_address(output_name, int(d_output))

print("🚀 开始极限压榨 (1000 次连续冲锋)...")
start_time = time.perf_counter()

for _ in range(1000):
    # 异步拷贝：内存 -> 显存
    cuda.memcpy_htod_async(d_input, h_input, stream)
    
    # 核心修正：改用 execute_async_v3
    context.execute_async_v3(stream_handle=stream.handle)
    
    # 异步拷贝：显存 -> 内存
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    
    # 等待同步 synchronize() 让 CPU 等待该流上的所有 GPU 工作完成  
    stream.synchronize()

end_time = time.perf_counter()
avg_time_ms = ((end_time - start_time) / 1000) * 1000

print(f"\n🎯 战果统计:")
print(f"⚡ 单张图片平均耗时: {avg_time_ms:.4f} 毫秒")
print(f"🚀 吞吐量 (FPS): {1000 / (end_time - start_time):.0f}")
print("=== 边缘部署巅峰已到达 ===")