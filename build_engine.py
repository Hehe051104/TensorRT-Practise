# Builder 创建空网络：network = builder.create_network(...) 此时 network 只是一个没有内容的“壳”
# Parser 填充网络：parser.parse(model.read())  Parser 把 ONNX 的结构一点点添加到 network 中，相当于往壳里填内容
# Builder 编译：builder.build_serialized_network(network, config) Builder 基于已经填满的 network 进行优化和编译


import tensorrt as trt
import os

print("=== 边缘重工：TensorRT 物理铸造炉启动 ===")

onnx_file_path = "resnet18_edge_v1.onnx"
engine_file_path = "resnet18_fp16.engine"

# 1. 设置 TRT 记录器
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# 2. 初始化构建器和网络定义
builder = trt.Builder(TRT_LOGGER)
# EXPLICIT_BATCH 是必须的，告诉 TRT 我们自己控制批次大小
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, TRT_LOGGER)  # 指明这个解析器未来要将Onnx结构塞进network

# 3. 解析 ONNX 图纸  把网络准备好，为了第五步编译
print(f"📖 正在解析图纸: {onnx_file_path}")
with open(onnx_file_path, 'rb') as model:
    if not parser.parse(model.read()):
        print("❌ 解析失败！")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        exit(1)

# 4. 配置优化参数（最优解的核心！）
config = builder.create_builder_config()

#告诉 TensorRT 输入张量的形状范围，即使在本例中是固定形状，也需要通过 optimization_profile 显式声明
profile = builder.create_optimization_profile()
# 假设你的输入节点名称是 'input' (可以通过 netron 查看，通常是 'input' 或 'input.1')
# 我们定义 Min, Opt, Max 三个维度，这里我们全部锁定为 (1, 3, 224, 224)
input_name = network.get_input(0).name
profile.set_shape(input_name, (1, 3, 224, 224), (1, 3, 224, 224), (1, 3, 224, 224))
config.add_optimization_profile(profile)

# 设置工作区内存池大小 (分配 2GB 给编译过程使用)
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * 1024 * 1024 * 1024)

# 开启 FP16 半精度极限加速 (如果硬件支持)
if builder.platform_has_fast_fp16:
    config.set_flag(trt.BuilderFlag.FP16)
    print("⚡ FP16 极限加速已开启！")

# 5. 开始物理编译
print("🔥 正在熔炼 TensorRT 专属预制板 (这可能需要几分钟，请耐心等待 4060 寻优)...")
engine_bytes = builder.build_serialized_network(network, config)

# 6. 保存编译结果
with open(engine_file_path, "wb") as f:
    f.write(engine_bytes)

print(f"✅ 铸造完成！边缘核武已保存至: {engine_file_path}")