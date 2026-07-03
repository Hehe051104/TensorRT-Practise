# netron resnet18_edge_v1.onnx --host 0.0.0.0 --port 8080      看模型架构图



import torch
import torchvision.models as models
import onnx

print("=== 边缘实验室：ResNet18 导出任务启动 ===")

# 1. 准备模型：使用 ResNet18 作为基础
# 严谨性提醒：必须调用 .eval()，否则权重会处于随机的训练状态
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()

# 2. 准备 4060 的张量
# 虽然导出可以在 CPU 上做，但我们在 GPU 环境里，直接用 CUDA 加速
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 3. 定义输入：边缘计算通常处理 224x224 的图像
dummy_input = torch.randn(1, 3, 224, 224).to(device)

# 4. 执行导出
onnx_filename = "resnet18_edge_v1.onnx"
print(f"正在将模型‘降维打击’为 {onnx_filename}...")

torch.onnx.export(
    model,
    dummy_input,               # 提供输入的形状和类型信息，执行一次前向传播来构建计算图  模型是"代码"（动态图，运行的时候才知道执行哪条语句）不是"静态图"（），
    onnx_filename,
    export_params=True,        # 打包所有权重参数
    opset_version=14,          # 算子集版本 14，边缘侧最稳健的选择
    do_constant_folding=True,  # 优化开关：折叠常量算子，减小体积  预先计算所有不依赖输入数据的运算
    input_names=['input'],     # 规范输入名称   便于多输入模型区分（如 ['image', 'mask']）
    output_names=['output'],   # 规范输出名称  多输出示例：output_names=['detection_boxes', 'detection_scores', 'num_detections']
    dynamic_axes={             # 前瞻性：允许以后输入不同大小的 Batch
        'input': {0: 'batch_size'},  # 输入的batch维度可变，命名为batch_size [1, 3, 224, 224] [8, 3, 224, 224]
        'output': {0: 'batch_size'}
    }
)

# 5. 严谨性校验：确保导出的模型没有逻辑断层
print("正在检查模型结构完整性...")
onnx_model = onnx.load(onnx_filename)
onnx.checker.check_model(onnx_model)

print(f"恭喜指挥官！{onnx_filename} 已就绪。")
print(f"当前文件归属：{torch.hub._get_torch_home()}") # 验证权限