#!/usr/bin/env python3
import torch
import torch.nn as nn
from torchvision import models
import os

# --- 从torch.onnx命名空间导入关键函数 ---
from torch.onnx import register_custom_op_symbolic

# ==============================================================================
# 1. 配置区域
# ==============================================================================
MODEL_PTH_PATH = "./models/rotnet_resnet18_mish_best.pth"
ONNX_EXPORT_PATH = "./models/rotnet_resnet18_with_custom_op.onnx"
IMAGE_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs("./models", exist_ok=True)

# ==============================================================================
# 2. 【核心】定义符号函数并注册
# ==============================================================================
def mish_symbolic(g, input, **kwargs):
    """
    定义当 ONNX 导出器遇到 'aten::mish' 操作时应该做什么。
    g: ONNX 计算图对象。
    input: 输入张量。
    我们告诉它：不要分解这个操作，而是直接在图中插入一个 'MyMish' 节点。
    """
    # 语法: g.op('domain::OperatorName', inputs...)
    # 域名和算子名必须与你的 C++ 实现完全一致。
    return g.op('com.mydomain::MyMish', input)

# 将我们定义的符号函数注册到 ONNX 导出器中。
# 'aten::mish' 是 PyTorch 中 nn.Mish 对应的底层 ATen 操作名。
# 9 是这个注册生效的起始 opset 版本。
#
# 这行代码是整个脚本的“魔法”所在。
register_custom_op_symbolic('aten::mish', mish_symbolic, 9)

# ==============================================================================
# 3. 新增辅助函数：替换激活层
# ==============================================================================
def replace_activations(module, old_activation, new_activation):
    """
    递归地遍历模型的所有子模块，并将指定的旧激活层替换为新激活层。
    """
    for name, child in module.named_children():
        if isinstance(child, old_activation):
            setattr(module, name, new_activation())
        else:
            replace_activations(child, old_activation, new_activation)

# ==============================================================================
# 4. 定义与训练时完全一致的模型结构
# ==============================================================================
class RotationPredictionModel(nn.Module):
    """
    这个模型定义应该与你用来训练和保存 .pth 文件的模型完全一样。
    注意：我们在这里直接使用标准的 nn.Mish。
    """
    def __init__(self, num_classes=4):
        super(RotationPredictionModel, self).__init__()
        self.encoder = models.resnet18(weights='IMAGENET1K_V1')
        num_features = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.Mish(),  # <-- 在这里使用标准的 nn.Mish，和训练时保持一致！
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.encoder(x)
        outputs = self.classifier(features)
        return outputs

# ==============================================================================
# 5. 执行加载和导出
# ==============================================================================

# 实例化模型
model = RotationPredictionModel(num_classes=4)

# 【最终修正】在加载权重前，将模型中所有的 ReLU 替换为 Mish
# 这是为了确保整个网络都使用我们想要优化的激活函数。
print("正在将模型中的所有 nn.ReLU 替换为 nn.Mish...")
replace_activations(model, nn.ReLU, nn.Mish)
print("替换完成！")

model.to(DEVICE)

# 加载你训练好的权重。
# 警告：此操作要求 .pth 文件本身也是在一个将 ReLU 全部替换为 Mish 的模型上训练得出的。
# 如果加载为 ReLU 训练的权重，模型行为将不正确。
if os.path.exists(MODEL_PTH_PATH):
    print(f"正在从 {MODEL_PTH_PATH} 加载权重...")
    model.load_state_dict(torch.load(MODEL_PTH_PATH, map_location=DEVICE))
else:
    print(f"警告：找不到权重文件 {MODEL_PTH_PATH}。将使用随机初始化的模型进行导出。")

model.eval()

# 创建一个符合模型输入的虚拟张量
dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)

print("开始导出模型到 ONNX (使用符号注册法)...")

try:
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_EXPORT_PATH,
        export_params=True,
        opset_version=14,  # <-- 推荐使用较新的 opset
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print(f"✅ 成功导出为 ONNX 文件：{ONNX_EXPORT_PATH}")
    print("   现在这个模型中的所有 Mish 节点都应该被正确地替换为了 'MyMish'。")
    print("   请使用 Netron 可视化工具进行最终确认！")

except Exception as e:
    print(f"❌ 导出失败：{e}")

