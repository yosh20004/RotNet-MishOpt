#!/usr/bin/env python3
import torch
import torch.nn as nn
from torchvision import models
import os

# --- 参数设置 ---
MODEL_PTH_PATH = "./models/rotnet_resnet18_mish_best.pth"
ONNX_EXPORT_PATH = "./models/rotnet_resnet18_mish.onnx"
IMAGE_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 定义模型结构 (与你训练时一致) ---
class RotationPredictionModel(nn.Module):
    def __init__(self, num_classes=4):
        super(RotationPredictionModel, self).__init__()
        self.encoder = models.resnet18(weights='IMAGENET1K_V1')
        num_features = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),  # ReLU will be replaced by Mish
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.encoder(x)
        outputs = self.classifier(features)
        return outputs

def replace_relu_with_mish(model):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_relu_with_mish(module)
        if isinstance(module, nn.ReLU):
            setattr(model, name, nn.Mish())

# --- 创建模型并加载权重 ---
model = RotationPredictionModel(num_classes=4).to(DEVICE)
replace_relu_with_mish(model)
model.load_state_dict(torch.load(MODEL_PTH_PATH, map_location=DEVICE))
model.eval()

# --- 创建 dummy 输入 ---
dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)

# --- 导出为 ONNX ---
torch.onnx.export(
    model,
    dummy_input,
    ONNX_EXPORT_PATH,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)

print(f"✅ 成功导出为 ONNX 文件：{ONNX_EXPORT_PATH}")
