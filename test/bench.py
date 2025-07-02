#!/usr/bin/env python3
"""
ONNX 推理 + Profiling + 性能分析 一体化脚本

步骤：
1. 加载模型并进行推理（含旋转）
2. 启用 ONNX Runtime profiler，生成 JSON 分析文件
3. 自动解析该文件并打印算子与节点耗时排名
"""

import os
import json
import time
from collections import defaultdict
from PIL import Image, ImageOps
import torch
from torchvision import transforms
import onnxruntime
import numpy as np

# --- 配置 ---
ONNX_MODEL_PATH = './models/rotnet_resnet18_mish.onnx'
TEST_IMAGE_DIR = './data'
IMAGE_SIZE = 224
ANGLES = [0, 90, 180, 270]
TOPK_OPS = 10
TOPK_NODES = 10

# --- 图像预处理 ---
preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --- Step 1: 加载 ONNX 模型，开启 profiler ---
sess_options = onnxruntime.SessionOptions()
sess_options.enable_profiling = True
sess_options.optimized_model_filepath = "optimized_model_origin.onnx"

try:
    session = onnxruntime.InferenceSession(
        ONNX_MODEL_PATH,
        sess_options=sess_options,
        providers=['CPUExecutionProvider']
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"✅ 成功加载模型: {ONNX_MODEL_PATH}")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    exit()

# --- Step 2: 推理 ---
if not os.path.isdir(TEST_IMAGE_DIR):
    print(f"❌ 找不到测试图像目录: {TEST_IMAGE_DIR}")
    exit()

image_files = [f for f in os.listdir(TEST_IMAGE_DIR) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
if not image_files:
    print(f"⚠️ 测试目录中没有图像: {TEST_IMAGE_DIR}")
    exit()

print(f"🚀 开始推理，总图像数: {len(image_files)} × {len(ANGLES)}")

for image_name in sorted(image_files):
    img_path = os.path.join(TEST_IMAGE_DIR, image_name)
    try:
        original_image = Image.open(img_path).convert("RGB")
        original_image = ImageOps.exif_transpose(original_image)

        for angle in ANGLES:
            rotated = original_image.rotate(angle, expand=True)
            tensor_img = preprocess(rotated).unsqueeze(0)
            np_img = tensor_img.numpy().astype(np.float32)
            _ = session.run([output_name], {input_name: np_img})
    except Exception as e:
        print(f"⚠️ 处理图像失败 {image_name}: {e}")

# --- Step 3: 获取 profiler 输出文件 ---
profile_path = session.end_profiling()
print(f"📁 Profiler 输出文件: {profile_path}")

# --- Step 4: 分析 profiler 文件 ---
def analyze_profile(json_path, topk_ops=10, topk_nodes=10):
    with open(json_path, 'r') as f:
        data = json.load(f)

    op_time = defaultdict(float)
    node_time = []

    for entry in data:
        if entry.get("cat") == "Node":
            op_type = entry["args"].get("op_name", "Unknown")
            dur = entry.get("dur", 0)
            node_name = entry.get("name", "Unnamed")
            provider = entry["args"].get("provider", "Unknown")

            op_time[op_type] += dur
            node_time.append((dur, node_name, op_type, provider))

    print(f"\n🧠 Top {topk_ops} 最耗时算子类型:")
    for op, dur_us in sorted(op_time.items(), key=lambda x: x[1], reverse=True)[:topk_ops]:
        print(f"{op:<20} {dur_us / 1000:.3f} ms")

    print(f"\n⚙️ Top {topk_nodes} 最耗时单个节点:")
    for dur, name, op, provider in sorted(node_time, reverse=True)[:topk_nodes]:
        print(f"{name:<30} {op:<15} {dur / 1000:.3f} ms  [{provider}]")

# --- 执行分析 ---
analyze_profile(profile_path, TOPK_OPS, TOPK_NODES)
