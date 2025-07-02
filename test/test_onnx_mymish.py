#!/usr/bin/env python3
"""
测试带自定义 Mish 激活函数的 RotNet ONNX 模型。
执行四种角度旋转分类任务。
"""

import onnxruntime as ort
import numpy as np
import os
import ctypes
from PIL import Image, ImageOps
import torch
from torchvision import transforms
import time

# --- 配置路径 ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

onnx_model_path = os.path.join(project_root, "models", "rotnet_resnet18_with_custom_op.onnx")
custom_op_library_path = os.path.join(project_root, "opti", "libmymish_avx2_omp.so")
ort_lib_path = os.path.join(project_root, "opti", "onnxruntime-linux-x64-1.22.0", "lib", "libonnxruntime.so")
test_image_dir = os.path.join(project_root, "data")

# --- 参数 ---
IMAGE_SIZE = 224
ANGLES = [0, 90, 180, 270]

# --- 预处理（与训练保持一致）---
preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --- 加载 ONNX Runtime 核心库 ---
try:
    ctypes.CDLL(ort_lib_path)
    print(f"✅ 成功加载 ONNX Runtime: {ort_lib_path}")
except Exception as e:
    print(f"❌ 加载 ONNX Runtime 库失败: {e}")
    exit()

# --- 加载自定义算子库并创建会话 ---
if not os.path.exists(onnx_model_path) or not os.path.exists(custom_op_library_path):
    print("❌ 模型或自定义算子库路径无效")
    exit()

sess_options = ort.SessionOptions()
try:
    sess_options.register_custom_ops_library(custom_op_library_path)
    print(f"✅ 注册自定义算子: {custom_op_library_path}")
except Exception as e:
    print(f"❌ 注册失败: {e}")
    exit()

try:
    session = ort.InferenceSession(onnx_model_path, sess_options, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"✅ 成功加载模型: {onnx_model_path}")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    exit()

# --- 加载测试图像 ---
if not os.path.isdir(test_image_dir):
    print(f"❌ 找不到测试图像目录: {test_image_dir}")
    exit()

image_files = [f for f in os.listdir(test_image_dir) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
if not image_files:
    print(f"⚠️ 测试目录中没有图像")
    exit()

# --- 执行推理测试 ---
total = 0
correct = 0
angle_stats = {angle: [0, 0] for angle in ANGLES}
sum_time = 0.0

for image_name in sorted(image_files):
    img_path = os.path.join(test_image_dir, image_name)
    try:
        original = Image.open(img_path).convert("RGB")
        original = ImageOps.exif_transpose(original)

        for true_angle in ANGLES:
            total += 1
            angle_stats[true_angle][1] += 1

            rotated = original.rotate(true_angle, expand=True)
            input_tensor = preprocess(rotated).unsqueeze(0).numpy().astype(np.float32)

            start_time = time.time()
            output = session.run([output_name], {input_name: input_tensor})
            end_time = time.time()
            
            pred_idx = np.argmax(output[0], axis=1)[0]
            pred_angle = ANGLES[pred_idx]

            correct_flag = (pred_angle == true_angle)
            angle_stats[true_angle][0] += int(correct_flag)
            correct += int(correct_flag)

            status = "✅" if correct_flag else "❌"
            
            sum_time += (end_time - start_time)
            print(f"{status} {image_name} rotated {true_angle}° → predicted {pred_angle}°, time = {(end_time - start_time) * 1000} ms")

    except Exception as e:
        print(f"⚠️ 跳过图像 {image_name}: {e}")

# --- 输出统计 ---
acc = (correct / total) * 100 if total > 0 else 0
print(f"\n--- 自定义算子模型测试完成 ---")
print(f"总图像数: {total}, 正确预测: {correct}, 准确率: {acc:.2f}%, 平均耗时: {sum_time * 1000 / total} ms")

print("\n--- 每个角度准确率 ---")
for angle in ANGLES:
    correct_angle, total_angle = angle_stats[angle]
    angle_acc = (correct_angle / total_angle) * 100 if total_angle > 0 else 0
    print(f"角度 {angle:<3}°: {correct_angle}/{total_angle} → {angle_acc:.2f}%")
