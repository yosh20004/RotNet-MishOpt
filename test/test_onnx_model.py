#!/usr/bin/env python3
"""
测试 RotNet ONNX 模型的可用性与准确性。
测试过程模仿 PyTorch 脚本，执行四种角度的旋转预测。
"""

import onnxruntime
import numpy as np
import os
from PIL import Image, ImageOps
import torch
from torchvision import transforms
import time

# --- 配置参数 ---
ONNX_MODEL_PATH = './models/rotnet_resnet18_mish.onnx'
TEST_IMAGE_DIR = './data'
IMAGE_SIZE = 224
ANGLES = [0, 90, 180, 270]

# --- 预处理（与训练/预测保持一致）---
preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --- 加载 ONNX 模型 ---
try:
    session = onnxruntime.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"✅ 成功加载 ONNX 模型: {ONNX_MODEL_PATH}")
except Exception as e:
    print(f"❌ 加载 ONNX 模型失败: {e}")
    exit()

# --- 加载测试图像 ---
if not os.path.isdir(TEST_IMAGE_DIR):
    print(f"❌ 找不到测试图像目录: {TEST_IMAGE_DIR}")
    exit()

image_files = [f for f in os.listdir(TEST_IMAGE_DIR) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
if not image_files:
    print(f"⚠️ 测试目录中没有图像: {TEST_IMAGE_DIR}")
    exit()

# --- 开始测试 ---
total = 0
correct = 0
angle_stats = {angle: [0, 0] for angle in ANGLES}  # {角度: [正确数，总数]}
sum_time = 0.0

for image_name in sorted(image_files):
    img_path = os.path.join(TEST_IMAGE_DIR, image_name)
    try:
        original_image = Image.open(img_path).convert("RGB")
        original_image = ImageOps.exif_transpose(original_image)

        for true_angle in ANGLES:
            total += 1
            angle_stats[true_angle][1] += 1

            # 1. 图像旋转 + 预处理
            rotated_img = original_image.rotate(true_angle, expand=True)
            tensor_img = preprocess(rotated_img).unsqueeze(0)  # shape: (1, 3, 224, 224)
            np_img = tensor_img.numpy().astype(np.float32)

            # 2. 推理
            start_time = time.time()
            outputs = session.run([output_name], {input_name: np_img})
            end_time = time.time()
            predicted_idx = np.argmax(outputs[0], axis=1)[0]
            predicted_angle = ANGLES[predicted_idx]

            # 3. 结果判断
            is_correct = (predicted_angle == true_angle)
            angle_stats[true_angle][0] += int(is_correct)
            correct += int(is_correct)

            result_str = "✅" if is_correct else "❌"
            
            sum_time += (end_time - start_time) 
            print(f"{result_str} {image_name} rotated {true_angle}° → predicted {predicted_angle}°, time = {(end_time - start_time) * 1000} ms")

    except Exception as e:
        print(f"⚠️ 无法处理图像 {image_name}: {e}")

# --- 打印统计结果 ---
accuracy = (correct / total) * 100 if total > 0 else 0
print(f"\n--- ONNX 模型测试完成 ---")
print(f"总测试数: {total}")
print(f"正确预测数: {correct}")
print(f"整体准确率: {accuracy:.2f}%")
print(f"平均耗时: {sum_time * 1000 / total} ms")

print("\n--- 各角度准确率 ---")
for angle in ANGLES:
    correct_angle, total_angle = angle_stats[angle]
    acc = (correct_angle / total_angle) * 100 if total_angle > 0 else 0
    print(f"角度 {angle:<3}°: {correct_angle}/{total_angle} 正确 → {acc:.2f}%")
