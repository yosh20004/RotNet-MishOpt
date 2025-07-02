#!/usr/bin/env python3
"""
对包含自定义 MyMish 算子的 ONNX 模型进行推理、性能评测和分析。

功能:
1. 加载包含自定义算子 (.so) 的 ONNX 模型。
2. 对指定目录的图像进行旋转推理。
3. 启用 ONNX Runtime profiler，记录每个算子和节点的耗时。
4. 自动解析性能数据并打印出最耗时的算子和节点排名。
"""

import os
import json
import ctypes
from collections import defaultdict
from PIL import Image, ImageOps
import torch
from torchvision import transforms
import onnxruntime as ort
import numpy as np

# ==============================================================================
# 1. 配置区域
# ==============================================================================

# --- 核心路径设置 ---
# 请确保这些路径指向您的文件
ONNX_MODEL_PATH = "./models/rotnet_resnet18_with_custom_op.onnx"
CUSTOM_OP_LIBRARY_PATH = "./opti/libmymish_avx2_omp.so"
ORT_LIBRARY_PATH = "./opti/onnxruntime-linux-x64-1.22.0/lib/libonnxruntime.so"
TEST_IMAGE_DIR = './data'

# --- 性能分析参数 ---
TOPK_OPS = 15      # 报告中显示的耗时最长的算子类型数量
TOPK_NODES = 15    # 报告中显示的耗时最长的节点数量

# --- 模型推理参数 ---
IMAGE_SIZE = 224
ANGLES = [0, 90, 180, 270]

# ==============================================================================
# 2. 预处理和辅助函数
# ==============================================================================

# 图像预处理 (与训练和标准测试保持一致)
preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def analyze_profile(json_path, topk_ops, topk_nodes):
    """解析 ONNX Runtime 生成的 JSON 性能分析文件。"""
    print("\n" + "="*80)
    print("PERFORMANCE ANALYSIS REPORT")
    print("="*80)
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ 无法读取或解析性能分析文件: {json_path}\nError: {e}")
        return

    op_time = defaultdict(float)
    node_time = []

    for entry in data:
        if entry.get("cat") == "Node" and "dur" in entry:
            op_type = entry["args"].get("op_name", "UnknownOp")
            dur_us = entry["dur"]
            node_name = entry.get("name", "UnnamedNode")
            provider = entry["args"].get("provider", "UnknownProvider")

            op_time[op_type] += dur_us
            node_time.append((dur_us, node_name, op_type, provider))

    total_time_ms = sum(op_time.values()) / 1000.0
    print(f"模型CPU总执行时间: {total_time_ms:.3f} ms\n")

    print(f"🧠 Top {topk_ops} 最耗时的算子类型 (按类型总耗时排名):")
    # Sort by duration, descending
    sorted_ops = sorted(op_time.items(), key=lambda x: x[1], reverse=True)
    for op, dur_us in sorted_ops[:topk_ops]:
        percentage = (dur_us / 1000.0 / total_time_ms) * 100 if total_time_ms > 0 else 0
        print(f"  - {op:<20} {dur_us / 1000:.3f} ms ({percentage:.2f}%)")

    print(f"\n⚙️ Top {topk_nodes} 最耗时的单个节点 (按节点单次耗时排名):")
    # Sort by duration, descending
    sorted_nodes = sorted(node_time, reverse=True)
    for dur_us, name, op, provider in sorted_nodes[:topk_nodes]:
        print(f"  - {name:<35} {op:<15} {dur_us / 1000:.3f} ms  [{provider}]")
    print("="*80)

# ==============================================================================
# 3. 主执行流程
# ==============================================================================

if __name__ == "__main__":
    # --- 步骤 1: 检查文件并加载核心库 ---
    for path in [ONNX_MODEL_PATH, CUSTOM_OP_LIBRARY_PATH, ORT_LIBRARY_PATH, TEST_IMAGE_DIR]:
        if not os.path.exists(path):
            print(f"❌ 关键文件或目录不存在: {path}")
            exit()
    try:
        ctypes.CDLL(ORT_LIBRARY_PATH)
        print(f"✅ 成功加载 ONNX Runtime 核心库: {ORT_LIBRARY_PATH}")
    except Exception as e:
        print(f"❌ 加载 ONNX Runtime 库失败: {e}")
        exit()

    # --- 步骤 2: 创建会话，注册自定义算子，并开启性能分析 ---
    sess_options = ort.SessionOptions()
    sess_options.enable_profiling = True  # <-- 开启性能分析
    sess_options.optimized_model_filepath = "optimized_model.onnx"

    try:
        sess_options.register_custom_ops_library(CUSTOM_OP_LIBRARY_PATH)
        print(f"✅ 成功注册自定义算子库: {CUSTOM_OP_LIBRARY_PATH}")
        
        session = ort.InferenceSession(
            ONNX_MODEL_PATH,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        print(f"✅ 成功加载模型并创建评测会话: {ONNX_MODEL_PATH}")
    except Exception as e:
        print(f"❌ 创建评测会话失败: {e}")
        exit()

    # --- 步骤 3: 执行推理以收集性能数据 ---
    image_files = [f for f in os.listdir(TEST_IMAGE_DIR) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
    if not image_files:
        print(f"⚠️ 测试目录中没有图像: {TEST_IMAGE_DIR}")
        exit()

    print(f"\n🚀 开始推理，收集性能数据 (总图像数: {len(image_files)} × {len(ANGLES)})...")
    for image_name in sorted(image_files):
        img_path = os.path.join(TEST_IMAGE_DIR, image_name)
        try:
            original_image = Image.open(img_path).convert("RGB")
            original_image = ImageOps.exif_transpose(original_image)
            for angle in ANGLES:
                rotated_img = original_image.rotate(angle, expand=True)
                input_tensor = preprocess(rotated_img).unsqueeze(0).numpy().astype(np.float32)
                _ = session.run([output_name], {input_name: input_tensor})
        except Exception as e:
            print(f"⚠️ 处理图像失败 {image_name}: {e}")
    print("...推理完成。")

    # --- 步骤 4: 结束评测并生成报告 ---
    profile_file_path = session.end_profiling()
    print(f"\n📁 性能分析文件已生成: {profile_file_path}")
    
    analyze_profile(profile_file_path, TOPK_OPS, TOPK_NODES)

