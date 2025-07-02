import onnx
import onnx.helper as helper
from onnx import TensorProto
import onnxruntime as ort
import numpy as np
import time
import os
import sys
import ctypes

INPUT_SIZE = [1, 2, 128, 512]

def build_single_op_model(op_type, output_path, **kwargs):
    """构建一个包含多个相同操作的ONNX模型，增加计算复杂度"""
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, INPUT_SIZE)  # 增大输入尺寸
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, INPUT_SIZE)

    # 创建多个相同的操作节点来增加计算量
    nodes = []
    current_input = 'X'
    for i in range(5):  # 连续5个相同操作
        output_name = f'intermediate_{i}' if i < 4 else 'Y'
        node = helper.make_node(op_type, [current_input], [output_name], **kwargs)
        nodes.append(node)
        current_input = output_name

    graph_def = helper.make_graph(nodes, 'multi-op-graph', [X], [Y])

    # 指定与ONNX Runtime兼容的opset版本
    opset_imports = [helper.make_opsetid("", 22)]

    # 指定与ONNX Runtime兼容的IR版本
    model_def = helper.make_model(graph_def, producer_name='onnx-test', ir_version=10, opset_imports=opset_imports)
    onnx.save(model_def, output_path)

def build_decomposed_mish_model(output_path):
    """构建一个用基础算子分解实现的Mish模型，增加计算复杂度"""
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, INPUT_SIZE)  # 增大输入尺寸
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, INPUT_SIZE)

    # 创建多个Mish操作来增加计算量
    nodes = []
    initializers = []
    current_input = 'X'

    for i in range(5):  # 连续5个Mish操作
        # Mish(x) = x * tanh(ln(1 + exp(x)))
        exp_out = f'exp_out_{i}'
        add_out = f'add_out_{i}'
        log_out = f'log_out_{i}'
        tanh_out = f'tanh_out_{i}'
        final_out = f'mish_out_{i}' if i < 4 else 'Y'

        one_const_name = f'one_const_{i}'
        one_const = helper.make_tensor(one_const_name, TensorProto.FLOAT, [], [1.0])
        initializers.append(one_const)

        nodes.extend([
            helper.make_node('Exp', [current_input], [exp_out]),
            helper.make_node('Add', [exp_out, one_const_name], [add_out]),
            helper.make_node('Log', [add_out], [log_out]),
            helper.make_node('Tanh', [log_out], [tanh_out]),
            helper.make_node('Mul', [current_input, tanh_out], [final_out])
        ])

        current_input = final_out

    graph_def = helper.make_graph(
        nodes,
        'multi-decomposed-mish-graph',
        [X],
        [Y],
        initializers
    )

    opset_imports = [helper.make_opsetid("", 22)]
    model_def = helper.make_model(graph_def, producer_name='onnx-test', ir_version=10, opset_imports=opset_imports)
    onnx.save(model_def, output_path)


def benchmark(session, input_data, num_runs=500):
    """改进的基准测试函数，增加预热和测试轮数"""
    # 增加预热轮数，确保缓存和优化生效
    print(f"      Warming up with 50 runs...")
    for _ in range(50):
        session.run(None, {'X': input_data})

    # 多次测量取最稳定的结果
    times = []
    for round_idx in range(15):  # 进行15轮测试
        start_time = time.perf_counter()
        for _ in range(num_runs):
            session.run(None, {'X': input_data})
        end_time = time.perf_counter()

        round_time_ms = (end_time - start_time) * 1000 / num_runs
        times.append(round_time_ms)
        print(f"      Round {round_idx + 1}: {round_time_ms:.6f} ms")

    # 返回中位数时间，更稳定
    times.sort()
    median_time_ms = times[2]  # 5次测试的中位数
    print(f"      Median time: {median_time_ms:.6f} ms")
    return median_time_ms

if __name__ == "__main__":
    # --- 关键修正: 解决 libonnxruntime.so 的加载问题 ---
    # 使用 ctypes 手动预加载 libonnxruntime.so。
    # 这会将其加载到进程的地址空间，从而让动态链接器在加载 libmymish.so 时能找到它。
    # 这是比修改 LD_LIBRARY_PATH 更健壮和可移植的方法。
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # 注意：在Linux上，实际的库文件通常是 libonnxruntime.so.1.22.0，
        # 而 libonnxruntime.so 是一个指向它的符号链接。我们直接加载.so文件即可。
        ort_lib_path = os.path.join(script_dir, "onnxruntime-linux-x64-1.22.0", "lib", "libonnxruntime.so")
        if os.path.exists(ort_lib_path):
             ctypes.CDLL(ort_lib_path)
             print(f"[*] Successfully pre-loaded ONNX Runtime library.")
        else:
             print(f"[!] Warning: ONNX Runtime library not found at '{ort_lib_path}'. MyMish test might fail.")
    except Exception as e:
        print(f"[!] FAILED to pre-load ONNX Runtime library: {e}")


    print("\n1. Generating ONNX models...")
    build_single_op_model('Relu', 'relu_model.onnx')
    build_decomposed_mish_model('onnx_mish_model.onnx')
    build_single_op_model('MyMish', 'mymish_model.onnx', domain='com.mydomain')
    print("   Models generated successfully.")

    # 增大输入数据尺寸以匹配新的模型
    input_data = np.random.randn(*INPUT_SIZE).astype(np.float32)
    print(f"   Input data shape: {input_data.shape}, size: {input_data.size:,} elements")
    
    print("\n2. Running benchmarks...")
    
    time_onnx_mish = 0.0
    time_mymish = 0.0

    try:
        sess_relu = ort.InferenceSession('relu_model.onnx')
        time_relu = benchmark(sess_relu, input_data)
        print(f"   - ONNX Relu (ideal): {time_relu:.6f} ms")
    except Exception as e:
        print(f"   - FAILED to test Relu: {e}")

    try:
        sess_onnx_mish = ort.InferenceSession('onnx_mish_model.onnx')
        time_onnx_mish = benchmark(sess_onnx_mish, input_data)
        print(f"   - ONNX Mish (Decomposed): {time_onnx_mish:.6f} ms")
    except Exception as e:
        print(f"   - FAILED to test Decomposed Mish: {e}")
    
    try:
        so = ort.SessionOptions()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        so_path = os.path.join(script_dir, "libmymish_avx2.so")
        
        if not os.path.exists(so_path):
            print(f"\n[ERROR] Custom op library not found at: {so_path}")
            raise FileNotFoundError(f"Custom op library not found at: {so_path}")
            
        so.register_custom_ops_library(so_path) 
        
        sess_mymish = ort.InferenceSession('mymish_model.onnx', sess_options=so)
        time_mymish = benchmark(sess_mymish, input_data)
        print(f"   - MyMish (Fused & C++ Optimized): {time_mymish:.6f} ms")
        
    except Exception as e:
        print(f"   - FAILED to test MyMish: {e}")

    # --- 打印结论 ---
    print("\n--- Benchmark Summary ---")
    if time_onnx_mish > 0 and time_mymish > 0:
        print(f"MyMish is {time_onnx_mish / time_mymish:.2f}x faster than the decomposed ONNX Mish.")
    else:
        print("Could not perform speed comparison due to errors in previous steps.")
