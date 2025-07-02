import onnx
import onnx.helper as helper
from onnx import TensorProto
import onnxruntime as ort
import numpy as np
import time
import os
import sys
import ctypes

def build_single_op_model(op_type, output_path, **kwargs):
    """构建一个只包含单个操作的ONNX模型"""
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 3, 224, 224])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 3, 224, 224])
    
    node_def = helper.make_node(op_type, ['X'], ['Y'], **kwargs)
    graph_def = helper.make_graph([node_def], 'single-op-graph', [X], [Y])
    
    opset_imports = [helper.make_opsetid("", 14)]
    
    if 'domain' in kwargs and kwargs['domain']:
        opset_imports.append(helper.make_opsetid(kwargs['domain'], 1))
        
    model_def = helper.make_model(graph_def, producer_name='onnx-benchmark-test', ir_version=10, opset_imports=opset_imports)
    onnx.checker.check_model(model_def)
    onnx.save(model_def, output_path)

def build_debug_mish_model(output_path):
    """构建一个分解式的Mish模型，并暴露所有中间张量作为输出以供调试。"""
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 3, 224, 224])
    
    # 定义所有中间和最终的输出张量信息
    Y_exp = helper.make_tensor_value_info('Y_exp', TensorProto.FLOAT, [1, 3, 224, 224])
    Y_add = helper.make_tensor_value_info('Y_add', TensorProto.FLOAT, [1, 3, 224, 224])
    Y_log = helper.make_tensor_value_info('Y_log', TensorProto.FLOAT, [1, 3, 224, 224])
    Y_tanh = helper.make_tensor_value_info('Y_tanh', TensorProto.FLOAT, [1, 3, 224, 224])
    Y_final = helper.make_tensor_value_info('Y_final', TensorProto.FLOAT, [1, 3, 224, 224])

    # 定义计算节点
    node_exp = helper.make_node('Exp', ['X'], ['Y_exp'])
    one_const_input_name = 'one_const_input'
    one_const = helper.make_tensor(one_const_input_name, TensorProto.FLOAT, [], [1.0])
    node_add = helper.make_node('Add', ['Y_exp', one_const_input_name], ['Y_add'])
    node_log = helper.make_node('Log', ['Y_add'], ['Y_log'])
    node_tanh = helper.make_node('Tanh', ['Y_log'], ['Y_tanh'])
    node_mul = helper.make_node('Mul', ['X', 'Y_tanh'], ['Y_final'])

    # 构建图，注意这里的输出列表包含了所有我们想观察的中间张量
    graph_def = helper.make_graph(
        [node_exp, node_add, node_log, node_tanh, node_mul],
        'debug-mish-graph',
        [X], 
        [Y_exp, Y_add, Y_log, Y_tanh, Y_final], # 将所有中间张量作为图的输出
        [one_const] 
    )
    
    opset_imports = [helper.make_opsetid("", 14)]
    model_def = helper.make_model(graph_def, producer_name='onnx-benchmark-test', ir_version=10, opset_imports=opset_imports)
    onnx.checker.check_model(model_def)
    onnx.save(model_def, output_path)

def benchmark(session, input_data, num_runs=100):
    """简单的Tick-Tock计时函数"""
    for _ in range(10):
        session.run(None, {'X': input_data})
        
    start_time = time.perf_counter()
    for _ in range(num_runs):
        session.run(None, {'X': input_data})
    end_time = time.perf_counter()
    
    avg_time_ms = (end_time - start_time) * 1000 / num_runs
    return avg_time_ms

def debug_and_verify(sess_ref_debug, sess_test, input_data):
    """验证MyMish的正确性，并在失败时打印出ONNX官方实现的完整中间步骤。"""
    # 1. 从调试模型中获取所有中间和最终输出
    ref_outputs = sess_ref_debug.run(None, {'X': input_data})
    ref_exp, ref_add, ref_log, ref_tanh, ref_final = ref_outputs

    # 2. 从待测试模型中获取最终输出
    test_final = sess_test.run(None, {'X': input_data})[0]

    # 3. 进行高精度比对
    is_close = np.allclose(ref_final, test_final, rtol=1e-4, atol=1e-5)
    
    if is_close:
        print("✅ Correctness Check PASSED: The outputs are numerically consistent.")
    else:
        print("❌ Correctness Check FAILED: The outputs are different!")
        abs_diff = np.abs(ref_final - test_final)
        
        # 找到误差最大的元素的位置
        mismatch_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
        
        # 获取该位置对应的输入值
        input_val = input_data[mismatch_idx]

        print("\n--- Debug Trace for Failing Input ---")
        print(f"Index of failure: {mismatch_idx}")
        print(f"Input X:                          {input_val:.8f}")
        print("-----------------------------------")
        print("Expected intermediate values (from ONNX):")
        print(f"  1. Y_exp = Exp(X):              {ref_exp[mismatch_idx]:.8f}")
        print(f"  2. Y_add = Y_exp + 1:           {ref_add[mismatch_idx]:.8f}")
        print(f"  3. Y_log = Log(Y_add):          {ref_log[mismatch_idx]:.8f}")
        print(f"  4. Y_tanh = Tanh(Y_log):        {ref_tanh[mismatch_idx]:.8f}")
        print(f"  5. Y_final = X * Y_tanh:      {ref_final[mismatch_idx]:.8f} (Expected Final)")
        print("-----------------------------------")
        print(f"Actual output from MyMish:        {test_final[mismatch_idx]:.8f} (Actual Final)")
        print(f"Absolute difference:              {abs_diff[mismatch_idx]:.8f}")
        print("-----------------------------------")
        print("\n[Action] Please use these intermediate values to debug your C++ implementation.")
        
    return is_close

if __name__ == "__main__":
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ort_lib_path = os.path.join(script_dir, "onnxruntime-linux-x64-1.22.0", "lib", "libonnxruntime.so")
        if os.path.exists(ort_lib_path):
             ctypes.CDLL(ort_lib_path)
             print(f"[*] Successfully pre-loaded ONNX Runtime library.")
        else:
             print(f"[!] Warning: ONNX Runtime library not found at '{ort_lib_path}'. MyMish test might fail.")
    except Exception as e:
        print(f"[!] FAILED to pre-load ONNX Runtime library: {e}")

    print("\n[Step 1] Generating ONNX models...")
    build_single_op_model('Relu', 'relu_model.onnx')
    build_debug_mish_model('debug_mish_model.onnx') # 改为生成调试模型
    build_single_op_model('MyMish', 'mymish_model.onnx', domain='com.mydomain')
    print("   ...Models generated successfully.")

    input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
    
    print("\n[Step 2] Running performance benchmarks...")
    
    time_onnx_mish = 0.0
    time_mymish = 0.0
    sess_debug_mish = None
    sess_mymish = None

    try:
        sess_relu = ort.InferenceSession('relu_model.onnx')
        time_relu = benchmark(sess_relu, input_data)
        print(f"   - ONNX Relu (ideal): {time_relu:.6f} ms")
    except Exception as e:
        print(f"   - FAILED to test Relu: {e}")

    try:
        # 加载新的调试模型来获取性能和参考值
        sess_debug_mish = ort.InferenceSession('debug_mish_model.onnx')
        # 注意：性能测试会稍微慢一点，因为它需要写多个输出
        time_onnx_mish = benchmark(sess_debug_mish, input_data)
        print(f"   - ONNX Mish (Decomposed): {time_onnx_mish:.6f} ms")
    except Exception as e:
        print(f"   - FAILED to test Decomposed Mish: {e}")
    
    try:
        so = ort.SessionOptions()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        so_path = os.path.join(script_dir, "libmymish_avx2_omp.so") 
        if not os.path.exists(so_path):
            raise FileNotFoundError(f"Custom op library not found at: {so_path}")
            
        so.register_custom_ops_library(so_path) 
        sess_mymish = ort.InferenceSession('mymish_model.onnx', sess_options=so)
        time_mymish = benchmark(sess_mymish, input_data)
        print(f"   - MyMish (Fused & C++ Optimized): {time_mymish:.6f} ms")
    except Exception as e:
        print(f"\n[ERROR] FAILED to test MyMish: {e}")

    print("\n[Step 3] Verifying numerical correctness...")
    if sess_debug_mish is not None and sess_mymish is not None:
        debug_and_verify(sess_debug_mish, sess_mymish, input_data) # 使用新的验证函数
    else:
        print("   Skipping correctness check due to errors in session creation.")

    print("\n--- Final Summary ---")
    if time_onnx_mish > 0 and time_mymish > 0:
        print(f"Performance: MyMish is {time_onnx_mish / time_mymish:.2f}x faster than the decomposed ONNX Mish.")
    else:
        print("Performance: Could not perform speed comparison due to errors.")
