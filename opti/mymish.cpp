#include <cmath>
#include <cstdio>
#include <xmmintrin.h> // for SSE
#include <immintrin.h> // for AVX2
#include <omp.h>       // for OpenMP
#include "avx_mathfun.h"

// ONNX Runtime C++ API 头文件
#include "onnxruntime-linux-x64-1.22.0/include/onnxruntime_cxx_api.h"
#include "onnxruntime-linux-x64-1.22.0/include/onnxruntime_c_api.h"

constexpr size_t PARALLEL_THRESHOLD = 400000;
constexpr int INNER_NUM_THREADS = 2;

// 定义自定义算子的命名空间
namespace MyCustomOps {

template <typename T>
struct MyMishKernel {
    MyMishKernel(const OrtApi& api, const OrtKernelInfo* info) {}

    // 核心计算函数
    void Compute(OrtKernelContext* context) {
        Ort::KernelContext ctx(context);
        auto input_X = ctx.GetInput(0);
        const T* x_data = input_X.GetTensorData<T>();

        auto tensor_info = input_X.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();
        size_t size = tensor_info.GetElementCount();

        auto output_Y = ctx.GetOutput(0, shape);
        T* y_data = output_Y.GetTensorMutableData<T>();

        // call MishKernel
        __Mish(y_data, x_data, size);
    }


private:
    [[gnu::always_inline]] void __Mish(T* y_data, const T* x_data, size_t size) {

        // 对于特殊情况的朴素实现(cpu上跑不了fp8 fp16)
        if constexpr (sizeof(T) != sizeof(float)) {
            #pragma omp parallel for if(size > 1024)
            for (size_t i = 0; i < size; ++i) {
                T x = x_data[i];
                T softplus_x = std::log(1 + std::exp(x));
                y_data[i] = x * std::tanh(softplus_x);
            }
            return;
        }

        const size_t stride = sizeof(__m256) / sizeof(T);
        const size_t vectorized_size = (size / stride) * stride;

        // printf("%d\n", size);

        // AVX2向量化部分 - 使用OpenMP并行化
        #pragma omp parallel for if(size > PARALLEL_THRESHOLD) num_threads(INNER_NUM_THREADS) schedule(static)
        for (size_t i = 0; i < vectorized_size; i += stride) {
            __m256 x_vec = _mm256_load_ps(reinterpret_cast<const float*>(x_data + i));

            // softplus(x) = log(1 + exp(x)) (以2为底数)
            __m256 exp_vec = exp256_ps(x_vec);
            __m256 add_vec = _mm256_add_ps(_mm256_set1_ps(1.0f), exp_vec);
            __m256 softplus_vec = log256_ps(add_vec);

            // mish(x) = x * tanh(softplus(x))
            auto _mm256_tanh_ps = [](__m256 in) -> __m256 {
                // tanh(y) = (1 - e^(-2y)) / (1 + e^(-2y))
                const __m256 one      = _mm256_set1_ps(1.0f);
                const __m256 neg_two  = _mm256_set1_ps(-2.0f);
                
                __m256 exp_arg = _mm256_mul_ps(in, neg_two);
                __m256 exp_val = exp256_ps(exp_arg); 
                
                __m256 num = _mm256_sub_ps(one, exp_val);
                __m256 den = _mm256_add_ps(one, exp_val);
                
                // 分母 den = 1 + exp(-2*in) 总是 >= 1, 不会发生除零错误。
                return _mm256_div_ps(num, den);
            };
            __m256 tanh_vec = _mm256_tanh_ps(softplus_vec);
            __m256 y_vec = _mm256_mul_ps(x_vec, tanh_vec);

            _mm256_store_ps(reinterpret_cast<float*>(y_data + i), y_vec);
        }

        // 处理剩余的元素（标量处理）
        #pragma omp parallel for if((size - vectorized_size) > 512)
        for (size_t i = vectorized_size; i < size; ++i) {
            T x = x_data[i];
            T softplus_x = std::log(1 + std::exp(x));
            y_data[i] = x * std::tanh(softplus_x);
        }
    }
};




// --- 下面的注册代码保持不变 ---
struct MyMishCustomOp : Ort::CustomOpBase<MyMishCustomOp, MyMishKernel<float>> {
    // 注册算子的名字(MyMish)
    const char* GetName() const { return "MyMish"; };

    // 注册算子的输入参数数量
    size_t GetInputTypeCount() const { return 1; };

    // 注册算子的输入类型(FloatTensor)
    ONNXTensorElementDataType GetInputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

    // 注册算子的输出参数数量
    size_t GetOutputTypeCount() const { return 1; };

    // 注册算子的输出类型(FloatTensor)
    ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

    // 创建算子核心对象
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
        return new MyMishKernel<float>(api, info);
    };
};

} // namespace MyCustomOps

extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api) {
    static MyCustomOps::MyMishCustomOp my_mish_op;
    static Ort::CustomOpDomain domain{"com.mydomain"};
    try {
        domain.Add(&my_mish_op);
        // 使用原始的 C 指针 options 来添加域
        // 需要获取 CustomOpDomain 的底层 C API 指针
        OrtStatus* status = Ort::GetApi().AddCustomOpDomain(options, domain);
        if (status != nullptr) {
            return status;  // 返回错误状态
        }
    } catch (const std::exception& e) {
        // 如果有异常，返回错误状态
        return Ort::GetApi().CreateStatus(ORT_RUNTIME_EXCEPTION, e.what());
    }
    return nullptr;
}

extern "C" const char* ORT_API_CALL GetOpDomain() {
    return "com.mydomain";
}