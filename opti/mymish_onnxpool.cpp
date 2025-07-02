#include <cmath>  
#include <cstdio>  
#include <xmmintrin.h> // for SSE  
#include <immintrin.h> // for AVX2  
#include "avx_mathfun.h"  
  
// ONNX Runtime C++ API 头文件  
#include "onnxruntime-linux-x64-1.22.0/include/onnxruntime_cxx_api.h"  
#include "onnxruntime-linux-x64-1.22.0/include/onnxruntime_c_api.h"  
  
constexpr size_t PARALLEL_THRESHOLD = 50000;  
// constexpr size_t BATCH_SIZE = 8192; // 每个批次处理的元素数量  
  
namespace MyCustomOps {  
  
template <typename T>  
struct MyMishKernel {  
    MyMishKernel(const OrtApi& api, const OrtKernelInfo* info) {}  
  
    void Compute(OrtKernelContext* context) {  
        Ort::KernelContext ctx(context);  
        auto input_X = ctx.GetInput(0);  
        const T* x_data = input_X.GetTensorData<T>();  
  
        auto tensor_info = input_X.GetTensorTypeAndShapeInfo();  
        auto shape = tensor_info.GetShape();  
        size_t size = tensor_info.GetElementCount();  
  
        auto output_Y = ctx.GetOutput(0, shape);  
        T* y_data = output_Y.GetTensorMutableData<T>();  
  
        // 使用ONNX Runtime的线程池进行并行计算  
        __MishWithOrtThreadPool(ctx, y_data, x_data, size);  
    }  
  
private:  
    struct MishTaskData {  
        const T* x_data;  
        T* y_data;  
        size_t total_size;  
        size_t batch_size;  
    };  
  
    // 使用ONNX Runtime线程池的Mish实现  
    void __MishWithOrtThreadPool(Ort::KernelContext& ctx, T* y_data, const T* x_data, size_t size) {  
        // 对于小数据量或非float类型，直接串行处理  
        if (sizeof(T) != sizeof(float)) {  
            __MishSerial(y_data, x_data, size);  
            return;  
        }  

        else if (size < PARALLEL_THRESHOLD) {
            ProcessBatch(y_data, x_data, size);
        }
  
        // 准备任务数据  
        const uint BATCH_SIZE = size / 32;
        MishTaskData task_data = {x_data, y_data, size, BATCH_SIZE};  
          
        // 计算需要的批次数量  
        size_t num_batches = (size + BATCH_SIZE - 1) / BATCH_SIZE;  
  
        // 使用ONNX Runtime的ParallelFor进行并行处理  
        ctx.ParallelFor(  
            [](void* user_data, size_t batch_idx) {  
                auto* data = static_cast<MishTaskData*>(user_data);  
                size_t start = batch_idx * data->batch_size;  
                size_t end = std::min(start + data->batch_size, data->total_size);  
                  
                // 处理这个批次的数据  
                ProcessBatch(data->y_data + start, data->x_data + start, end - start);  
            },  
            num_batches,  
            0, // num_batch参数设为0，让ONNX Runtime自动决定批次大小  
            &task_data  
        );  
    }  
  
    // 批次处理函数 - 使用AVX2向量化  
    static void ProcessBatch(T* y_data, const T* x_data, size_t size) {  
        if constexpr (sizeof(T) == sizeof(float)) {  
            const size_t stride = sizeof(__m256) / sizeof(T);  
            const size_t vectorized_size = (size / stride) * stride;  
  
            // AVX2向量化处理  
            for (size_t i = 0; i < vectorized_size; i += stride) {  
                __m256 x_vec = _mm256_loadu_ps(reinterpret_cast<const float*>(x_data + i));  
  
                // softplus(x) = log(1 + exp(x))  
                __m256 exp_vec = exp256_ps(x_vec);  
                __m256 add_vec = _mm256_add_ps(_mm256_set1_ps(1.0f), exp_vec);  
                __m256 softplus_vec = log256_ps(add_vec);  
  
                // tanh(softplus(x))  
                __m256 tanh_vec = TanhAVX(softplus_vec);  
                  
                // mish(x) = x * tanh(softplus(x))  
                __m256 y_vec = _mm256_mul_ps(x_vec, tanh_vec);  
  
                _mm256_storeu_ps(reinterpret_cast<float*>(y_data + i), y_vec);  
            }  
  
            // 处理剩余元素  
            for (size_t i = vectorized_size; i < size; ++i) {  
                T x = x_data[i];  
                T softplus_x = std::log(1 + std::exp(x));  
                y_data[i] = x * std::tanh(softplus_x);  
            }  
        } else {  
            // 非float类型的串行处理  
            for (size_t i = 0; i < size; ++i) {  
                T x = x_data[i];  
                T softplus_x = std::log(1 + std::exp(x));  
                y_data[i] = x * std::tanh(softplus_x);  
            }  
        }  
    }  
  
    // AVX2 tanh实现  
    static __m256 TanhAVX(__m256 in) {  
        // tanh(y) = (1 - e^(-2y)) / (1 + e^(-2y))  
        const __m256 one = _mm256_set1_ps(1.0f);  
        const __m256 neg_two = _mm256_set1_ps(-2.0f);  
          
        __m256 exp_arg = _mm256_mul_ps(in, neg_two);  
        __m256 exp_val = exp256_ps(exp_arg);   
          
        __m256 num = _mm256_sub_ps(one, exp_val);  
        __m256 den = _mm256_add_ps(one, exp_val);  
          
        return _mm256_div_ps(num, den);  
    }  
  
    // 串行处理函数  
    void __MishSerial(T* y_data, const T* x_data, size_t size) {  
        for (size_t i = 0; i < size; ++i) {  
            T x = x_data[i];  
            T softplus_x = std::log(1 + std::exp(x));  
            y_data[i] = x * std::tanh(softplus_x);  
        }  
    }  
};  
  
// 自定义算子定义  
struct MyMishCustomOp : Ort::CustomOpBase<MyMishCustomOp, MyMishKernel<float>> {  
    const char* GetName() const { return "MyMish"; }  
    size_t GetInputTypeCount() const { return 1; }  
    ONNXTensorElementDataType GetInputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }  
    size_t GetOutputTypeCount() const { return 1; }  
    ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }  
      
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {  
        return new MyMishKernel<float>(api, info);  
    }  
};  
  
} // namespace MyCustomOps  
  
// 注册函数  
extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api) {  
    static MyCustomOps::MyMishCustomOp my_mish_op;  
    static Ort::CustomOpDomain domain{"com.mydomain"};  
      
    try {  
        domain.Add(&my_mish_op);  
        OrtStatus* status = Ort::GetApi().AddCustomOpDomain(options, domain);  
        return status;  
    } catch (const std::exception& e) {  
        return Ort::GetApi().CreateStatus(ORT_RUNTIME_EXCEPTION, e.what());  
    }  
}  
  
extern "C" const char* ORT_API_CALL GetOpDomain() {  
    return "com.mydomain";  
}