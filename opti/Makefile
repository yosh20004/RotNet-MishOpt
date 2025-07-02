# 简单的Makefile来编译mymish.cpp到so文件

# 编译器设置
CXX = g++

# 基础编译标志
BASE_CXXFLAGS = -std=c++17 -fPIC -Wall -Wextra

# OpenMP支持选项 (on/off)
OPENMP ?= on

# SIMD指令集选项 (none/sse4/avx/avx2/avx512)
SIMD ?= avx2

# 根据SIMD选项设置编译标志
ifeq ($(SIMD), avx2)
    SIMD_FLAGS = -mavx2 -mfma -DUSE_AVX2
    BUILD_SUFFIX = _avx2
else ifeq ($(SIMD), avx)
    SIMD_FLAGS = -mavx -DUSE_AVX
    BUILD_SUFFIX = _avx
else ifeq ($(SIMD), sse4)
    SIMD_FLAGS = -msse4.1 -msse4.2 -DUSE_SSE4
    BUILD_SUFFIX = _sse4
else ifeq ($(SIMD), avx512)
    SIMD_FLAGS = -mavx512f -mavx512dq -DUSE_AVX512
    BUILD_SUFFIX = _avx512
else
    SIMD_FLAGS = -march=native
    BUILD_SUFFIX =
endif

# 根据OpenMP选项设置编译标志
ifeq ($(OPENMP), on)
    OPENMP_FLAGS = -fopenmp -DUSE_OPENMP
    OPENMP_LIBS = -lgomp
    BUILD_SUFFIX := $(BUILD_SUFFIX)_omp
else
    OPENMP_FLAGS =
    OPENMP_LIBS =
endif

# 构建模式 (debug/release)
BUILD_MODE ?= release

ifeq ($(BUILD_MODE), debug)
    OPT_FLAGS = -g -O0 -DDEBUG
else
    OPT_FLAGS = -O3 -DNDEBUG
endif

# 最终编译标志
CXXFLAGS = $(BASE_CXXFLAGS) $(OPT_FLAGS) $(SIMD_FLAGS) $(OPENMP_FLAGS)

# ONNX Runtime路径
ONNX_ROOT = ./onnxruntime-linux-x64-1.22.0
INCLUDE_DIR = $(ONNX_ROOT)/include
LIB_DIR = $(ONNX_ROOT)/lib

# 包含路径和库路径
INCLUDES = -I$(INCLUDE_DIR)

# --- 核心修改点 ---
# 增加了 LDFLAGS 变量来专门存放链接器参数
# -Wl,-rpath,'$ORIGIN/./onnxruntime-linux-x64-1.22.0/lib'
# $ORIGIN 指代的是最终生成的 .so 文件所在的目录。
# 根据你的项目结构，.so 和 onnxruntime-* 文件夹都在 opti 目录下，是同级关系。
# 因此路径是 './onnxruntime-linux-x64-1.22.0/lib'
LDFLAGS = -L$(LIB_DIR) -Wl,-rpath,'$ORIGIN/onnxruntime-linux-x64-1.22.0/lib'
LIBS = -lonnxruntime $(OPENMP_LIBS)

# 目标文件
TARGET = libmymish$(BUILD_SUFFIX).so
SOURCE = mymish_onnxpool.cpp

# 默认目标
all: $(TARGET)

# 编译规则
# 将 LDFLAGS 和 LIBS 分开，逻辑更清晰
$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -shared -o $(TARGET) $(SOURCE) $(LDFLAGS) $(LIBS)

# 清理
clean:
	rm -f libmymish_*.so mymish.o

# --- 后续所有目标保持不变 ---

# 测试编译（只编译不链接）
test-compile:
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $(SOURCE) -o mymish.o

# 便捷的SIMD构建目标
avx2:
	$(MAKE) SIMD=avx2

avx:
	$(MAKE) SIMD=avx

sse4:
	$(MAKE) SIMD=sse4

avx512:
	$(MAKE) SIMD=avx512

native:
	$(MAKE) SIMD=none

# 调试版本
debug:
	$(MAKE) BUILD_MODE=debug

# OpenMP版本控制
openmp:
	$(MAKE) OPENMP=on

no-openmp:
	$(MAKE) OPENMP=off

# 组合构建目标
avx2-openmp:
	$(MAKE) SIMD=avx2 OPENMP=on

debug-openmp:
	$(MAKE) BUILD_MODE=debug OPENMP=on

# 显示当前配置
info:
	@echo "编译配置信息:"
	@echo "   编译器: $(CXX)"
	@echo "   SIMD指令集: $(SIMD)"
	@echo "   构建模式: $(BUILD_MODE)"
	@echo "   OpenMP支持: $(OPENMP)"
	@echo "   SIMD标志: $(SIMD_FLAGS)"
	@echo "   OpenMP标志: $(OPENMP_FLAGS)"
	@echo "   优化标志: $(OPT_FLAGS)"
	@echo "   目标文件: $(TARGET)"

# 检测CPU支持的指令集
check-cpu:
	@echo "检测CPU支持的指令集:"
	@if grep -q "avx512" /proc/cpuinfo; then echo "   ✓ AVX-512"; else echo "   ✗ AVX-512"; fi
	@if grep -q "avx2" /proc/cpuinfo; then echo "   ✓ AVX2"; else echo "   ✗ AVX2"; fi
	@if grep -q "avx" /proc/cpuinfo; then echo "   ✓ AVX"; else echo "   ✗ AVX"; fi
	@if grep -q "sse4_1" /proc/cpuinfo; then echo "   ✓ SSE4.1"; else echo "   ✗ SSE4.1"; fi
	@if grep -q "sse4_2" /proc/cpuinfo; then echo "   ✓ SSE4.2"; else echo "   ✗ SSE4.2"; fi

# 构建所有SIMD版本
all-simd:
	@echo "构建所有SIMD版本..."
	$(MAKE) SIMD=sse4
	$(MAKE) SIMD=avx
	$(MAKE) SIMD=avx2
	@if grep -q "avx512" /proc/cpuinfo; then $(MAKE) SIMD=avx512; fi

# 帮助信息
help:
	@echo "可用的构建目标:"
	@echo "   all         - 默认构建 (AVX2 + OpenMP)"
	@echo "   avx2        - 构建AVX2版本"
	@echo "   avx         - 构建AVX版本"
	@echo "   sse4        - 构建SSE4版本"
	@echo "   avx512      - 构建AVX-512版本"
	@echo "   native      - 构建原生优化版本"
	@echo "   debug       - 构建调试版本"
	@echo "   openmp      - 构建OpenMP版本"
	@echo "   no-openmp   - 构建无OpenMP版本"
	@echo "   avx2-openmp - 构建AVX2+OpenMP版本"
	@echo "   debug-openmp - 构建调试+OpenMP版本"
	@echo "   all-simd    - 构建所有SIMD版本"
	@echo "   check-cpu   - 检测CPU支持的指令集"
	@echo "   info        - 显示编译配置"
	@echo "   clean       - 清理构建文件"
	@echo "   help        - 显示此帮助"
	@echo ""
	@echo "编译选项:"
	@echo "   SIMD=avx2|avx|sse4|avx512|none  - 指定SIMD指令集"
	@echo "   OPENMP=on|off                - 启用/禁用OpenMP"
	@echo "   BUILD_MODE=release|debug       - 指定构建模式"
	@echo ""
	@echo "示例:"
	@echo "   make avx2-openmp             # 构建AVX2+OpenMP优化版本"
	@echo "   make debug SIMD=avx2 OPENMP=on # 构建AVX2+OpenMP调试版本"
	@echo "   make OPENMP=off              # 构建无OpenMP版本"

.PHONY: all clean test-compile avx2 avx sse4 avx512 native debug openmp no-openmp avx2-openmp debug-openmp info check-cpu all-simd help
