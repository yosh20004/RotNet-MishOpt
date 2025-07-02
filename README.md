# RotNet with Custom Mish Activation

ResNet18-based rotation angle prediction model with custom optimized Mish activation function.

## Features

- **Custom Mish Operator**: Fused multiple basic operators with AVX2 optimization
- **ONNX Export**: Supports both standard and custom operator export
- **Performance Optimization**: 10% performance improvement over standard implementation

## Installation

```bash
pip install -r requirements.txt
```

## Model Files

Download model files from [Releases](../../releases) to `models/` directory:
- `rotnet_resnet18_mish_best.pth` - PyTorch model weights
- `rotnet_resnet18_mish.onnx` - Standard ONNX model
- `rotnet_resnet18_with_custom_op.onnx` - Custom operator ONNX model

## Usage

### 1. Export ONNX Models

```bash
# Standard ONNX model
python scripts/export_to_onnx.py

# Custom operator model
python scripts/export_to_onnx_mymish.py
```

### 2. Run Tests

Prepare test images in `data/` directory, then:

```bash
# Test standard model
python test/test_onnx_model.py

# Test custom operator model
python test/test_onnx_mymish.py
```

### 3. Performance Testing

```bash
# Performance comparison
python opti/test.py
```

## Custom Operator

MyMish operator fuses multiple basic operations of Mish activation function into a single operator with AVX2 optimization:

- Fuses Exp, Add, Log, Tanh, Mul operators
- AVX2 SIMD parallel computation
- Reduces memory access and intermediate results

Recompile:
```bash
cd opti/
make
```


