# AetherNet: High-Performance Super-Resolution

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Version](https://img.shields.io/badge/pytorch-1.13%2B-orange.svg)](https://pytorch.org/)

AetherNet is a Single-Image Super-Resolution (SISR) network designed for a superior balance of **speed, quality, and deployment-readiness**. The name "Aether" reflects the goal of achieving performance so efficient it feels almost ethereal, bridging the gap between high-quality research models and practical, real-world applications.

This project was created by **Philip Hofmann** with architectural analysis, refinement, and documentation assistance from AI.

---

## 🚀 Key Features & Strengths

AetherNet is built from the ground up with production in mind. It's not just another research architecture; it's a deployable tool.

* **Deployment-First Design**: Every component, from the normalization layers to the convolutions, is designed to be **fused** into a simple, blazing-fast inference graph.
* **Structural Reparameterization**: Utilizes a hybrid of large and small convolution kernels during training (`ReparamLargeKernelConv`) that are mathematically fused into a single, highly efficient large-kernel convolution for deployment, maximizing performance on modern hardware like GPUs with TensorRT.
* **Holistic Quantization Support**: Full support for Quantization-Aware Training (QAT) is a core feature, not an afterthought. The architecture includes quantization-safe modules and a streamlined workflow to produce high-quality `FP32`, `FP16`, and `INT8` models.
* **Fusable Custom Normalization**: A custom `DeploymentNorm` layer acts like LayerNorm during training but fuses into a simple scale/shift operation for inference, eliminating a significant bottleneck present in many other models.
* **Efficient and Modern Blocks**: The core `AetherBlock` uses a `GatedConvFFN` (a convolutional GLU variant) and optional channel/spatial attention mechanisms to ensure high-quality feature representation with minimal computational overhead.
* **Multi-Stage Feature Fusion**: Instead of a U-Net-like encoder-decoder, AetherNet processes features at a single high resolution across multiple stages and fuses their outputs. This preserves spatial detail and provides rich features for reconstruction.

## 🏛️ Architectural Design

For the expert, AetherNet's novelty lies in its synthesis of modern techniques into a single, cohesive architecture laser-focused on deployable performance.

#### Overall Flow

The network follows a clean and effective flow:

1.  **Shallow Feature Extraction**: A single `3x3` convolution maps the input image to the feature space.
2.  **Deep Feature Extraction (Body)**: The core of the network consists of several `stages`. Each stage is a sequence of `AetherBlock`s. Unlike U-Nets, the spatial resolution is maintained throughout. A skip connection is maintained from the shallow features.
3.  **Multi-Stage Fusion**: After each stage, features are processed by a `1x1` convolution. These processed features from all stages are then concatenated and fused by the `QuantFusion` layer. This result is added to the initial shallow features via a global residual connection.
4.  **Reconstruction**: A final set of convolutional layers, including an `AdaptiveUpsample` module using `PixelShuffle`, upscales the features to the target resolution.

#### Core Components

* **`ReparamLargeKernelConv`**: During training, this module runs two parallel branches: a large `11x11` depth-wise convolution and a smaller `3x3` one. Before deployment, the `fuse_model()` method mathematically merges the weights of the small kernel into the large one, resulting in a single convolution with zero overhead at inference time. This captures the benefits of a large receptive field while maintaining robust gradient flow during training.

* **`DeploymentNorm`**: This custom normalization is key to AetherNet's speed. It normalizes per-channel across spatial dimensions (similar to LayerNorm) but tracks statistics using an Exponential Moving Average (EMA), like BatchNorm. This allows its statistics to be "baked" into the preceding and following convolutions, effectively disappearing at inference time.

* **`GatedConvFFN`**: This feed-forward network replaces standard `Linear` layers with `1x1` convolutions in a Gated Linear Unit (GLU) structure. This is often more effective for vision tasks and preserves spatial information within the block. It includes a trainable `temperature` parameter to scale the gate activation, offering finer control over information flow.

## 🔧 How to Use

The AetherNet implementation is straightforward to use.

### 1. Basic Inference

First, instantiate a model variant and load your pre-trained weights.

```python
import torch
from aether_arch import aether_medium

# Instantiate a medium-sized model for 4x super-resolution
model = aether_medium(scale=4)

# Load your trained checkpoint
# model.load_state_dict(torch.load('aether_medium_x4.pth'))
model.eval()

# Create a dummy low-resolution image (B, C, H, W) in range [0, 1]
lr_image = torch.rand(1, 3, 128, 128)

# Run inference
with torch.no_grad():
    sr_image = model(lr_image)

print("Output shape:", sr_image.shape)
# Expected output shape: torch.Size([1, 3, 512, 512])
```

### 2. Model Configurations

Three standard configurations are provided for different use cases:

```python
from aether_arch import aether_small, aether_medium, aether_large

# Small and fast, no attention
model_s = aether_small(scale=4)

# Balanced quality and speed (default)
model_m = aether_medium(scale=4)

# Highest quality, more parameters
model_l = aether_large(scale=4)
```

### 3. Deployment Workflow (Fuse & Export)

AetherNet is designed for a simple and powerful deployment pipeline.

#### Step A: Fuse for Inference (FP32/FP16)

Before any export, **fuse the model**. This dramatically speeds up inference by merging reparameterizable layers.

```python
# Assuming 'model' is your trained AetherNet instance
model.eval()
model.fuse_model()

# Now the model is ready for high-performance FP32/FP16 inference or export
torch.save(model.state_dict(), 'aether_medium_x4_fused.pth')
```

#### Step B: Quantization-Aware Training (QAT) & INT8 Export

For maximum performance on supported hardware, you can train with QAT and export to INT8.

```python
from aether_arch import export_onnx

# 1. Start with a trained floating-point model
model = aether_medium(scale=4)
# model.load_state_dict(...)

# 2. Prepare the model for QAT. This inserts quantization observers.
model.prepare_qat()

# 3. Fine-tune the model for a few epochs with your training pipeline.
#    This allows the model to adapt to the quantization noise.
#    ... your QAT training loop ...

# 4. Convert the QAT-trained model to a true INT8 model
model.eval()
quantized_model = model.convert_to_quantized()

# 5. Export the fused, quantized model to ONNX
export_onnx(quantized_model, scale=4, precision='int8')
```

#### Step C: Exporting to ONNX

The `export_onnx` utility handles FP32, FP16, and INT8 export. Always make sure your model is fused.

```python
from aether_arch import export_onnx

# Load a trained, fused FP32 model
model = aether_medium(scale=4)
# model.load_state_dict(torch.load('aether_medium_x4_fused.pth'))
model.eval()
model.fuse_model() # Ensure fusion is called

# Export to FP32 ONNX
export_onnx(model, scale=4, precision='fp32')

# Export to FP16 ONNX
export_onnx(model, scale=4, precision='fp16')
```

## 📜 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

```
MIT License

Copyright (c) 2025 Philip Hofmann

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```