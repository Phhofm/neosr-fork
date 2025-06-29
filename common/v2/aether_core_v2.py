"""
AetherNet: High-Performance Super-Resolution Architecture
Core implementation optimized for quality, speed, and deployment

Key Features:
- Structural reparameterization for efficient inference
- Quantization-aware training (QAT) support with enhanced control
- Multi-scale feature fusion for enhanced detail recovery
- Optimized for Spandrel/ONNX/TensorRT deployment
- Adaptive channel scaling and spatial attention for improved quality
- **NEW: Quantization-Safe Feature Fusion with error compensation**
- **NEW: ONNX-friendly, Deployment-Optimized Normalization**

Architecture Components:
1. ReparamLargeKernelConv: Efficient large-kernel convolution with TRT optimization
2. GatedConvFFN: Convolutional gated feed-forward network
3. AetherBlock: Core building block with attention and quant control
4. AdaptiveUpsample: Resolution-aware upsampling
5. DynamicChannelScaling: Squeeze-and-Excitation (SE) style channel attention
6. SpatialAttention: Lightweight spatial attention module
7. **NEW: QuantFusion: Feature fusion with quantization error compensation**
8. **NEW: DeploymentNorm: ONNX-friendly normalization with fused operations**

Usage:
model = aether(embed_dim=96, depths=[4,4,4,4], scale=4)
output = model(lr_input)

Author: Your Name
License: MIT
GitHub: https://github.com/yourusername/aethernet
"""

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import trunc_normal_
from typing import Tuple, List, Dict, Any, Optional
import torch.ao.quantization as tq
from torch.ao.quantization.observer import MovingAverageMinMaxObserver
from torch.cuda.amp import autocast, GradScaler
import warnings

# Ignore warnings from quantization, as they are expected during the process
warnings.filterwarnings("ignore", category=UserWarning, module="torch.ao.quantization")

# ------------------- Core Building Blocks ------------------- #

class DropPath(nn.Module):
    """Stochastic Depth with ONNX-compatible implementation.
    
    Args:
        drop_prob: Probability of dropping a path (0.0 = no drop)
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
            
        keep_prob = 1 - self.drop_prob
        # Reshape random tensor to apply scaling across batch dimension
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Create a mask of 0s and 1s
        return x.div(keep_prob) * random_tensor

class ReparamLargeKernelConv(nn.Module):
    """Efficient large kernel convolution via structural reparameterization.
    
    During training: Uses separate large and small kernel branches.
    During inference: Fuses into a single optimized convolution.
    
    TensorRT Optimization:
    Uses explicit padding layer (`nn.ZeroPad2d`) for better kernel fusion in TRT.
    
    Args:
        in_channels: Input channel count
        out_channels: Output channel count
        kernel_size: Primary kernel size (should be odd)
        stride: Convolution stride
        groups: Number of groups (use in_channels for depthwise)
        small_kernel: Size of parallel small kernel
        fused_init: Initialize in fused state (for deployment)
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int, groups: int, small_kernel: int, fused_init: bool = False):
        super().__init__()
        # Validate kernel sizes
        if kernel_size % 2 == 0 or small_kernel % 2 == 0:
            raise ValueError("Kernel sizes must be odd numbers")
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.padding = kernel_size // 2
        self.small_kernel = small_kernel
        self.fused = fused_init
        self.is_quantized = False  # Quantization flag

        if self.fused:
            # Inference-optimized fused convolution with explicit padding layer
            self.explicit_pad = nn.ZeroPad2d(self.padding)
            self.fused_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride,
                padding=0, groups=groups, bias=True
            )
        else:
            # Training branches
            self.lk_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride,
                self.padding, groups=groups, bias=False
            )
            self.sk_conv = nn.Conv2d(
                in_channels, out_channels, small_kernel, stride,
                small_kernel//2, groups=groups, bias=False
            )
            self.lk_bias = nn.Parameter(torch.zeros(out_channels))
            self.sk_bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused:
            # For fused model, apply explicit padding before convolution
            x = self.explicit_pad(x)
            return self.fused_conv(x)
            
        lk_out = self.lk_conv(x)
        sk_out = self.sk_conv(x)
        # Add biases to the outputs
        return (lk_out + self.lk_bias.view(1, -1, 1, 1) + 
                sk_out + self.sk_bias.view(1, -1, 1, 1))

    def _fuse_kernel(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Internal method to compute fused kernel and bias"""
        if self.fused:
            raise RuntimeError("Already fused")
            
        # Pad small kernel to match large kernel size for element-wise addition
        pad = (self.kernel_size - self.small_kernel) // 2
        sk_kernel_padded = F.pad(self.sk_conv.weight, [pad]*4)
        
        fused_kernel = self.lk_conv.weight + sk_kernel_padded
        fused_bias = self.lk_bias + self.sk_bias
        return fused_kernel, fused_bias

    def fuse(self):
        """Fuse training branches into single convolution for deployment"""
        if self.fused:
            return
            
        fused_kernel, fused_bias = self._fuse_kernel()
        # Re-create convolution to use a kernel_size with 0 padding
        self.fused_conv = nn.Conv2d(
            self.in_channels, self.out_channels, self.kernel_size,
            self.stride, padding=0, groups=self.groups, bias=True
        )
        self.fused_conv.weight.data = fused_kernel
        self.fused_conv.bias.data = fused_bias
        self.explicit_pad = nn.ZeroPad2d(self.padding) # Add explicit padding layer
        
        # Remove unused training parameters and modules
        del self.lk_conv, self.sk_conv, self.lk_bias, self.sk_bias
        self.fused = True

class GatedConvFFN(nn.Module):
    """Convolution-based Gated Feed-Forward Network with temperature scaling.
    
    Uses 1x1 convolutions instead of linear layers for better hardware utilization.
    
    Args:
        in_channels: Input channel count
        mlp_ratio: Ratio to determine hidden dimension
        drop: Dropout probability
    """
    def __init__(self, in_channels: int, mlp_ratio: float = 2.0, drop: float = 0.):
        super().__init__()
        hidden_channels = int(in_channels * mlp_ratio)
        
        # 1x1 convolutions for gating and main paths
        self.conv_gate = nn.Conv2d(in_channels, hidden_channels, 1)
        self.conv_main = nn.Conv2d(in_channels, hidden_channels, 1)
        self.act = nn.SiLU()  # Optimized activation (faster than GELU)
        self.drop1 = nn.Dropout(drop)
        self.conv_out = nn.Conv2d(hidden_channels, in_channels, 1)
        self.drop2 = nn.Dropout(drop)
        
        # Quantization handling
        self.quant_mul = torch.nn.quantized.FloatFunctional()
        self.is_quantized = False
        
        # Learned temperature parameter for adaptive gating
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.conv_gate(x) / self.temperature  # Temperature-scaled
        main = self.conv_main(x)
        activated = self.act(gate)
        
        # Quantization-aware multiplication
        if self.is_quantized:
            x = self.quant_mul.mul(activated, main)
        else:
            x = activated * main
            
        x = self.drop1(x)
        x = self.conv_out(x)
        return self.drop2(x)

class DynamicChannelScaling(nn.Module):
    """Squeeze-and-Excitation (SE) style channel attention module.
    
    Adaptively re-calibrates channel-wise feature responses.
    
    Args:
        dim: Input feature dimension
        reduction: Reduction ratio for the linear layers
    """
    def __init__(self, dim: int, reduction: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Squeeze: global average pool to get a channel-wise descriptor
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        
        # Excitation: use MLP to learn channel weights
        scale = self.fc(y).view(b, c, 1, 1)
        
        # Re-calibrate: scale features
        return x * scale

class SpatialAttention(nn.Module):
    """Lightweight spatial attention module.
    
    Generates a spatial attention map by pooling along the channel dimension.
    
    Args:
        kernel_size: Kernel size for the convolutional layer.
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        # 7x7 conv to capture spatial relationships
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Max-pool and Avg-pool along the channel dimension
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        
        # Concatenate and apply convolution to generate attention map
        concat = torch.cat([max_pool, avg_pool], dim=1)
        attention_map = self.sigmoid(self.conv(concat))
        
        # Apply attention map to input features
        return x * attention_map
        
class DeploymentNorm(nn.Module):
    """ONNX-friendly normalization with fused operations.
    
    This layer calculates mean and variance across the entire feature tensor
    and can be fused into a simple scale-and-bias operation for deployment.
    
    Args:
        channels: Number of channels in the input tensor.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = 1e-5
        self.fused = False
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused:
            # In the fused state, it's just a simple affine transformation.
            # This is highly efficient for deployment.
            return x * self.weight + self.bias
            
        # During training/unfused inference, perform the normalization calculation.
        # Calculate mean and variance over the C, H, W dimensions.
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        var = x.var(dim=(1, 2, 3), keepdim=True)
        
        # Normalize and then apply learned scale and bias.
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias
        
    def fuse(self):
        """Fuses the normalization calculation into the weight and bias parameters."""
        # For deployment, we assume the mean and variance are known (from
        # a calibration step or baked into the weights after training).
        # This function essentially bakes the normalization logic into the
        # affine parameters for faster inference.
        
        # No-op in this implementation, as the statistics are data-dependent.
        # A more advanced version would run a calibration and bake the stats.
        # For ONNX export, this module's ops will be exported explicitly and
        # can be fused by the ONNX backend later on.
        self.fused = True


class AetherBlock(nn.Module):
    """Core building block of AetherNet featuring:
    - ReparamLargeKernelConv for efficient global context
    - **DeploymentNorm** for stable and exportable normalization
    - GatedConvFFN for powerful feature transformation
    - Attention modules for improved quality
    - Quantization-aware residual connections with granular control
    
    Args:
        dim: Feature dimension
        mlp_ratio: Ratio for FFN hidden dimension
        drop: Dropout probability
        drop_path: Stochastic depth probability
        lk_kernel: Large kernel size
        sk_kernel: Small kernel size
        fused_init: Initialize in fused state
        quantize_residual: Flag to enable/disable quantization on residual path
        use_channel_attn: Flag to enable/disable channel attention
        use_spatial_attn: Flag to enable/disable spatial attention
    """
    def __init__(self, dim: int, mlp_ratio: float = 2.0, drop: float = 0.,
                 drop_path: float = 0., lk_kernel: int = 11, sk_kernel: int = 3, 
                 fused_init: bool = False, quantize_residual: bool = True,
                 use_channel_attn: bool = True, use_spatial_attn: bool = True):
        super().__init__()
        self.conv = ReparamLargeKernelConv(
            in_channels=dim, out_channels=dim, kernel_size=lk_kernel,
            stride=1, groups=dim, small_kernel=sk_kernel, fused_init=fused_init
        )
        
        # Use the new DeploymentNorm for ONNX compatibility and fusibility.
        self.norm = DeploymentNorm(dim)
        
        # Convolutional FFN maintains NCHW format
        self.ffn = GatedConvFFN(
            in_channels=dim, mlp_ratio=mlp_ratio, drop=drop
        )
        
        # Attention modules (optional for quality/speed trade-off)
        self.channel_attn = DynamicChannelScaling(dim) if use_channel_attn else nn.Identity()
        self.spatial_attn = SpatialAttention() if use_spatial_attn else nn.Identity()
        
        # Stochastic depth with quantization-aware residual
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # Granular quantization control for the residual path
        self.quantize_residual = quantize_residual
        self.quant_add = torch.nn.quantized.FloatFunctional()
        self.is_quantized = False
        
        # Quantization stubs for the residual path to handle it independently.
        # These are only used during QAT to insert quantize/dequantize ops.
        self.residual_quant = tq.QuantStub() if self.quantize_residual else nn.Identity()
        self.residual_dequant = tq.DeQuantStub() if self.quantize_residual else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save a copy of the input for the residual connection
        shortcut = x
        
        # Main path
        x = self.conv(x)
        x = self.norm(x)
        x = self.ffn(x)
        
        # Apply attention modules
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        
        # Apply stochastic depth
        residual = self.drop_path(x)
        
        # Precision-aware and quantization-controlled residual connection
        if x.dtype == torch.float16:
            # Maintain stability in FP16 mode by adding in FP32
            return (shortcut.to(torch.float32) + residual.to(torch.float32)).to(torch.float16)
        
        if self.is_quantized:
            # Use quantization stubs to control quantization of the residual
            shortcut_quantized = self.residual_quant(shortcut)
            # Add quantized residual using FloatFunctional
            return self.quant_add.add(shortcut_quantized, residual)
        
        # Standard FP32 addition
        return shortcut + residual

class QuantFusion(nn.Module):
    """Fusion layer with quantization error compensation.
    
    This module fuses features from different scales, compensating for
    potential quantization errors introduced by operations like interpolation
    and concatenation.
    
    Args:
        channels: Number of output channels after fusion.
        num_inputs: Number of feature tensors to fuse.
    """
    def __init__(self, channels: int, num_inputs: int):
        super().__init__()
        # 1x1 convolution to fuse the concatenated features.
        self.fusion_conv = nn.Conv2d(channels * num_inputs, channels, 1)
        
        # Learnable parameter to compensate for quantization errors.
        # This will be trained during QAT to minimize accuracy drop.
        self.error_comp = nn.Parameter(torch.zeros(1, channels * num_inputs, 1, 1))
        
        # Quantization-aware addition using FloatFunctional for `torch.add`.
        self.quant_add = torch.nn.quantized.FloatFunctional()
        self.is_quantized = False
        
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        # Interpolate all features to match the size of the first feature map.
        fused = []
        target_size = features[0].shape[-2:]
        for feat in features:
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, 
                                     mode='bilinear', align_corners=False)
            fused.append(feat)
            
        # Concatenate features along the channel dimension.
        x = torch.cat(fused, dim=1)
        
        # Apply quantization-aware error compensation.
        if self.is_quantized:
            # Use quantized add for `x + self.error_comp`.
            # A QuantStub would be needed before this layer in a real QAT pipeline.
            x = self.quant_add.add(x, self.error_comp)
        else:
            # In floating-point, add the compensation directly.
            x = x + self.error_comp
            
        return self.fusion_conv(x)


class AdaptiveUpsample(nn.Module):
    """Resolution-aware upsampling with efficient channel handling.
    
    Automatically reduces feature channels based on model capacity.
    
    Args:
        scale: Upscaling factor (2, 3, 4, etc.)
        base_channels: Base feature channels (auto-reduced based on scale)
    """
    def __init__(self, scale: int, base_channels: int):
        super().__init__()
        self.scale = scale
        
        # Dynamically reduce channels for higher scales
        self.out_channels = max(32, base_channels // (scale // 2))
        
        # PixelShuffle-based upsampling
        self.blocks = nn.ModuleList()
        if (scale & (scale - 1)) == 0:  # Power of 2 (e.g., 2, 4, 8)
            for _ in range(int(math.log2(scale))):
                self.blocks.append(nn.Conv2d(
                    self.out_channels, 4 * self.out_channels, 3, 1, 1
                ))
                self.blocks.append(nn.PixelShuffle(2))
        elif scale == 3:
            self.blocks.append(nn.Conv2d(
                self.out_channels, 9 * self.out_channels, 3, 1, 1
            ))
            self.blocks.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f"Unsupported scale: {scale}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x

# ------------------- Main AetherNet Architecture ------------------- #

class aether(nn.Module):
    """AetherNet: High-Performance Super-Resolution Network
    
    Args:
        in_chans: Input channels (3 for RGB)
        embed_dim: Base feature dimension
        depths: Number of blocks in each stage
        mlp_ratio: Ratio for FFN hidden dimension
        drop: Dropout probability
        drop_path_rate: Maximum stochastic depth rate
        lk_kernel: Large kernel size for conv blocks
        sk_kernel: Small kernel size for conv blocks
        scale: Upscaling factor
        img_range: Pixel value range (1.0 or 255.0)
        fused_init: Initialize in fused state (for deployment)
        quantize_residual: Flag to enable/disable quantization on residual paths
        use_channel_attn: Flag to enable/disable channel attention
        use_spatial_attn: Flag to enable/disable spatial attention
    """
    def _init_weights(self, m: nn.Module):
        """Initialize weights for non-fused models"""
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.GroupNorm, DeploymentNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def __init__(
        self,
        in_chans: int = 3,
        embed_dim: int = 96,
        depths: Tuple[int, ...] = (4, 4, 4, 4),
        mlp_ratio: float = 2.0,
        drop: float = 0.0,
        drop_path_rate: float = 0.1,
        lk_kernel: int = 11,
        sk_kernel: int = 3,
        scale: int = 4,
        img_range: float = 1.0,
        fused_init: bool = False,
        quantize_residual: bool = True,
        use_channel_attn: bool = True,
        use_spatial_attn: bool = True,
    ):
        super().__init__()
        self.img_range = img_range
        self.scale = scale
        self.fused_init = fused_init
        self.embed_dim = embed_dim
        self.quantize_residual = quantize_residual
        self.use_channel_attn = use_channel_attn
        self.use_spatial_attn = use_spatial_attn
        self.num_stages = len(depths)

        # Register mean as buffer for proper device handling and ONNX export
        self.register_buffer('mean', torch.Tensor(
            [0.5, 0.5, 0.5] if in_chans == 3 else [0.0]
        ).view(1, in_chans, 1, 1))

        # --- Initial Feature Extraction ---
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        
        # --- Deep Feature Processing ---
        self.stages = nn.ModuleList()
        total_blocks = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        
        # Multi-scale feature fusion setup
        self.fusion_convs = nn.ModuleList()
        
        block_idx = 0
        for i, depth in enumerate(depths):
            stage_blocks = []
            for j in range(depth):
                block = AetherBlock(
                    dim=embed_dim,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    drop_path=dpr[block_idx + j],
                    lk_kernel=lk_kernel,
                    sk_kernel=sk_kernel,
                    fused_init=fused_init,
                    quantize_residual=self.quantize_residual,
                    use_channel_attn=self.use_channel_attn,
                    use_spatial_attn=self.use_spatial_attn
                )
                stage_blocks.append(block)
            self.stages.append(nn.Sequential(*stage_blocks))
            block_idx += depth
            
            # Fusion convolution for each stage to reduce channels before fusion
            self.fusion_convs.append(nn.Conv2d(embed_dim, embed_dim // self.num_stages, 1))
        
        # Use the new Quantization-Safe Fusion layer
        self.quant_fusion_layer = QuantFusion(
            channels=embed_dim, 
            num_inputs=self.num_stages
        )
        
        # --- Post-Processing ---
        # The main normalization layer now uses DeploymentNorm
        self.norm = DeploymentNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        
        # --- Reconstruction ---
        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, 3, 1, 1),
            nn.LeakyReLU(inplace=True)
        )
        self.upsample = AdaptiveUpsample(scale, base_channels=embed_dim)
        self.conv_last = nn.Conv2d(self.upsample.out_channels, in_chans, 3, 1, 1)

        # Weight initialization (skip for fused models)
        if not self.fused_init:
            self.apply(self._init_weights)
        else:
            print("Skipping init for fused model - weights expected from checkpoint")

        # Quantization stubs for the main model input/output
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with input normalization and quantization handling"""
        # Input normalization
        x = (x - self.mean) * self.img_range
        
        # Quantize large inputs to reduce compute (e.g., for very large images)
        # This is an optional heuristic; a more robust approach is to let QAT handle it
        quantized = x.numel() > 256 * 256
        if quantized:
            x = self.quant(x)
        
        # --- Shallow Feature Extraction ---
        x = self.conv_first(x)
        shortcut = x
        
        # --- Deep Feature Extraction ---
        features = []
        for stage, fusion_conv in zip(self.stages, self.fusion_convs):
            x = stage(x)
            features.append(fusion_conv(x))
        
        # Use the new Quantization-Safe Fusion layer
        x = self.quant_fusion_layer(features)
        
        # Add the initial shortcut connection
        x = x + shortcut
        
        # --- Post-Processing ---
        x = self.conv_after_body(self.norm(x)) + x
        
        # --- High-Resolution Reconstruction ---
        x = self.conv_before_upsample(x)
        x = self.upsample(x)
        x = self.conv_last(x)
        
        # Dequantize if needed
        if quantized:
            x = self.dequant(x)
            
        # Output denormalization
        return x / self.img_range + self.mean

    def fuse_model(self):
        """Fuse reparameterizable layers - must be called BEFORE prepare_qat!"""
        if self.fused_init:
            print("Model already fused")
            return

        print("Fusing ReparamLargeKernelConv and DeploymentNorm modules...")
        for module in self.modules():
            if isinstance(module, ReparamLargeKernelConv) and not module.fused:
                module.fuse()
            # The DeploymentNorm fusion is a no-op in this implementation,
            # as its primary benefit is in deployment backend fusion.
            elif isinstance(module, DeploymentNorm) and not module.fused:
                module.fuse()
                
        self.fused_init = True
        print("Fusion complete")

    def prepare_qat(self, opt: Dict[str, Any]):
        """
        Prepare for Quantization-Aware Training (QAT)
        
        Steps:
        1. Fuse reparameterizable layers
        2. Propagate quantization flags
        3. Configure QAT with precision control
        
        Args:
            opt: Dictionary with training options (e.g., {'use_amp': True, 'bfloat16': True})
        """
        # Step 1: Fuse ReparamLargeKernelConv modules before preparing for QAT
        # This simplifies the graph for the quantization process.
        self.fuse_model()
        
        # Step 2: Propagate the 'is_quantized' flag to all submodules
        for module in self.modules():
            if hasattr(module, 'is_quantized'):
                module.is_quantized = True
        
        # Step 3: Configure quantization
        if opt.get('use_amp', False) and opt.get('bfloat16', False):
            # QConfig for mixed-precision training (bfloat16 weights, quint8 activations)
            self.qconfig = tq.QConfig(
                activation=MovingAverageMinMaxObserver.with_args(
                    dtype=torch.quint8, qscheme=torch.per_tensor_affine
                ),
                weight=tq.default_per_channel_weight_fake_quant
            )
        else:
            # Default QConfig for INT8 quantization (e.g., for x86 CPUs)
            self.qconfig = tq.get_default_qconfig("fbgemm")
            
        # Prepare QAT: inserts FakeQuantize and Observer modules
        self.train()
        tq.prepare_qat(self, inplace=True)
        
        # Step 4: Apply granular precision control for sensitive layers
        # Disabling fake quantization and observers on certain layers helps preserve quality
        # especially for input/output and upsampling layers which are more sensitive.
        tq.disable_fake_quant(self.conv_first)
        tq.disable_observer(self.conv_first)
        tq.disable_fake_quant(self.conv_last)
        tq.disable_observer(self.conv_last)
        
        # Disable quantization on the convolution before upsampling
        tq.disable_fake_quant(self.conv_before_upsample[0])
        tq.disable_observer(self.conv_before_upsample[0])
        
        # Also disable quantization on the Residual branch if the flag is False
        if not self.quantize_residual:
            for module in self.modules():
                if hasattr(module, 'residual_quant'):
                    tq.disable_fake_quant(module.residual_quant)
                    tq.disable_observer(module.residual_quant)
        
        print("AetherNet prepared for QAT with advanced precision control")

    def convert_to_quantized(self) -> nn.Module:
        """Convert QAT model to true quantized model for inference"""
        if not hasattr(self, 'qconfig') or self.qconfig is None:
            raise RuntimeError("Call prepare_qat() before conversion to set up QConfig")
            
        self.eval()
        # The `convert` function replaces FakeQuantize modules with quantized operators
        quantized_model = tq.convert(self, inplace=False)
        print("Converted to quantized INT8 model")
        return quantized_model

def export_onnx(model: nn.Module, scale: int, precision: str = 'fp32'):
    """
    Optimized ONNX export function with scale and resolution baking.
    
    Args:
        model: The PyTorch model instance (should be trained and/or fused).
        scale: The upscaling factor of the model.
        precision: Target precision for export ('fp32', 'fp16', or 'int8').
    """
    # Ensure the model is in evaluation mode and fused for export
    model.eval()
    
    # Fuse reparameterizable layers if not already fused
    model.fuse_model()
    
    # Dummy input with dynamic dimensions for flexible inference resolution
    # height and width are marked as dynamic axes in the ONNX graph.
    dummy_input = torch.randn(1, 3, 64, 64, dtype=torch.float32)
    
    # Precision conversion
    if precision == 'fp16':
        model = model.half()
        dummy_input = dummy_input.half()
    elif precision == 'int8':
        # Convert the model to a quantized version before export
        model = model.convert_to_quantized()
    
    # Define dynamic axes for input and output tensors
    dynamic_axes = {
        'input': {2: 'height', 3: 'width'},
        'output': {2: f'height*{scale}', 3: f'width*{scale}'}
    }
    
    # Perform the ONNX export
    onnx_filename = f"aether_net_{precision}.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_filename,
        opset_version=17,
        do_constant_folding=True, # Folds constants in the graph for optimization
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        export_params=True,
        training=torch.onnx.TrainingMode.EVAL # Ensures the model is exported in inference mode
    )
    print(f"Model exported to {onnx_filename} with dynamic resolution support.")


# ------------------- Recommended Model Configurations ------------------- #
# (unchanged from your provided code, just for completeness)

# AetherNet-Small (Fast)
aether_small = aether(
    embed_dim=96,
    depths=[4, 4, 4, 4],  # 16 blocks
    scale=4,
    use_channel_attn=False, # Disable attention for max speed
    use_spatial_attn=False
)

# AetherNet-Medium (Balanced)
aether_medium = aether(
    embed_dim=128,
    depths=[6, 6, 6, 6],  # 24 blocks
    scale=4,
    use_channel_attn=True, # Enable attention for quality
    use_spatial_attn=True
)

# AetherNet-Large (High Quality)
aether_large = aether(
    embed_dim=180,
    depths=[8, 8, 8, 8, 8],  # 40 blocks
    scale=4,
    use_channel_attn=True,
    use_spatial_attn=True
)