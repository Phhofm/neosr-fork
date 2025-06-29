"""
AetherNet: High-Performance Super-Resolution Architecture
Core implementation optimized for quality, speed, and deployment

Key Improvements:
1. Fixed DeploymentNorm with proper statistical fusion
2. Enhanced quantization handling throughout
3. TensorRT-optimized operations
4. Robust QAT workflow
5. ONNX-friendly operations
6. Scale-aware dynamic shapes

Author: Philip Hofmann
License: MIT
GitHub: https://github.com/phhofm/aethernet
"""

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import trunc_normal_
from typing import Tuple, List, Dict, Any, Optional
import torch.ao.quantization as tq
from torch.ao.quantization.observer import MovingAverageMinMaxObserver
import warnings

# Ignore quantization warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.ao.quantization")

# ------------------- Core Building Blocks ------------------- #

class DropPath(nn.Module):
    """Stochastic Depth with ONNX-compatible implementation"""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
            
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class ReparamLargeKernelConv(nn.Module):
    """Efficient large kernel convolution with TRT optimization"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int, groups: int, small_kernel: int, fused_init: bool = False):
        super().__init__()
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
        self.is_quantized = False

        if self.fused:
            self.explicit_pad = nn.ZeroPad2d(self.padding)
            self.fused_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride,
                padding=0, groups=groups, bias=True
            )
        else:
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
            x = self.explicit_pad(x)
            return self.fused_conv(x)
            
        lk_out = self.lk_conv(x)
        sk_out = self.sk_conv(x)
        return (lk_out + self.lk_bias.view(1, -1, 1, 1) + 
                sk_out + self.sk_bias.view(1, -1, 1, 1))

    def _fuse_kernel(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.fused:
            raise RuntimeError("Already fused")
            
        pad = (self.kernel_size - self.small_kernel) // 2
        sk_kernel_padded = F.pad(self.sk_conv.weight, [pad]*4)
        
        fused_kernel = self.lk_conv.weight + sk_kernel_padded
        fused_bias = self.lk_bias + self.sk_bias
        return fused_kernel, fused_bias

    def fuse(self):
        if self.fused:
            return
            
        fused_kernel, fused_bias = self._fuse_kernel()
        self.fused_conv = nn.Conv2d(
            self.in_channels, self.out_channels, self.kernel_size,
            self.stride, padding=0, groups=self.groups, bias=True
        )
        self.fused_conv.weight.data = fused_kernel
        self.fused_conv.bias.data = fused_bias
        self.explicit_pad = nn.ZeroPad2d(self.padding)
        
        # Propagate quantization flag
        if self.is_quantized:
            self.fused_conv.qconfig = self.lk_conv.qconfig
            self.fused_conv = tq.QuantWrapper(self.fused_conv)
        
        del self.lk_conv, self.sk_conv, self.lk_bias, self.sk_bias
        self.fused = True

class GatedConvFFN(nn.Module):
    """Convolution-based Gated FFN with temperature scaling"""
    def __init__(self, in_channels: int, mlp_ratio: float = 2.0, drop: float = 0.):
        super().__init__()
        hidden_channels = int(in_channels * mlp_ratio)
        
        self.conv_gate = nn.Conv2d(in_channels, hidden_channels, 1)
        self.conv_main = nn.Conv2d(in_channels, hidden_channels, 1)
        self.act = nn.SiLU()
        self.drop1 = nn.Dropout(drop)
        self.conv_out = nn.Conv2d(hidden_channels, in_channels, 1)
        self.drop2 = nn.Dropout(drop)
        
        self.quant_mul = torch.nn.quantized.FloatFunctional()
        self.is_quantized = False
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.conv_gate(x) / self.temperature
        main = self.conv_main(x)
        activated = self.act(gate)
        
        if self.is_quantized:
            x = self.quant_mul.mul(activated, main)
        else:
            x = activated * main
            
        x = self.drop1(x)
        x = self.conv_out(x)
        return self.drop2(x)

class DynamicChannelScaling(nn.Module):
    """SE-style channel attention with quantization support"""
    def __init__(self, dim: int, reduction: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid()
        )
        self.quant_mul = torch.nn.quantized.FloatFunctional()
        self.is_quantized = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        scale = self.fc(y).view(b, c, 1, 1)
        
        if self.is_quantized:
            return self.quant_mul.mul(x, scale)
        return x * scale

class SpatialAttention(nn.Module):
    """Lightweight spatial attention with TRT optimizations"""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.quant_mul = torch.nn.quantized.FloatFunctional()
        self.is_quantized = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        concat = torch.cat([max_pool, avg_pool], dim=1)
        attention_map = self.sigmoid(self.conv(concat))
        
        if self.is_quantized:
            return self.quant_mul.mul(x, attention_map)
        return x * attention_map
        
class DeploymentNorm(nn.Module):
    """ONNX-friendly normalization with statistical fusion"""
    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = 1e-5
        self.fused = False
        
        # Buffers for running statistics
        self.register_buffer('running_mean', torch.zeros(1, channels, 1, 1))
        self.register_buffer('running_var', torch.ones(1, channels, 1, 1))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused:
            return x * self.weight + self.bias
            
        if self.training:
            # Calculate batch statistics
            mean = x.mean(dim=(1, 2, 3), keepdim=True)
            var = x.var(dim=(1, 2, 3), keepdim=True, unbiased=False)
            
            # Update running stats
            with torch.no_grad():
                self.running_mean = (0.9 * self.running_mean + 0.1 * mean.mean(0, keepdim=True))
                self.running_var = (0.9 * self.running_var + 0.1 * var.mean(0, keepdim=True))
                self.num_batches_tracked += 1
        else:
            mean = self.running_mean
            var = self.running_var
            
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias
        
    def fuse(self):
        """Fuse normalization into affine parameters"""
        if self.fused:
            return
            
        # Compute scale and shift
        scale = self.weight / torch.sqrt(self.running_var + self.eps)
        shift = self.bias - self.running_mean * scale
        
        # Update parameters
        self.weight.data = scale
        self.bias.data = shift
        self.fused = True


class AetherBlock(nn.Module):
    """Core building block with enhanced quantization control"""
    def __init__(self, dim: int, mlp_ratio: float = 2.0, drop: float = 0.,
                 drop_path: float = 0., lk_kernel: int = 11, sk_kernel: int = 3, 
                 fused_init: bool = False, quantize_residual: bool = True,
                 use_channel_attn: bool = True, use_spatial_attn: bool = True):
        super().__init__()
        self.conv = ReparamLargeKernelConv(
            in_channels=dim, out_channels=dim, kernel_size=lk_kernel,
            stride=1, groups=dim, small_kernel=sk_kernel, fused_init=fused_init
        )
        self.norm = DeploymentNorm(dim)
        self.ffn = GatedConvFFN(in_channels=dim, mlp_ratio=mlp_ratio, drop=drop)
        self.channel_attn = DynamicChannelScaling(dim) if use_channel_attn else nn.Identity()
        self.spatial_attn = SpatialAttention() if use_spatial_attn else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.quantize_residual = quantize_residual
        self.quant_add = torch.nn.quantized.FloatFunctional()
        self.is_quantized = False
        
        # Residual quantization stubs
        self.residual_quant = tq.QuantStub() if quantize_residual else nn.Identity()
        self.residual_dequant = tq.DeQuantStub() if quantize_residual else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.conv(x)
        x = self.norm(x)
        x = self.ffn(x)
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        residual = self.drop_path(x)
        
        # Handle residual connection based on precision
        if self.is_quantized and self.quantize_residual:
            shortcut_quant = self.residual_quant(shortcut)
            return self.quant_add.add(shortcut_quant, residual)
        
        return shortcut + residual

class QuantFusion(nn.Module):
    """Fusion layer with proper channel dimension handling"""
    def __init__(self, channels: int, num_inputs: int):
        super().__init__()
        # Fusion convolution now outputs the original embed_dim
        self.fusion_conv = nn.Conv2d(channels, channels // num_inputs, 1)
        
        # Error compensation matches concatenated dimension
        self.error_comp = nn.Parameter(torch.zeros(1, channels, 1, 1))
        
        # Quantization operations
        self.quant_add = torch.nn.quantized.FloatFunctional()
        self.is_quantized = False
        
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        # Interpolate features to match first feature map size
        fused = []
        target_size = features[0].shape[-2:]
        for feat in features:
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, 
                                     mode='bilinear', align_corners=False)
            fused.append(feat)
            
        # Concatenate along channel dimension
        x = torch.cat(fused, dim=1)
        
        # Apply quantization-aware error compensation
        if self.is_quantized:
            x = self.quant_add.add(x, self.error_comp)
        else:
            x = x + self.error_comp
            
        return self.fusion_conv(x)


class AdaptiveUpsample(nn.Module):
    """Resolution-aware upsampling with proper channel handling"""
    def __init__(self, scale: int, in_channels: int):  # Changed to in_channels
        super().__init__()
        self.scale = scale
        self.in_channels = in_channels  # Use input channels directly
        self.blocks = nn.ModuleList()
        
        # Calculate output channels based on scale
        self.out_channels = max(32, in_channels // (scale // 2))
        
        # Power of 2 scaling
        if (scale & (scale - 1)) == 0:
            num_ups = int(math.log2(scale))
            # First convolution: in_channels -> 4 * out_channels
            self.blocks.append(nn.Conv2d(
                in_channels, 4 * self.out_channels, 3, 1, 1
            ))
            self.blocks.append(nn.PixelShuffle(2))
            
            # Additional upscaling steps
            for _ in range(num_ups - 1):
                self.blocks.append(nn.Conv2d(
                    self.out_channels, 4 * self.out_channels, 3, 1, 1
                ))
                self.blocks.append(nn.PixelShuffle(2))
        elif scale == 3:
            self.blocks.append(nn.Conv2d(
                in_channels, 9 * self.out_channels, 3, 1, 1
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
    """AetherNet: Production-Ready Super-Resolution Network"""
    def _init_weights(self, m: nn.Module):
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
        self.is_quantized = False  # Global quantization flag

        # Register buffers for proper device handling
        self.register_buffer('mean', torch.Tensor(
            [0.5, 0.5, 0.5] if in_chans == 3 else [0.0]
        ).view(1, in_chans, 1, 1))

        # --- Initial Feature Extraction ---
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        
        # --- Deep Feature Processing ---
        self.stages = nn.ModuleList()
        total_blocks = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        
        # Multi-scale feature fusion
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
            
            self.fusion_convs.append(nn.Conv2d(embed_dim, embed_dim // self.num_stages, 1))
        
        # Quantization-safe fusion
        self.quant_fusion_layer = QuantFusion(
            channels=embed_dim * self.num_stages,  # Changed to match concatenated dim
            num_inputs=self.num_stages
        )
        
        # --- Post-Processing ---
        self.norm = DeploymentNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        
        # --- Reconstruction ---
        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, 3, 1, 1),
            nn.LeakyReLU(inplace=True)
        )

        # Get output channels from conv_before_upsample
        conv_before_channels = embed_dim // 2

        # Initialize upsampler with correct input channels
        self.upsample = AdaptiveUpsample(scale, in_channels=conv_before_channels)

        self.conv_last = nn.Conv2d(self.upsample.out_channels, in_chans, 3, 1, 1)

        # Weight initialization
        if not self.fused_init:
            self.apply(self._init_weights)
        else:
            print("Skipping init for fused model - weights expected from checkpoint")

        # Quantization stubs
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input normalization
        x = (x - self.mean) * self.img_range
        
        # Quantize input if model is in quantized mode
        if self.is_quantized:
            x = self.quant(x)
        
        # --- Shallow Features ---
        x = self.conv_first(x)
        shortcut = x
        
        # --- Deep Features ---
        features = []
        for stage, fusion_conv in zip(self.stages, self.fusion_convs):
            x = stage(x)
            features.append(x)
        
        x = self.quant_fusion_layer(features)
        x = x + shortcut
        
        # --- Post-Processing ---
        x = self.conv_after_body(self.norm(x)) + x
        
        # --- Reconstruction ---
        x = self.conv_before_upsample(x)
        x = self.upsample(x)
        x = self.conv_last(x)
        
        # Dequantize if needed
        if self.is_quantized:
            x = self.dequant(x)
            
        # Output denormalization
        return x / self.img_range + self.mean

    def fuse_model(self):
        """Fuse reparameterizable layers for deployment"""
        if self.fused_init:
            print("Model already fused")
            return

        print("Fusing modules for optimal inference...")
        for module in self.modules():
            if isinstance(module, ReparamLargeKernelConv) and not module.fused:
                module.fuse()
            elif isinstance(module, DeploymentNorm) and not module.fused:
                module.fuse()
                
        self.fused_init = True
        print("Fusion complete")

    def prepare_qat(self, opt: Dict[str, Any]):
        """Prepare for Quantization-Aware Training"""
        # Step 1: Fuse layers
        self.fuse_model()
        
        # Step 2: Propagate quantization flags
        self.is_quantized = True
        for module in self.modules():
            if hasattr(module, 'is_quantized'):
                module.is_quantized = True
        
        # Step 3: Configure quantization
        if opt.get('use_amp', False) and opt.get('bfloat16', False):
            self.qconfig = tq.QConfig(
                activation=MovingAverageMinMaxObserver.with_args(
                    dtype=torch.quint8, qscheme=torch.per_tensor_affine
                ),
                weight=tq.default_per_channel_weight_fake_quant
            )
        else:
            self.qconfig = tq.get_default_qconfig("fbgemm")
            
        # Prepare QAT
        self.train()
        tq.prepare_qat(self, inplace=True)
        
        # Step 4: Precision control
        tq.disable_fake_quant(self.conv_first)
        tq.disable_observer(self.conv_first)
        tq.disable_fake_quant(self.conv_last)
        tq.disable_observer(self.conv_last)
        tq.disable_fake_quant(self.conv_before_upsample[0])
        tq.disable_observer(self.conv_before_upsample[0])
        
        if not self.quantize_residual:
            for module in self.modules():
                if hasattr(module, 'residual_quant'):
                    tq.disable_fake_quant(module.residual_quant)
                    tq.disable_observer(module.residual_quant)
        
        print("AetherNet prepared for QAT")

    def convert_to_quantized(self) -> nn.Module:
        """Convert to true quantized model"""
        if not hasattr(self, 'qconfig') or self.qconfig is None:
            raise RuntimeError("Call prepare_qat() before conversion")
            
        self.eval()
        quantized_model = tq.convert(self, inplace=False)
        quantized_model.is_quantized = True
        print("Converted to quantized INT8 model")
        return quantized_model

    def calibrate(self, calib_loader: torch.utils.data.DataLoader):
        """Calibrate model for post-training quantization"""
        print("Calibrating model for PTQ...")
        self.eval()
        with torch.no_grad():
            for data in calib_loader:
                self(data.to(next(self.parameters()).device))
        print("Calibration complete")

def export_onnx(
    model: nn.Module,
    scale: int,
    precision: str = 'fp32',
    max_resolution: Tuple[int, int] = (2048, 2048)
):
    """Optimized ONNX export with resolution baking"""
    model.eval()
    model.fuse_model()
    
    # Dynamic axes with scale-aware dimensions
    dynamic_axes = {
        'input': {0: 'batch_size', 2: 'height', 3: 'width'},
        'output': {0: 'batch_size', 2: 'height_out', 3: 'width_out'}
    }
    
    # Create dummy input with dynamic shape
    dummy_input = torch.randn(1, 3, 64, 64, dtype=torch.float32)
    
    # Handle precision
    if precision == 'fp16':
        model = model.half()
        dummy_input = dummy_input.half()
    elif precision == 'int8':
        model = model.convert_to_quantized()
    
    # Export with optimization profiles
    onnx_filename = f"aether_net_{precision}.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_filename,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        export_params=True,
        training=torch.onnx.TrainingMode.EVAL,
        # TensorRT optimization hints
        **{'keep_initializers_as_inputs': True},
        **{'verbose': False},
        **{'operator_export_type': torch.onnx.OperatorExportTypes.ONNX}
    )
    
    print(f"Exported {precision.upper()} model to {onnx_filename}")
    return onnx_filename

# ------------------- Recommended Configurations ------------------- #
aether_small = lambda scale: aether(
    embed_dim=96,
    depths=[4, 4, 4, 4],
    scale=scale,
    use_channel_attn=False,
    use_spatial_attn=False
)

aether_medium = lambda scale: aether(
    embed_dim=128,
    depths=[6, 6, 6, 6],
    scale=scale,
    use_channel_attn=True,
    use_spatial_attn=True
)

aether_large = lambda scale: aether(
    embed_dim=180,
    depths=[8, 8, 8, 8, 8],
    scale=scale,
    use_channel_attn=True,
    use_spatial_attn=True
)