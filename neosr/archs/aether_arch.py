# aether_arch.py for neosr - AetherNet Architecture with QAT and Registry Support (Corrected BFloat16 Compatible QAT)
#
# This file combines the AetherNet architecture, Quantization-Aware Training (QAT) integration,
# and correct registration with neosr's ARCH_REGISTRY.
# Fixes AttributeError: module 'torch.ao.quantization' has no attribute 'default_per_channel_weight_fake_quant_fn'
# by using 'default_per_channel_weight_fake_quant'.

import math
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from typing import List, Union, Any, Dict

# Import specific QAT observers for bfloat16 compatibility
import torch.ao.quantization as tq
from torch.ao.quantization.observer import MovingAverageMinMaxObserver, HistogramObserver

# neosr-specific imports for model registration and global options.
from neosr.archs.arch_util import net_opt, to_2tuple
from neosr.utils.registry import ARCH_REGISTRY

# Retrieve the global upscale factor from neosr's configuration.
upscale_opt, __ = net_opt() 


# --- Core Utility Functions and Modules ---

def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Applies Stochastic Depth (DropPath) to the input tensor.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """A PyTorch module wrapper for the drop_path function."""
    def __init__(self, drop_prob: float = 0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class ReparamLargeKernelConv(nn.Module):
    """
    A structural re-parameterization block for efficient large kernel convolutions.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int, groups: int, small_kernel: int, fused_init: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.padding = kernel_size // 2
        self.small_kernel = small_kernel
        
        self.fused = fused_init

        if self.fused:
            self.fused_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, self.padding, groups=groups, bias=True
            )
        else:
            self.lk_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, self.padding, groups=groups, bias=False
            )
            self.sk_conv = nn.Conv2d(
                in_channels, out_channels, small_kernel, stride, small_kernel // 2, groups=groups, bias=False
            )
            self.lk_bias = nn.Parameter(torch.zeros(out_channels))
            self.sk_bias = nn.Parameter(torch.zeros(out_channels))
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused:
            return self.fused_conv(x)
        
        lk_out = self.lk_conv(x)
        sk_out = self.sk_conv(x)
        return lk_out + self.lk_bias.view(1, -1, 1, 1) + sk_out + self.sk_bias.view(1, -1, 1, 1)

    def _fuse_kernel(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.fused:
            raise RuntimeError("Cannot fuse an already fused ReparamLargeKernelConv module.")

        pad = (self.kernel_size - self.small_kernel) // 2
        sk_kernel_padded = torch.nn.functional.pad(self.sk_conv.weight, [pad] * 4)
        
        fused_kernel = self.lk_conv.weight + sk_kernel_padded
        fused_bias = self.lk_bias + self.sk_bias
        
        return fused_kernel, fused_bias

    def fuse(self):
        if self.fused:
            return
            
        fused_kernel, fused_bias = self._fuse_kernel()
        
        self.fused_conv = nn.Conv2d(
            self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, groups=self.groups, bias=True
        )
        
        self.fused_conv.weight.data = fused_kernel
        self.fused_conv.bias.data = fused_bias
        
        self.__delattr__('lk_conv')
        self.__delattr__('sk_conv')
        self.__delattr__('lk_bias')
        self.__delattr__('sk_bias')
        
        self.fused = True


class GatedFFN(nn.Module):
    """Gated Feed-Forward Network (FFN) with GELU activation."""
    def __init__(self, in_features: int, hidden_features: int = None, out_features: int = None, drop: float = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1_gate = nn.Linear(in_features, hidden_features)
        self.fc1_main = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.fc1_gate(x)
        main = self.fc1_main(x)
        x = self.act(gate) * main
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class AetherBlock(nn.Module):
    """The core building block of AetherNet."""
    def __init__(self, dim: int, mlp_ratio: float = 2.0, drop: float = 0., 
                 drop_path: float = 0., lk_kernel: int = 11, sk_kernel: int = 3, fused_init: bool = False):
        super().__init__()
        
        self.conv = ReparamLargeKernelConv(
            in_channels=dim, out_channels=dim, kernel_size=lk_kernel, 
            stride=1, groups=dim, small_kernel=sk_kernel, fused_init=fused_init
        )
        
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.ffn = GatedFFN(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.conv(x)
        
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        
        x = self.ffn(x)
        
        x = x.permute(0, 3, 1, 2)
        
        x = shortcut + self.drop_path(x)
        
        return x


class Upsample(nn.Sequential):
    """Upsample module using PixelShuffle."""
    def __init__(self, scale: int, num_feat: int):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log2(scale))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f"Upscale factor {scale} is not supported. Only 2, 3, 4, 8... are supported.")
        super(Upsample, self).__init__(*m)


@ARCH_REGISTRY.register()
class aether(nn.Module):
    r"""
    AetherNet: A high-performance Single Image Super-Resolution (SISR) network.
    """
    def _init_weights(self, m: nn.Module):
        """Initializes weights for various module types."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def __init__(
        self,
        in_chans: int = 3,
        embed_dim: int = 128,
        depths: tuple[int, ...] = (6, 6, 6, 6, 6, 6), # Default to medium variant depths
        mlp_ratio: float = 2.0,
        drop_rate: float = 0.,
        drop_path_rate: float = 0.1,
        lk_kernel: int = 11,
        sk_kernel: int = 3,
        scale: int = upscale_opt, # Uses the global neosr 'scale' from net_opt() as default
        img_range: float = 1.0, 
        fused_init: bool = False, # Controls whether to initialize in fused state
        **kwargs, # Catch-all for extra config parameters
    ):
        super().__init__()

        self.img_range = img_range
        self.upscale = scale # Store the effective upscale factor internally
        self.fused_init = fused_init # Retain fused initialization state
        
        # Mean tensor for pixel normalization/denormalization.
        self.mean = torch.Tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1) if in_chans == 3 else torch.zeros(1, 1, 1, 1)
            
        # 1. Shallow feature extraction: Initial convolution layer
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        # 2. Deep feature extraction: Stack of AetherBlocks
        self.num_layers = len(depths)
        self.layers = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        cur_drop_path_idx = 0
        for i_layer in range(self.num_layers):
            layer_blocks = []
            for i in range(depths[i_layer]):
                layer_blocks.append(AetherBlock(
                    dim=embed_dim,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=dpr[cur_drop_path_idx + i],
                    lk_kernel=lk_kernel,
                    sk_kernel=sk_kernel,
                    fused_init=self.fused_init
                ))
            self.layers.append(nn.Sequential(*layer_blocks))
            cur_drop_path_idx += depths[i_layer]
        
        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        num_feat_upsample = 64
        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(embed_dim, num_feat_upsample, 3, 1, 1), 
            nn.LeakyReLU(inplace=True)
        )
        self.upsample = Upsample(self.upscale, num_feat_upsample)
        self.conv_last = nn.Conv2d(num_feat_upsample, in_chans, 3, 1, 1)

        if not self.fused_init:
            self.apply(self._init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean.device != x.device:
            self.mean = self.mean.to(x.device)
        self.mean = self.mean.type_as(x)

        x = (x - self.mean) * self.img_range
        
        # --- QAT Stub: Marks the start of the quantizable part of the network ---
        if hasattr(self, 'quant'): # Check if QAT is prepared
            x = self.quant(x)
        
        x_first = self.conv_first(x)
        
        res = x_first
        for layer in self.layers:
            res = layer(res)
        
        res = res.permute(0, 2, 3, 1) 
        res = self.norm(res)
        res = res.permute(0, 3, 1, 2)

        res = self.conv_after_body(res)
        res += x_first

        x = self.conv_before_upsample(res)
        x = self.conv_last(self.upsample(x))

        # --- QAT DeQuantStub: Marks the end of the quantizable part of the network ---
        if hasattr(self, 'dequant'): # Check if QAT is prepared
            x = self.dequant(x)

        return x / self.img_range + self.mean
        
    def fuse_model(self):
        """
        Fuses all `ReparamLargeKernelConv` modules in the network into single
        convolutional layers. This process is irreversible.
        """
        if self.fused_init:
            print("Model already initialized in a fused state. Skipping fuse_model().")
            return
            
        print("Performing in-place fusion of ReparamLargeKernelConv modules...")
        for module in self.modules():
            if isinstance(module, ReparamLargeKernelConv):
                if not module.fused:
                    module.fuse()
        self.fused_init = True
        print("Fusion complete.")

    # --- Method for QAT Preparation ---
    def prepare_qat(self, opt: Dict[str, Any]):
        """
        Prepares the model for Quantization-Aware Training.
        This method instruments the model with 'FakeQuantize' modules,
        allowing the model's weights and activations to adapt to quantization.
        It dynamically selects QConfig based on bfloat16 setting from the provided 'opt' dictionary.
        """
        # Determine QConfig based on whether bfloat16 is used
        if opt.get('use_amp', False) and opt.get('bfloat16', False):
            print("Using MovingAverageMinMaxObserver for QAT due to bfloat16 enablement.")
            # QConfig for bfloat16 compatible QAT (using MovingAverageMinMaxObserver for activations)
            self.qconfig = tq.QConfig(
                activation=MovingAverageMinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine),
                # CORRECTED: Changed 'default_per_channel_weight_fake_quant_fn' to 'default_per_channel_weight_fake_quant'
                weight=tq.default_per_channel_weight_fake_quant 
            )
        else:
            print("Using default FBGEMM QConfig for QAT (HistogramObserver for activations).")
            # Default QConfig for 'fbgemm' backend (typically uses HistogramObserver for activations)
            self.qconfig = tq.get_default_qconfig("fbgemm")

        self.train() # Model must be in training mode for QAT preparation.
        
        # Insert QuantStub and DeQuantStub explicitly for whole model quantization.
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

        # `torch.quantization.prepare_qat` will now use `self.qconfig`
        # and insert FakeQuantize modules based on it.
        tq.prepare_qat(self, inplace=True)
        
        print("AetherNet has been prepared for Quantization-Aware Training (QAT).")

    # --- Method to convert the QAT-trained model to a truly quantized model ---
    def convert_to_quantized(self):
        """
        Converts the QAT-prepared (and trained) model into a truly quantized model (e.g., INT8).
        This should be called *after* QAT training is complete.
        """
        self.eval() # Important: Switch to evaluation mode before conversion
        
        quantized_model = tq.convert(self, inplace=False)
        
        print("AetherNet has been converted to a quantized model.")
        return quantized_model


# --- Model Variants (Registered for neosr) ---

@ARCH_REGISTRY.register()
def aether_small(**kwargs) -> 'aether':
    """AetherNet Small variant, optimized for smaller models."""
    return aether(
        embed_dim=96,
        depths=(4, 4, 4, 4),
        mlp_ratio=2.0,
        **kwargs
    )

@ARCH_REGISTRY.register()
def aether_medium(**kwargs) -> 'aether':
    """AetherNet Medium variant, a balanced choice for performance."""
    return aether(
        embed_dim=128,
        depths=(6, 6, 6, 6, 6, 6),
        mlp_ratio=2.0,
        **kwargs
    )

@ARCH_REGISTRY.register()
def aether_large(**kwargs) -> 'aether':
    """AetherNet Large variant, for higher capacity and potentially better results."""
    return aether(
        embed_dim=180,
        depths=(8, 8, 8, 8, 8, 8, 8, 8),
        mlp_ratio=2.5,
        lk_kernel=13,
        **kwargs
    )
