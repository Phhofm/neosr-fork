# aether_arch.py for neosr
# This file defines the AetherNet architecture for use with the neosr framework.
# It now includes methods for Quantization-Aware Training (QAT) with a warning fix.

import math
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from typing import Any

# Import specific QAT observers and functions for bfloat16 compatibility
import torch.ao.quantization as tq
from torch.ao.quantization.observer import MovingAverageMinMaxObserver, HistogramObserver

# neosr-specific imports
from neosr.archs.arch_util import net_opt, to_2tuple
from neosr.utils.registry import ARCH_REGISTRY

# Ensure upscale_opt is defined for module-level use
upscale_opt, __ = net_opt() 

# --- Import from common.aether_core ---
# This import expects 'common' to be added to sys.path by train.py
# from common.aether_core import aether as AetherNetCore # Alias to avoid name conflict with class aether below
# Removed this specific import as AetherNetCore is not used in this file directly,
# and it simplifies potential pathing issues if 'common' is added later.

# --- Core AetherNet Modules (Identical across frameworks) ---

def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ReparamLargeKernelConv(nn.Module):
    """
    A structural re-parameterization block with a large kernel and a parallel small kernel.
    The two branches are fused into a single convolution during inference for high speed.
    `fused_init` allows direct initialization in a fused state,
    useful for loading already fused models.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, small_kernel, fused_init=False):
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
        

    def forward(self, x):
        if self.fused:
            return self.fused_conv(x)
        
        lk_out = self.lk_conv(x)
        sk_out = self.sk_conv(x)
        return lk_out + self.lk_bias.view(1, -1, 1, 1) + sk_out + self.sk_bias.view(1, -1, 1, 1)

    def _fuse_kernel(self):
        """Internal method to fuse the kernels and biases of the two branches."""
        if self.fused:
            raise RuntimeError("Cannot fuse an already fused ReparamLargeKernelConv module.")

        pad = (self.kernel_size - self.small_kernel) // 2
        sk_kernel_padded = torch.nn.functional.pad(self.sk_conv.weight, [pad] * 4)
        
        fused_kernel = self.lk_conv.weight + sk_kernel_padded
        fused_bias = self.lk_bias + self.sk_bias
        
        return fused_kernel, fused_bias

    def fuse(self):
        """
        Switch the module to inference mode by fusing the branches.
        This operation is done in-place and modifies the module's structure.
        """
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
    """Gated Feed-Forward Network with GELU activation."""
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_gate = nn.Linear(in_features, hidden_features)
        self.fc1_main = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        gate = self.fc1_gate(x)
        main = self.fc1_main(x)
        x = self.act(gate) * main
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class AetherBlock(nn.Module):
    """
    The core building block of AetherNet.
    It uses a re-parameterized large kernel convolution for global context
    and a Gated FFN for feature mixing.
    `fused_init` allows direct initialization in a fused state.
    """
    def __init__(self, dim, mlp_ratio=2.0, drop=0., drop_path=0., lk_kernel=11, sk_kernel=3, fused_init=False):
        super().__init__()
        
        self.conv = ReparamLargeKernelConv(
            in_channels=dim, out_channels=dim, kernel_size=lk_kernel, 
            stride=1, groups=dim, small_kernel=sk_kernel, fused_init=fused_init
        )
        
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.ffn = GatedFFN(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.conv(x)
        
        x = x.permute(0, 2, 3, 1) # (B, H, W, C)
        x = self.norm(x)
        
        x = self.ffn(x)
        
        x = x.permute(0, 3, 1, 2) # (B, C, H, W)
        
        x = shortcut + self.drop_path(x)
        
        return x


class Upsample(nn.Sequential):
    """Upsample module using PixelShuffle."""
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n (e.g., 2, 4, 8)
            for _ in range(int(math.log2(scale))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f"Upscale factor {scale} is not supported.")
        super(Upsample, self).__init__(*m)


@ARCH_REGISTRY.register()
class aether(nn.Module):
    r"""AetherNet: A high-performance Single Image Super-Resolution (SISR) network.
    
    This architecture utilizes structural re-parameterization (ReparamLargeKernelConv)
    and a Gated Feed-Forward Network (GatedFFN) for efficient feature extraction.
    
    Args:
        in_chans (int): Number of input image channels (e.g., 3 for RGB).
        embed_dim (int): Feature dimension of the network.
        depths (tuple[int]): Number of AetherBlocks in each layer group.
        mlp_ratio (float): Ratio of MLP hidden features to embedding dimension.
        drop_rate (float): Dropout rate for FFN.
        drop_path_rate (float): Stochastic depth rate.
        lk_kernel (int): Large kernel size for ReparamLargeKernelConv.
        sk_kernel (int): Small kernel size for ReparamLargeKernelConv.
        scale (int): Upscale factor (e.g., 2, 3, 4). This parameter name
                     is critical for neosr to pass the global 'scale'.
                     Defaults to the value provided by neosr's net_opt().
        img_range (float): Pixel value range (e.g., 1.0 for [0,1] input).
        fused_init (bool): If True, initializes the network with fused convolutions directly,
                           suitable for loading pre-fused models for inference.
    """
    def _init_weights(self, m):
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
        in_chans=3,
        embed_dim=128,
        depths=(6, 6, 6, 6, 6, 6), # Default to medium variant depths
        mlp_ratio=2.0,
        drop_rate=0.,
        drop_path_rate=0.1,
        lk_kernel=11,
        sk_kernel=3,
        scale=upscale_opt, # <--- Changed default to upscale_opt
        img_range=1.0, 
        fused_init=False, # Key parameter for fusion
        **kwargs, # Accepts any additional parameters from the config
    ):
        super().__init__()

        self.img_range = img_range
        self.upscale = scale # Store internally as self.upscale
        self.fused_init = fused_init # Store fused_init state for internal logic
        
        # Mean subtraction/addition for pixel normalization
        if in_chans == 3:
            self.mean = torch.Tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1) # For grayscale or other channel counts
            
        # 1. Shallow feature extraction: Initial convolution layer
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        # 2. Deep feature extraction: Stack of AetherBlocks
        self.num_layers = len(depths)
        self.layers = nn.ModuleList()
        # Calculate stochastic depth decay for each block
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        cur_drop_path_idx = 0
        for i_layer in range(self.num_layers):
            layer_blocks = []
            for i in range(depths[i_layer]):
                layer_blocks.append(AetherBlock(
                    dim=embed_dim,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=dpr[cur_drop_path_idx + i], # Assign individual drop_path rates
                    lk_kernel=lk_kernel,
                    sk_kernel=sk_kernel,
                    fused_init=self.fused_init # Pass the fused_init flag down to blocks
                ))
            self.layers.append(nn.Sequential(*layer_blocks))
            cur_drop_path_idx += depths[i_layer]
        
        # Normalization and final convolution after the main body
        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # 3. High-quality image reconstruction
        num_feat_upsample = 64 # Consistent feature dimension before upsampling
        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(embed_dim, num_feat_upsample, 3, 1, 1), 
            nn.LeakyReLU(inplace=True)
        )
        self.upsample = Upsample(self.upscale, num_feat_upsample) # Use self.upscale (derived from 'scale' param)
        self.conv_last = nn.Conv2d(num_feat_upsample, in_chans, 3, 1, 1)

        # Apply initial weights only if not in fused_init mode
        # Fused models typically load pre-calculated weights
        if not self.fused_init:
            self.apply(self._init_weights)

    def forward(self, x):
        # Ensure mean tensor is on the same device and has the same dtype as input x
        if self.mean.device != x.device:
            self.mean = self.mean.to(x.device)
        self.mean = self.mean.type_as(x)

        # Normalize input
        x = (x - self.mean) * self.img_range
        
        x_first = self.conv_first(x)
        
        res = x_first
        for layer in self.layers:
            res = layer(res)
        
        # Apply LayerNorm and permute back
        res = res.permute(0, 2, 3, 1) # (B, H, W, C) -> for LayerNorm
        res = self.norm(res)
        res = res.permute(0, 3, 1, 2) # (B, C, H, W) -> back for Conv2d

        # Residual connection after body
        res = self.conv_after_body(res)
        res += x_first

        # Upsampling and final convolution
        x = self.conv_before_upsample(res)
        x = self.conv_last(self.upsample(x))

        # Denormalize output
        return x / self.img_range + self.mean
        
    def fuse_model(self):
        """
        Call this method after loading weights to fuse the re-parameterization blocks
        for fast inference. This method changes the model's internal structure in-place.
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

    # --- QAT Methods (added for Quantization-Aware Training) ---
    def prepare_qat(self, opt: dict[str, Any]):
        """
        Prepares the AetherNet model for Quantization-Aware Training (QAT).
        This method inserts observers and fake quantization modules into the model.
        It supports bfloat16 quantization based on 'bfloat16' flag in opt.
        """
        self.train() # Ensure the model is in training mode for QAT preparation

        if opt.get("bfloat16", False):
            # QAT configuration for bfloat16 with reduce_range explicitly set
            qconfig = tq.QConfig(
                activation=tq.FakeQuantize.with_args(
                    observer=MovingAverageMinMaxObserver, 
                    dtype=torch.bfloat16, 
                    reduce_range=False # FIX: Explicitly setting reduce_range to suppress warning
                ),
                weight=tq.FakeQuantize.with_args(
                    observer=MovingAverageMinMaxObserver, 
                    dtype=torch.bfloat16, 
                    qscheme=torch.per_tensor_symmetric, 
                    reduce_range=False # FIX: Explicitly setting reduce_range to suppress warning
                )
            )
            print("Preparing AetherNet for BFloat16 Quantization-Aware Training (QAT).")
        else:
            # Default QAT configuration for int8 (typically 'fbgemm' or 'qnnpack')
            qconfig = tq.get_default_qat_qconfig('fbgemm') # You might choose 'qnnpack' depending on deployment target
            print("Preparing AetherNet for INT8 Quantization-Aware Training (QAT) with FBGEMM.")

        self.qconfig = qconfig # Store qconfig on the model instance
        tq.prepare_qat(self, inplace=True)
        print("AetherNet has been prepared for Quantization-Aware Training (QAT).")

    def convert_to_quantized(self):
        """
        Converts the QAT-prepared (and trained) model into a truly quantized model (e.g., INT8).
        This should be called *after* QAT training is complete.
        """
        self.eval() # Important: Switch to evaluation mode before conversion
        quantized_model = tq.convert(self, inplace=False)
        print("AetherNet has been converted to a quantized model.")
        return quantized_model


# --- Model Variants (registered for neosr) ---

@ARCH_REGISTRY.register()
def aether_small(**kwargs):
    """AetherNet Small variant, registered with neosr's ARCH_REGISTRY."""
    return aether(
        embed_dim=96,
        depths=(4, 4, 4, 4),
        mlp_ratio=2.0,
        **kwargs # Pass along other parameters from neosr config, including 'scale' if present
    )

@ARCH_REGISTRY.register()
def aether_medium(**kwargs):
    """AetherNet Medium variant, registered with neosr's ARCH_REGISTRY."""
    return aether(
        embed_dim=128,
        depths=(6, 6, 6, 6, 6, 6),
        mlp_ratio=2.0,
        **kwargs # Pass along other parameters from neosr config, including 'scale' if present
    )

@ARCH_REGISTRY.register()
def aether_large(**kwargs):
    """AetherNet Large variant, registered with neosr's ARCH_REGISTRY."""
    return aether(
        embed_dim=180,
        depths=(8, 8, 8, 8, 8, 8, 8, 8),
        mlp_ratio=2.5,
        lk_kernel=13, # Larger kernel for the large model variant
        **kwargs # Pass along other parameters from neosr config, including 'scale' if present
    )
