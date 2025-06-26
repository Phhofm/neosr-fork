# aethernet_post_train_converter.py
# This script provides a complete post-training workflow for AetherNet models:
# 1. Loads a PyTorch model (assumed to be QAT-trained, but still in FP32/BF16).
# 2. Fuses the re-parameterization blocks (ReparamLargeKernelConv) into single convolutions
#    for faster inference, saving this as a fused FP32/BF16 .pth model.
# 3. Converts the fused model into a truly INT8 quantized PyTorch model.
# 4. Exports this INT8 PyTorch model to an INT8 ONNX format, supporting dynamic shapes.
# It includes the necessary AetherNet architecture definition to be self-contained.

# Example command
# python aethernet_post_train_converter.py \
#     --input_model_path /path/to/your_qat_trained_model.pth \
#     --model_type aether_medium \
#     --scale 2 \
#     --output_fused_pth_path ./fused_aether_medium_x2.pth \
#     --output_int8_onnx_path ./aether_medium_x2_int8.onnx \
#     --device cuda \
#     --dynamic_shapes \
#     --opt_height 64 --opt_width 64 \
#     --verify_onnx

import torch
import torch.nn as nn
import math
import argparse
import os
from typing import Dict, Any, Tuple

# Import PyTorch Quantization modules
import torch.ao.quantization as tq
from torch.ao.quantization.observer import MovingAverageMinMaxObserver, HistogramObserver

try:
    import onnxruntime as ort
    _ONNXRUNTIME_AVAILABLE = True
except ImportError:
    _ONNXRUNTIME_AVAILABLE = False
    print("Warning: onnxruntime not found. ONNX verification will be skipped.")

# ==============================================================================
# AetherNet Architecture Definition (Self-Contained for this script)
# This section is a direct copy of the core AetherNet PyTorch modules,
# including methods for fusion and quantization.
# ==============================================================================

def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Applies Stochastic Depth (DropPath) to the input tensor."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
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
    Structural re-parameterization block. Fuses large and small kernels for inference.
    Designed to be initialized in a fused state for this script's purpose.
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

    def _fuse_kernel(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Internal method to compute the fused kernel and bias."""
        if self.fused:
            raise RuntimeError("Cannot fuse an already fused ReparamLargeKernelConv module.")
        
        pad = (self.kernel_size - self.small_kernel) // 2
        sk_kernel_padded = torch.nn.functional.pad(self.sk_conv.weight, [pad] * 4)
        
        fused_kernel = self.lk_conv.weight + sk_kernel_padded
        fused_bias = self.lk_bias + self.sk_bias
        
        return fused_kernel, fused_bias

    def fuse(self):
        """Fuses the ReparamLargeKernelConv layers in-place."""
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
            raise ValueError(f"Upscale factor {scale} is not supported.")
        super(Upsample, self).__init__(*m)


class aether(nn.Module):
    r"""
    AetherNet: A high-performance Single Image Super-Resolution (SISR) network.
    """
    def _init_weights(self, m: nn.Module):
        """Initializes weights for various module types. (Not directly used if fused_init=True)."""
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def __init__(
        self,
        in_chans: int = 3,
        embed_dim: int = 128,
        depths: Tuple[int, ...] = (6, 6, 6, 6),
        mlp_ratio: float = 2.0,
        drop_rate: float = 0.,
        drop_path_rate: float = 0.1,
        lk_kernel: int = 11,
        sk_kernel: int = 3,
        upscale: int = 4, 
        img_range: float = 1.0, 
        fused_init: bool = False, # Key: True when loading a fused model directly
        **kwargs,
    ):
        super().__init__()

        self.img_range = img_range
        self.upscale = upscale
        self.fused_init = fused_init
        
        self.mean = torch.Tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1) if in_chans == 3 else torch.zeros(1, 1, 1, 1)
            
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

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
                    fused_init=self.fused_init # Pass fused_init to ReparamLargeKernelConv
                ))
            self.layers.append(nn.Sequential(*layer_blocks))
            cur_drop_path_idx += depths[i_layer]
        
        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        num_feat_upsample = 64
        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(embed_dim, num_feat_upsample, 3, 1, 1), nn.LeakyReLU(inplace=True)
        )
        self.upsample = Upsample(self.upscale, num_feat_upsample)
        self.conv_last = nn.Conv2d(num_feat_upsample, in_chans, 3, 1, 1)

        if not self.fused_init:
            self.apply(self._init_weights)

        # QAT stubs for the entire model. These are crucial for the `convert_to_quantized` method.
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean.device != x.device:
            self.mean = self.mean.to(x.device)
        self.mean = self.mean.type_as(x)

        x = (x - self.mean) * self.img_range
        
        # Apply QuantStub. This will be replaced by actual quantization during conversion.
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

        # Apply DeQuantStub. This will be replaced by actual dequantization during conversion.
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

    def prepare_qat(self, opt: Dict[str, Any]):
        """
        Prepares the model for Quantization-Aware Training (or sets QConfig for post-QAT conversion).
        This method is adapted from aether_arch.py and is used here to correctly set the qconfig
        before calling convert_to_quantized.
        """
        # Determine QConfig based on whether bfloat16 was used during training
        if opt.get('use_amp', False) and opt.get('bfloat16', False):
            print("Setting QConfig for bfloat16-compatible QAT (MovingAverageMinMaxObserver for activations).")
            self.qconfig = tq.QConfig(
                activation=MovingAverageMinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine),
                weight=tq.default_per_channel_weight_fake_quant 
            )
        else:
            print("Setting QConfig for default FBGEMM QConfig (HistogramObserver for activations).")
            self.qconfig = tq.get_default_qconfig("fbgemm")

        print("AetherNet QConfig has been set.")

    def convert_to_quantized(self):
        """
        Converts the QAT-prepared (and trained) model into a truly quantized model (e.g., INT8).
        This should be called *after* QAT training is complete and the model is loaded.
        Assumes `self.qconfig` has been set by `prepare_qat` or manually.
        """
        self.eval() # Important: Switch to evaluation mode before conversion
        
        if not hasattr(self, 'qconfig') or self.qconfig is None:
            raise RuntimeError("QConfig is not set. Call model.prepare_qat() or manually set model.qconfig before conversion.")

        quantized_model = tq.convert(self, inplace=False)
        
        print("AetherNet has been converted to a truly quantized model (e.g., INT8).")
        return quantized_model


# --- Helper functions for model instantiation ---

def get_aether_model(model_type_str: str, upscale_factor: int, in_chans: int = 3, fused_init: bool = False, **kwargs) -> aether:
    """
    Returns an AetherNet model instance based on string identifier for use in
    this script.
    """
    # Define parameters for each variant. These should match your training configs.
    if model_type_str == "aether_small":
        return aether(
            in_chans=in_chans, embed_dim=96, depths=(4, 4, 4, 4), mlp_ratio=2.0,
            upscale=upscale_factor, fused_init=fused_init, **kwargs
        )
    elif model_type_str == "aether_medium":
        return aether(
            in_chans=in_chans, embed_dim=128, depths=(6, 6, 6, 6, 6, 6), mlp_ratio=2.0,
            upscale=upscale_factor, fused_init=fused_init, **kwargs
        )
    elif model_type_str == "aether_large":
        return aether(
            in_chans=in_chans, embed_dim=180, depths=(8, 8, 8, 8, 8, 8, 8, 8), mlp_ratio=2.5,
            lk_kernel=13, # Aether Large uses a slightly larger kernel
            upscale=upscale_factor, fused_init=fused_init, **kwargs
        )
    else:
        raise ValueError(f"Unknown AetherNet model type: {model_type_str}. "
                         "Supported types are 'aether_small', 'aether_medium', 'aether_large'.")

# ==============================================================================
# Main Conversion Logic
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="AetherNet Post-Training Converter: Fuses model, converts to INT8 PyTorch, and exports to INT8 ONNX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required Arguments
    parser.add_argument("--input_model_path", type=str, required=True,
                        help="Path to the trained AetherNet PyTorch model checkpoint (.pth). "
                             "This should be a QAT-trained but unfused floating-point model.")
    parser.add_argument("--model_type", type=str, default="aether_medium",
                        choices=["aether_small", "aether_medium", "aether_large"],
                        help="Type of AetherNet model (e.g., 'aether_medium'). "
                             "Must match the variant used during training.")
    parser.add_argument("--scale", type=int, default=2, choices=[1, 2, 3, 4],
                        help="Upscale factor of the model (e.g., 2 for 2x SR). "
                             "Must match the factor the model was trained for.")
    parser.add_argument("--output_fused_pth_path", type=str, required=True,
                        help="Path to save the fused PyTorch model checkpoint (.pth). "
                             "This will be the FP32/BF16 fused version.")
    parser.add_argument("--output_int8_onnx_path", type=str, required=True,
                        help="Path to save the final INT8 ONNX model (.onnx).")
    
    # Optional Arguments
    parser.add_argument("--input_channels", type=int, default=3,
                        help="Number of input image channels (e.g., 3 for RGB).")
    parser.add_argument("--bfloat16_qconfig", action="store_true",
                        help="Set this flag if bfloat16 (with MovingAverageMinMaxObserver) "
                             "was enabled during QAT training. Affects INT8 conversion QConfig.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to perform operations on (e.g., 'cuda', 'cpu').")
    
    # Dynamic Shapes for ONNX Export
    parser.add_argument("--dynamic_shapes", action="store_true",
                        help="Export ONNX model with dynamic batch size, height, and width.")
    parser.add_argument("--opt_batch_size", type=int, default=1, help="Optimal batch size for ONNX export.")
    parser.add_argument("--min_batch_size", type=int, default=1, help="Minimum batch size for dynamic ONNX export.")
    parser.add_argument("--max_batch_size", type=int, default=4, help="Maximum batch size for dynamic ONNX export.")
    parser.add_argument("--opt_height", type=int, default=64, help="Optimal LR height for ONNX export.")
    parser.add_argument("--min_height", type=int, default=32, help="Minimum LR height for dynamic ONNX export.")
    parser.add_argument("--max_height", type=int, default=256, help="Maximum LR height for dynamic ONNX export.")
    parser.add_argument("--opt_width", type=int, default=64, help="Optimal LR width for ONNX export.")
    parser.add_argument("--min_width", type=int, default=32, help="Minimum LR width for dynamic ONNX export.")
    parser.add_argument("--max_width", type=int, default=256, help="Maximum LR width for dynamic ONNX export.")
    
    # ONNX Export Specifics
    parser.add_argument("--opset_version", type=int, default=17,
                        help="The ONNX opset version to use for export.")
    parser.add_argument("--verify_onnx", action="store_true",
                        help="Perform verification by comparing PyTorch and ONNX outputs.")
    parser.add_argument("--atol", type=float, default=5e-2, # Loosened default for INT8
                        help="Absolute tolerance for ONNX output verification. "
                             "Adjust for FP16/INT8 models.")
    parser.add_argument("--rtol", type=float, default=5e-2, # Loosened default for INT8
                        help="Relative tolerance for ONNX output verification. "
                             "Adjust for FP16/INT8 models.")
    
    args = parser.parse_args()

    # --- Setup Output Directories ---
    output_fused_dir = os.path.dirname(args.output_fused_pth_path)
    if output_fused_dir and not os.path.exists(output_fused_dir):
        os.makedirs(output_fused_dir, exist_ok=True)
        print(f"Created output directory for fused .pth: {output_fused_dir}")

    output_onnx_dir = os.path.dirname(args.output_int8_onnx_path)
    if output_onnx_dir and not os.path.exists(output_onnx_dir):
        os.makedirs(output_onnx_dir, exist_ok=True)
        print(f"Created output directory for INT8 ONNX: {output_onnx_dir}")

    print(f"\n--- AetherNet Post-Training Conversion Workflow ---")
    print(f"Input Model: {args.input_model_path}")
    print(f"Output Fused PyTorch Model: {args.output_fused_pth_path}")
    print(f"Output INT8 ONNX Model: {args.output_int8_onnx_path}")
    print(f"Model Type: {args.model_type}, Scale: {args.scale}x, Device: {args.device}")
    print(f"BFloat16 QConfig used during training: {args.bfloat16_qconfig}")

    try:
        # 1. Load the initial QAT-trained PyTorch Model (unfused, floating-point)
        print("\n--- Step 1: Loading initial QAT-trained PyTorch model ---")
        model = get_aether_model(
            model_type_str=args.model_type,
            upscale_factor=args.scale,
            in_chans=args.input_channels,
            fused_init=False # Load as unfused to fuse it here
        ).to(args.device)
        model.eval() # Set to evaluation mode

        checkpoint = torch.load(args.input_model_path, map_location=args.device)
        
        # Determine the actual state_dict within the checkpoint
        state_dict_to_load = None
        if isinstance(checkpoint, dict):
            if 'params_g' in checkpoint:
                state_dict_to_load = checkpoint['params_g']
                print("Found model state_dict under key 'params_g'.")
            elif 'params' in checkpoint:
                state_dict_to_load = checkpoint['params']
                print("Found model state_dict under key 'params'.")
            elif 'net_g' in checkpoint:
                state_dict_to_load = checkpoint['net_g']
                print("Found model state_dict under key 'net_g'.")
            elif 'state_dict' in checkpoint:
                state_dict_to_load = checkpoint['state_dict']
                print("Found model state_dict under key 'state_dict'.")
            else: # Fallback: assume the dict itself is the state_dict
                state_dict_to_load = checkpoint
                print("Checkpoint is a dict, but no common state_dict key found. Assuming dict is state_dict.")
        else: # Assume the loaded object is directly the state_dict
            state_dict_to_load = checkpoint
            print("Loaded file is assumed to be the raw state_dict.")

        if state_dict_to_load is None:
            raise ValueError("Could not find model's state dictionary in the loaded checkpoint.")

        # Load weights into the model
        model.load_state_dict(state_dict_to_load, strict=True)
        print("Initial PyTorch model loaded successfully.")

        # 2. Fuse the model and save the fused FP32/BF16 .pth model
        print("\n--- Step 2: Fusing model for inference and saving FP32/BF16 .pth ---")
        model.fuse_model() # This modifies the model in-place
        torch.save(model.state_dict(), args.output_fused_pth_path)
        print(f"Fused FP32/BF16 PyTorch model saved to: {args.output_fused_pth_path}")
        
        # Validation of fused .pth file
        print("Verifying fused PyTorch model file...")
        fused_model_check = get_aether_model(args.model_type, args.scale, args.input_channels, fused_init=True).to(args.device)
        fused_model_check.load_state_dict(torch.load(args.output_fused_pth_path, map_location=args.device), strict=True)
        fused_model_check.eval()
        
        # Check if the ReparamLargeKernelConv instances are actually fused
        is_fused_correctly = True
        for module in fused_model_check.modules():
            if isinstance(module, ReparamLargeKernelConv):
                if not hasattr(module, 'fused_conv') or not isinstance(module.fused_conv, nn.Conv2d):
                    is_fused_correctly = False
                    break
                if hasattr(module, 'lk_conv') or hasattr(module, 'sk_conv'):
                    is_fused_correctly = False
                    break
        
        if is_fused_correctly:
            print("Fused PyTorch model verification successful: All ReparamLargeKernelConv modules are fused.")
        else:
            print("Fused PyTorch model verification FAILED: Not all ReparamLargeKernelConv modules are fused correctly.")
            print("Warning: The saved fused .pth file might not be in the expected fused state.")


        # 3. Convert the fused model to a truly INT8 quantized PyTorch model
        print("\n--- Step 3: Converting fused model to truly INT8 PyTorch model ---")
        # Prepare a dummy 'opt' dict for setting the qconfig for conversion
        qat_opt_dummy = {
            'use_amp': True if args.bfloat16_qconfig else False,
            'bfloat16': args.bfloat16_qconfig
        }
        model.prepare_qat(qat_opt_dummy) # Set the qconfig based on training options
        model = model.convert_to_quantized() # Perform the INT8 conversion
        model.eval() # Ensure the converted model is in evaluation mode
        print("Fused PyTorch model converted to INT8 successfully.")


        # 4. Export the INT8 PyTorch model to INT8 ONNX
        print("\n--- Step 4: Exporting INT8 PyTorch model to INT8 ONNX ---")
        dummy_input = torch.randn(
            args.opt_batch_size,
            args.input_channels,
            args.opt_height,
            args.opt_width,
            dtype=torch.float32 # Quantized models still expect float inputs
        ).to(args.device)

        dynamic_axes = {}
        if args.dynamic_shapes:
            dynamic_axes['input'] = {0: 'batch_size', 2: 'height', 3: 'width'}
            dynamic_axes['output'] = {0: 'batch_size', 
                                      2: 'output_height', 
                                      3: 'output_width'} 
        
        should_save_onnx = True
        
        if args.verify_onnx:
            if not _ONNXRUNTIME_AVAILABLE:
                print("ONNX verification skipped: onnxruntime not installed.")
                should_save_onnx = False # Do not save if verification cannot be performed
            else:
                print("Performing ONNX model output verification...")
                
                # Run inference with PyTorch model
                torch_output = model(dummy_input).detach().cpu().numpy()
                print("PyTorch INT8 model inference complete.")

                # Temporarily export to a temp ONNX file for verification
                temp_onnx_path = args.output_int8_onnx_path + ".tmp_verify.onnx"
                print(f"Temporarily exporting to {temp_onnx_path} for verification...")
                with torch.no_grad():
                    torch.onnx.export(
                        model,
                        dummy_input,
                        temp_onnx_path,
                        export_params=True,
                        opset_version=args.opset_version,
                        do_constant_folding=True,
                        input_names=['input'],
                        output_names=['output'],
                        dynamic_axes=dynamic_axes,
                        verbose=False
                    )
                print("Temporary ONNX export complete for verification.")

                # Load and run inference with ONNX model
                ort_session = ort.InferenceSession(temp_onnx_path)
                ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
                ort_output = ort_session.run(None, ort_inputs)[0]
                print("ONNX INT8 model inference complete.")

                # Compare outputs and display actual tolerances
                max_abs_diff = torch.max(torch.abs(torch.from_numpy(torch_output) - torch.from_numpy(ort_output))).item()
                # Use a small epsilon to avoid division by zero for relative difference
                max_rel_diff = torch.max(torch.abs(
                    (torch.from_numpy(torch_output) - torch.from_numpy(ort_output)) / 
                    torch.from_numpy(torch_output).abs().clamp(min=1e-8)
                )).item()

                print(f"Actual Max Absolute Difference: {max_abs_diff:.6f}")
                print(f"Actual Max Relative Difference: {max_rel_diff:.6f}")

                if torch.allclose(torch.from_numpy(torch_output), torch.from_numpy(ort_output), 
                                  atol=args.atol, rtol=args.rtol):
                    print(f"Verification successful: PyTorch INT8 and ONNX INT8 outputs match "
                          f"within specified tolerances (atol={args.atol}, rtol={args.rtol}).")
                    should_save_onnx = True
                else:
                    print(f"Verification FAILED: PyTorch INT8 and ONNX INT8 outputs do NOT match "
                          f"within specified tolerances (atol={args.atol}, rtol={args.rtol}).")
                    should_save_onnx = False # Do NOT save if verification fails
                
                if os.path.exists(temp_onnx_path):
                    os.remove(temp_onnx_path)
                    print(f"Removed temporary ONNX file: {temp_onnx_path}")
        
        if should_save_onnx:
            print(f"\nFinal export to ONNX (opset_version={args.opset_version})...")
            with torch.no_grad():
                torch.onnx.export(
                    model,
                    dummy_input,
                    args.output_int8_onnx_path,
                    export_params=True,
                    opset_version=args.opset_version,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes=dynamic_axes,
                    verbose=False
                )
            print(f"Final INT8 ONNX model saved to: {args.output_int8_onnx_path}")
        else:
            print("\nFinal INT8 ONNX model NOT saved due to verification failure or disabled onnxruntime.")
            exit(1)

    except Exception as e:
        print(f"\nAn unexpected error occurred during the process: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        exit(1)

    print("\n--- AetherNet Post-Training Conversion Workflow Completed ---")
    print(f"Fused FP32/BF16 PyTorch model is at: {args.output_fused_pth_path}")
    if should_save_onnx:
        print(f"Quantized INT8 ONNX model is at: {args.output_int8_onnx_path}")
    else:
        print("No INT8 ONNX model was saved.")


if __name__ == "__main__":
    main()
