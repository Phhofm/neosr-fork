# fuse_aethernet.py
# This script loads an unfused AetherNet model, fuses its ReparamLargeKernelConv
# modules into single convolutions for faster inference, and saves the fused model.
# It is designed to be self-contained and universally usable,
# independent of the training framework (e.g., neosr, traiNNer-redux).

import torch
import torch.nn as nn
import math
import argparse
import os
from typing import Dict, Any # For type hints in helper functions


# ==============================================================================
# AetherNet Architecture Definition (Self-Contained for this script)
# This section is a direct copy of the core AetherNet PyTorch modules.
# It ensures the script can instantiate the model without external framework dependencies.
# ==============================================================================

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
    Structural re-parameterization block. Fuses large and small kernels for inference.
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
        """Initializes weights for various module types."""
        if isinstance(m, nn.Linear):
            # Explicitly using nn.init.trunc_normal_
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            # Explicitly using nn.init.trunc_normal_
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def __init__(
        self,
        in_chans: int = 3,
        embed_dim: int = 128,
        depths: tuple[int, ...] = (6, 6, 6, 6),
        mlp_ratio: float = 2.0,
        drop_rate: float = 0.,
        drop_path_rate: float = 0.1,
        lk_kernel: int = 11,
        sk_kernel: int = 3,
        upscale: int = 4, 
        img_range: float = 1.0, 
        fused_init: bool = False, 
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
                    fused_init=self.fused_init
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean.device != x.device:
            self.mean = self.mean.to(x.device)
        self.mean = self.mean.type_as(x)

        x = (x - self.mean) * self.img_range
        
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

        return x / self.img_range + self.mean
        
    def fuse_model(self):
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


# --- Helper functions for model instantiation in fusion script ---

def get_aether_model(model_type_str: str, upscale_factor: int, fused_init: bool = False, **kwargs) -> aether:
    """
    Returns an AetherNet model instance based on string identifier for use in
    the fusion script.
    """
    if model_type_str == "aether_small":
        return aether(
            embed_dim=96,
            depths=(4, 4, 4, 4),
            mlp_ratio=2.0,
            upscale=upscale_factor,
            fused_init=fused_init,
            **kwargs
        )
    elif model_type_str == "aether_medium":
        return aether(
            embed_dim=128,
            depths=(6, 6, 6, 6, 6, 6),
            mlp_ratio=2.0,
            upscale=upscale_factor,
            fused_init=fused_init,
            **kwargs
        )
    elif model_type_str == "aether_large":
        return aether(
            embed_dim=180,
            depths=(8, 8, 8, 8, 8, 8, 8, 8),
            mlp_ratio=2.5,
            lk_kernel=13,
            upscale=upscale_factor,
            fused_init=fused_init,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown AetherNet model type: {model_type_str}. "
                         "Supported types are 'aether_small', 'aether_medium', 'aether_large'.")

# ==============================================================================
# Main Fusion Script Logic
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fuse AetherNet model for inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help
    )
    
    # Required Arguments
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the unfused PyTorch model checkpoint (.pth).")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save the fused model checkpoint (.pth).")
    
    # Arguments with default values and choices
    parser.add_argument("--model_type", type=str, default="aether_medium",
                        choices=["aether_small", "aether_medium", "aether_large"],
                        help="Type of AetherNet model (e.g., 'aether_medium'). "
                             "Must match the variant trained. This is used to correctly "
                             "initialize the model architecture before loading weights.")
    parser.add_argument("--upscale", type=int, default=4,
                        help="Upscale factor of the model (e.g., 2, 3, 4). "
                             "Must match the factor the model was trained for. "
                             "This is used to correctly initialize the model architecture.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to load the model on for fusion (e.g., 'cuda', 'cpu').")
    
    args = parser.parse_args()

    # --- Pre-check arguments ---
    if not os.path.exists(args.model_path):
        print(f"Error: Input model path '{args.model_path}' does not exist.")
        exit(1) # Exit with an error code

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")

    print(f"--- AetherNet Fusion Script ---")
    print(f"Source Model: {args.model_path}")
    print(f"Output Model: {args.output_path}")
    print(f"Model Type: {args.model_type}, Upscale: {args.upscale}x, Device: {args.device}")

    try:
        # Instantiate the AetherNet model in its unfused state.
        model = get_aether_model(args.model_type, args.upscale, fused_init=False).to(args.device)
        model.eval() # Set to evaluation mode; fusion typically happens for inference.

        # Load the checkpoint file
        checkpoint = torch.load(args.model_path, map_location=args.device)
        
        # --- IMPORTANT: Extract the actual state_dict from the checkpoint ---
        if isinstance(checkpoint, dict) and 'params' in checkpoint:
            # This handles the case where the state_dict is nested under a 'params' key,
            # which is common for neosr/BasicSR checkpoints.
            state_dict_to_load = checkpoint['params']
            print("Detected 'params' key in checkpoint. Loading state_dict from 'params'.")
        elif isinstance(checkpoint, dict) and 'net_g' in checkpoint:
            # Another common key for the generator's state_dict in some frameworks
            state_dict_to_load = checkpoint['net_g']
            print("Detected 'net_g' key in checkpoint. Loading state_dict from 'net_g'.")
        else:
            # Assume the loaded object is the state_dict itself (raw state_dict)
            state_dict_to_load = checkpoint
            print("Loaded checkpoint is assumed to be the raw state_dict.")

        # Load weights into the instantiated model. Strict=True ensures all keys match.
        model.load_state_dict(state_dict_to_load, strict=True)
        print("Unfused model state_dict loaded successfully.")

        # Perform the fusion operation. This modifies the model in-place.
        model.fuse_model()

        # Save the fused model's state_dict.
        torch.save(model.state_dict(), args.output_path)
        print(f"Fused model saved successfully to: {args.output_path}")

        # --- Optional Verification Step ---
        # Load the newly saved fused model to ensure it's valid and fused correctly.
        print("\nPerforming quick verification of the saved fused model...")
        # Initialize in fused state for verification.
        fused_model_check = get_aether_model(args.model_type, args.upscale, fused_init=True).to(args.device)
        
        # Load the saved fused model (which is now just the raw state_dict)
        fused_state_dict_path = args.output_path
        fused_model_check.load_state_dict(torch.load(fused_state_dict_path, map_location=args.device), strict=True)
        fused_model_check.eval()
        
        # Assert that a characteristic fused layer exists in the loaded model.
        first_block_conv = fused_model_check.layers[0][0].conv
        assert hasattr(first_block_conv, 'fused_conv') and isinstance(first_block_conv.fused_conv, nn.Conv2d), \
            "Verification failed: Fused model does NOT contain fused_conv layers after loading."
        print("Verification successful: The saved model correctly loads as a fused AetherNet.")

    except Exception as e:
        print(f"\nAn error occurred during fusion: {e}")
        # Optionally re-raise the exception for detailed debugging: raise
        exit(1) # Exit with an error code if an exception occurs


if __name__ == "__main__":
    main()
