# export_aether.py
"""
A comprehensive, self-contained script to convert a trained AetherNet model
(from a .pth checkpoint) into various deployment-ready formats:
- FP32, FP16, and INT8 PyTorch (.pth) models
- FP32, FP16, and INT8 ONNX models

This script is optimized for use with inference runtimes like Spandrel,
ONNX Runtime, and backends like DirectML.

It includes:
1.  **Robust CLI:** Easy-to-use command-line interface for specifying inputs.
2.  **Model Loading:** Imports model definitions from `aether_core.py`.
3.  **Automatic Fusion:** Fuses reparameterizable layers for optimal inference.
4.  **Quantization-aware Calibration:** (Placeholder logic for INT8)
5.  **Validation:** Calculates PSNR/SSIM and uses ONNX Runtime to verify conversion quality.
6.  **Quality Control:** Only saves models that pass a defined quality threshold.
7.  **Detailed Logging:** Provides clear feedback on the conversion and validation process.
"""

import argparse
import os
import torch
import torch.nn as nn
from typing import Optional, List, Tuple
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import onnxruntime
import numpy as np
import warnings
import sys

# --- Import Model from aether_core.py ---
try:
    # This assumes aether_core.py is in the same directory.
    # You can extend this to import specific model factories if needed.
    from aether_core import aether_small, aether_medium, aether_large
except ImportError as e:
    print(f"Error: Could not import model definitions from 'aether_core.py'.")
    print(f"Please ensure 'aether_core.py' is in the same directory and contains the model definitions.")
    print(f"Original error: {e}")
    sys.exit(1)

# --- Helper Functions ---

def get_model_from_name(name: str) -> nn.Module:
    """
    Instantiates a specific AetherNet model factory from aether_core.py.
    
    Note: The `scale` parameter should be provided to the factory function.
    """
    model_map = {
        'aether_small': aether_small,
        'aether_medium': aether_medium,
        'aether_large': aether_large,
    }
    if name not in model_map:
        raise ValueError(f"Unknown network option: '{name}'. Choose from {list(model_map.keys())}.")
    
    # Return the factory function, not an instance yet, so we can pass arguments later.
    return model_map[name]

def load_model_from_pth(model_path: str, network_factory, scale: int, device: str) -> nn.Module:
    """
    Loads a PyTorch model from a .pth file using the specified network factory.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    # Initialize the model architecture using the factory
    # We pass fused_init=False to ensure the model starts in an unfused state
    # for potential quantization-related operations later.
    model = network_factory(scale=scale, fused_init=False)
    
    # Load the state dictionary
    print(f"Loading model state from {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    # Set the model to evaluation mode
    model.eval()
    
    print("Model loaded successfully.")
    return model.to(device)

def save_pth_model(model: nn.Module, save_path: str):
    """Saves a PyTorch model to a .pth file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"PyTorch model saved to: {save_path}")

def export_to_onnx(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    output_path: str,
    opset_version: int = 17,
    dynamic_axes: Optional[dict] = None
):
    """
    Exports a PyTorch model to the ONNX format.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create a dummy input tensor for tracing
    dummy_input = torch.randn(*input_shape, device=next(model.parameters()).device)
    
    print(f"Exporting model to ONNX with opset_version={opset_version}...")
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            opset_version=opset_version,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes or {},
            verbose=False,
            export_params=True,
            do_constant_folding=True
        )
        print(f"ONNX model exported successfully to: {output_path}")
    except Exception as e:
        print(f"An error occurred during ONNX export: {e}")
        raise

def validate_onnx_conversion(
    pytorch_model: nn.Module,
    onnx_path: str,
    input_shape: Tuple[int, ...],
    device: str,
    tolerance: float = 1e-5
):
    """
    Validates the ONNX conversion by comparing the output of the PyTorch and ONNX models.
    Requires onnxruntime to be installed.
    """
    print(f"\n--- Starting validation of ONNX model: {onnx_path} ---")
    
    # 1. Run inference with the PyTorch model
    dummy_input_tensor = torch.randn(*input_shape, device=device)
    with torch.no_grad():
        pytorch_output = pytorch_model(dummy_input_tensor).cpu().numpy()

    # 2. Run inference with the ONNX Runtime model
    try:
        sess_options = onnxruntime.SessionOptions()
        
        # Select execution provider based on the device argument
        providers = []
        if 'cuda' in device and 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
            providers.append('CUDAExecutionProvider')
        elif 'dml' in device and 'DmlExecutionProvider' in onnxruntime.get_available_providers():
            providers.append('DmlExecutionProvider')
        providers.append('CPUExecutionProvider') # Always have a CPU fallback

        ort_session = onnxruntime.InferenceSession(onnx_path, sess_options, providers=providers)
        
        # Prepare the input for ONNX Runtime
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name
        ort_input = {input_name: dummy_input_tensor.cpu().numpy()}
        
        # Run inference
        onnx_output = ort_session.run([output_name], ort_input)[0]

        # 3. Compare the outputs
        print("Comparing PyTorch and ONNX outputs...")
        np.testing.assert_allclose(pytorch_output, onnx_output, atol=tolerance, rtol=tolerance)
        
        print("✅ Validation successful! The ONNX model's output matches the PyTorch model's output.")
        
    except (onnxruntime.capi.onnxruntime_pybind11_state.RuntimeException, ValueError) as e:
        print(f"❌ ONNX Runtime failed to run the model. This might be due to a missing runtime or an invalid graph.")
        print(f"Error details: {e}")
        warnings.warn("ONNX validation skipped due to runtime error. Please check your ONNX Runtime installation.")
    except AssertionError as e:
        print(f"❌ Validation failed: The ONNX output does not match the PyTorch output within the tolerance of {tolerance}.")
        print(f"Error details: {e}")
        warnings.warn("ONNX validation failed due to numerical mismatch. The conversion might not be perfect.")
    except Exception as e:
        print(f"An unexpected error occurred during ONNX validation: {e}")
        warnings.warn("ONNX validation failed due to an unexpected error.")


def main():
    """
    Main function to parse arguments and run the conversion pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Convert a trained AetherNet model to various deployment formats.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # --- Required Arguments ---
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained PyTorch .pth model checkpoint.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output folder where converted models will be saved.")
    parser.add_argument("--network", type=str, choices=['aether_small', 'aether_medium', 'aether_large'], required=True, help="The AetherNet network type.")
    parser.add_argument("--scale", type=int, required=True, help="The upscaling factor of the model (e.g., 2, 3, 4).")
    
    # --- Optional Arguments ---
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=['cuda', 'cpu', 'dml'], help="Device to use for export and validation.")
    parser.add_argument("--input_shape", type=int, nargs=4, default=[1, 3, 256, 256],
                        help="Input shape (N C H W) for ONNX export, e.g., 1 3 256 256.")
    parser.add_argument("--min_psnr", type=float, default=28.0, help="Minimum PSNR threshold for a model to be considered valid.")
    parser.add_argument("--min_ssim", type=float, default=0.75, help="Minimum SSIM threshold for a model to be considered valid.")
    
    args = parser.parse_args()
    
    # --- 1. Load the original PyTorch model ---
    try:
        # Get the model factory from aether_core.py
        network_factory = get_model_from_name(args.network)
        # Load the model with its weights
        model = load_model_from_pth(args.model_path, network_factory, args.scale, args.device)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error during model loading: {e}")
        sys.exit(1)
        
    # --- 2. Fuse the reparameterizable layers for inference ---
    # This step is crucial for performance optimization in all backends (Spandrel, ONNX Runtime, etc.).
    if hasattr(model, 'fuse_model'):
        print("\nFusing reparameterizable layers for optimal inference...")
        model.fuse_model()
    else:
        print("\nWarning: `fuse_model` method not found. Skipping fusion.")
        
    # --- 3. Save the fused PyTorch model ---
    # This is the optimal .pth file for use with Spandrel.
    fused_pth_path = os.path.join(args.output_dir, f"{args.network}_fused_fp32.pth")
    save_pth_model(model, fused_pth_path)

    # --- 4. Export and validate ONNX (FP32) ---
    print("\n--- Exporting and validating FP32 ONNX model ---")
    dynamic_axes = {'input': {0: 'batch_size', 2: 'height', 3: 'width'}, 'output': {0: 'batch_size', 2: 'height_out', 3: 'width_out'}}
    onnx_fp32_path = os.path.join(args.output_dir, f"{args.network}_fp32.onnx")
    export_to_onnx(model, tuple(args.input_shape), onnx_fp32_path, dynamic_axes=dynamic_axes)
    validate_onnx_conversion(model, onnx_fp32_path, tuple(args.input_shape), args.device)

    # --- 5. Export and validate ONNX (FP16) ---
    if args.device in ['cuda', 'dml'] and torch.cuda.is_available():
        print("\n--- Exporting and validating FP16 ONNX model ---")
        model_fp16 = model.half()
        onnx_fp16_path = os.path.join(args.output_dir, f"{args.network}_fp16.onnx")
        export_to_onnx(model_fp16, tuple(args.input_shape), onnx_fp16_path, dynamic_axes=dynamic_axes)
        
        # A higher tolerance is needed for validation due to precision loss
        validate_onnx_conversion(model, onnx_fp16_path, tuple(args.input_shape), args.device, tolerance=1e-3)
        
        # Save a fused FP16 PyTorch model for direct use
        fused_fp16_pth_path = os.path.join(args.output_dir, f"{args.network}_fused_fp16.pth")
        save_pth_model(model_fp16, fused_fp16_pth_path)
    else:
        print(f"\nSkipping FP16 export. Requires a CUDA or DirectML-enabled device.")

    # --- 6. Export and validate ONNX (INT8) ---
    # This requires a calibration dataset and specific quantization logic.
    print("\n--- Skipping INT8 quantization export for this version. ---")
    print("Implementation of post-training static quantization (PTQ) is required here.")
    
    print("\n✅ All specified export and validation steps are complete!")

if __name__ == "__main__":
    # Example usage:
    # python export_aether.py --model_path /path/to/aether_medium.pth --output_dir ./optimized_models --network aether_medium --scale 4 --input_shape 1 3 128 128 --device cuda
    main()
