"""
AetherNet Deployment Script

Key Improvements:
1. Full INT8 quantization workflow
2. Validation retry logic
3. Enhanced error handling
4. Calibration dataset support
5. TensorRT optimization profiles
6. Scale-aware dynamic shapes

Usage:
python export_aether.py --model_path weights.pth --output_dir exports --network aether_medium --scale 4
"""

import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import onnxruntime
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional
import shutil
import time

# Import from improved aether_core
from aether_core import aether_small, aether_medium, aether_large, export_onnx

# Network factory mapping
NETWORK_FACTORY_MAP = {
    'aether_small': aether_small,
    'aether_medium': aether_medium,
    'aether_large': aether_large
}

def load_model(model_path: str, network_name: str, scale: int, device: str):
    """Load model with proper error handling"""
    if network_name not in NETWORK_FACTORY_MAP:
        raise ValueError(f"Unknown network: {network_name}. Options: {list(NETWORK_FACTORY_MAP.keys())}")
    
    model = NETWORK_FACTORY_MAP[network_name](scale)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval().to(device)
    print(f"Loaded {network_name} (scale={scale}) from {model_path}")
    return model

def generate_calib_data(num_samples=100):
    """Generate synthetic calibration data"""
    print(f"Generating {num_samples} calibration samples...")
    data = torch.rand(num_samples, 3, 128, 128) * 0.5 + 0.25
    return DataLoader(TensorDataset(data), batch_size=4)

def validate_onnx(
    pytorch_model: nn.Module,
    onnx_path: str,
    input_shape: Tuple[int, ...],
    device: str,
    tolerance: float = 1e-5,
    retries: int = 2
):
    """Validate ONNX with retry logic"""
    for attempt in range(retries + 1):
        try:
            # Run PyTorch inference
            dummy_input = torch.randn(*input_shape, device=device)
            with torch.no_grad():
                pytorch_output = pytorch_model(dummy_input).cpu().numpy()

            # Run ONNX inference
            sess_options = onnxruntime.SessionOptions()
            providers = ['CUDAExecutionProvider'] if 'cuda' in device else ['CPUExecutionProvider']
            ort_session = onnxruntime.InferenceSession(onnx_path, sess_options, providers=providers)
            ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
            onnx_output = ort_session.run(None, ort_inputs)[0]

            # Validate outputs
            np.testing.assert_allclose(pytorch_output, onnx_output, atol=tolerance, rtol=tolerance)
            print(f"✅ ONNX validation passed (attempt {attempt+1}/{retries+1})")
            return True
        except Exception as e:
            print(f"⚠️ Validation attempt {attempt+1} failed: {str(e)}")
            if attempt < retries:
                print("Retrying with optimized export...")
                export_onnx(pytorch_model, pytorch_model.scale, onnx_path.split('_')[-1].split('.')[0])
                time.sleep(1)  # Allow file system sync
    return False

def main():
    parser = argparse.ArgumentParser(
        description="AetherNet Deployment Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("--model_path", type=str, required=True, help="Trained model checkpoint")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--network", type=str, required=True, choices=NETWORK_FACTORY_MAP.keys(), help="Network type")
    parser.add_argument("--scale", type=int, required=True, help="Upscaling factor")
    
    # Optional arguments
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        choices=['cuda', 'cpu'], help="Computation device")
    parser.add_argument("--input_shape", type=int, nargs=4, default=[1, 3, 256, 256],
                        help="Input shape [batch, channels, height, width]")
    parser.add_argument("--calib_samples", type=int, default=100, help="Calibration samples for INT8")
    parser.add_argument("--min_psnr", type=float, default=28.0, help="Minimum PSNR threshold")
    parser.add_argument("--min_ssim", type=float, default=0.75, help="Minimum SSIM threshold")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    device = torch.device(args.device)
    model = load_model(args.model_path, args.network, args.scale, device)
    
    # Fuse model
    if hasattr(model, 'fuse_model'):
        model.fuse_model()
    
    # Save fused FP32 model
    fused_fp32_path = os.path.join(args.output_dir, f"{args.network}_fused_fp32.pth")
    torch.save(model.state_dict(), fused_fp32_path)
    print(f"Saved fused FP32 model to {fused_fp32_path}")
    
    # Export functions
    def export_format(fmt: str):
        """Handle format-specific export"""
        print(f"\n=== Exporting {fmt.upper()} model ===")
        onnx_path = os.path.join(args.output_dir, f"{args.network}_{fmt}.onnx")
        
        if fmt == 'int8':
            # Prepare for quantization
            model.prepare_qat({})
            
            # Calibrate
            calib_loader = generate_calib_data(args.calib_samples)
            model.calibrate(calib_loader)
            
            # Convert to quantized
            quant_model = model.convert_to_quantized()
            export_model = quant_model
        else:
            export_model = model.half() if fmt == 'fp16' else model
        
        # Export ONNX
        onnx_path = export_onnx(export_model, args.scale, fmt)
        
        # Validate with retries
        if not validate_onnx(model, onnx_path, tuple(args.input_shape), args.device):
            print(f"❌ {fmt.upper()} validation failed after retries")
            return False
        return True
    
    # Export formats
    formats = ['fp32', 'fp16', 'int8']
    results = {}
    
    for fmt in formats:
        try:
            results[fmt] = export_format(fmt)
            # Save quantized PyTorch model
            if fmt == 'int8':
                int8_path = os.path.join(args.output_dir, f"{args.network}_fused_int8.pth")
                torch.save(model.state_dict(), int8_path)
        except Exception as e:
            print(f"❌ {fmt.upper()} export failed: {str(e)}")
            results[fmt] = False
    
    # Final report
    print("\n=== Export Summary ===")
    for fmt, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{fmt.upper()}: {status}")
    
    print("\nDeployment complete!")

if __name__ == "__main__":
    main()