"""
export_aether.py

A comprehensive, self-contained script to convert a trained AetherNet model
(from a .pth checkpoint) into various deployment-ready formats:
- FP32, FP16, and INT8 PyTorch (.pth) models
- FP32, FP16, and INT8 ONNX models

This script includes:
1.  **Robust CLI:** Easy-to-use command-line interface for specifying inputs.
2.  **Model Loading:** Infers model type (small, medium, large) from input arguments.
3.  **Automatic Fusion:** Fuses reparameterizable layers for optimal inference.
4.  **Quantization-aware Calibration:** Uses a calibration dataset for accurate INT8 conversion.
5.  **Validation:** Calculates PSNR and SSIM for converted models against a validation dataset.
6.  **Quality Control:** Only saves models that pass a defined quality threshold.
7.  **Spandrel/ONNX/TRT/DML Compatibility:** Exports to formats suitable for various backends.
8.  **Detailed Logging:** Provides clear feedback on the conversion and validation process.
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np
import torch.ao.quantization as tq
from typing import Dict, Any, Tuple
import warnings
import sys

# Import your AetherNet core implementation
try:
    from aether_core import aether, aether_small, aether_medium, aether_large
    # Import validation metrics
    import torchmetrics
    
    # Silence specific user warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.ao.quantization")

except ImportError as e:
    print(f"Error: Could not import necessary modules. Make sure 'aether_core.py' is in the same directory and torchmetrics is installed (pip install torchmetrics).")
    print(f"Original error: {e}")
    sys.exit(1)

# --- 1. Helper Functions ---

def get_model_from_name(name: str, scale: int) -> torch.nn.Module:
    """Instantiates a specific AetherNet model by name."""
    model_map = {
        'aether_small': aether_small,
        'aether_medium': aether_medium,
        'aether_large': aether_large,
    }
    if name not in model_map:
        raise ValueError(f"Unknown network option: {name}. Choose from {list(model_map.keys())}.")
    
    # Instantiate the model with the given scale, in unfused state for QAT.
    # Note: `fused_init=False` is important for `prepare_qat`.
    return model_map[name](scale=scale, fused_init=False)

def calculate_psnr_ssim(preds: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> Tuple[float, float]:
    """
    Calculates PSNR and SSIM for a batch of images.
    
    Args:
        preds: Predicted high-resolution tensor (B, C, H, W).
        target: Ground truth high-resolution tensor (B, C, H, W).
        data_range: The range of the image data (e.g., 1.0 for [0, 1]).
        
    Returns:
        A tuple of (PSNR, SSIM) values.
    """
    psnr_metric = torchmetrics.image.PeakSignalNoiseRatio(data_range=data_range, reduction='mean').to(preds.device)
    ssim_metric = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=data_range, reduction='mean').to(preds.device)

    psnr = psnr_metric(preds, target).item()
    ssim = ssim_metric(preds, target).item()
    
    return psnr, ssim

# --- 2. Custom Dataset for Calibration and Validation ---

class SRDataset(Dataset):
    """
    A simple dataset to load low-resolution and high-resolution image pairs
    for calibration and validation.
    """
    def __init__(self, lr_dir: str, hr_dir: str, scale: int):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.scale = scale
        self.lr_images = sorted([f for f in os.listdir(lr_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.hr_images = sorted([f for f in os.listdir(hr_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        if len(self.lr_images) != len(self.hr_images):
            warnings.warn("Mismatch between LR and HR image counts. Ensure datasets are aligned.")

        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        lr_path = os.path.join(self.lr_dir, self.lr_images[idx])
        hr_path = os.path.join(self.hr_dir, self.hr_images[idx])
        
        lr_img = Image.open(lr_path).convert('RGB')
        hr_img = Image.open(hr_path).convert('RGB')
        
        # Check if HR image size is correct
        lr_w, lr_h = lr_img.size
        hr_w, hr_h = hr_img.size
        if hr_w != lr_w * self.scale or hr_h != lr_h * self.scale:
            warnings.warn(f"Size mismatch for image {self.lr_images[idx]}. HR size: {hr_img.size}, expected: ({lr_w*self.scale}, {lr_h*self.scale})")
        
        return self.to_tensor(lr_img), self.to_tensor(hr_img)

def create_dataloader(data_path: str, scale: int, batch_size: int, is_calibration: bool) -> DataLoader:
    """
    Creates a DataLoader from a dataset path containing 'LR' and 'HR' subfolders.
    """
    lr_dir = os.path.join(data_path, 'LR')
    hr_dir = os.path.join(data_path, 'HR')
    
    if not os.path.isdir(lr_dir) or not os.path.isdir(hr_dir):
        raise FileNotFoundError(f"Dataset path '{data_path}' must contain 'LR' and 'HR' subfolders.")
        
    dataset = SRDataset(lr_dir, hr_dir, scale)
    
    # Use a small batch size for calibration to avoid memory issues with large images
    # and to ensure a diverse set of examples.
    if is_calibration:
        batch_size = min(batch_size, 16)
        
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def validate_model(model: torch.nn.Module, dataloader: DataLoader, device: str, 
                   min_psnr: float, min_ssim: float) -> Tuple[bool, float, float]:
    """
    Validates a model's conversion quality using PSNR and SSIM.
    
    Returns:
        A tuple (success_flag, average_psnr, average_ssim).
    """
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for lr, hr in dataloader:
            lr, hr = lr.to(device), hr.to(device)
            
            # Predict SR image
            sr = model(lr)
            
            # Clamp outputs to the valid range for metrics
            sr = torch.clamp(sr, 0.0, 1.0)
            
            # Calculate metrics for the batch
            psnr, ssim = calculate_psnr_ssim(sr, hr, data_range=1.0)
            
            total_psnr += psnr
            total_ssim += ssim
            num_batches += 1
            
    avg_psnr = total_psnr / num_batches
    avg_ssim = total_ssim / num_batches
    
    print(f"  Validation Metrics: PSNR = {avg_psnr:.4f} dB, SSIM = {avg_ssim:.4f}")
    
    # Check if the model passes the quality thresholds
    if avg_psnr >= min_psnr and avg_ssim >= min_ssim:
        print("  Validation PASSED. Quality thresholds met.")
        return True, avg_psnr, avg_ssim
    else:
        print("  Validation FAILED. Quality thresholds NOT met.")
        return False, avg_psnr, avg_ssim


# --- 3. Main Conversion Logic ---

def convert_and_validate(
    model: torch.nn.Module,
    model_name: str,
    scale: int,
    precision: str,
    output_dir: str,
    validation_dataloader: DataLoader,
    calibration_dataloader: Optional[DataLoader],
    device: str,
    min_psnr: float,
    min_ssim: float,
    max_retries: int = 3
):
    """
    Main function to convert, validate, and save a model.
    Includes retry logic for failed validation.
    """
    
    # Define output file paths
    pth_path = os.path.join(output_dir, f"{model_name}_{precision}.pth")
    onnx_path = os.path.join(output_dir, f"{model_name}_{precision}.onnx")
    
    success = False
    attempt = 0
    
    while not success and attempt < max_retries:
        print(f"\n--- Attempt {attempt + 1}/{max_retries}: Converting model to {precision.upper()} ---")
        
        converted_model = None
        
        try:
            # --- Conversion Step ---
            if precision == 'fp32':
                converted_model = model.eval().to(device)
                converted_model.fuse_model() # Fuse layers for deployment
            
            elif precision == 'fp16':
                converted_model = model.half().eval().to(device)
                converted_model.fuse_model()
            
            elif precision == 'int8':
                if calibration_dataloader is None:
                    raise ValueError("Calibration dataset is required for INT8 conversion.")
                    
                # Re-instantiate the model to get a fresh, unfused version
                # This is crucial for PTQ where observers need to be inserted.
                model_for_quant = get_model_from_name(model_name, scale=scale).to(device)
                # Load the QAT-trained weights
                model_for_quant.load_state_dict(model.state_dict())
                model_for_quant.eval()
                
                # --- Post-Training Quantization (PTQ) ---
                print("  Preparing model for static quantization (PTQ)...")
                
                # Use fbgemm for x86 CPUs or qnnpack for ARM/mobile
                # We can also use 'x86' which is a recommended default in newer PyTorch versions.
                # Here we use a more general qconfig.
                model_for_quant.qconfig = tq.get_default_qconfig("fbgemm")
                
                # Prepare the model by inserting observers and fake quantizers.
                # `prepare` calls the `fuse_model` function inside `aether_core`.
                tq.prepare(model_for_quant, inplace=True)
                
                # --- Calibration ---
                print("  Running calibration to collect activation statistics...")
                with torch.no_grad():
                    for i, (lr, _) in enumerate(calibration_dataloader):
                        model_for_quant(lr.to(device))
                        if i >= 128: # Use a limited number of batches for calibration
                            break

                # --- Conversion to INT8 ---
                print("  Converting calibrated model to INT8...")
                converted_model = tq.convert(model_for_quant, inplace=False)
                # Move back to the target device
                converted_model = converted_model.to(device)
            
            else:
                raise ValueError(f"Unsupported precision: {precision}")

            # --- Validation Step ---
            print(f"  Validating converted {precision.upper()} model...")
            dummy_input_for_validation = torch.randn(1, 3, 64, 64).to(device)
            # Run a dummy forward pass to check for runtime errors
            _ = converted_model(dummy_input_for_validation)
            
            # Run full validation with PSNR and SSIM
            passes_validation, avg_psnr, avg_ssim = validate_model(
                converted_model, validation_dataloader, device, min_psnr, min_ssim
            )
            
            if passes_validation:
                success = True
                
                # --- Saving Step ---
                # Save as a PyTorch .pth model
                print("  Saving converted PyTorch model...")
                # For quantized models, save the state dict.
                if precision == 'int8':
                    torch.save(converted_model.state_dict(), pth_path)
                else:
                    # Save fused float model
                    torch.save(converted_model.state_dict(), pth_path)
                print(f"  Saved PyTorch model to: {pth_path}")

                # Save as an ONNX model
                print("  Exporting to ONNX...")
                # The `export_onnx` function in `aether_core.py` is called here.
                # It handles dynamic axes and is a key part of the pipeline.
                
                # Set up dynamic axes for ONNX export
                dummy_input_onnx = torch.randn(1, 3, 64, 64, device=device, dtype=torch.float16 if precision == 'fp16' else torch.float32)
                dynamic_axes = {
                    'input': {2: 'height', 3: 'width'},
                    'output': {2: 'height_out', 3: 'width_out'}
                }
                
                torch.onnx.export(
                    converted_model,
                    dummy_input_onnx,
                    onnx_path,
                    opset_version=17,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes=dynamic_axes,
                    export_params=True,
                    training=torch.onnx.TrainingMode.EVAL
                )
                print(f"  Saved ONNX model to: {onnx_path}")
                print(f"  Conversion to {precision.upper()} successful!")
                
            else:
                # Validation failed, try again
                print("  Validation failed. Re-attempting conversion...")
                attempt += 1

        except Exception as e:
            print(f"  An error occurred during {precision.upper()} conversion attempt {attempt + 1}: {e}")
            attempt += 1

    if not success:
        print(f"\n! Conversion to {precision.upper()} failed after {max_retries} attempts. Skipping this format.")

def main():
    """
    Main function to parse arguments and run the conversion pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Convert a trained AetherNet model to various deployment formats.",
        formatter_class=argparse.ArgumentDefaultsHelpiveHelpFormatter
    )
    
    # --- Required Arguments ---
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help="Path to the input QAT-trained PyTorch .pth model checkpoint."
    )
    parser.add_argument(
        '--output_folder',
        type=str,
        required=True,
        help="Path to the output folder where converted models will be saved."
    )
    parser.add_argument(
        '--network',
        type=str,
        choices=['aether_small', 'aether_medium', 'aether_large'],
        required=True,
        help="The AetherNet network type of the input model."
    )
    parser.add_argument(
        '--scale',
        type=int,
        required=True,
        help="The upscaling factor of the model (e.g., 2, 3, 4)."
    )
    parser.add_argument(
        '--validation_dataset_path',
        type=str,
        required=True,
        help="Path to a validation dataset for quality checks. Must contain 'LR' and 'HR' subfolders."
    )
    parser.add_argument(
        '--calibration_dataset_path',
        type=str,
        required=False,
        help="Path to a calibration dataset for INT8 quantization. Must contain 'LR' and 'HR' subfolders. (Required for INT8 conversion)."
    )
    
    # --- Optional Arguments ---
    parser.add_argument(
        '--min_psnr',
        type=float,
        default=28.0,
        help="Minimum PSNR threshold for a model to be considered valid."
    )
    parser.add_argument(
        '--min_ssim',
        type=float,
        default=0.75,
        help="Minimum SSIM threshold for a model to be considered valid."
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help="Batch size for validation and calibration."
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        choices=['cuda', 'cpu'],
        help="Device to use for conversion and validation."
    )
    
    args = parser.parse_args()

    # --- Setup ---
    print("--- Starting AetherNet Model Conversion Pipeline ---")
    print(f"Input model: {args.model_path}")
    print(f"Output folder: {args.output_folder}")
    print(f"Network: {args.network}, Scale: {args.scale}")
    print(f"Device: {args.device.upper()}")
    print("-" * 50)
    
    # Create output directory
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Load the trained model
    print("Loading the trained QAT model checkpoint...")
    
    # Instantiate the model in unfused state for QAT conversion to work correctly
    model = get_model_from_name(args.network, args.scale)
    
    try:
        # Load the checkpoint's state dictionary
        state_dict = torch.load(args.model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.to(args.device)
        model.eval()
        print("Model checkpoint loaded successfully.")
    except Exception as e:
        print(f"Error loading model from {args.model_path}: {e}")
        print("Please ensure the checkpoint corresponds to the specified network type and scale.")
        sys.exit(1)
        
    # --- Prepare DataLoaders ---
    print("\nPreparing validation and calibration datasets...")
    try:
        validation_dataloader = create_dataloader(args.validation_dataset_path, args.scale, args.batch_size, is_calibration=False)
        print(f"Validation dataset loaded from: {args.validation_dataset_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
        
    calibration_dataloader = None
    if args.calibration_dataset_path:
        try:
            calibration_dataloader = create_dataloader(args.calibration_dataset_path, args.scale, args.batch_size, is_calibration=True)
            print(f"Calibration dataset loaded from: {args.calibration_dataset_path}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("INT8 conversion will be skipped.")

    # --- Conversion Loop ---
    # Define a list of precisions to convert to
    precisions_to_convert = ['fp32', 'fp16']
    if calibration_dataloader is not None:
        precisions_to_convert.append('int8')
    
    for precision in precisions_to_convert:
        # Pass the original model (or a fresh copy for INT8) to the conversion function
        convert_and_validate(
            model=model,
            model_name=args.network,
            scale=args.scale,
            precision=precision,
            output_dir=args.output_folder,
            validation_dataloader=validation_dataloader,
            calibration_dataloader=calibration_dataloader,
            device=args.device,
            min_psnr=args.min_psnr,
            min_ssim=args.min_ssim
        )
        
    print("\n--- AetherNet Model Conversion Pipeline Finished ---")
    print(f"Check the '{args.output_folder}' directory for your optimized models.")

if __name__ == '__main__':
    # For demonstration, you would typically run this from the command line.
    # Example usage:
    # python export_aether.py --model_path /path/to/your/trained_aether_medium.pth --output_folder ./optimized_models --network aether_medium --scale 4 --validation_dataset_path /path/to/val_dataset --calibration_dataset_path /path/to/cal_dataset --device cuda
    main()
