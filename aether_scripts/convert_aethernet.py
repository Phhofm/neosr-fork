# This script converts a QAT-trained PyTorch AetherNet model into various
# optimized formats for deployment: a fused PyTorch .pth, FP32 ONNX,
# FP16 ONNX, and an INT8 ONNX.
#
# Usage Example:
# python convert_aethernet.py \
#     --input_pth_path "path/to/your/aether_small_qat_trained.pth" \
#     --output_dir "converted_models" \
#     --scale 2 \
#     --network aether_small \
#     --img_size 64 \
#     --dynamic_shapes # Add this flag for dynamic ONNX exports
#     --atol 1e-4 
#     --rtol 1e-3

import argparse
import os
import sys
import logging
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn

# Set flag to True so INT8 export is attempted
# NOTE: The quantize_qat API is deprecated in ONNX Runtime >= 1.22.0,
# but the exported ONNX model from PyTorch is already in the correct QDQ format.
try:
    from onnxruntime.quantization import QuantFormat, QuantType
    ONNX_RUNTIME_QUANTIZER_AVAILABLE = True
    # Suppress ONNX Runtime info logs
    logging.getLogger('onnxruntime').setLevel(logging.WARNING)
except ImportError:
    logging.warning("onnxruntime.quantization not found. INT8 ONNX export will be skipped.")
    ONNX_RUNTIME_QUANTIZER_AVAILABLE = False

# Conditional import for ONNX Runtime
try:
    import onnxruntime as ort
    ONNX_RUNTIME_AVAILABLE = True
except ImportError:
    ONNX_RUNTIME_AVAILABLE = False
    logging.warning("ONNX Runtime is not installed. Validation and inference will be skipped.")


# Ensure the 'common' directory (containing aether_core.py) is in sys.path
current_file_abs_path = Path(__file__).resolve()
# FIX: Get the parent of the current script's parent directory.
project_root_directory = current_file_abs_path.parent.parent
common_dir_path = project_root_directory / "common"

if str(common_dir_path) not in sys.path:
    sys.path.insert(0, str(common_dir_path))
    print(f"Added '{common_dir_path}' to sys.path for common modules.")

# Import the core AetherNet model from the common module
try:
    from aether_core import aether # Assumes aether_core.py is directly importable via sys.path
except ImportError:
    print(f"Error: Could not import 'aether' from 'aether_core'.")
    print(f"Please ensure 'aether_core.py' is in '{common_dir_path}' and that directory is correctly added to sys.path.")
    sys.exit(1)

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def validate_onnx_model(
    onnx_path: str,
    pytorch_output: np.ndarray,
    dummy_input_np: np.ndarray,
    atol: float,
    rtol: float,
) -> bool:
    """
    Validates an ONNX model by running inference and comparing the output with
    the PyTorch model's output using a specified tolerance.
    
    Args:
        onnx_path (str): Path to the ONNX model file.
        pytorch_output (np.ndarray): The numpy array output from the PyTorch model.
        dummy_input_np (np.ndarray): The numpy array of the dummy input.
        atol (float): Absolute tolerance for comparison.
        rtol (float): Relative tolerance for comparison.
    
    Returns:
        bool: True if the outputs are close, False otherwise.
    """
    if not ONNX_RUNTIME_AVAILABLE:
        logger.warning(f"ONNX Runtime is not available. Skipping validation for {onnx_path}.")
        return False

    logger.info(f"Validating {onnx_path}...")
    try:
        # Create an ONNX Runtime session
        ort_session = ort.InferenceSession(onnx_path, providers=ort.get_available_providers())
        
        # Get input and output names
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input_np}
        ort_outputs = ort_session.run(None, ort_inputs)
        
        # Get the first output from ONNX Runtime
        onnx_output = ort_outputs[0]
        
        # Compare outputs
        is_close = np.allclose(pytorch_output, onnx_output, atol=atol, rtol=rtol)
        
        if is_close:
            logger.info(f"✅ Validation Passed for {onnx_path}!")
            logger.info(f"   Max absolute difference: {np.max(np.abs(pytorch_output - onnx_output)):.6f}")
            logger.info(f"   Used tolerances: atol={atol}, rtol={rtol}")
        else:
            logger.error(f"❌ Validation FAILED for {onnx_path}!")
            max_abs_diff = np.max(np.abs(pytorch_output - onnx_output))
            logger.error(f"   Max absolute difference: {max_abs_diff:.6f}")
            logger.error(f"   Used tolerances: atol={atol}, rtol={rtol}")
            logger.error(f"   This difference exceeds the allowed tolerance. The model outputs are not close enough.")
            
        return is_close
        
    except Exception as e:
        logger.error(f"Error during validation of {onnx_path}: {e}")
        return False


# --- Main Conversion Function ---
def convert_model(
    input_pth_path: str,
    output_dir: str,
    scale: int,
    network_type: str,
    img_size: int,
    dynamic_shapes: bool,
    opset_version: int,
    fp_mode: str,
    min_batch_size: int,
    opt_batch_size: int,
    max_batch_size: int,
    min_height: int,
    opt_height: int,
    max_height: int,
    min_width: int,
    opt_width: int,
    max_width: int,
    img_range: float,
    atol: float,
    rtol: float,
) -> None:
    """
    Converts a QAT-trained PyTorch AetherNet model to various release-ready formats:
    Fused PyTorch .pth, FP32 ONNX, FP16 ONNX.

    Args:
        input_pth_path (str): Path to the input PyTorch .pth checkpoint file.
        output_dir (str): Directory to save all exported models.
        scale (int): Upscale factor (e.g., 2, 3, 4).
        network_type (str): Type of AetherNet model ('aether_small', 'aether_medium', 'aether_large').
        img_size (int): Input image size (H or W) for dummy input.
        dynamic_shapes (bool): If True, export ONNX with dynamic batch, height, and width.
        opset_version (int): ONNX opset version for export.
        fp_mode (str): Floating-point precision for ONNX export ('fp32' or 'fp16').
        min_batch_size (int): Minimum batch size for dynamic ONNX.
        opt_batch_size (int): Optimal batch size for dynamic ONNX.
        max_batch_size (int): Maximum batch size for dynamic ONNX.
        min_height (int): Minimum input height for dynamic ONNX.
        opt_height (int): Optimal input height for dynamic ONNX.
        max_height (int): Maximum input height for dynamic ONNX.
        min_width (int): Minimum input width for dynamic ONNX.
        opt_width (int): Optimal input width for dynamic ONNX.
        max_width (int): Maximum input width for dynamic ONNX.
        img_range (float): The maximum pixel value range (e.g., 1.0 for [0,1] input).
        atol (float): Absolute tolerance for output comparison.
        rtol (float): Relative tolerance for output comparison.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Load PyTorch Model ---
    logger.info(f"Loading PyTorch model: {network_type} from {input_pth_path}")

    # Map network_type to AetherNet parameters
    network_configs = {
        'aether_small': {'embed_dim': 96, 'depths': (4, 4, 4, 4), 'mlp_ratio': 2.0, 'lk_kernel': 11, 'sk_kernel': 3},
        'aether_medium': {'embed_dim': 128, 'depths': (6, 6, 6, 6, 6, 6), 'mlp_ratio': 2.0, 'lk_kernel': 11, 'sk_kernel': 3},
        'aether_large': {'embed_dim': 180, 'depths': (8, 8, 8, 8, 8, 8, 8, 8), 'mlp_ratio': 2.5, 'lk_kernel': 13, 'sk_kernel': 3},
    }

    config = network_configs.get(network_type)
    if not config:
        logger.error(f"Unknown network type: {network_type}")
        sys.exit(1)

    # Instantiate the AetherNet model in unfused_init=False mode first to load weights correctly,
    # then fuse it explicitly. Or, if the checkpoint itself represents a fused model,
    # it can be loaded directly with fused_init=True. For general case, loading unfused then fusing is safer.
    model = aether(
        in_chans=3, # Assuming RGB input
        scale=scale,
        img_range=img_range,
        fused_init=False, # Initialize as unfused for loading, then fuse below
        **config
    )
    model.eval() # Set model to evaluation mode

    # Load the state dictionary
    checkpoint = torch.load(input_pth_path, map_location='cpu')
    
    # Handle various common checkpoint structures (e.g., from neosr or raw state_dict)
    model_state_dict = None
    if 'net_g' in checkpoint:
        if isinstance(checkpoint['net_g'], dict): # neosr's model wrapper
            model_state_dict = checkpoint['net_g']
            logger.info("Loaded state_dict from 'net_g' key in checkpoint.")
        else: # If 'net_g' is the model object itself
            model_state_dict = checkpoint['net_g'].state_dict()
            logger.info("Loaded state_dict from 'net_g' model object in checkpoint.")
    elif 'params' in checkpoint: # Common for some PyTorch/BasicSR checkpoints
        model_state_dict = checkpoint['params']
        logger.info("Loaded state_dict from 'params' key in checkpoint.")
    else: # Assume the checkpoint itself is the state_dict
        model_state_dict = checkpoint
        logger.info("Loaded raw state_dict from checkpoint.")

    # Remove 'module.' prefix if it exists (for DataParallel trained models)
    if any(k.startswith('module.') for k in model_state_dict.keys()):
        model_state_dict = {k[7:]: v for k, v in model_state_dict.items()}
        logger.info("Removed 'module.' prefix from state_dict keys.")

    model.load_state_dict(model_state_dict, strict=True)
    logger.info("PyTorch model weights loaded successfully.")


    # --- 2. Fuse Model Layers for Inference ---
    # This ensures the ReparamLargeKernelConv modules are converted to single convs.
    model.fuse_model()
    model.cpu() # Ensure model is on CPU for ONNX export for better compatibility


    # --- 3. Save Fused PyTorch Model (.pth) ---
    fused_pth_path = os.path.join(output_dir, f"{network_type}_{scale}x_fused.pth")
    try:
        torch.save(model.state_dict(), fused_pth_path)
        logger.info(f"Fused PyTorch model saved to {fused_pth_path}")
    except Exception as e:
        logger.error(f"Error saving fused PyTorch model: {e}")


    # --- 4. Prepare Dummy Input for ONNX Export ---
    # This dummy input represents the *optimal* shape for tracing the ONNX graph.
    # Dynamic axes will handle the min/max shapes.
    dummy_input = torch.randn(opt_batch_size, 3, opt_height, opt_width, dtype=torch.float32)

    # Define dynamic axes for ONNX export
    onnx_dynamic_axes = {
        'input': {0: 'batch_size', 2: 'height', 3: 'width'},
        'output': {0: 'batch_size', 2: 'height', 3: 'width'}
    } if dynamic_shapes else None

    # Get the PyTorch model's output to use as the ground truth for validation
    with torch.no_grad():
        pytorch_output = model(dummy_input).numpy()
    dummy_input_np = dummy_input.numpy()

    # --- 5. Export to FP32 ONNX ---
    fp32_onnx_path = os.path.join(output_dir, f"{network_type}_{scale}x_fp32_int8.onnx")
    logger.info(f"Exporting FP32 ONNX model to {fp32_onnx_path}")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            fp32_onnx_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=onnx_dynamic_axes,
            training=torch.onnx.TrainingMode.EVAL # Critical for ensuring correct QAT node behavior
        )
        logger.info("FP32 ONNX model exported successfully.")
        
        # Validate the exported FP32 ONNX model
        validate_onnx_model(fp32_onnx_path, pytorch_output, dummy_input_np, atol=atol, rtol=rtol)

    except Exception as e:
        logger.error(f"Error exporting FP32 ONNX model: {e}")
        sys.exit(1) # Cannot proceed if FP32 ONNX fails


    # --- 6. Export to FP16 ONNX ---
    fp16_onnx_path = os.path.join(output_dir, f"{network_type}_{scale}x_fp16.onnx")
    logger.info(f"Exporting FP16 ONNX model to {fp16_onnx_path}")
    try:
        model_fp16 = model.half() # Convert model to FP16 precision
        dummy_input_fp16 = dummy_input.half() # Match dummy input dtype
        torch.onnx.export(
            model_fp16,
            dummy_input_fp16,
            fp16_onnx_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=onnx_dynamic_axes,
            training=torch.onnx.TrainingMode.EVAL
        )
        logger.info("FP16 ONNX model exported successfully.")
        
        # Validate the exported FP16 ONNX model.
        # Note: FP16 validation might require different tolerances.
        validate_onnx_model(fp16_onnx_path, pytorch_output, dummy_input_np.astype(np.float16), atol=atol*10, rtol=rtol*10)
        
    except Exception as e:
        logger.warning(f"Error exporting FP16 ONNX model: {e}. Skipping FP16 ONNX export.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a QAT-trained AetherNet PyTorch model to Fused PTH, FP32/FP16/INT8 ONNX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help
    )
    
    # Required Arguments
    parser.add_argument("--input_pth_path", type=str, required=True,
                        help="Path to the input QAT-trained PyTorch .pth checkpoint file.")
    parser.add_argument("--output_dir", type=str, default="converted_aethernet_models",
                        help="Directory to save all exported models (Fused PTH, ONNX).")
    
    # Model Configuration Arguments
    parser.add_argument("--scale", type=int, required=True,
                        help="Upscale factor of the model (e.g., 2, 3, 4).")
    parser.add_argument("--network", type=str, required=True,
                        choices=['aether_small', 'aether_medium', 'aether_large'],
                        help="Type of AetherNet model to convert.")
    parser.add_argument("--img_size", type=int, default=64, # This is for dummy input height/width
                        help="Input image size (height and width) for dummy input for ONNX tracing. "
                             "This should correspond to a typical patch size or smallest expected input.")
    parser.add_argument("--img_range", type=float, default=1.0,
                        help="Pixel value range (e.g., 1.0 for [0,1] input). Should match training.")

    # ONNX Export Configuration
    parser.add_argument("--dynamic_shapes", action='store_true',
                        help="If set, export ONNX with dynamic batch size, height, and width.")
    parser.add_argument("--opset_version", type=int, default=17,
                        help="ONNX opset version for export.")
    parser.add_argument("--fp_mode", type=str, default="fp32", choices=["fp32", "fp16"],
                        help="Floating-point precision for ONNX model export (fp32 or fp16).")
    
    # Dynamic Shape Specific Arguments (if --dynamic_shapes is used)
    parser.add_argument("--min_batch_size", type=int, default=1,
                        help="Minimum batch size for dynamic ONNX export.")
    parser.add_argument("--opt_batch_size", type=int, default=1,
                        help="Optimal batch size for dynamic ONNX export.")
    parser.add_argument("--max_batch_size", type=int, default=4,
                        help="Maximum batch size for dynamic ONNX export.")
    parser.add_argument("--min_height", type=int, default=32,
                        help="Minimum input height for dynamic ONNX export.")
    parser.add_argument("--opt_height", type=int, default=64,
                        help="Optimal input height for dynamic ONNX export.")
    parser.add_argument("--max_height", type=int, default=256,
                        help="Maximum input height for dynamic ONNX export.")
    parser.add_argument("--min_width", type=int, default=32,
                        help="Minimum input width for dynamic ONNX export.")
    parser.add_argument("--opt_width", type=int, default=64,
                        help="Optimal input width for dynamic ONNX export.")
    parser.add_argument("--max_width", type=int, default=256,
                        help="Maximum input width for dynamic ONNX export.")

    # Validation Arguments
    parser.add_argument("--atol", type=float, default=1e-5,
                        help="Absolute tolerance for validating ONNX outputs against PyTorch.")
    parser.add_argument("--rtol", type=float, default=1e-4,
                        help="Relative tolerance for validating ONNX outputs against PyTorch.")


    args = parser.parse_args()

    # --- Argument Validation and Pre-checks ---
    if not os.path.exists(args.input_pth_path):
        logger.error(f"Input PyTorch model path '{args.input_pth_path}' does not exist.")
        sys.exit(1)

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)


    logger.info(f"--- AetherNet Model Conversion Script ---")
    logger.info(f"Input PyTorch Model: {args.input_pth_path}")
    logger.info(f"Output Directory: {args.output_dir}")
    logger.info(f"Network Type: {args.network}, Upscale Factor: {args.scale}x")
    logger.info(f"ONNX Dynamic Shapes: {args.dynamic_shapes}")
    logger.info(f"ONNX Opset Version: {args.opset_version}")
    logger.info(f"ONNX Floating Point Mode: {args.fp_mode}")
    logger.info(f"Validation Tolerances: atol={args.atol}, rtol={args.rtol}")

    convert_model(
        input_pth_path=args.input_pth_path,
        output_dir=args.output_dir,
        scale=args.scale,
        network_type=args.network,
        img_size=args.img_size, # Used as fixed H/W for dummy input if not dynamic, or opt H/W if dynamic
        dynamic_shapes=args.dynamic_shapes,
        opset_version=args.opset_version,
        fp_mode=args.fp_mode,
        min_batch_size=args.min_batch_size,
        opt_batch_size=args.opt_batch_size,
        max_batch_size=args.max_batch_size,
        min_height=args.min_height,
        opt_height=args.opt_height,
        max_height=args.max_height,
        min_width=args.min_width,
        opt_width=args.opt_width,
        max_width=args.max_width,
        img_range=args.img_range,
        atol=args.atol,
        rtol=args.rtol,
    )

    logger.info("\nConversion process completed. Generated files are in the output directory.")
    logger.info("You can now use these ONNX files to build TensorRT engines.")


if __name__ == "__main__":
    main()