# This script converts a QAT-trained PyTorch AetherNet model into various
# optimized formats for deployment: a fused PyTorch .pth, FP32 ONNX,
# FP16 ONNX, and an INT8 ONNX, as well as FP16 and INT8 fused PyTorch .pth models.
#
# Usage Example:
# python convert_aethernet.py \
#     --input_pth_path "path/to/your/aether_small_qat_trained.pth" \
#     --output_dir "converted_models" \
#     --scale 2 \
#     --network aether_small \
#     --img_size 64 \
#     --calibration_data_dir "/path/to/your/training_images/LR" \
#     --calibration_dataset_size 100
#     # --dynamic_shapes is now default (True)
#     # Add --static flag for static shapes
#     # --atol and --rtol now have default values

import argparse
import os
import sys
import logging
from pathlib import Path
import numpy as np
import copy # Import copy module for deepcopy
import glob # For listing image files
from PIL import Image # For image loading and preprocessing

import torch
import torch.nn as nn

# Set flag to True so INT8 export is attempted
# NOTE: The quantize_qat API is deprecated in ONNX Runtime >= 1.22.0,
# but the exported ONNX model from PyTorch is already in the correct QDQ format.
try:
    # Corrected import: CalibrationDataReader needs to be explicitly imported
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static, CalibrationDataReader
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
# This assumes a structure like: project_root/scripts/convert_aethernet.py and project_root/common/aether_core.py
project_root_directory = current_file_abs_path.parent.parent
common_dir_path = project_root_directory / "common"

# If the script is run from the same directory as aether_core.py, adjust the path
if not (common_dir_path / "aether_core.py").exists() and (current_file_abs_path.parent / "aether_core.py").exists():
    common_dir_path = current_file_abs_path.parent

if str(common_dir_path) not in sys.path:
    sys.path.insert(0, str(common_dir_path))
    print(f"Added '{common_dir_path}' to sys.path for common modules.")

# Import the core AetherNet model from the common module
try:
    from aether_core import aether # Assumes aether_core.py is directly importable via sys.path
    # Import quantization utilities used by aether_core.py
    import torch.ao.quantization as tq
    from torch.ao.quantization.observer import MovingAverageMinMaxObserver
except ImportError as e:
    print(f"Error: Could not import 'aether' from 'aether_core' or quantization utilities. Details: {e}")
    print(f"Please ensure 'aether_core.py' is in '{common_dir_path}' and that directory is correctly added to sys.path.")
    sys.exit(1)

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class CustomCalibrationDataReader(CalibrationDataReader):
    """
    A calibration data reader for ONNX Runtime static quantization that loads
    actual images from a specified directory. This can also be adapted for
    PyTorch's native quantization calibration.
    """
    def __init__(self, lr_image_dir: str, target_height: int, target_width: int,
                 img_range: float, calibration_dataset_size: int, input_name: str = "input"):
        self.image_paths = []
        # Support common image extensions
        for ext in ['png', 'jpg', 'jpeg', 'bmp', 'tif']:
            self.image_paths.extend(glob.glob(os.path.join(lr_image_dir, '**', f'*.{ext}'), recursive=True))

        if not self.image_paths:
            raise ValueError(f"No images found in {lr_image_dir} for calibration. Please check the path and image types.")

        # Shuffle and select a subset for calibration if the dataset is large
        if len(self.image_paths) > calibration_dataset_size:
            np.random.seed(42) # For reproducibility
            np.random.shuffle(self.image_paths)
            self.image_paths = self.image_paths[:calibration_dataset_size]
            logger.info(f"Randomly selected {len(self.image_paths)} images for calibration from {lr_image_dir}.")
        else:
            logger.info(f"Using all {len(self.image_paths)} images found in {lr_image_dir} for calibration.")


        self.input_name = input_name
        self.target_height = target_height
        self.target_width = target_width
        self.img_range = img_range
        # Initialize the generator directly, it will be iterated by get_next()
        self.data_generator = self._preprocess_data()
        self.current_batch = None # To store the last yielded batch for rewind

    def _preprocess_image(self, img_path: str) -> np.ndarray:
        """Loads and preprocesses a single image."""
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((self.target_width, self.target_height), Image.BICUBIC) # Resize
            img_np = np.array(img).astype(np.float32)

            # Normalize pixel values
            if self.img_range == 1.0:
                img_np /= 255.0
            # If img_range is 255.0, assume input is already [0, 255] and no division needed
            # Add other normalization schemes if your model expects them (e.g., mean/std)

            # Convert HWC to CHW
            img_np = np.transpose(img_np, (2, 0, 1))
            # Add batch dimension (for a single image calibration)
            img_np = np.expand_dims(img_np, axis=0)
            return img_np
        except Exception as e:
            logger.error(f"Error processing image {img_path}: {e}")
            return None # Return None or raise an error as appropriate

    def _preprocess_data(self):
        """Generator to preprocess and yield data dictionaries."""
        for img_path in self.image_paths:
            processed_img = self._preprocess_image(img_path)
            if processed_img is not None:
                yield {self.input_name: processed_img}

    def get_next(self):
        """Returns the next calibration data sample. Stores it for rewind."""
        try:
            self.current_batch = next(self.data_generator)
            return self.current_batch
        except StopIteration:
            self.current_batch = None # No more data
            return None

    def rewind(self):
        """Rewinds the data reader to the beginning."""
        self.data_generator = self._preprocess_data()
        self.current_batch = None # Reset current batch


class DummyCalibrationDataReader(CalibrationDataReader):
    """
    A dummy calibration data reader for ONNX Runtime static quantization.
    Used as a fallback if no real calibration data directory is provided.
    """
    def __init__(self, data: np.ndarray, input_name: str = "input"):
        self.data_for_calibration = [{input_name: data}] # Expects data to be a single preprocessed numpy array
        self.enum_data = iter(self.data_for_calibration)
        self.data_len = len(self.data_for_calibration)

    def get_next(self):
        try:
            return next(self.enum_data)
        except StopIteration:
            return None

    def rewind(self):
        self.enum_data = iter(self.data_for_calibration)


def validate_onnx_model(
    onnx_path: str,
    pytorch_output_np: np.ndarray, # Reference output from PyTorch
    dummy_input_np: np.ndarray,
    atol: float,
    rtol: float,
) -> bool:
    """
    Validates an ONNX model by running inference and comparing the output with
    the PyTorch model's output using a specified tolerance.

    Args:
        onnx_path (str): Path to the ONNX model file.
        pytorch_output_np (np.ndarray): The numpy array output from the original PyTorch model.
        dummy_input_np (np.ndarray): The numpy array of the dummy input for ONNX.
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
        session_options = ort.SessionOptions()
        # session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        ort_session = ort.InferenceSession(onnx_path, sess_options=session_options, providers=ort.get_available_providers())

        # Get input and output names
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input_np}
        ort_outputs = ort_session.run(None, ort_inputs)

        # Get the first output from ONNX Runtime
        onnx_output = ort_outputs[0]

        # Compare outputs
        is_close = np.allclose(pytorch_output_np, onnx_output, atol=atol, rtol=rtol)

        if is_close:
            logger.info(f"✅ Validation Passed for {onnx_path}!")
            logger.info(f"   Max absolute difference: {np.max(np.abs(pytorch_output_np - onnx_output)):.6f}")
            logger.info(f"   Used tolerances: atol={atol}, rtol={rtol}")
        else:
            logger.error(f"❌ Validation FAILED for {onnx_path}!")
            max_abs_diff = np.max(np.abs(pytorch_output_np - onnx_output))
            logger.error(f"   Max absolute difference: {max_abs_diff:.6f}")
            logger.error(f"   Used tolerances: atol={atol}, rtol={rtol}")
            logger.error(f"   This difference exceeds the allowed tolerance. The model outputs are not close enough.")

        return is_close

    except Exception as e:
        logger.error(f"Error during validation of {onnx_path}: {e}")
        return False


def validate_pytorch_model(
    model_to_validate: torch.nn.Module,
    reference_output_np: np.ndarray,
    dummy_input: torch.Tensor,
    atol: float,
    rtol: float,
    model_name: str
) -> bool:
    """
    Validates a PyTorch model by running inference and comparing the output with
    a reference output using specified tolerances.

    Args:
        model_to_validate (torch.nn.Module): The PyTorch model to validate.
        reference_output_np (np.ndarray): The numpy array output from the original FP32 PyTorch model.
        dummy_input (torch.Tensor): The dummy input tensor for inference.
        atol (float): Absolute tolerance for comparison.
        rtol (float): Relative tolerance for comparison.
        model_name (str): A descriptive name for the model being validated (e.g., "FP16 Fused PyTorch").

    Returns:
        bool: True if the outputs are close, False otherwise.
    """
    logger.info(f"Validating PyTorch model: {model_name}...")
    original_device = dummy_input.device # Store original device

    try:
        model_to_validate.eval() # Ensure eval mode for consistent behavior

        # Force INT8 Fused PyTorch model to CPU for validation to bypass CUDA-specific quantized op issues.
        # For other models (FP32, FP16 PyTorch), attempt CUDA if available, else use CPU.
        if "INT8 Fused PyTorch" in model_name:
            logger.info(f"Forcing {model_name} validation to CPU to avoid potential CUDA quantized operator issues.")
            model_to_validate = model_to_validate.cpu()
            dummy_input = dummy_input.cpu()
        elif torch.cuda.is_available():
            try:
                model_to_validate = model_to_validate.cuda()
                dummy_input = dummy_input.cuda()
                logger.info(f"Moved {model_name} and input to CUDA for validation.")
            except Exception as e:
                logger.warning(f"Could not move {model_to_validate} or input to CUDA: {e}. Attempting CPU validation.")
                model_to_validate = model_to_validate.cpu() # Fallback to CPU
                dummy_input = dummy_input.cpu()
        else:
            logger.info(f"CUDA not available. Validating {model_name} on CPU.")
            model_to_validate = model_to_validate.cpu() # Ensure on CPU
            dummy_input = dummy_input.cpu()


        with torch.no_grad():
            output_tensor = model_to_validate(dummy_input)
            # Ensure output is float32 for comparison, as quantized model's output might be float
            # and move to CPU for numpy conversion
            output_np = output_tensor.float().cpu().numpy()

        is_close = np.allclose(reference_output_np, output_np, atol=atol, rtol=rtol)

        if is_close:
            logger.info(f"✅ Validation Passed for {model_name}!")
            logger.info(f"   Max absolute difference: {np.max(np.abs(reference_output_np - output_np)):.6f}")
            logger.info(f"   Used tolerances: atol={atol}, rtol={rtol}")
        else:
            logger.error(f"❌ Validation FAILED for {model_name}!")
            max_abs_diff = np.max(np.abs(reference_output_np - output_np))
            logger.error(f"   Max absolute difference: {max_abs_diff:.6f}")
            logger.error(f"   Used tolerances: atol={atol}, rtol={rtol}")
            logger.error(f"   This difference exceeds the allowed tolerance. The model outputs are not close enough.")

        return is_close

    except Exception as e:
        logger.error(f"Error during validation of PyTorch model {model_name}: {e}")
        logger.error("This often indicates missing operator implementations for the specific PyTorch build/backend "
                     "(e.g., 'QuantizedCPU' or 'QuantizedCUDA'). Consider reinstalling PyTorch from official channels "
                     "or checking your PyTorch build configuration for quantization support.")
        return False
    finally:
        # Move model and input back to original device (CPU in this script's context)
        # Only move back if it was moved from original_device (CPU) to CUDA
        if "INT8 Fused PyTorch" not in model_name and torch.cuda.is_available() and str(original_device) == 'cpu':
            try:
                model_to_validate.to(original_device)
                dummy_input.to(original_device)
            except Exception as e:
                logger.warning(f"Could not move model or input back to original device: {e}")


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
    calibration_data_dir: str, # New argument
    calibration_dataset_size: int, # New argument
) -> None:
    """
    Converts a QAT-trained PyTorch AetherNet model to various release-ready formats:
    Fused PyTorch .pth (FP32, FP16, INT8), FP32 ONNX, FP16 ONNX, and INT8 ONNX.

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
        rtol (float): Relative tolerance for comparison.
        calibration_data_dir (str): Path to the directory containing low-resolution images for INT8 ONNX calibration.
        calibration_dataset_size (int): Number of images to use for INT8 ONNX calibration.
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
    # then fuse it explicitly.
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


    # --- 2. Fuse Model Layers for Inference (ReparamLargeKernelConv fusion) ---
    # This ensures the ReparamLargeKernelConv modules are converted to single convs.
    model.fuse_model()
    model.cpu() # Ensure model is on CPU for ONNX export and initial dummy input processing


    # --- 3. Prepare Dummy Input for ONNX Export and PyTorch Validation ---
    # This dummy input represents the *optimal* shape for tracing the ONNX graph
    # and for validating PyTorch models.
    dummy_input_fp32 = torch.randn(opt_batch_size, 3, opt_height, opt_width, dtype=torch.float32)

    # Get the original FP32 PyTorch model's output to use as the ground truth for validation
    with torch.no_grad():
        pytorch_output_fp32_np = model(dummy_input_fp32).numpy()
    dummy_input_fp32_np = dummy_input_fp32.numpy()


    # --- 4. Save FP32 Fused PyTorch Model (.pth) ---
    fused_pth_path = os.path.join(output_dir, f"{network_type}_{scale}x_fp32_fused.pth")
    try:
        torch.save(model.state_dict(), fused_pth_path)
        logger.info(f"Fused FP32 PyTorch model saved to {fused_pth_path}")
        # Validate the FP32 fused model against itself (should pass with very high accuracy)
        validate_pytorch_model(
            model,
            pytorch_output_fp32_np,
            dummy_input_fp32,
            atol=1e-7, # Very strict tolerance
            rtol=1e-6, # Very strict tolerance
            model_name="FP32 Fused PyTorch"
        )
    except Exception as e:
        logger.error(f"Error saving or validating FP32 fused PyTorch model: {e}")


    # --- 5. Save FP16 Fused PyTorch Model (.pth) ---
    logger.info("--- Exporting FP16 Fused PyTorch Model ---")
    # Create a deep copy of the model to convert to FP16, leaving the original FP32 model intact
    fp16_model = copy.deepcopy(model).half()
    fp16_fused_pth_path = os.path.join(output_dir, f"{network_type}_{scale}x_fp16_fused.pth")
    try:
        torch.save(fp16_model.state_dict(), fp16_fused_pth_path)
        logger.info(f"FP16 Fused PyTorch model saved to {fp16_fused_pth_path}")

        # Validate FP16 PyTorch model
        # Dummy input needs to be in FP16 for validation with the FP16 model
        dummy_input_fp16 = dummy_input_fp32.half()
        validate_pytorch_model(
            fp16_model,
            pytorch_output_fp32_np, # Compare against original FP32 output
            dummy_input_fp16,
            atol=atol * 20, # Loosen tolerance for FP16 comparison
            rtol=rtol * 20, # Loosen tolerance for FP16 comparison
            model_name="FP16 Fused PyTorch"
        )
    except Exception as e:
        logger.error(f"Error saving or validating FP16 fused PyTorch model: {e}")


    # --- 6. Save INT8 Fused PyTorch Model (.pth) ---
    logger.info("--- Exporting INT8 Fused PyTorch Model ---")
    try:
        # Create a deep copy of the model for INT8 conversion.
        # Explicitly ensure it's in float32 before preparing for QAT.
        model_for_int8_conversion = copy.deepcopy(model).float()
        # Ensure the model is in eval mode before preparing for conversion
        model_for_int8_conversion.eval()

        # Set default QConfig for PyTorch's native quantization if not already set.
        if not hasattr(model_for_int8_conversion, 'qconfig') or model_for_int8_conversion.qconfig is None:
            model_for_int8_conversion.qconfig = tq.get_default_qconfig('fbgemm')
            logger.info("Set default QConfig for INT8 PyTorch model conversion.")

        # Call prepare_qat to instrument the model and set qconfig
        dummy_qat_opt = {'use_amp': False, 'bfloat16': False}
        model_for_int8_conversion.prepare_qat(dummy_qat_opt)
        logger.info("Model prepared for quantization.")

        # --- Calibration for PyTorch INT8 Model ---
        # Use CustomCalibrationDataReader for PyTorch INT8 model calibration if path is provided
        if calibration_data_dir and os.path.isdir(calibration_data_dir):
            logger.info("Performing calibration for PyTorch INT8 model using provided real data.")
            # Create a separate instance of CustomCalibrationDataReader for PyTorch calibration
            # This ensures multiple passes if needed and isolation from ONNX calibration reader.
            pytorch_calibration_reader = CustomCalibrationDataReader(
                lr_image_dir=calibration_data_dir,
                target_height=opt_height,
                target_width=opt_width,
                img_range=img_range,
                calibration_dataset_size=calibration_dataset_size,
                input_name="input" # Matches dummy_input expected name
            )
            # Iterate through calibration data to allow observers to collect statistics
            sample_count = 0
            while True:
                data_batch = pytorch_calibration_reader.get_next()
                if data_batch is None:
                    break
                # Convert numpy array from data_batch to torch tensor for inference
                input_tensor = torch.from_numpy(data_batch["input"]).float()
                with torch.no_grad():
                    _ = model_for_int8_conversion(input_tensor)
                sample_count += 1
            logger.info(f"Performed {sample_count} inference passes for PyTorch INT8 model calibration.")
            if sample_count == 0:
                logger.warning("No calibration samples processed for PyTorch INT8 model. "
                               "Ensure calibration_data_dir contains valid images and parameters are correct.")
        else:
            logger.warning("No valid --calibration_data_dir provided for PyTorch INT8 model calibration. "
                           "Falling back to dummy input for calibration. This may lead to suboptimal INT8 model accuracy.")
            # Fallback to dummy inference pass if no real data dir is provided
            with torch.no_grad():
                _ = model_for_int8_conversion(dummy_input_fp32)
            logger.info("Performed dummy inference pass to collect observer statistics for INT8 conversion (PyTorch).")


        # Convert the prepared model to a truly quantized (INT8) model
        int8_model = model_for_int8_conversion.convert_to_quantized()
        logger.info("Model converted to INT8.")

        int8_fused_pth_path = os.path.join(output_dir, f"{network_type}_{scale}x_int8_fused.pth")
        torch.save(int8_model.state_dict(), int8_fused_pth_path)
        logger.info(f"INT8 Fused PyTorch model saved to {int8_fused_pth_path}")

        # Validate INT8 PyTorch model
        # The INT8 model's forward pass expects float32 input (handles quant/dequant internally)
        validate_pytorch_model(
            int8_model,
            pytorch_output_fp32_np, # Compare against original FP32 output
            dummy_input_fp32,       # Use original FP32 dummy input for the INT8 model
            atol=atol * 50, # Significantly loosen tolerance for INT8 due to precision loss
            rtol=rtol * 50, # Significantly loosen tolerance for INT8 due to precision loss
            model_name="INT8 Fused PyTorch"
        )
    except Exception as e:
        logger.error(f"Error saving or validating INT8 fused PyTorch model: {e}")
        logger.warning("Note: INT8 PyTorch model export requires the input .pth to have been trained "
                       "with Quantization-Aware Training (QAT) or proper Post-Training Quantization steps. "
                       "If you encounter errors during INT8 conversion, ensure your input model is QAT-compatible, "
                       "or consider PyTorch's native Post-Training Dynamic/Static Quantization workflows.")


    # --- 7. Export to FP32 ONNX --- (Original ONNX export logic, modified to use new dummy input)
    onnx_dynamic_axes = {
        'input': {0: 'batch_size', 2: 'height', 3: 'width'},
        'output': {0: 'batch_size', 2: 'height', 3: 'width'}
    } if dynamic_shapes else None

    # Use a fresh deep copy of the model for ONNX export to ensure it's exactly FP32
    fp32_onnx_export_model = copy.deepcopy(model).float() # Ensure float32 for export
    fp32_onnx_path = os.path.join(output_dir, f"{network_type}_{scale}x_fp32_int8.onnx")
    logger.info(f"Exporting FP32 ONNX model to {fp32_onnx_path}")
    try:
        torch.onnx.export(
            fp32_onnx_export_model, # Use the fresh FP32 model for ONNX export
            dummy_input_fp32,
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
        validate_onnx_model(
            fp32_onnx_path,
            pytorch_output_fp32_np,
            dummy_input_fp32_np,
            atol=atol,
            rtol=rtol
        )

    except Exception as e:
        logger.error(f"Error exporting FP32 ONNX model: {e}")
        sys.exit(1) # Cannot proceed if FP32 ONNX fails


    # --- 8. Export to FP16 ONNX --- (Original ONNX export logic, modified to use new dummy input)
    fp16_onnx_path = os.path.join(output_dir, f"{network_type}_{scale}x_fp16.onnx")
    logger.info(f"Exporting FP16 ONNX model to {fp16_onnx_path}")
    try:
        # Create a deep copy and convert to FP16 for this specific ONNX export
        model_fp16_onnx = copy.deepcopy(model).half()
        dummy_input_fp16_onnx = dummy_input_fp32.half() # Match dummy input dtype
        torch.onnx.export(
            model_fp16_onnx,
            dummy_input_fp16_onnx,
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
        validate_onnx_model(
            fp16_onnx_path,
            pytorch_output_fp32_np, # Compare against original FP32 output
            dummy_input_fp16_onnx.numpy(),
            atol=atol * 20, # Increased tolerance for FP16 comparison
            rtol=rtol * 20  # Increased tolerance for FP16 comparison
        )

    except Exception as e:
        logger.warning(f"Error exporting FP16 ONNX model: {e}. Skipping FP16 ONNX export.")


    # --- 9. Export to INT8 ONNX --- (NEW)
    if ONNX_RUNTIME_QUANTIZER_AVAILABLE:
        int8_onnx_path = os.path.join(output_dir, f"{network_type}_{scale}x_int8.onnx")
        logger.info(f"Exporting INT8 ONNX model to {int8_onnx_path}")
        try:
            # Determine which calibration data reader to use for ONNX
            if calibration_data_dir and os.path.isdir(calibration_data_dir):
                onnx_calibration_reader = CustomCalibrationDataReader(
                    lr_image_dir=calibration_data_dir,
                    target_height=opt_height, # Pass optimal height
                    target_width=opt_width,   # Pass optimal width
                    img_range=img_range,
                    calibration_dataset_size=calibration_dataset_size,
                    input_name="input" # Must match the input name used in torch.onnx.export
                )
            else:
                logger.warning("No valid --calibration_data_dir provided or directory not found. "
                               "Falling back to dummy input for INT8 ONNX calibration. "
                               "This may lead to suboptimal INT8 model accuracy. "
                               "For best results, provide a path to your low-resolution training images.")
                onnx_calibration_reader = DummyCalibrationDataReader(dummy_input_fp32_np)


            # The input model for quantize_static should be the FP32 ONNX model that contains
            # the QDQ (Quantize/Dequantize) nodes from PyTorch's QAT export.
            # The 'fp32_onnx_path' from step 7 serves this purpose.
            quantize_static(
                fp32_onnx_path,           # Input ONNX model (FP32 with QDQ nodes)
                int8_onnx_path,           # Output ONNX model (INT8)
                onnx_calibration_reader,  # Calibration data reader
                quant_format=QuantFormat.QDQ, # Use QDQ format, which PyTorch exports
                # Per-tensor is generally suitable for activations, per-channel for weights.
                # Since PyTorch's QAT typically results in per-channel weights, QDQ handles this.
                weight_type=QuantType.QInt8 # Quantize weights to INT8
            )
            logger.info("INT8 ONNX model exported successfully.")

            # Validate the exported INT8 ONNX model.
            validate_onnx_model(
                int8_onnx_path,
                pytorch_output_fp32_np, # Compare against original FP32 output
                dummy_input_fp32_np,    # Use original FP32 dummy input for ONNX Runtime inference
                atol=atol * 100, # Significantly loosen tolerance for INT8 ONNX
                rtol=rtol * 100  # Significantly loosen tolerance for INT8 ONNX
            )

        except Exception as e:
            logger.error(f"Error exporting INT8 ONNX model: {e}")
            logger.warning("Note: ONNX Runtime INT8 quantization (especially static) requires specific setup, "
                           "including a calibration dataset. If this fails, review ONNX Runtime's documentation "
                           "on quantization or consider simpler dynamic quantization if applicable.")
    else:
        logger.warning("Skipping INT8 ONNX export because onnxruntime.quantization is not available.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a QAT-trained AetherNet PyTorch model to Fused PTH (FP32, FP16, INT8), FP32 ONNX, FP16 ONNX, and INT8 ONNX.",
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
    parser.add_argument("--img_size", type=int, default=32, # This is for dummy input height/width
                        help="Input image size (height and width) for dummy input for ONNX tracing. "
                             "This should correspond to a typical patch size or smallest expected input.")
    parser.add_argument("--img_range", type=float, default=1.0,
                        help="Pixel value range (e.g., 1.0 for [0,1] input). Should match training.")

    # ONNX Export Configuration
    parser.add_argument("--dynamic_shapes", action='store_true', default=True, # Now default is True
                        help="If set, export ONNX with dynamic batch size, height, and width.")
    parser.add_argument("--static", action='store_true', # New flag for static shapes
                        help="If set, force ONNX export to use static shapes (overrides --dynamic_shapes).")
    parser.add_argument("--opset_version", type=int, default=17,
                        help="ONNX opset version for export.")
    parser.add_argument("--fp_mode", type=str, default="fp32", choices=["fp32", "fp16"],
                        help="Floating-point precision for ONNX model export (fp32 or fp16).")

    # Dynamic Shape Specific Arguments (if --dynamic_shapes is used)
    parser.add_argument("--min_batch_size", type=int, default=1,
                        help="Minimum batch size for dynamic ONNX export.")
    parser.add_argument("--opt_batch_size", type=int, default=1,
                        help="Optimal batch size for dynamic ONNX export.")
    parser.add_argument("--max_batch_size", type=int, default=16,
                        help="Maximum batch size for dynamic ONNX export.")
    parser.add_argument("--min_height", type=int, default=32,
                        help="Minimum input height for dynamic ONNX export.")
    parser.add_argument("--opt_height", type=int, default=256,
                        help="Optimal input height for dynamic ONNX export.")
    parser.add_argument("--max_height", type=int, default=512,
                        help="Maximum input height for dynamic ONNX export.")
    parser.add_argument("--min_width", type=int, default=32,
                        help="Minimum input width for dynamic ONNX export.")
    parser.add_argument("--opt_width", type=int, default=256,
                        help="Optimal input width for dynamic ONNX export.")
    parser.add_argument("--max_width", type=int, default=512,
                        help="Maximum input width for dynamic ONNX export.")

    # Calibration Arguments (NEW)
    parser.add_argument("--calibration_data_dir", type=str, default=None,
                        help="Path to the directory containing low-resolution images for INT8 ONNX calibration. "
                             "If not provided, a dummy calibration will be used, potentially reducing INT8 accuracy.")
    parser.add_argument("--calibration_dataset_size", type=int, default=100,
                        help="Number of images to use for INT8 ONNX calibration. A random subset will be selected.")


    # Validation Arguments
    parser.add_argument("--atol", type=float, default=1e-4, # Changed default
                        help="Absolute tolerance for validating ONNX outputs against PyTorch.")
    parser.add_argument("--rtol", type=float, default=1e-3, # Changed default
                        help="Relative tolerance for validating ONNX outputs against PyTorch.")


    args = parser.parse_args()

    # --- Override dynamic_shapes if --static is provided ---
    if args.static:
        args.dynamic_shapes = False
        logger.info("Static shapes forced by --static flag.")

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
    logger.info(f"Calibration Data Directory: {args.calibration_data_dir}")
    logger.info(f"Calibration Dataset Size: {args.calibration_dataset_size}")


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
        calibration_data_dir=args.calibration_data_dir, # Pass new args
        calibration_dataset_size=args.calibration_dataset_size, # Pass new args
    )

    logger.info("\nConversion process completed. Generated files are in the output directory.")
    logger.info("You can now use these ONNX files to build TensorRT engines.")


if __name__ == "__main__":
    main()
