import os
import time
import argparse
import math
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

# Import TensorRT and related libraries
# IMPORTANT: Ensure TensorRT and PyCUDA are installed and configured in your environment.
# Typically:
# pip install numpy Pillow
# (Then install TensorRT Python bindings via NVIDIA's instructions, e.g., using official wheel files)
# pip install pycuda
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    # Removed pycuda.autoinit from global scope. We will manage context explicitly.
except ImportError as e:
    print(f"Error importing TensorRT or PyCUDA: {e}")
    print("Please ensure TensorRT and PyCUDA are installed and configured correctly.")
    print("Refer to NVIDIA's TensorRT documentation for installation steps for your specific version.")
    print("You might need to install `pycuda` (pip install pycuda).")
    exit(1) # Exit if core dependencies are not met

# Set up TensorRT logger
# TRT_LOGGER.WARNING is recommended for production to avoid excessive verbose output.
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# --- Configuration for Tiling ---
# This overlap value is for the *input* tiles (e.g., 20 pixels on each side for a 256x256 tile)
# It will be scaled for the output tiles.
TILE_OVERLAP = 20 # pixels

def load_engine(engine_path: Path) -> trt.ICudaEngine:
    """
    Loads a TensorRT engine from a specified file path.

    Args:
        engine_path (Path): Absolute or relative path to the TensorRT engine file (.engine).

    Returns:
        trt.ICudaEngine: The deserialized TensorRT engine.

    Raises:
        FileNotFoundError: If the engine file does not exist.
        RuntimeError: If the engine cannot be deserialized.
    """
    if not engine_path.is_file():
        raise FileNotFoundError(f"TensorRT engine file not found at: {engine_path}")

    print(f"Attempting to load TensorRT engine from: {engine_path}")
    with open(engine_path, 'rb') as f:
        # Create a TensorRT runtime. This manages the engine's execution.
        runtime = trt.Runtime(TRT_LOGGER)
        # Deserialize the engine from the byte stream
        engine = runtime.deserialize_cuda_engine(f.read())
    
    if not engine:
        raise RuntimeError(f"Failed to deserialize TensorRT engine from {engine_path}.")
    print(f"Successfully loaded TensorRT engine.")
    return engine

def get_engine_io_info(engine: trt.ICudaEngine, profile_idx: int = 0) -> Tuple[list, list, int, int, int]:
    """
    Extracts input and output binding information (names, shapes, dtypes, optimal shapes).
    Infers the model's optimal input dimensions (tile size) and super-resolution scale factor.

    Args:
        engine (trt.ICudaEngine): The TensorRT engine.
        profile_idx (int): The index of the optimization profile to query (default is 0 for most engines).

    Returns:
        tuple: (inputs_info, outputs_info, input_opt_h, input_opt_w, scale_factor)
            - inputs_info (list of dict): Details for each input binding.
            - outputs_info (list of dict): Details for each output binding.
            - input_opt_h (int): Optimal input height (tile height) for the engine.
            - input_opt_w (int): Optimal input width (tile width) for the engine.
            - scale_factor (int): The super-resolution upscale factor of the model.

    Raises:
        RuntimeError: If inputs/outputs cannot be determined or if shapes are ambiguous.
        ValueError: If input/output shapes are not 4D (NCHW) or unexpected.
        AttributeError: If the loaded engine object does not have expected TensorRT attributes.
    """
    inputs_info = []
    outputs_info = []
    input_opt_h, input_opt_w = -1, -1
    scale_factor = -1

    # --- Defensive checks for engine attributes ---
    # The 'num_bindings' error is critical. We re-check it here with a more specific message.
    if not hasattr(engine, 'num_bindings'):
        raise AttributeError(
            f"Loaded engine object of type '{type(engine)}' does not have 'num_bindings' attribute. "
            "This indicates a severe issue with TensorRT Python bindings or environment. "
            "Possible causes: TensorRT Python package version mismatch with native libraries, "
            "conflicting `LD_LIBRARY_PATH` entries, or a corrupted engine file. "
            "Please ensure your 'tensorrt' Python package (pip show tensorrt) matches "
            "the TensorRT version that built the engine (trtexec output)."
        )
    if not hasattr(engine, 'get_binding_name') or \
       not hasattr(engine, 'binding_is_input') or \
       not hasattr(engine, 'get_binding_dtype') or \
       not hasattr(engine, 'get_profile_shape'):
        raise AttributeError(
            f"Loaded engine object of type '{type(engine)}' is missing essential TensorRT binding methods. "
            "Verify your TensorRT installation and environment setup (e.g., `LD_LIBRARY_PATH`)."
        )
    # --- End defensive checks ---


    # Iterate through all bindings to identify inputs and outputs
    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        is_input = engine.binding_is_input(i)
        dtype = trt.nptype(engine.get_binding_dtype(name))

        # For dynamic shapes, get_profile_shape returns (min_shape, opt_shape, max_shape)
        # Ensure profile_idx is valid for engine.num_optimization_profiles
        if profile_idx >= engine.num_optimization_profiles:
            raise ValueError(f"Profile index {profile_idx} is out of bounds. Engine has only {engine.num_optimization_profiles} optimization profiles.")
        
        min_shape, opt_shape, max_shape = engine.get_profile_shape(profile_idx, i)
        
        info = {
            'name': name,
            'is_input': is_input,
            'dtype': dtype,
            'shape': opt_shape, # Use optimal shape for buffer allocation in allocate_buffers_for_context
            'min_shape': min_shape,
            'max_shape': max_shape,
            'binding_idx': i
        }

        if is_input:
            inputs_info.append(info)
            if name == "input": # Assuming the input tensor is named "input"
                # A typical image model input shape is (N, C, H, W)
                if len(opt_shape) == 4:
                    input_opt_h = opt_shape[2]
                    input_opt_w = opt_shape[3]
                else:
                    raise ValueError(f"Unexpected input shape for binding '{name}': {opt_shape}. Expected (N, C, H, W).")
        else:
            outputs_info.append(info)
    
    if not inputs_info or not outputs_info:
        raise RuntimeError("Could not determine input or output bindings from the engine. Ensure model has clear inputs/outputs.")
    
    # Infer scale factor from optimal input and output shapes
    # Assuming single input and single output, and output is always N C H*scale W*scale
    if len(inputs_info) == 1 and len(outputs_info) == 1:
        output_opt_shape = outputs_info[0]['shape']
        if len(output_opt_shape) == 4 and input_opt_h > 0 and input_opt_w > 0:
            output_h = output_opt_shape[2]
            output_w = output_opt_shape[3]
            # Calculate scale based on height. Width scale should be consistent.
            if input_opt_h > 0 and input_opt_w > 0:
                scale_h = output_h / input_opt_h
                scale_w = output_w / input_opt_w
                if scale_h == scale_w and scale_h == int(scale_h): # Check if it's an integer scale
                    scale_factor = int(scale_h)
                else:
                    print(f"[WARNING] Non-integer or inconsistent scale factor detected: H_scale={scale_h}, W_scale={scale_w}. Using {int(round(scale_h))} as scale.")
                    scale_factor = int(round(scale_h)) # Fallback to rounded value if slight float deviation
        else:
            raise ValueError(f"Unexpected output shape for binding '{outputs_info[0]['name']}': {output_opt_shape}. Expected (N, C, H, W).")
    else:
        print("[WARNING] Multiple inputs/outputs detected or input/output shape not 4D. Cannot infer scale factor automatically.")
        # User might need to manually verify the scale factor if auto-detection fails in complex models.

    print(f"Engine I/O Info: Input Optimal Shape: (1, 3, {input_opt_h}, {input_opt_w}), Inferred Scale Factor: {scale_factor}x")
    return inputs_info, outputs_info, input_opt_h, input_opt_w, scale_factor


def allocate_buffers_for_context(engine: trt.ICudaEngine, context: trt.IExecutionContext, profile_idx: int = 0):
    """
    Allocates host and device buffers based on the active optimization profile
    and the set binding shapes in the context. These buffers will be used for
    data transfer between host (CPU) and device (GPU) during inference.

    Args:
        engine (trt.ICudaEngine): The TensorRT engine.
        context (trt.IExecutionContext): The TensorRT execution context.
        profile_idx (int): The index of the optimization profile used.

    Returns:
        tuple: (inputs, outputs, bindings, stream)
            - inputs (list of dict): Host and device buffers for inputs.
            - outputs (list of dict): Host and device buffers for outputs.
            - bindings (list of int): Device memory pointers for all bindings.
            - stream (cuda.Stream): CUDA stream for asynchronous operations.

    Raises:
        ValueError: If dynamic dimensions are not properly resolved by the context,
                    or if buffer allocation fails.
    """
    inputs = []
    outputs = []
    bindings = [None] * engine.num_bindings # Initialize bindings list with None
    stream = cuda.Stream() # Create a CUDA stream for asynchronous memory ops and kernel execution

    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        is_input = engine.binding_is_input(i)
        
        # Get the actual shape that the context has set for this binding.
        # This will be the resolved shape for dynamic inputs.
        binding_shape = context.get_binding_shape(i)
        
        dtype = trt.nptype(engine.get_binding_dtype(name))
        
        # Check for unresolved dynamic dimensions (indicated by -1).
        # This means set_binding_shape was not called correctly for dynamic inputs.
        if any(d == -1 for d in binding_shape):
            raise ValueError(
                f"Dynamic dimension found for binding '{name}' at index {i} with shape {binding_shape}. "
                "Ensure `context.set_binding_shape()` was called correctly for this binding's input shape "
                "before allocating buffers."
            )

        # Calculate the total size in bytes for the buffer
        size_bytes = trt.volume(binding_shape) * engine.get_binding_dtype(name).itemsize

        # Allocate pinned (page-locked) host memory for faster transfers
        host_mem = cuda.pagelocked_empty(binding_shape, dtype)
        # Allocate device (GPU) memory
        device_mem = cuda.mem_alloc(size_bytes)
        
        # Store the device memory pointer (required by execute_async_v2) at the correct binding index
        bindings[i] = int(device_mem)

        # Categorize buffers into inputs and outputs lists
        if is_input:
            inputs.append({'host': host_mem, 'device': device_mem, 'name': name, 'binding_idx': i})
        else:
            outputs.append({'host': host_mem, 'device': device_mem, 'name': name, 'binding_idx': i})

    return inputs, outputs, bindings, stream


def preprocess_tile_np(image_np_tile: np.ndarray, img_range: float = 1.0) -> np.ndarray:
    """
    Preprocesses a NumPy image tile for AetherNet inference.
    Normalizes pixel values and transposes to NCHW format.
    Assumes input `image_np_tile` is already HWC (Height, Width, Channels).

    Args:
        image_np_tile (np.ndarray): Input tile as a NumPy array (HWC, float32 or uint8).
        img_range (float): The maximum pixel value range for normalization (e.g., 1.0 for [0,1]).

    Returns:
        np.ndarray: Preprocessed tile as a NumPy array (NCHW format, float32).
    """
    if image_np_tile.dtype == np.uint8:
        # Convert to float32 if starting from uint8
        img_np = image_np_tile.astype(np.float32)
    else:
        img_np = image_np_tile

    # Normalize pixel values to the expected model input range (e.g., [0, 1])
    img_np = img_np / (255.0 / img_range) 

    # Convert from HWC (Height, Width, Channels) to NCHW (Batch, Channels, Height, Width)
    img_np = img_np.transpose(2, 0, 1) # HWC -> CHW
    img_np = np.expand_dims(img_np, axis=0) # CHW -> NCHW (add batch dimension, N=1)

    return img_np

def postprocess_tile_np(output_array: np.ndarray, img_range: float = 1.0) -> np.ndarray:
    """
    Postprocesses the TensorRT output NumPy array for stitching.
    Denormalizes pixel values and converts from NCHW to HWC.

    Args:
        output_array (np.ndarray): Output array from TensorRT (NCHW format, float32).
        img_range (float): The pixel value range the model was trained with (e.g., 1.0).

    Returns:
        np.ndarray: Postprocessed tile as a NumPy array (HWC, float32), denormalized to [0, 255].
    """
    # Remove batch dimension if present (NCHW -> CHW)
    if output_array.ndim == 4:
        output_array = output_array.squeeze(0)

    # Convert from CHW (Channels, Height, Width) to HWC (Height, Width, Channels)
    output_array = output_array.transpose(1, 2, 0)

    # Denormalize pixel values back to [0, 255] range (still float32 for blending)
    output_array = output_array * (255.0 / img_range)
    # No clipping or uint8 conversion here, as it will be blended before final conversion.
    return output_array

def pad_image_for_tiling(image: Image.Image, tile_h: int, tile_w: int, overlap: int) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Pads an input PIL image (converted to NumPy) to make its dimensions
    compatible with tiling, ensuring full tiles can be extracted and stitched.
    Padding is done using reflection mode to minimize artifacts.

    Args:
        image (Image.Image): The input PIL Image.
        tile_h (int): The height of each tile (optimal input height for the engine).
        tile_w (int): The width of each tile (optimal input width for the engine).
        overlap (int): The overlap in pixels between adjacent tiles.

    Returns:
        tuple: (padded_image_np, padding_offsets)
            - padded_image_np (np.ndarray): The padded image as a NumPy array (HWC, float32).
            - padding_offsets (tuple): (top_pad, bottom_pad, left_pad, right_pad) indicating
                                       the amount of padding added to each side.
    """
    img_np = np.array(image).astype(np.float32) # Convert to NumPy, HWC
    original_h, original_w = img_np.shape[0], img_np.shape[1]

    # Calculate stride for tiling
    stride_h = tile_h - 2 * overlap
    stride_w = tile_w - 2 * overlap

    # Ensure stride is at least 1
    if stride_h <= 0 or stride_w <= 0:
        raise ValueError(f"Tile overlap ({overlap}) is too large for tile dimensions ({tile_h}x{tile_w}). "
                         "Stride must be positive. Reduce overlap or use larger tile dimensions.")

    # Calculate required padding
    # The padded image should have dimensions that allow for `N` full tiles
    # plus the `2 * overlap` border for the last tile's overlap.
    # We need to calculate how much the image needs to extend to cleanly fit tiles.
    
    # Calculate effective content size that must be covered by strides
    content_h = max(0, original_h - overlap) # Content after first overlap region
    content_w = max(0, original_w - overlap) # Content after first overlap region

    # Calculate number of strides needed for content, then total required size
    num_strides_h = math.ceil(content_h / stride_h) if content_h > 0 else 0
    num_strides_w = math.ceil(content_w / stride_w) if content_w > 0 else 0

    # Calculate total required dimensions for the padded image
    required_padded_h = (num_strides_h * stride_h) + overlap
    required_padded_w = (num_strides_w * stride_w) + overlap

    # If original image is smaller than a single tile, pad to tile size (plus minimal overlap for stitching)
    if original_h <= overlap: required_padded_h = tile_h
    if original_w <= overlap: required_padded_w = tile_w

    # Calculate padding amounts for each side
    top_pad = overlap
    left_pad = overlap
    
    bottom_pad = max(0, required_padded_h - original_h - top_pad)
    right_pad = max(0, required_padded_w - original_w - left_pad)

    # Pad the image using reflection mode
    padded_img_np = np.pad(img_np, 
                           ((top_pad, bottom_pad), (left_pad, right_pad), (0,0)), 
                           mode='reflect')
    
    padding_offsets = (top_pad, bottom_pad, left_pad, right_pad)
    print(f"Padded image from {original_h}x{original_w} to {padded_img_np.shape[0]}x{padded_img_np.shape[1]} with padding: {padding_offsets}")
    return padded_img_np, padding_offsets

def create_blending_kernel(tile_h_upscaled: int, tile_w_upscaled: int, overlap_upscaled: int) -> np.ndarray:
    """
    Generates a 2D blending kernel (mask) for seamless stitching of upscaled tiles.
    The kernel has a value of 1 in the central non-overlapping region and
    linearly ramps down to 0 in the overlapping border regions.

    Args:
        tile_h_upscaled (int): Upscaled tile height.
        tile_w_upscaled (int): Upscaled tile width.
        overlap_upscaled (int): Upscaled overlap in pixels.

    Returns:
        np.ndarray: A 2D NumPy array representing the blending kernel (float32, values 0-1).
    """
    kernel_x = np.linspace(0, 1, overlap_upscaled)
    kernel_x = np.concatenate([kernel_x, np.ones(tile_w_upscaled - 2 * overlap_upscaled), kernel_x[::-1]])
    
    kernel_y = np.linspace(0, 1, overlap_upscaled)
    kernel_y = np.concatenate([kernel_y, np.ones(tile_h_upscaled - 2 * overlap_upscaled), kernel_y[::-1]])

    blending_kernel = np.outer(kernel_y, kernel_x)
    return blending_kernel.astype(np.float32)

def run_inference(engine_path: Path, input_folder: Path, output_folder: Path, model_input_range: float = 1.0):
    """
    Main function to run super-resolution inference on all images in a folder
    using a TensorRT engine. This function implements a tiling strategy to
    handle images of arbitrary dimensions and ensure seamless upscaling.

    Args:
        engine_path (Path): Path to the TensorRT engine file (.engine).
        input_folder (Path): Path to the directory containing input images.
        output_folder (Path): Path to the directory where upscaled images will be saved.
        model_input_range (float): Pixel value range (e.g., 1.0 for [0,1]) the model expects.
    """
    output_folder.mkdir(parents=True, exist_ok=True) # Ensure output directory exists

    # Ensure a CUDA context is active before loading engine
    # This explicit context management helps with stability
    try:
        # Get the current CUDA context, assuming autoinit or a previous context setup.
        # If no context is active, pycuda.autoinit typically creates one.
        ctx = cuda.Context.get_current()
        print(f"Active CUDA Context: {ctx.get_device().name()} (ID: {ctx.get_device().pci_bus_id()})")
    except Exception as e:
        print(f"[CRITICAL ERROR] Failed to get or create CUDA context: {e}")
        print("Please ensure your NVIDIA GPU drivers are installed and CUDA is functioning correctly.")
        exit(1) # Exit if no CUDA context can be established

    # Now, with a context active, proceed to load the engine
    try:
        engine = load_engine(engine_path)
    except (FileNotFoundError, RuntimeError, AttributeError) as e:
        print(f"[CRITICAL ERROR] Failed to load or initialize TensorRT engine: {e}")
        print("Please verify your TensorRT installation, Python package version compatibility, and the engine file's integrity.")
        return

    # Create execution context. This manages runtime inference.
    context = engine.create_execution_context()

    # Get engine's I/O info and optimal input dimensions (which will be our tile size)
    try:
        inputs_info, outputs_info, opt_input_h, opt_input_w, scale_factor = get_engine_io_info(engine)
    except (ValueError, RuntimeError, AttributeError) as e:
        print(f"[CRITICAL ERROR] Error getting engine I/O info: {e}")
        print("Could not determine optimal input shapes or scale factor from the engine.")
        # Clean up context and engine before returning
        del context
        del engine
        return
    
    # Check for expected number of inputs/outputs for a typical SR model
    if len(inputs_info) != 1 or len(outputs_info) != 1:
        print(f"[ERROR] Expected 1 input and 1 output, but found {len(inputs_info)} inputs and {len(outputs_info)} outputs.")
        print("This script is designed for models with a single image input and single image output.")
        # Clean up context and engine before returning
        del context
        del engine
        return

    # Ensure the determined optimal input size is valid for tiling
    if opt_input_h <= 0 or opt_input_w <= 0:
        print(f"[ERROR] Invalid optimal input dimensions from engine: {opt_input_h}x{opt_input_w}. Cannot proceed with tiling.")
        del context
        del engine
        return

    # Set the dynamic input shape for the current execution context to the optimal tile size.
    try:
        input_binding_idx = inputs_info[0]['binding_idx']
        context.set_binding_shape(input_binding_idx, (1, 3, opt_input_h, opt_input_w))
        print(f"Engine's active input binding shape set to its optimal: (1, 3, {opt_input_h}, {opt_input_w})")
    except Exception as e:
        print(f"[ERROR] Failed to set binding shape for input. Ensure the engine supports dynamic input shapes and the optimal shape is valid. Error: {e}")
        del context
        del engine
        return

    # Allocate host and device buffers based on the determined shapes in the context
    try:
        inputs_buffers, outputs_buffers, bindings, stream = allocate_buffers_for_context(engine, context)
    except ValueError as e:
        print(f"[ERROR] Failed to allocate buffers: {e}")
        print("This might happen if dynamic input shapes were not properly resolved by the context.")
        del context
        del engine
        return
    
    # Gather image files from the input folder
    image_files = sorted(list(input_folder.glob('*.png')) + 
                         list(input_folder.glob('*.jpg')) + 
                         list(input_folder.glob('*.jpeg')))
    
    if not image_files:
        print(f"No supported image files found in '{input_folder}'. Supported formats: .png, .jpg, .jpeg")
        del context
        del engine
        return

    print(f"\nFound {len(image_files)} images in '{input_folder}'. Starting inference...")

    total_inference_time = 0
    num_processed_images = 0

    # Benchmark variables
    warmup_runs = min(5, len(image_files)) # Number of initial inferences to warm up the GPU/engine
    print(f"Running {warmup_runs} warmup inferences...")
    # Warm-up runs to ensure GPU clocks are stable and caches are populated
    for i in range(warmup_runs):
        try:
            # For warmup, we use the first image's actual size for padding calculation, but only process one tile.
            # This is a simplified warmup; a more rigorous one might process a full small image.
            dummy_img = Image.open(image_files[0]).convert('RGB')
            dummy_padded_np, _ = pad_image_for_tiling(dummy_img, opt_input_h, opt_input_w, TILE_OVERLAP)
            # Take just the first tile for warmup
            dummy_tile_np = dummy_padded_np[0:opt_input_h, 0:opt_input_w, :]
            
            input_data = preprocess_tile_np(dummy_tile_np, model_input_range)
            np.copyto(inputs_buffers[0]['host'], input_data.ravel())
            cuda.memcpy_htod_async(inputs_buffers[0]['device'], inputs_buffers[0]['host'], stream)
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            cuda.memcpy_dtoh_async(outputs_buffers[0]['host'], outputs_buffers[0]['device'], stream)
            stream.synchronize()
        except Exception as e:
            print(f"[WARNING] Warmup run {i+1} failed: {e}")
            # Do not exit on warmup failure, just note it.

    print(f"\nStarting tiled inference on {len(image_files)} images...")
    # Main inference loop for benchmarking and saving results
    for i, img_path in enumerate(image_files):
        print(f"Processing image {i+1}/{len(image_files)}: '{img_path.name}'")
        try:
            pil_image = Image.open(img_path).convert('RGB')
            original_h, original_w = pil_image.size[1], pil_image.size[0] # PIL.size is (width, height)

            # 1. Pad image for tiling
            padded_img_np, padding_offsets = pad_image_for_tiling(pil_image, opt_input_h, opt_input_w, TILE_OVERLAP)
            padded_h, padded_w, _ = padded_img_np.shape

            # Calculate effective stride (non-overlapping part of tile)
            tile_stride_h = opt_input_h - 2 * TILE_OVERLAP
            tile_stride_w = opt_input_w - 2 * TILE_OVERLAP

            # Create blending kernel for upscaled tiles
            upscaled_overlap = TILE_OVERLAP * scale_factor
            blending_kernel_upscaled = create_blending_kernel(
                opt_input_h * scale_factor, 
                opt_input_w * scale_factor, 
                upscaled_overlap
            )
            
            # Initialize canvases for stitching
            output_canvas_h = padded_h * scale_factor
            output_canvas_w = padded_w * scale_factor
            output_canvas_upscaled = np.zeros((output_canvas_h, output_canvas_w, 3), dtype=np.float32)
            weights_canvas_upscaled = np.zeros((output_canvas_h, output_canvas_w), dtype=np.float32)

            # Measure inference time per image (including tiling overhead)
            start_time = time.perf_counter()

            # 2. Iterate and process tiles
            # Loop with coordinates for slicing the padded input image
            y_coords = list(range(0, padded_h - TILE_OVERLAP, tile_stride_h)) # Adjust end to include last tile
            if (padded_h - TILE_OVERLAP) % tile_stride_h != 0:
                y_coords.append(padded_h - opt_input_h) # Ensure the last tile covers the bottom edge

            x_coords = list(range(0, padded_w - TILE_OVERLAP, tile_stride_w)) # Adjust end
            if (padded_w - TILE_OVERLAP) % tile_stride_w != 0:
                x_coords.append(padded_w - opt_input_w) # Ensure the last tile covers the right edge

            # Make coordinates unique and sorted in case the last tile calculation overlaps with a stride
            y_coords = sorted(list(set(y_coords)))
            x_coords = sorted(list(set(x_coords)))


            for y in y_coords:
                for x in x_coords:
                    # Extract input tile
                    tile_np = padded_img_np[y : y + opt_input_h, x : x + opt_input_w, :]

                    # Preprocess tile
                    preprocessed_tile = preprocess_tile_np(tile_np, model_input_range)
                    
                    # Copy input data to host buffer
                    np.copyto(inputs_buffers[0]['host'], preprocessed_tile.ravel())

                    # Transfer input data to device, execute inference, and transfer output back to host
                    cuda.memcpy_htod_async(inputs_buffers[0]['device'], inputs_buffers[0]['host'], stream)
                    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
                    cuda.memcpy_dtoh_async(outputs_buffers[0]['host'], outputs_buffers[0]['device'], stream)
                    stream.synchronize() # Wait for operations to complete for this tile

                    # Postprocess output tile (denormalize, HWC, float32)
                    postprocessed_tile_upscaled = postprocess_tile_np(outputs_buffers[0]['host'], model_input_range)

                    # 3. Stitch tiles using blending kernel
                    # Calculate destination coordinates in the upscaled canvas
                    dest_y_start = y * scale_factor
                    dest_x_start = x * scale_factor
                    dest_y_end = dest_y_start + (opt_input_h * scale_factor)
                    dest_x_end = dest_x_start + (opt_input_w * scale_factor)

                    # Add weighted upscaled tile to canvas
                    output_canvas_upscaled[dest_y_start:dest_y_end, dest_x_start:dest_x_end, :] += \
                        postprocessed_tile_upscaled * blending_kernel_upscaled[:, :, np.newaxis]
                    
                    # Add blending weights to the weights canvas
                    weights_canvas_upscaled[dest_y_start:dest_y_end, dest_x_start:dest_x_end] += \
                        blending_kernel_upscaled

            end_time = time.perf_counter()
            inference_time = end_time - start_time
            total_inference_time += inference_time
            num_processed_images += 1

            # 4. Final averaging and cropping
            # Handle potential division by zero for unvisited areas (shouldn't happen with proper tiling)
            # Add small epsilon to prevent division by zero in case of weights_canvas_upscaled == 0
            final_upscaled_padded_np = output_canvas_upscaled / (weights_canvas_upscaled[:, :, np.newaxis] + 1e-6)
            
            # Calculate crop dimensions based on original image size and scale factor
            crop_top = padding_offsets[0] * scale_factor
            crop_bottom = final_upscaled_padded_np.shape[0] - (padding_offsets[1] * scale_factor)
            crop_left = padding_offsets[2] * scale_factor
            crop_right = final_upscaled_padded_np.shape[1] - (padding_offsets[3] * scale_factor)

            # Ensure crop dimensions are positive
            if crop_top < 0: crop_top = 0
            if crop_left < 0: crop_left = 0
            if crop_bottom > final_upscaled_padded_np.shape[0]: crop_bottom = final_upscaled_padded_np.shape[0]
            if crop_right > final_upscaled_padded_np.shape[1]: crop_right = final_upscaled_padded_np.shape[1]


            final_upscaled_np = final_upscaled_padded_np[
                crop_top:crop_bottom, 
                crop_left:crop_right, 
                :
            ]
            
            # Convert to uint8 and save as PIL Image
            output_image = Image.fromarray(np.clip(final_upscaled_np, 0, 255).astype(np.uint8), 'RGB')

            output_filename = output_folder / f"{img_path.stem}_upscaled.png"
            output_image.save(output_filename)
            print(f"  Saved upscaled image to: '{output_filename}'")

        except Exception as e:
            print(f"[ERROR] Failed to process '{img_path.name}': {e}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging
            continue # Continue to the next image even if one fails

    if num_processed_images > 0:
        avg_inference_time = total_inference_time / num_processed_images
        avg_fps = num_processed_images / total_inference_time
        print("\n--- Inference Summary ---")
        print(f"Total images successfully processed: {num_processed_images}")
        print(f"Total inference time (excluding warm-up): {total_inference_time:.4f} seconds")
        print(f"Average inference time per image: {avg_inference_time:.4f} seconds")
        print(f"Average FPS (Frames Per Second): {avg_fps:.2f} fps")
    else:
        print("\nNo images were successfully processed. Please check input folder and image formats.")

    # Explicitly delete context and engine to release GPU resources
    del context
    del engine

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AetherNet Super-Resolution inference using a TensorRT engine with tiling.")
    parser.add_argument(
        "--engine_path",
        type=str,
        required=True,
        help="Absolute or relative path to the TensorRT engine file (.engine). "
             "Example: './converted_models_dynamic/aether_small_2x_int8_dynamic.engine'."
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Path to the folder containing input images for super-resolution. Can contain images of various sizes."
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="./output_images_upscaled_trt",
        help="Path to the folder where upscaled images will be saved. Defaults to './output_images_upscaled_trt'."
    )
    parser.add_argument(
        "--model_input_range",
        type=float,
        default=1.0,
        help="The pixel value range the AetherNet model was trained with (e.g., 1.0 for [0,1], 255.0 for [0,255]). "
             "Defaults to 1.0 (recommended for AetherNet when using neosr's default normalization)."
    )

    args = parser.parse_args()

    # Convert string paths from argparse to pathlib.Path objects
    engine_path = Path(args.engine_path) 
    input_folder_path = Path(args.input_folder)
    output_folder_path = Path(args.output_folder)

    # Example of creating a dummy input folder and images if it doesn't exist
    # for initial testing. REMEMBER TO REPLACE THESE WITH YOUR ACTUAL IMAGES FOR ACCURATE BENCHMARKS!
    if not input_folder_path.exists() and not input_folder_path.is_file():
        print(f"Input folder '{input_folder_path}' not found. Creating a dummy one for demonstration.")
        input_folder_path.mkdir(parents=True, exist_ok=True)
        # Create dummy images of various sizes
        for i in range(2): # Two 240x240 images
            dummy_img = Image.new('RGB', (240, 240), color = (i*60 % 255, i*120 % 255, i*180 % 255))
            dummy_img.save(input_folder_path / f'dummy_image_240_{i+1}.png')
        dummy_img_odd = Image.new('RGB', (71, 97), color = (100, 50, 200)) # Odd dimensions
        dummy_img_odd.save(input_folder_path / 'dummy_image_71x97.png')
        dummy_img_large = Image.new('RGB', (500, 400), color = (20, 150, 80)) # Larger image
        dummy_img_large.save(input_folder_path / 'dummy_image_500x400.png')
        print(f"Created dummy images in '{input_folder_path}'. Please replace with your real images for accurate benchmarks.")

    # Run the inference
    run_inference(engine_path, input_folder_path, output_folder_path, args.model_input_range)
