# This script orchestrates the training process for super-resolution models
# within the neosr framework. It handles configuration parsing, data loading,
# model building, optimization, logging, validation, and checkpoint management.

import datetime
import logging
import math
import re
import sys
import time
from os import path as osp
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import torch
from torch.utils import data
from torch.utils.data.sampler import Sampler

from neosr.data import build_dataloader, build_dataset
from neosr.data.data_sampler import EnlargedSampler
from neosr.data.prefetch_dataloader import CUDAPrefetcher
from neosr.models import build_model
from neosr.utils import (
    AvgTimer,
    MessageLogger,
    check_disk_space,
    check_resume,
    get_root_logger,
    get_time_str,
    init_tb_logger,
    init_wandb_logger,
    make_exp_dirs,
    mkdir_and_rename,
    scandir,
    tc,
)
from neosr.utils.options import copy_opt_file, parse_options

# Ensure minimum supported Python version is 3.13
if sys.version_info.major != 3 or sys.version_info.minor != 13:
    msg = f"{tc.red}Python version 3.13 is required.{tc.end}"
    raise ValueError(msg)


def init_tb_loggers(opt: Dict[str, Any]) -> Union[Any, None]:
    """
    Initializes Weights & Biases (WandB) and TensorBoard loggers based on the
    provided options dictionary. WandB is initialized before TensorBoard to
    allow proper synchronization.

    Args:
        opt (Dict[str, Any]): The options dictionary parsed from the configuration file,
                              containing logging-related settings.

    Returns:
        Union[Any, None]: An initialized TensorBoard logger object if `use_tb_logger`
                          is enabled, otherwise None.
    """
    # Initialize WandB logger if enabled and project name is provided
    if (
        (opt["logger"].get("wandb") is not None)
        and (opt["logger"]["wandb"].get("project") is not None)
        and ("debug" not in opt["name"])
    ):
        # Assert that TensorBoard is also enabled when using WandB, as WandB syncs with TB
        assert opt["logger"].get("use_tb_logger") is True, (
            "TensorBoard should be enabled when using WandB for proper synchronization."
        )
        init_wandb_logger(opt)

    tb_logger = None
    # Initialize TensorBoard logger if enabled and not in debug mode
    if opt["logger"].get("use_tb_logger") and "debug" not in opt["name"]:
        # Construct the log directory path relative to the project root
        tb_logger_log_dir = Path(opt["root_path"]) / "experiments" / "tb_logger" / opt["name"]
        tb_logger = init_tb_logger(log_dir=tb_logger_log_dir)
    return tb_logger


def create_train_val_dataloader(
    opt: Dict[str, Any], logger: logging.Logger
) -> Tuple[Union[data.DataLoader, None], Sampler, List[data.DataLoader], int, int]:
    """
    Creates training and validation dataloaders based on the configuration options.

    Args:
        opt (Dict[str, Any]): The options dictionary.
        logger (logging.Logger): The logger instance for logging informational messages.

    Returns:
        Tuple[Union[data.DataLoader, None], Sampler, List[data.DataLoader], int, int]:
            A tuple containing:
            - train_loader (Union[data.DataLoader, None]): The training DataLoader.
            - train_sampler (Sampler): The sampler used for the training dataset.
            - val_loaders (List[data.DataLoader]): A list of validation DataLoaders.
            - total_epochs (int): The total number of training epochs.
            - total_iters (int): The total number of training iterations.

    Raises:
        SystemExit: If an unknown dataset phase is encountered in the configuration.
    """
    train_loader, val_loaders = None, []

    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train":
            dataset_enlarge_ratio = dataset_opt.get("dataset_enlarge_ratio", 1)
            # Add degradations section to dataset_opt if present in global options
            if opt.get("degradations") is not None:
                dataset_opt.update(opt["degradations"])
            
            train_set = build_dataset(dataset_opt)
            train_sampler = EnlargedSampler(
                train_set, opt["world_size"], opt["rank"], dataset_enlarge_ratio
            )
            num_gpu = opt.get("num_gpu", "auto")
            train_loader = build_dataloader(
                train_set,  # type: ignore[reportArgumentType]
                dataset_opt,
                num_gpu=num_gpu,
                dist=opt["dist"],
                sampler=train_sampler,
                seed=opt["manual_seed"],
            )

            accumulate = dataset_opt.get("accumulate", 1)
            num_iter_per_epoch = math.ceil(
                len(train_set)  # type: ignore[reportArgumentType]
                * dataset_enlarge_ratio
                / (dataset_opt["batch_size"] * accumulate * opt["world_size"])
            )
            total_iters = int(opt["logger"].get("total_iter", 1000000) * accumulate)
            total_epochs: int = math.ceil(total_iters / num_iter_per_epoch)
            
            logger.info(
                "Training information:"
                f"\n-------- Starting model: {opt['name']}"
                f"\n-------- GPUs detected: {opt['world_size']}"
                f"\n-------- Patch size: {dataset_opt['patch_size']}"
                f"\n-------- Dataset size: {len(train_set)}"  # type: ignore[reportArgumentType]
                f"\n-------- Batch size per GPU: {dataset_opt['batch_size']}"
                f"\n-------- Accumulated batches: {dataset_opt['batch_size'] * accumulate}"
                f"\n-------- Required iterations per epoch: {num_iter_per_epoch}"
                f"\n-------- Total epochs: {total_epochs} for total iterations {total_iters // accumulate}."
            )
        elif phase.split("_")[0] == "val":
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(
                val_set,  # type: ignore[reportArgumentType]
                dataset_opt,
                num_gpu=opt["num_gpu"],
                dist=opt["dist"],
                sampler=None,
                seed=opt["manual_seed"],
            )
            logger.info(f"Number of validation images/folders: {len(val_set)}")  # type: ignore[reportArgumentType]
            val_loaders.append(val_loader)
        else:
            msg = f"{tc.red}Dataset phase '{phase}' is not recognized. Please check your configuration.{tc.end}"
            logger.error(msg)
            sys.exit(1)

    # Ensure train_loader and train_sampler are initialized, if not, raise an error or handle
    if train_loader is None or train_sampler is None: # type: ignore[reportUnboundVariable]
        raise ValueError("Training dataloader or sampler could not be created. Check 'train' dataset configuration.")

    return train_loader, train_sampler, val_loaders, total_epochs, total_iters


def load_resume_state(opt: Dict[str, Any]) -> Union[Dict[str, Any], None]:
    """
    Loads the training resume state from a checkpoint file if auto-resume is enabled
    or a specific resume path is provided in the options.

    Args:
        opt (Dict[str, Any]): The options dictionary containing resume settings.

    Returns:
        Union[Dict[str, Any], None]: The loaded resume state dictionary (e.g., epoch, iter)
                                     if successful, otherwise None.
    """
    resume_state_path = None
    if opt["auto_resume"]:
        # Path to the directory where training states are saved
        state_path = opt["path"]["training_states"] 
        if Path(state_path).is_dir():
            # Find all '.state' files and pick the one with the highest iteration number
            states = list(
                scandir(state_path, suffix="state", recursive=False, full_path=False)
            )
            if len(states) != 0:
                states = [float(v.split(".state")[0]) for v in states]
                resume_state_path = Path(state_path) / f"{max(states):.0f}.state"
                opt["path"]["resume_state"] = resume_state_path # Update opt for consistency

    elif opt["path"].get("resume_state"):
        # Use the explicitly provided resume state path, ensuring it's absolute
        resume_state_path = Path(opt["root_path"]) / opt["path"]["resume_state"]
        
    resume_state = None
    if resume_state_path:
        print(f"{tc.light_green}Attempting to load resume state from: {resume_state_path}{tc.end}")
        try:
            # Load the state dictionary, mapping it to the current CUDA device
            resume_state = torch.load(
                resume_state_path, map_location=torch.device("cuda")
            )
            # Perform a consistency check to ensure the loaded state matches current opt
            check_resume(opt, resume_state["iter"])
            print(f"{tc.light_green}Successfully loaded resume state. "
                  f"Epoch: {resume_state.get('epoch', 'N/A')}, "
                  f"Iteration: {resume_state.get('iter', 'N/A')}{tc.end}")
        except Exception as e:
            # Log error if loading fails and set resume_state to None to start from scratch
            print(f"{tc.red}Error loading resume state from {resume_state_path}: {e}{tc.end}")
            resume_state = None 
    return resume_state

def generate_calib_data(opt, calib_iters: int) -> data.DataLoader:
    """Efficient calibration data with proper device handling"""
    # Create synthetic calibration images (0.25-0.75 range)
    num_samples = calib_iters * opt['datasets']['train']['batch_size']
    calib_data = torch.rand(
        num_samples,
        opt.get('in_chans', 3),
        opt['datasets']['train']['patch_size'],
        opt['datasets']['train']['patch_size']
    ) * 0.5 + 0.25
    
    # Build dataset wrapper
    class CalibDataset(data.Dataset):
        def __len__(self): 
            return num_samples
        
        def __getitem__(self, idx):
            return calib_data[idx]  # Return tensor directly
    
    # Create dataloader without pin_memory
    return data.DataLoader(
        CalibDataset(),
        batch_size=opt['datasets']['train']['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=False  # Critical fix: disable pinning
    )

def train_pipeline(project_root: str) -> None:
    """
    Main training pipeline for neosr. This function orchestrates the entire
    training process from setup to completion.

    Args:
        project_root (str): The absolute path to the project's root directory
                            (e.g., '/home/user/Documents/GitHub/neosr-fork').

    Raises:
        NotImplementedError: If PyTorch version is too old, CUDA is unavailable,
                             or compilation is attempted on Windows.
        RuntimeError: If system CUDA version is older than PyTorch's target CUDA.
        ValueError: If a crucial configuration file cannot be copied.
        SystemExit: For critical errors that prevent training from proceeding.
    """
    # Verify PyTorch version is 2.4 or newer
    if int(torch.__version__.split(".")[1]) < 4:
        msg = f"{tc.red}Pytorch >=2.4 is required, please upgrade.{tc.end}"
        raise NotImplementedError(msg)

    # Verify CUDA availability
    if not torch.cuda.is_available():
        msg = f"{tc.red}CUDA not available. Please install PyTorch with CUDA support.{tc.end}"
        raise NotImplementedError(msg)

    # Check for CUDA version consistency between system and PyTorch
    try:
        # Get NVIDIA CUDA driver version
        nvcc_cmd = "nvcc --version"
        nvcc_cuda_version_match = re.search(r"release (\d+\.\d+)", popen(nvcc_cmd).read()) # noqa: S605
        if nvcc_cuda_version_match:
            nvcc_cuda = nvcc_cuda_version_match[1]
            torch_cuda = torch.version.cuda
            # Compare versions as tuples for correct numerical comparison
            if tuple(map(int, torch_cuda.split("."))) > tuple(map(int, nvcc_cuda.split("."))):
                msg = (
                    f"{tc.red}Your system CUDA version ({nvcc_cuda}) "
                    f"appears to be older than PyTorch's target CUDA ({torch_cuda})! "
                    "This might lead to compatibility issues.{tc.end}"
                )
                raise RuntimeError(msg)
    except Exception:
        # Suppress errors if nvcc command fails or version check is not possible
        pass

    # Set default device to CUDA for all new tensors
    torch.set_default_device("cuda")

    # Parse command-line options and configuration files.
    # The 'project_root' is passed to `parse_options` to ensure correct path resolution.
    opt, args = parse_options(project_root, is_train=True) 
    opt["root_path"] = project_root # Explicitly set/confirm the project root in options

    # Dynamically set absolute paths for experiment-related directories
    # This ensures all outputs go into the correct 'experiments' subfolder
    opt['path']['experiments_root'] = Path(opt['root_path']) / "experiments" / opt['name']
    opt['path']['models'] = Path(opt['path']['experiments_root']) / "models"
    opt['path']['log'] = opt['path']['experiments_root'] # Log files reside in the experiment folder
    opt['path']['validation'] = Path(opt['path']['experiments_root']) / "validation"
    opt['path']['visualization'] = Path(opt['path']['experiments_root']) / "visualization"
    opt['path']['training_states'] = Path(opt['path']['experiments_root']) / "training_states"

    # Check for Triton compatibility on Windows if compilation is enabled
    if sys.platform.startswith("win") and opt.get("compile", False) is True:
        msg = f"{tc.red}PyTorch compilation (Triton backend) is not supported on Windows. Please disable 'compile' in your configuration file.{tc.end}"
        raise NotImplementedError(msg)

    # Configure Automatic Mixed Precision (AMP) and BFloat16 for faster matrix multiplication
    if opt.get("fast_matmul", False):
        torch.set_float32_matmul_precision("medium") # Enables TF32 for compatible GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_fp16_accumulation = True

    # Load previous training state if resuming
    resume_state = load_resume_state(opt)

    # Create experiment directories if not resuming, or if they don't exist
    if resume_state is None:
        make_exp_dirs(opt) # Creates experiment_root, models, log, etc.
        # Create TensorBoard log directory separately if enabled
        if (
            opt["logger"].get("use_tb_logger")
            and "debug" not in opt["name"]
            and opt["rank"] == 0 # Only create for rank 0 in distributed training
        ):
            mkdir_and_rename(Path(opt["root_path"]) / "experiments" / "tb_logger" / opt["name"])

    # Copy the training configuration file to the experiment root for reproducibility
    try:
        copy_opt_file(args.opt, opt["path"]["experiments_root"])
    except Exception as e:
        msg = (
            f"{tc.red}Failed to copy option file. "
            "Ensure the option 'name' in your config file is unique or matches "
            f"the name of an existing resumable experiment. Error: {e}{tc.end}"
        )
        raise ValueError(msg)

    # Initialize the root logger after directory setup
    log_file = Path(opt["path"]["log"]) / f"train_{opt['name']}_{get_time_str()}.log"
    logger = get_root_logger(
        logger_name="neosr", log_level=logging.INFO, log_file=str(log_file)
    )

    # Log GPU driver version
    smi_cmd = "nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits"
    try:
        driver_version = popen(smi_cmd).read().strip() # noqa: S605
    except Exception:
        driver_version = "N/A (could not retrieve)"

    logger.info(
        f"\n------------------------ neosr ------------------------"
        f"\nPyTorch Version: {torch.__version__}. Running on GPU: {torch.cuda.get_device_name()}, "
        f"with driver: {driver_version}."
    )

    # Initialize WandB and TensorBoard loggers
    tb_logger = init_tb_loggers(opt)

    # Create training and validation dataloaders
    train_loader, train_sampler, val_loaders, total_epochs, total_iters = create_train_val_dataloader(opt, logger)

    # Build the model based on the configuration
    model = build_model(opt)

    # --- Quantization-Aware Training Setup ---
    if opt['train'].get('enable_qat', False):
        logger.info(f"{tc.light_green}Initializing Quantization-Aware Training...{tc.end}")
        
        # Step 1: Fuse model if not already fused
        if hasattr(model.net_g, 'fuse_model'):
            model.net_g.fuse_model()
        
        # Step 2: Prepare QAT configuration
        qat_config = {
            'type': opt['train'].get('qat_type', 'int8'),
            'fused_init': opt['train'].get('fused_init', True),
            'quantize_residual': opt['train'].get('quantize_residual', True),
            'calibration_iters': opt['train'].get('calibration_iters', 0)
        }
        
        # Step 3: Prepare QAT
        model.net_g.prepare_qat(qat_config)
        
        # Step 4: Calibration if specified
        calib_iters = opt['train'].get('calibration_iters', 0)
        if calib_iters > 0:
            logger.info(f"Running calibration for {calib_iters} iterations")
            calib_loader = generate_calib_data(opt, calib_iters)
            model.net_g.calibrate(calib_loader, iters=calib_iters)

        # Set QAT flag on model
        model.is_qat_prepared = True
        
        # Reinitialize EMA model to match QAT structure
        if hasattr(model, 'net_g_ema'):
            logger.info("Reinitializing EMA model for QAT compatibility")
            model._init_ema()

    # Resume training from the loaded state or start fresh
    if resume_state:
        # Resume model's optimizer, scheduler, and other training states
        model.resume_training(resume_state)  # type: ignore[reportAttributeAccessIssue,attr-defined]
        # Adjust iteration log for accumulation steps
        resumed_iter_log = int(resume_state['iter'] // opt['datasets']['train'].get('accumulate', 1))
        logger.info(
            f"{tc.light_green}Resuming training from epoch: {resume_state['epoch']}, "
            f"iteration (adjusted for accumulation): {resumed_iter_log}{tc.end}"
        )
        start_epoch = resume_state["epoch"]
        current_iter = int(resume_state["iter"]) # Raw iteration count for internal tracking
        torch.cuda.empty_cache() # Clear CUDA cache to free up memory
    else:
        start_epoch = 0
        current_iter = 0

    # Initialize message logger for formatted console output
    msg_logger = MessageLogger(opt, tb_logger, current_iter // opt['datasets']['train'].get('accumulate', 1))

    # Initialize dataloader prefetcher for efficient data transfer to GPU
    if train_loader is not None:
        prefetcher = CUDAPrefetcher(train_loader, opt)
    else:
        logger.error(f"{tc.red}Training dataloader could not be created. Exiting.{tc.end}")
        sys.exit(1)

    # Log AMP (Automatic Mixed Precision) and BFloat16 status
    if opt.get("use_amp", False) and opt.get("bfloat16", False):
        logger.info("AMP enabled with BF16.")
    elif opt.get("use_amp", False) and not opt.get("bfloat16", False):
        logger.info("AMP enabled (FP16).")
    else:
        logger.info("AMP disabled.")

    # Error if BFloat16 is enabled without AMP
    if not opt.get("use_amp", False) and opt.get("bfloat16", False):
        msg = f"{tc.red}bfloat16 option has no effect without 'use_amp'. Please enable 'use_amp' for bfloat16 to function.{tc.end}"
        logger.error(msg)
        sys.exit(1)

    # Detect GPU architecture and provide recommendations/warnings
    major_cuda_version, minor_cuda_version = 0, 0
    if torch.cuda.is_available():
        major_cuda_version, minor_cuda_version = torch.cuda.get_device_capability()

    is_ampere = major_cuda_version >= 8
    is_turing = major_cuda_version == 7
    is_pascal_or_older = major_cuda_version <= 6

    # Recommend BF16 on Ampere or newer GPUs if not already enabled
    if not opt.get("use_amp", False) and is_ampere:
        msg = f"{tc.light_yellow}Modern GPU detected (Ampere or newer). Consider enabling AMP with bfloat16 for potential speedup.{tc.end}"
        logger.warning(msg)

    # Warn/Error about BF16 on older architectures
    if opt.get("bfloat16", False) and is_turing:
        msg = f"{tc.light_yellow}Turing GPU detected. bfloat16 support might be limited or inefficient. Consider disabling bfloat16.{tc.end}"
        logger.warning(msg)
    elif opt.get("bfloat16", False) and is_pascal_or_older:
        msg = f"{tc.red}Pascal or older GPU detected. bfloat16 is NOT supported on this architecture. Please disable bfloat16.{tc.end}"
        logger.error(msg)
        sys.exit(1)

    # Warn about AMP on Pascal or older GPUs
    if opt.get("use_amp", False) and is_pascal_or_older:
        msg = f"{tc.light_yellow}Pascal GPU doesn't have Tensor Cores. Consider disabling AMP as it may not provide benefits.{tc.end}"
        logger.warning(msg)

    # Log deterministic mode status
    if opt["deterministic"]:
        logger.info("Deterministic mode enabled.")

    # Training loop parameters
    accumulate = opt["datasets"]["train"].get("accumulate", 1)
    print_freq = opt["logger"].get("print_freq", 100)
    save_checkpoint_freq = opt["logger"]["save_checkpoint_freq"]
    val_freq = opt["val"]["val_freq"] if opt.get("val") is not None else 100

    logger.info(
        f"{tc.light_green}Starting training from epoch: {start_epoch}, "
        f"iteration (adjusted for accumulation): {int(current_iter / accumulate)}{tc.end}"
    )
    iter_timer = AvgTimer() # Timer for average iteration time
    start_time = time.time() # Start time for total training duration

    try:
        for epoch in range(start_epoch, total_epochs + 1):
            # Set epoch for distributed sampler to ensure data shuffling
            train_sampler.set_epoch(epoch)  # type: ignore[attr-defined]
            prefetcher.reset() # Reset prefetcher for new epoch
            train_data = prefetcher.next() # Fetch first batch for the epoch

            while train_data is not None:
                current_iter += 1
                if current_iter > total_iters:
                    break # Stop training if total iterations exceeded

                # Feed data to the model and optimize parameters
                model.feed_data(train_data)  # type: ignore[reportAttributeAccessIssue,attr-defined]
                model.optimize_parameters(current_iter)  # type: ignore[reportFunctionMemberAccess,attr-defined]
                
                # Update learning rate based on schedule
                model.update_learning_rate(  # type: ignore[reportFunctionMemberAccess,attr-defined]
                    current_iter, warmup_iter=opt["train"].get("warmup_iter", -1)
                )
                
                iter_timer.record() # Record time for current iteration
                if current_iter == 1:
                    msg_logger.reset_start_time() # Reset logger's timer on first iteration

                # Log training progress if current iteration is a multiple of print_freq
                if current_iter >= accumulate: # Only log after accumulation steps are done
                    current_iter_log = current_iter / accumulate
                else:
                    current_iter_log = current_iter # Log raw iter if still accumulating

                if current_iter_log % print_freq == 0:
                    log_vars = {"epoch": epoch, "iter": current_iter_log}
                    log_vars.update({"lrs": model.get_current_learning_rate()})  # type: ignore[reportFunctionMemberAccess,attr-defined]
                    log_vars.update({"time": iter_timer.get_avg_time()})
                    log_vars.update(model.get_current_log())  # type: ignore[reportFunctionMemberAccess,attr-defined]
                    msg_logger(log_vars)

                # Save models and training states periodically
                if current_iter_log % save_checkpoint_freq == 0:
                    free_space = check_disk_space()
                    if free_space < 500: # Check for at least 500 MB free space
                        msg = (
                            f"{tc.red}Not enough free disk space in {Path.cwd()}. "
                            "Please free up at least 500 MB of space. "
                            "Attempting to save current progress...{tc.end}"
                        )
                        logger.error(msg)
                        model.save(epoch, int(current_iter_log))  # type: ignore[reportFunctionMemberAccess,attr-defined]
                        sys.exit(1) # Exit if disk space is critically low after saving

                    logger.info(f"{tc.light_green}Saving models and training states.{tc.end}")
                    model.save(epoch, int(current_iter_log))  # type: ignore[reportFunctionMemberAccess,attr-defined]

                # Perform validation periodically if configured
                if opt.get("val") is not None and (current_iter_log % val_freq == 0):
                    for val_loader in val_loaders:
                        model.validation(  # type: ignore[reportFunctionMemberAccess,attr-defined]
                            val_loader,
                            int(current_iter_log),
                            tb_logger,
                            opt["val"].get("save_img", True),
                        )

                iter_timer.start() # Restart timer for the next iteration
                train_data = prefetcher.next() # Fetch next batch
            # End of inner (iteration) loop

        # End of outer (epoch) loop
        
        # Log total training time and save the latest model at the end of training
        consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
        logger.info(
            f"{tc.light_green}End of training. Time consumed: {consumed_time}{tc.end}"
        )
        logger.info(f"{tc.light_green}Saving the latest model.{tc.end}")
        model.save(epoch=-1, current_iter=-1) # -1 stands for the latest checkpoint

    except KeyboardInterrupt:
        # Handle graceful exit on KeyboardInterrupt (Ctrl+C)
        msg = f"{tc.light_green}Training interrupted by user. Saving latest models.{tc.end}"
        logger.info(msg)
        # Save current progress before exiting
        model.save(epoch, int(current_iter_log))  # type: ignore[reportFunctionMemberAccess,attr-defined]
        sys.exit(0)
    except Exception as e:
        # Catch any other unexpected exceptions during the training loop
        logger.error(f"{tc.red}An unexpected error occurred during training: {e}{tc.end}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        sys.exit(1) # Exit with an error code

    # Final validation after training completes
    if opt.get("val") is not None:
        accumulate = opt["datasets"]["train"].get("accumulate", 1)
        for val_loader in val_loaders:
            model.validation(  # type: ignore[reportFunctionMemberAccess,attr-defined]
                val_loader,
                int(current_iter / accumulate),
                tb_logger,
                opt["val"].get("save_img", True),
            )
    
    # Close TensorBoard logger if it was initialized
    if tb_logger:
        tb_logger.close()


if __name__ == "__main__":
    # This block ensures that the script's entry point correctly sets up
    # the Python path and starts the training pipeline.
    
    # Determine the absolute path of the current file (train.py)
    current_file_abs_path = Path(__file__).resolve()
    
    # The project root (e.g., 'neosr-fork/') is the parent directory of train.py
    project_root_directory = current_file_abs_path.parent
    
    # The 'common' directory is expected to be a direct subfolder of the project root
    common_dir_path = project_root_directory / "common"

    # Add the 'common' directory to Python's system path if it's not already there.
    # This is crucial for importing modules from the 'common' directory.
    if str(common_dir_path) not in sys.path:
        sys.path.insert(0, str(common_dir_path))
        print(f"Added '{common_dir_path}' to sys.path for common modules (explicitly added from train.py).")

    # Start the main training pipeline, passing the absolute project root path.
    train_pipeline(str(project_root_directory))

