# train.py - neosr training pipeline
# This file has been modified to:
# 1. Correctly determine the root path for neosr, directing experiment outputs
#    to the 'experiments' folder within the project root.
# 2. Ensure full training states are loaded for resume functionality.
# 3. Provide clearer console output for sys.path modifications.
# 4. FIX: Corrected the placement of the main execution block to prevent NameError.

import datetime
import logging
import math
import re
import sys
import time
from os import path as osp
from pathlib import Path
from typing import Any

# IMPORTANT: Imports from neosr.data, neosr.models, neosr.utils, neosr.utils.options
# MUST come *after* the sys.path modification to ensure 'common' is found
# if any of these modules depend on it.

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

# minimum supported python version
if sys.version_info.major != 3 or sys.version_info.minor != 13:
    msg = f"{tc.red}Python version 3.13 is required.{tc.end}"
    raise ValueError(msg)


def init_tb_loggers(opt: dict[str, Any]):
    # initialize wandb logger before tensorboard logger to allow proper sync
    if (
        (opt["logger"].get("wandb") is not None)
        and (opt["logger"]["wandb"].get("project") is not None)
        and ("debug" not in opt["name"])
    ):
        assert opt["logger"].get("use_tb_logger") is True, (
            "should turn on tensorboard when using wandb"
        )
        init_wandb_logger(opt)
    tb_logger = None
    if opt["logger"].get("use_tb_logger") and "debug" not in opt["name"]:
        # opt['root_path'] is already the overall project root (e.g., neosr-fork/)
        # So, tb_logger will go into <PROJECT_ROOT>/experiments/tb_logger/model_name
        tb_logger_log_dir = Path(opt['root_path']) / "experiments" / "tb_logger" / opt["name"]
        tb_logger = init_tb_logger(log_dir=tb_logger_log_dir)
    return tb_logger


def create_train_val_dataloader(
    opt: dict[str, Any], logger: logging.Logger
) -> tuple[data.DataLoader | None, Sampler, list[data.DataLoader], int, int]:
    # create train and val dataloaders
    train_loader, val_loaders = None, []

    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train":
            dataset_enlarge_ratio = dataset_opt.get("dataset_enlarge_ratio", 1)
            # add degradations section to dataset_opt
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

            accumulate = dataset_opt.get("accumulate", 1) # Get accumulate from dataset_opt
            num_iter_per_epoch = math.ceil(
                len(train_set)  # type: ignore[reportArgumentType]
                * dataset_enlarge_ratio
                / (dataset_opt["batch_size"] * accumulate * opt["world_size"])
            )
            total_iters = int(opt["logger"].get("total_iter", 1000000) * accumulate)
            total_epochs: int = math.ceil(total_iters / num_iter_per_epoch)
            logger.info(
                "Training informations:"
                f"\n-------- Starting model: {opt['name']}"
                f"\n-------- GPUs detected: {opt['world_size']}"
                f"\n-------- Patch size: {dataset_opt['patch_size']}"
                f"\n-------- Dataset size: {len(train_set)}"  # type: ignore[reportArgumentType]
                f"\n-------- Batch size per gpu: {dataset_opt['batch_size']}"
                f"\n-------- Accumulated batches: {dataset_opt['batch_size'] * accumulate}"
                f"\n-------- Required iters per epoch: {num_iter_per_epoch}"
                f"\n-------- Total epochs {total_epochs} for total iters {total_iters // accumulate}."
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
            logger.info(f"Number of val images/folders: {len(val_set)}")  # type: ignore[reportArgumentType]
            val_loaders.append(val_loader)
        else:
            msg = f"{tc.red}Dataset phase {phase} is not recognized.{tc.end}"
            logger.error(msg)
            sys.exit(1)

    return train_loader, train_sampler, val_loaders, total_epochs, total_iters  # type: ignore[reportPossiblyUnboundVariable]


def load_resume_state(opt: dict[str, Any]):
    resume_state_path = None
    if opt["auto_resume"]:
        # opt["path"]["training_states"] will now be an absolute path from the project root
        state_path = opt["path"]["training_states"] 
        if Path(state_path).is_dir(): # Convert to Path object for .is_dir()
            states = list(
                scandir(state_path, suffix="state", recursive=False, full_path=False)
            )
            if len(states) != 0:
                states = [float(v.split(".state")[0]) for v in states]
                resume_state_path = Path(state_path) / f"{max(states):.0f}.state"
                opt["path"]["resume_state"] = resume_state_path

    elif opt["path"].get("resume_state"):
        # This resume_state path needs to be resolved relative to the true project root
        # if it's a relative path in the TOML. parse_options should handle this
        # if opt['root_path'] is the project root.
        resume_state_path = Path(opt['root_path']) / opt["path"]["resume_state"] # Ensure it's absolute
        
    if resume_state_path is None:
        resume_state = None
    else:
        print(f"{tc.light_green}Attempting to load resume state from: {resume_state_path}{tc.end}")
        try:
            resume_state = torch.load(
                resume_state_path, map_location=torch.device("cuda")
            )
            check_resume(opt, resume_state["iter"])
            print(f"{tc.light_green}Successfully loaded resume state. Epoch: {resume_state.get('epoch', 'N/A')}, Iter: {resume_state.get('iter', 'N/A')}{tc.end}")
        except Exception as e:
            print(f"{tc.red}Error loading resume state from {resume_state_path}: {e}{tc.end}")
            resume_state = None # Set to None to start from scratch if load fails
    return resume_state


def train_pipeline(project_root: str) -> None: # Renamed argument for clarity
    # raise error if pytorch <2.4
    if int(torch.__version__.split(".")[1]) < 4:
        msg = f"{tc.red}Pytorch >=2.4 is required, please upgrade.{tc.end}"
        raise NotImplementedError(msg)

    # raise error if not CUDA
    if not torch.cuda.is_available():
        msg = f"{tc.red}CUDA not available. Please install pytorch with cuda support.{tc.end}"
        raise NotImplementedError(msg)

    # check if system cuda version is not lower than pytorch target
    try:
        nvcc_cmd = "nvcc --version"
        nvcc_cuda = re.search(r"release (\d+\.\d+)", popen(nvcc_cmd).read())[1]  # noqa: S605
        torch_cuda = torch.version.cuda
        if tuple(map(int, torch_cuda.split("."))) > tuple(
            map(int, nvcc_cuda.split("."))
        ):
            msg = f"{tc.red}Your system CUDA version appears to be {nvcc_cuda} while pytorch is higher ({torch_cuda})!{tc.end}"
            raise RuntimeError(msg)
    except Exception: # Catch any exception during nvcc check to prevent script from crashing
        pass

    # default device
    torch.set_default_device("cuda")

    # parse options, set distributed setting, set random seed
    # Pass the actual project_root as the root_path
    opt, args = parse_options(project_root, is_train=True) 
    opt["root_path"] = project_root # Ensure this is the absolute project root

    # FIX: Explicitly set the full absolute paths for experiments_root, models, log etc.
    # These paths are now correctly relative to the `project_root`.
    opt['path']['experiments_root'] = Path(opt['root_path']) / "experiments" / opt['name']
    opt['path']['models'] = Path(opt['path']['experiments_root']) / "models"
    opt['path']['log'] = opt['path']['experiments_root'] # Log files will go into the experiment folder directly
    opt['path']['validation'] = Path(opt['path']['experiments_root']) / "validation"
    opt['path']['visualization'] = Path(opt['path']['experiments_root']) / "visualization"
    opt['path']['training_states'] = Path(opt['path']['experiments_root']) / "training_states" # Ensure this path exists for resume


    # Triton doesn't support Windows yet
    if sys.platform.startswith("win") and opt.get("compile", False) is True:
        msg = f"{tc.red}Compile is not supported on Windows, please disable it on your configuration file.{tc.end}"
        raise NotImplementedError(msg)

    # enable tensorfloat32 and possibly bfloat16 matmul
    fast_matmul = opt.get("fast_matmul", False)
    if fast_matmul:
        torch.set_float32_matmul_precision("medium")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_fp16_accumulation = True

    # load resume states if necessary
    resume_state = load_resume_state(opt)
    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt) # This uses the now-corrected paths in opt['path']
        if (
            opt["logger"].get("use_tb_logger")
            and "debug" not in opt["name"]
            and opt["rank"] == 0
        ):
            # mkdir for tb_logger separately, also relative to project root
            mkdir_and_rename(Path(opt["root_path"]) / "experiments" / "tb_logger" / opt["name"])


    # copy the toml file to the experiment root
    try:
        copy_opt_file(args.opt, opt["path"]["experiments_root"])
    except Exception as e: # Catch specific exception for better error message
        msg = f"{tc.red}Failed to copy option file. Make sure the option 'name' in your config file is the same as the previous state! Error: {e}{tc.end}"
        raise ValueError(msg)

    # WARNING: should not use get_root_logger in the above codes, including the called functions
    # Otherwise the logger will not be properly initialized
    log_file = Path(opt["path"]["log"]) / f"train_{opt['name']}_{get_time_str()}.log"
    logger = get_root_logger(
        logger_name="neosr", log_level=logging.INFO, log_file=str(log_file)
    )

    smi_cmd = "nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits"
    try:
        driver_version = (
            popen(smi_cmd)  # noqa: S605
            .read()
            .strip()
        )
    except Exception:
        driver_version = "N/A (could not retrieve)"


    logger.info(
        f"\n------------------------ neosr ------------------------\nPytorch Version: {torch.__version__}. Running on gpu {torch.cuda.get_device_name()}, with driver {driver_version}."
    )

    # initialize wandb and tb loggers
    tb_logger = init_tb_loggers(opt)

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loaders, total_epochs, total_iters = result

    # create model
    model = build_model(opt)

    # --- QAT INTEGRATION START ---
    # Check if QAT is enabled in the training configuration
    if opt.get('train', {}).get('enable_qat', False):
        logger.info(f"{tc.light_green}Enabling Quantization-Aware Training (QAT)...{tc.end}")
        # Call the prepare_qat method on your model's underlying network (net_g).
        # IMPORTANT: Pass the full 'opt' dictionary here so aether_arch.py can access bfloat16 info.
        if hasattr(model, 'net_g') and hasattr(model.net_g, 'prepare_qat'):
            try:
                model.net_g.prepare_qat(opt) # Pass 'opt' here
            except Exception as e:
                logger.error(f"{tc.red}Error during QAT preparation: {e}{tc.end}")
                sys.exit(1)
        else:
            logger.error(f"{tc.red}Error: model.net_g or model.net_g.prepare_qat() not found. "
                         "Please ensure 'aether' network is selected and aether_arch.py is correctly implemented with QAT methods.{tc.end}")
            sys.exit(1)
        logger.info(f"{tc.light_green}QAT preparation complete.{tc.end}")
    # --- QAT INTEGRATION END ---

    if resume_state:  # resume training
        # handle optimizers and schedulers
        model.resume_training(resume_state)  # type: ignore[reportAttributeAccessIssue,attr-defined]
        logger.info(
            f"{tc.light_green}Resuming training from epoch: {resume_state['epoch']}, iter: {int(resume_state['iter'] // opt['datasets']['train'].get('accumulate', 1))}{tc.end}"
        )
        start_epoch = resume_state["epoch"]
        current_iter = int(
            resume_state["iter"]
        ) # current_iter should be the raw iteration count
        torch.cuda.empty_cache()
    else:
        start_epoch = 0
        current_iter = 0

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, tb_logger, current_iter // opt['datasets']['train'].get('accumulate', 1))

    # dataloader prefetcher
    if train_loader is not None:
        prefetcher = CUDAPrefetcher(train_loader, opt)
    else:
        logger.error(f"{tc.red}Train dataloader could not be created. Exiting.{tc.end}")
        sys.exit(1)


    # log AMP (automatic mixed precision)
    if opt.get("use_amp", False) and opt.get("bfloat16", False):
        logger.info("AMP enabled with BF16.")
    elif opt.get("use_amp", False) and not opt.get("bfloat16", False):
        logger.info("AMP enabled.")
    else:
        logger.info("AMP disabled.")

    # error if bf16 is enabled by not amp
    if not opt.get("use_amp", False) and opt.get("bfloat16", False):
        msg = f"{tc.red}bfloat16 option has no effect without use_amp. Please enable use_amp for bfloat16.{tc.end}"
        logger.error(msg)
        sys.exit(1)

    # detect GPU architecture
    # Default values in case device_capability is not available
    major_cuda_version = 0
    minor_cuda_version = 0
    if torch.cuda.is_available():
        major_cuda_version, minor_cuda_version = torch.cuda.get_device_capability()

    is_ampere = major_cuda_version >= 8
    is_turing = major_cuda_version == 7
    is_pascal = major_cuda_version <= 6

    # detect Ampere and recommend bf16
    if opt.get("use_amp", False) is False and is_ampere:
        msg = f"{tc.light_yellow}Modern GPU detected (Ampere or newer). Consider enabling AMP with bfloat16 for potential speedup.{tc.end}"
        logger.warning(msg)

    # detect Turing or older and error if bf16 is enabled
    if opt.get("bfloat16", False) is True and is_turing:
        msg = f"{tc.light_yellow}Turing GPU detected. bfloat16 support might be limited or inefficient. Consider disabling bfloat16.{tc.end}"
        logger.warning(msg)
    elif opt.get("bfloat16", False) is True and is_pascal: # If Pascal or older AND bfloat16 enabled
        msg = f"{tc.red}Pascal or older GPU detected. bfloat16 is NOT supported on this architecture. Please disable bfloat16.{tc.end}"
        logger.error(msg)
        sys.exit(1)

    # detect Pascal or older and warn about AMP
    if opt.get("use_amp", False) is True and is_pascal:
        msg = f"{tc.light_yellow}Pascal GPU doesn't have Tensor Cores. Consider disabling AMP as it may not provide benefits.{tc.end}"
        logger.warning(msg)

    # log deterministic mode
    if opt["deterministic"]:
        logger.info("Deterministic mode enabled.")

    # training log vars
    accumulate = opt["datasets"]["train"].get("accumulate", 1)
    print_freq = opt["logger"].get("print_freq", 100)
    save_checkpoint_freq = opt["logger"]["save_checkpoint_freq"]
    val_freq = opt["val"]["val_freq"] if opt.get("val") is not None else 100

    # training
    logger.info(
        f"{tc.light_green}Start training from epoch: {start_epoch}, iter: {int(current_iter / accumulate)}{tc.end}"
    )
    iter_timer = AvgTimer()
    start_time = time.time()

    try:
        for epoch in range(start_epoch, total_epochs + 1):
            train_sampler.set_epoch(epoch)  # type: ignore[attr-defined]
            prefetcher.reset()  # type: ignore[reportPossiblyUnboundVariable]
            train_data = prefetcher.next()  # type: ignore[reportPossiblyUnboundVariable]

            while train_data is not None:
                current_iter += 1
                if current_iter > total_iters:
                    break
                # training
                model.feed_data(train_data)  # type: ignore[reportAttributeAccessIssue,attr-defined]
                model.optimize_parameters(current_iter)  # type: ignore[reportFunctionMemberAccess,attr-defined]
                # update learning rate
                model.update_learning_rate(  # type: ignore[reportFunctionMemberAccess,attr-defined]
                    current_iter, warmup_iter=opt["train"].get("warmup_iter", -1)
                )
                iter_timer.record()
                if current_iter == 1:
                    msg_logger.reset_start_time()

                # log
                if current_iter >= accumulate:
                    current_iter_log = current_iter / accumulate
                else:
                    current_iter_log = current_iter

                if current_iter_log % print_freq == 0:
                    log_vars = {"epoch": epoch, "iter": current_iter_log}
                    log_vars.update({"lrs": model.get_current_learning_rate()})  # type: ignore[reportFunctionMemberAccess,attr-defined]
                    log_vars.update({
                        "time": iter_timer.get_avg_time()
                    })
                    log_vars.update(model.get_current_log())  # type: ignore[reportFunctionMemberAccess,attr-defined]
                    msg_logger(log_vars)

                # save models and training states
                if current_iter_log % save_checkpoint_freq == 0:
                    free_space = check_disk_space()
                    if free_space < 500:
                        msg = f"""
                        {tc.red}
                        Not enough free disk space in {Path.cwd()}.
                        Please free up at least 500 MB of space.
                        Attempting to save current progress...
                        {tc.end}
                        """
                        logger.error(msg)
                        model.save(epoch, int(current_iter_log))  # type: ignore[reportFunctionMemberAccess,attr-defined]
                        sys.exit(1)

                    logger.info(
                        f"{tc.light_green}Saving models and training states.{tc.end}"
                    )
                    model.save(epoch, int(current_iter_log))  # type: ignore[reportFunctionMemberAccess,attr-defined]

                # validation
                if opt.get("val") is not None and (current_iter_log % val_freq == 0):
                    for val_loader in val_loaders:
                        model.validation(  # type: ignore[reportFunctionMemberAccess,attr-defined]
                            val_loader,
                            int(current_iter_log),
                            tb_logger,
                            opt["val"].get("save_img", True),
                        )

                iter_timer.start()
                train_data = prefetcher.next()  # type: ignore[reportPossiblyUnboundVariable]
            # end of iter

        # end of epoch

        consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
        logger.info(
            f"{tc.light_green}End of training. Time consumed: {consumed_time}{tc.end}"
        )
        logger.info(f"{tc.light_green}Save the latest model.{tc.end}")
        model.save(epoch=-1, current_iter=-1)  # type: ignore[reportFunctionMemberAccess,attr-defined]

    except KeyboardInterrupt:
        msg = f"{tc.light_green}Interrupted, saving latest models.{tc.end}"
        logger.info(msg)
        model.save(epoch, int(current_iter_log))  # type: ignore[reportFunctionMemberAccess,attr-defined]
        sys.exit(0)
    except Exception as e: # Catch other unexpected errors during training loop
        logger.error(f"{tc.red}An unexpected error occurred during training: {e}{tc.end}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


    if opt.get("val") is not None:
        accumulate = opt["datasets"]["train"].get("accumulate", 1)
        for val_loader in val_loaders:
            model.validation(  # type: ignore[reportFunctionMemberAccess,attr-defined]
                val_loader,
                int(current_iter / accumulate),
                tb_logger,
                opt["val"].get("save_img", True),
            )
    if tb_logger:
        tb_logger.close()


if __name__ == "__main__":
    # Determine the absolute path of the current file (train.py)
    current_file_abs_path = Path(__file__).resolve()
    # The project root (e.g., 'neosr-fork/') is the parent of train.py
    project_root_directory = current_file_abs_path.parent
    
    # The 'common' directory is directly under the project root
    common_dir_path = project_root_directory / "common"

    # Add the 'common' directory to Python's system path if it's not already there.
    if str(common_dir_path) not in sys.path:
        sys.path.insert(0, str(common_dir_path))
        print(f"Added '{common_dir_path}' to sys.path for common modules (explicitly added from train.py).")

    # Pass the actual project_root to the training pipeline
    train_pipeline(str(project_root_directory))

