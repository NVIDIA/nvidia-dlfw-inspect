# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import logging

import torch
import torch.distributed

from .logging import BaseLogger, get_logger, initialize_logger
from .utils import APICacheIdentifier, SingleMessageLogger

DEBUG_MANAGER = None


def initialize(
    config_file="",
    feature_dirs=None,
    statistics_logger: BaseLogger | None = None,
    log_dir=".",
    init_training_step=0,
    **kwargs,
):
    """
    API to initialize debug tool. Must be called once on every rank in global context.

    Parameters:
    config_file: str, path to config.yaml file containing features to enabled and layer names.
                default = ""
    feature_dirs: List[str] | str, A list of directories containing features to load and register. All files and folders in the directory are read recursively.
    statistics_logger: (Union[BaseLogger, None]): The logger to use for logging tensor statistics. If provided, it should follow reference implementation `BaseLogger`.
                If statistics logging is enabled, a default file logger will be created if logger is None.
                default = None
    log_dir: str, path to directory which will hold the 'debug_logs' and 'debug_statistics_logs'
                default = "."
    init_training_step: int, set training step iteration
                default = 0
    **kwargs:
        tb_writer (TensorBoardWriter, optional): A TensorBoard writer to use for logging.
                default = None
        default_logging_enabled (bool, optional): Whether to enable default logging to file if statistics collection is enabled.
                default = False
    """
    global DEBUG_MANAGER

    from .debug_manager import DebugManager

    if DEBUG_MANAGER is not None:
        raise ValueError("[NVDLFW INSPECT ERROR] Already initialized.")
    initialize_logger(log_dir)
    DEBUG_MANAGER = DebugManager(
        config_file,
        feature_dirs=feature_dirs,
        statistics_logger=statistics_logger,
        log_dir=log_dir,
        init_training_step=init_training_step,
        **kwargs,
    )
    DEBUG_MANAGER.load_config()


def log_message(msg, layer_name=None, level=logging.INFO, extra_cachable_args=None):
    global DEBUG_MANAGER
    if not DEBUG_MANAGER:
        raise RuntimeError(
            "[NVDLFW INSPECT ERROR] Debug manager not initialized. Call initialize first."
        )
    APICacheIdentifier.save_call_details()
    SingleMessageLogger.log_message_once(msg, layer_name, level, extra_cachable_args)


def end_debug():
    global DEBUG_MANAGER
    if DEBUG_MANAGER is not None:
        DEBUG_MANAGER.close()
        DEBUG_MANAGER = None


def initialize_training_step(train_step: int):
    global DEBUG_MANAGER
    if DEBUG_MANAGER is not None:
        DEBUG_MANAGER.initialize_training_step(train_step)


def step():
    global DEBUG_MANAGER
    if DEBUG_MANAGER is not None:
        DEBUG_MANAGER.step()


def list_features():
    global DEBUG_MANAGER
    DEBUG_MANAGER.list_features()


def explain_features(features=None):
    global DEBUG_MANAGER
    DEBUG_MANAGER.explain_features(features)


def infer_and_assign_layer_names(model: torch.nn.Module | list):
    """
    Infer and assign layer names to all modules in a model.
    """

    def _helper(module, prefix=""):
        module_name = f"{prefix}"

        # If ModuleList is found, loop over children and get layer number for each child module.
        # NOTE: This assumes that layer_number is an attribute of the module as it is in megatron-core.
        # If layer number is None, it falls back to default name which will be the same on all PP ranks.
        if isinstance(module, torch.nn.ModuleList):
            for name, child in module.named_children():
                layer_number = getattr(child, "layer_number", name)
                _helper(child, f"{module_name}.{layer_number}")

        elif isinstance(module, torch.nn.Module):
            module.name = module_name
            get_logger().log_message(
                f"Assigned layer name: {module_name}", level=logging.INFO
            )
            for name, child in module.named_children():
                _helper(child, f"{module_name}.{name}")

        else:
            m = module.module
            if m is not None:
                _helper(m, prefix)
            else:
                raise ValueError

    if isinstance(model, list):
        for m in model:
            _helper(m, "model")
    else:
        _helper(model, "model")


def set_tensor_reduction_group(group: torch.distributed.ProcessGroup, layer_name=None):
    global DEBUG_MANAGER
    if DEBUG_MANAGER is None:
        raise ValueError(
            "[NVDLFW INSPECT ERROR] Debug Tool should be initialized using initialize()."
        )
    DEBUG_MANAGER._tensor_reduction_group = group
    log_message(
        "Reduction group initialized for tensor reduction before logging statistics. \
                If per-rank statistics are required, pass `skip_reduction=True` when invoking the API. \
                To pass another reduction group, use `reduction_group` kwarg when invoking the API.",
        level=logging.WARNING,
    )


def get_tensor_reduction_group():
    global DEBUG_MANAGER
    if DEBUG_MANAGER is None:
        raise ValueError(
            "[NVDLFW INSPECT ERROR] Debug Tool should be initialized using initialize()."
        )
    return DEBUG_MANAGER._tensor_reduction_group


def __getattr__(name):
    global DEBUG_MANAGER
    if not DEBUG_MANAGER:
        raise RuntimeError(
            "[NVDLFW INSPECT ERROR] Debug manager not initialized. Call initialize first."
        )
    APICacheIdentifier.save_call_details()
    return DEBUG_MANAGER.get_extension_api(name)
