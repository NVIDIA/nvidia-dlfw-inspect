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

import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime

import torch.distributed as dist

LOGGER = None


class BaseLogger(ABC):
    """
    Reference implementation for Logger.
    Tensorboard and WandB loggers should follow this implementation.
    """

    @abstractmethod
    def log_scalar(self, name: str, value: float | int, iteration: int, **kwargs):
        """
        Log scalar.
        Similar to Tensorboard's add_scalar API.
        """
        pass


class FileLogger(BaseLogger):
    def __init__(self, name: str, root_log_dir: str):
        super().__init__()
        self.logger = logging.getLogger(name=name)
        self.root_log_dir = root_log_dir
        self.name = name.lower()
        log_level = os.getenv("NVDLFW_INSPECT_LOG_LEVEL", "INFO").upper()
        numeric_level = getattr(logging, log_level, logging.INFO)
        self.logger.setLevel(numeric_level)
        self.logged_api_executed = set()
        self.logged_api_encountered = set()

    def _get_log_file_name(
        self, log_dir, overwrite: bool = True, add_timestamp: bool = False
    ):
        rank = 0
        if dist.is_initialized():
            rank = dist.get_rank()

        log_file_name = self.name
        if "NVDLFW_INSPECT_ENABLE_LOG_TIMESTAMP" in os.environ:
            add_timestamp = int(os.environ["NVDLFW_INSPECT_ENABLE_LOG_TIMESTAMP"])

        if "NVDLFW_INSPECT_ENABLE_LOG_OVERWRITE" in os.environ:
            overwrite = int(os.environ["NVDLFW_INSPECT_ENABLE_LOG_OVERWRITE"])

        if add_timestamp:
            if "NVDLFW_INSPECT_LOG_TIMESTAMP" not in os.environ:
                os.environ["NVDLFW_INSPECT_LOG_TIMESTAMP"] = datetime.now().strftime(
                    "%Y%m%d_%H%M%S"
                )  # noqa: DTZ005
            timestamp = os.environ["NVDLFW_INSPECT_LOG_TIMESTAMP"]
            log_file_name += f"_{timestamp}"

        log_file_name += f"_globalrank-{rank}.log"

        log_file_name = os.path.join(self.root_log_dir, log_dir, log_file_name)
        if overwrite:
            open(log_file_name, "w").close()

        return log_file_name

    def initialize(
        self, log_dir: str, overwrite: bool = True, add_timestamp: bool = False
    ):
        os.makedirs(os.path.join(self.root_log_dir, log_dir), exist_ok=True)

        log_file_name = self._get_log_file_name(log_dir, overwrite, add_timestamp)

        # file handler
        file_handler = logging.FileHandler(log_file_name)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )

        # stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(
            logging.Formatter(
                "NVDLFW INSPECT - %(asctime)s - %(levelname)s - %(message)s"
            )
        )
        stream_handler.setLevel(logging.WARNING)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

        self.logger.propagate = False

        return self

    def _initialize_stats_logger(
        self, log_dir, overwrite: bool = True, add_timestamp: bool = False
    ):
        os.makedirs(os.path.join(self.root_log_dir, log_dir), exist_ok=True)
        log_file_name = self._get_log_file_name(log_dir, overwrite, add_timestamp)

        file_handler = logging.FileHandler(log_file_name)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )

        self.stats_logger = logging.getLogger(name="Default Statistics")
        self.stats_logger.addHandler(file_handler)
        self.stats_logger.setLevel(logging.INFO)
        self.stats_logger.propagate = False

    def log_scalar(self, name: str, value: float | int, iteration: int, **kwargs):
        if not hasattr(self, "stats_logger"):
            raise AttributeError(
                "[NVDLFW INSPECT ERROR] Default stats logger not initialized."
            )

        scalar_message_template = "{} \t\t\t\t iteration={:06d} \t\t\t\t value={:.4f}"
        level = kwargs.get("level", logging.INFO)
        self.stats_logger.log(
            level, scalar_message_template.format(name, iteration, value)
        )

    def log_message(self, message: str, **kwargs):
        level = kwargs.get("level", logging.INFO)
        self.logger.log(level, message)


class MetricLogger:
    enabled_loggers = []  # noqa: RUF012

    @staticmethod
    def log_scalar(name: str, value: float | int, iteration: int, **kwargs):
        for logger in MetricLogger.enabled_loggers:
            logger.log_scalar(name, value, iteration, **kwargs)

    @staticmethod
    def add_logger(_logger: BaseLogger):
        MetricLogger.enabled_loggers.append(_logger)


class TensorboardLogger(BaseLogger):
    """
    This is used when the user provides a TB writer.
    """

    def __init__(self, tb_writer):
        super().__init__()
        self.tb_writer = tb_writer

    def log_scalar(self, name: str, value: float | int, iteration: int, **kwargs):
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(name, value, iteration)


def wrap_tensorboard_writer(tb_writer) -> TensorboardLogger:
    """
    Helper function to wrap a Tensorboard SummaryWriter into BaseLogger format for logging.
    """
    return TensorboardLogger(tb_writer)


def print_n(*args):
    for arg in args:
        print(arg, end=" ")
    print()


def print_rank_0(*message):
    """If distributed is initialized, print only on rank 0."""
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print_n(*message)
    else:
        print_n(*message)


def enable_logging_at_current_step(
    cur_step, freq, start_step, end_step, start_end_list
):
    """
    Returns true if logging should be enabled for current_step

    Args:
    cur_step: int, current train step
    freq: int, logging frequency, default = 1.
    start_step: int, logging start step, default = 0,
    end_step: int, logging end step, default = -1 (don't stop logging once started).
    start_end_list: list([int, int]), list of start and end pairs that are non-overlapping and incremental, default = None
    """
    if start_end_list:
        for start, end in start_end_list:
            if (
                cur_step >= start
                and ((cur_step < end) or end == -1)
                and cur_step % freq == 0
            ):
                return True
    else:
        if cur_step >= start_step and cur_step % freq == 0:
            return True if end_step == -1 else cur_step <= end_step
    return False


def initialize_logger(log_dir):
    global LOGGER
    GENERIC_LOG_DIR = "nvdlfw_inspect_logs"  # noqa: N806
    LOGGER = FileLogger("nvdlfw_inspect", log_dir)
    LOGGER.initialize(GENERIC_LOG_DIR)


def get_logger():
    global LOGGER
    if LOGGER is None:
        raise RuntimeError(
            "[NVDLFW INSPECT ERROR] LOGGER not initialized. Call initialize first."
        )
    return LOGGER


def make_default_statistics_logger():
    global LOGGER
    if LOGGER is None:
        raise RuntimeError(
            "[NVDLFW INSPECT ERROR] LOGGER not initialized. Call initialize first."
        )
    STATISTICS_LOG_DIR = "nvdlfw_inspect_statistics_logs"  # noqa: N806
    LOGGER._initialize_stats_logger(STATISTICS_LOG_DIR)
    return LOGGER


def custom_assert(condition, message, level=logging.ERROR):
    """
    Custom assert function that logs the message before raising an AssertionError.

    Args:
        condition (bool): The condition to assert.
        message (str): The message to log and include in the AssertionError.
        layer_name (str, optional): The layer name to include in the log message.
        level (int, optional): The logging level.
    """
    if not condition:
        LOGGER.log_message(message, level=level)
        raise AssertionError(message)
