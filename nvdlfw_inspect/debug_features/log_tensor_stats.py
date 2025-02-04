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

import torch

import nvdlfw_inspect.api as debug_api
from nvdlfw_inspect.debug_features.generic_feature_api import GenericConfigAPIMapper
from nvdlfw_inspect.logging import MetricLogger, enable_logging_at_current_step
from nvdlfw_inspect.registry import Registry, api_method
from nvdlfw_inspect.utils import (
    append_parent_docstring,
    gather_along_first_dim,
    gather_tensor_on_last_rank,
)


@Registry.register_feature(namespace="base")
@append_parent_docstring(parent=GenericConfigAPIMapper)
class LogTensorStats(GenericConfigAPIMapper):
    """
    Log tensor statistics.

    APIs:
    1. log_tensor_stats
    - When using this api, you would need to pass args:
    -- layer_name: this is matched with the layer description in the config file
    -- tensor_name: this is matched with one of the tensors in the config field, and passed as a kwarg. For example, tensor_name='tensor1'
    -- tensor: the tensor to process, and passed as a kwarg. For example, tensor={torch tensor}
    - Optional kwargs:
    -- skip_reduction (default: False): skip reduction of tensor stats across GPU ranks. Each GPU rank will log its local stats.
    if skip_reduction is not set, this api only checks for DDP and reduces tensor on last rank.
    -- reduction_group: (default: None): Provide torch distributed process group for collecting tensor from GPU ranks that are part of this group.
    (For enabling tensor reduction across different parallelisms DP, TP, PP, etc)
    -- iteration: option to pass training step for logging. if using step() api of this tool in the training loop, this arg is not needed.

    Config:

    To enable the feature in yaml config:
    LogTensorStats:
      enabled: True
      feature_properties:
      ...

    Config fields:
    This feature works at a tensor level, you can set the following properties for each tensor:
    - stats: List[str], type of statistics to log. Options: {min, max, mean, std, l1_norm, l2_norm}
    - freq: int, logging frequency in training steps. Default = 1.
    - start_step: int, train step to start logging. Default = 0.
    - end_step: int, train step to end logging. Default = -1 (don't stop logging once started)
    - start_end_list: list([int, int]), non-overlaping list of (start, end) pairs in incremental order. Default = None. If not None, will ignore start_step and end_step
    """

    return_tensor = False

    def _enable_return_tensor(self):
        self.return_tensor = True

    def _disable_return_tensor(self):
        self.return_tensor = False

    def _is_tensor_return_enabled(self):
        return self.return_tensor

    def _check_log_frequency(self, config, **kwargs):
        if config.get("freq", None) is None:
            debug_api.log_message(
                "Frequency of logging is not provided. Using freq = 1 train step as default.",
                level=logging.WARNING,
            )
            freq = 1
        else:
            freq = int(config["freq"])

        iteration = self._get_current_iteration(**kwargs)
        return enable_logging_at_current_step(
            iteration,
            freq,
            config.get("start_step", 0),
            config.get("end_step", -1),
            config.get("start_end_list", None),
        )

    def _check_params(self, config, layer_name, **kwargs):
        return self._check_log_frequency(config, **kwargs)

    def _check_and_gather_tensor(self, config, layer_name, **kwargs):
        skip_reduction = kwargs.get("skip_reduction", False)
        if skip_reduction:
            return kwargs["tensor"], None

        # override global group
        reduction_group = debug_api.get_tensor_reduction_group()
        if kwargs.get("reduction_group", None) is not None:
            reduction_group = kwargs["reduction_group"]

        if not reduction_group:
            debug_api.log_message(
                "`reduction_group` not found and `skip_reduction` is False. "
                + "Tensor will be only reduced along DP group if initialzed. "
                + "If this not the desired behavior, pass `reduction_group` when using `log_tensor_stats` feature API. "
                + "Per-GPU stats are logged in `nvdlfw_inspect_statistics_logs`",
                layer_name=layer_name,
                level=logging.WARNING,
            )
            return gather_tensor_on_last_rank(kwargs["tensor"])

        debug_api.log_message(
            f"Feature={self.__class__.__name__}, API=log_tensor_stats: {kwargs['tensor_name']}: Reduction group found",
            layer_name=layer_name,
            level=logging.DEBUG,
        )
        return gather_along_first_dim(kwargs["tensor"], process_group=reduction_group)

    def _get_supported_stats_list(self):
        return {"min", "max", "mean", "std", "l1_norm", "l2_norm"}

    def _get_current_iteration(self, **kwargs):
        return kwargs.get("iteration", debug_api.DEBUG_MANAGER._trainer_iteration_count)

    @api_method
    def log_tensor_stats(self, config, layer_name, **kwargs):  # noqa: C901
        debug_api.log_message(
            f"Feature={self.__class__.__name__}, API=log_tensor_stats, TENSOR={kwargs['tensor_name']}: Called",
            layer_name=layer_name,
            level=logging.INFO,
        )
        if not self._check_params(config, layer_name, **kwargs):
            if self._is_tensor_return_enabled():
                return {}, None
            else:
                return {}

        gathered_tensor, _ = self._check_and_gather_tensor(config, layer_name, **kwargs)
        iteration = self._get_current_iteration(**kwargs)

        stats = {}
        non_supported_stats_list = []
        if torch.is_tensor(gathered_tensor):
            for stat in config.get("stats", []):
                if stat.lower() not in self._get_supported_stats_list():
                    non_supported_stats_list.append(stat.lower())
                    continue

                if stat.lower() == "min":
                    stats["min"] = gathered_tensor.min().float()
                    MetricLogger.log_scalar(
                        f"{layer_name}_{kwargs['tensor_name']}_min",
                        stats["min"],
                        iteration,
                    )
                elif stat.lower() == "max":
                    stats["max"] = gathered_tensor.max().float()
                    MetricLogger.log_scalar(
                        f"{layer_name}_{kwargs['tensor_name']}_max",
                        stats["max"],
                        iteration,
                    )
                elif stat.lower() == "mean":
                    stats["mean"] = gathered_tensor.mean().float()
                    MetricLogger.log_scalar(
                        f"{layer_name}_{kwargs['tensor_name']}_mean",
                        stats["mean"],
                        iteration,
                    )
                elif stat.lower() == "std":
                    stats["std"] = gathered_tensor.std().float()
                    MetricLogger.log_scalar(
                        f"{layer_name}_{kwargs['tensor_name']}_std",
                        stats["std"],
                        iteration,
                    )
                elif stat.lower() == "l1_norm":
                    stats["l1_norm"] = torch.norm(gathered_tensor, p=1).float()
                    MetricLogger.log_scalar(
                        f"{layer_name}_{kwargs['tensor_name']}_l1_norm",
                        stats["l1_norm"],
                        iteration,
                    )
                elif stat.lower() == "l2_norm":
                    stats["l2_norm"] = torch.norm(gathered_tensor, p=2).float()
                    MetricLogger.log_scalar(
                        f"{layer_name}_{kwargs['tensor_name']}_l2_norm",
                        stats["l2_norm"],
                        iteration,
                    )

        if non_supported_stats_list:
            debug_api.log_message(
                f"Statistic: [{non_supported_stats_list}] are not supported",
                layer_name=layer_name,
                level=logging.ERROR,
            )
            raise ValueError(
                f"Statistic: [{non_supported_stats_list}] are not supported"
            )

        if self._is_tensor_return_enabled():
            return stats, gathered_tensor
        else:
            return stats
