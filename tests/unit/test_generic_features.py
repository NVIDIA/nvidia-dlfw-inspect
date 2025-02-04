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

import pathlib

import torch
from utils import reset_debug_log

import nvdlfw_inspect.api as debug_api


def test_statistics_collection():
    debug_api.initialize(
        config_file=pathlib.Path(__file__).resolve().parent
        / "test_configs/stats_collection_test_config.yaml",
        default_logging_enabled=False,
    )

    tensor1 = torch.rand((100, 100, 5))
    tensor2 = torch.rand((100, 5))
    tensor3 = torch.rand((100, 50))

    activation_stats = debug_api.log_tensor_stats(
        "decoder.1.mlp.fc1", tensor=tensor1, tensor_name="activation", iteration=101
    )
    gradient_stats = debug_api.log_tensor_stats(
        "decoder.1.mlp.fc1", tensor=tensor2, tensor_name="gradient", iteration=66
    )
    weight_stats = debug_api.log_tensor_stats(
        "decoder.2.mlp.fc1", tensor=tensor3, tensor_name="weight", iteration=90
    )

    for stat in ["mean", "std", "l1_norm", "l2_norm"]:
        assert stat in activation_stats

    for stat in ["max", "l1_norm", "min"]:
        assert stat in gradient_stats

    for stat in ["mean", "l1_norm"]:
        assert stat in weight_stats

    assert activation_stats["mean"] == tensor1.mean()
    assert gradient_stats["max"] == tensor2.max()
    assert weight_stats["mean"] == tensor3.mean()

    activation_stats = debug_api.log_tensor_stats(
        "decoder.1.mlp.fc1", tensor=tensor1, tensor_name="activation", iteration=11
    )
    gradient_stats = debug_api.log_tensor_stats(
        "decoder.1.mlp.fc1", tensor=tensor2, tensor_name="gradient", iteration=65
    )
    weight_stats = debug_api.log_tensor_stats(
        "decoder.2.mlp.fc1", tensor=tensor3, tensor_name="weight", iteration=190
    )

    assert activation_stats == {}
    assert gradient_stats == {}
    assert weight_stats == {}

    debug_api.end_debug()
    reset_debug_log()
