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
from contextlib import suppress

import torch

import yaml

import nvdlfw_inspect.api as debug_api
from nvdlfw_inspect.config_manager import ConfigManager, ConfigSpec, is_layer_in_cfg
from nvdlfw_inspect.debug_features.generic_feature_api import GenericConfigAPIMapper
from nvdlfw_inspect.registry import Registry


def test_layer_selection():
    # Layer number selection
    config1 = {"enabled": True, "layers": {"layer_numbers": [1, 2, 3]}}

    config_spec = ConfigSpec()
    config_spec.initialize(config1)
    assert is_layer_in_cfg("decoder.1.mlp.fc1", config_spec)
    assert is_layer_in_cfg("decoder.2.mlp.fc1", config_spec)
    assert not is_layer_in_cfg("decoder.12.mlp.fc1", config_spec)
    config_spec.reset()

    # Layer number + layer type selection
    config2 = {
        "enabled": True,
        "layers": {"layer_numbers": [1, 2, 3], "layer_types": ["mlp", "qkv"]},
    }

    config_spec.initialize(config2)
    assert is_layer_in_cfg("decoder.1.mlp.fc1", config_spec)
    assert not is_layer_in_cfg("decoder.12.mlp.fc1", config_spec)
    assert not is_layer_in_cfg("decoder.3.attn.proj", config_spec)
    assert is_layer_in_cfg("decoder.3.attn.qkv", config_spec)
    config_spec.reset()

    # All layers selection
    config3 = {
        "enabled": True,
        "layers": {"layer_numbers": ["all"], "layer_types": ["all"]},
    }

    config_spec.initialize(config3)
    assert is_layer_in_cfg("decoder.1.mlp.fc1", config_spec)
    assert is_layer_in_cfg("decoder.12.mlp.fc1", config_spec)
    assert is_layer_in_cfg("decoder.3.attn.proj", config_spec)
    assert is_layer_in_cfg("decoder.3.attn.qkv", config_spec)
    config_spec.reset()

    # Regex pattern selection
    config4 = {
        "enabled": True,
        "layers": {"layer_name_regex_pattern": r".*1\..*attn..*(proj|qkv)"},
    }

    config_spec.initialize(config4)
    assert not is_layer_in_cfg("decoder.1.mlp.fc1", config_spec)
    assert not is_layer_in_cfg("decoder.12.mlp.fc1", config_spec)
    assert is_layer_in_cfg("decoder.1.attn.qkv", config_spec)
    assert not is_layer_in_cfg("decoder.3.attn.proj", config_spec)
    assert not is_layer_in_cfg("decoder.12.attn.proj", config_spec)
    config_spec.reset()
    Registry.reset()


def test_multiple_configs():
    debug_api.initialize(
        config_file=str(
            pathlib.Path(__file__).resolve().parent
            / "test_configs/basic_stat_collection.yaml"
        )
    )

    cfg1 = ConfigManager.get_config_for_layer("decoder.1.self_attention.qkv")
    cfg2 = ConfigManager.get_config_for_layer("decoder.2.mlp.fc1")

    assert cfg1["base"]["LogTensorStats"]["tensors"] == ["gradient"]
    assert cfg2["base"]["LogTensorStats"]["tensors"] == ["activation"]

    with suppress(ValueError):
        # This should fail.
        cfg2 = ConfigManager.get_config_for_layer("decoder.2.mlp.fc2")

    debug_api.end_debug()


def test_multiple_configs_dict():
    # Read the YAML file and convert it to a Python dict
    config_path = (
        pathlib.Path(__file__).resolve().parent
        / "test_configs/basic_stat_collection.yaml"
    )
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    debug_api.initialize(config_file=config_dict)

    cfg1 = ConfigManager.get_config_for_layer("decoder.1.self_attention.qkv")
    cfg2 = ConfigManager.get_config_for_layer("decoder.2.mlp.fc1")

    assert cfg1["base"]["LogTensorStats"]["tensors"] == ["gradient"]
    assert cfg2["base"]["LogTensorStats"]["tensors"] == ["activation"]

    with suppress(ValueError):
        # This should fail.
        cfg2 = ConfigManager.get_config_for_layer("decoder.2.mlp.fc2")

    debug_api.end_debug()


def test_config_parsing():
    debug_api.initialize(
        config_file=pathlib.Path(__file__).resolve().parent
        / "test_configs/stats_collection_test_config.yaml"
    )

    cfg_fc1 = ConfigManager.get_config_for_layer("decoder.1.mlp.fc1")["base"]
    cfg_fc2 = ConfigManager.get_config_for_layer("decoder.2.mlp.fc2")["base"]
    assert cfg_fc1 and cfg_fc2

    tensor_parsing = True
    tensor = torch.randn(10, 10)

    ret, _ = GenericConfigAPIMapper().parse_config_and_api(
        cfg_fc1["LogTensorStats"],
        tensor=tensor,
        tensor_parsing=tensor_parsing,
        tensor_name="weight",
    )
    assert not ret

    ret, _ = GenericConfigAPIMapper().parse_config_and_api(
        cfg_fc2["LogTensorStats"],
        tensor=tensor,
        tensor_parsing=tensor_parsing,
        tensor_name="activation",
    )
    assert not ret

    ret, parsed_cfg_fc1 = GenericConfigAPIMapper().parse_config_and_api(
        cfg_fc1["LogTensorStats"],
        tensor=tensor,
        tensor_parsing=tensor_parsing,
        tensor_name="activation",
    )
    assert ret
    assert parsed_cfg_fc1 == {
        "tensor": "activation",
        "stats": ["mean", "std", "l1_norm", "l2_norm"],
        "freq": 1,
        "start_step": 100,
        "end_step": 500,
    }

    debug_api.end_debug()


def test_config_parsing_dict():
    config_path = (
        pathlib.Path(__file__).resolve().parent
        / "test_configs/stats_collection_test_config.yaml"
    )
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    debug_api.initialize(config_file=config_dict)

    cfg_fc1 = ConfigManager.get_config_for_layer("decoder.1.mlp.fc1")["base"]
    cfg_fc2 = ConfigManager.get_config_for_layer("decoder.2.mlp.fc2")["base"]
    assert cfg_fc1 and cfg_fc2

    tensor_parsing = True
    tensor = torch.randn(10, 10)

    ret, _ = GenericConfigAPIMapper().parse_config_and_api(
        cfg_fc1["LogTensorStats"],
        tensor=tensor,
        tensor_parsing=tensor_parsing,
        tensor_name="weight",
    )
    assert not ret

    ret, _ = GenericConfigAPIMapper().parse_config_and_api(
        cfg_fc2["LogTensorStats"],
        tensor=tensor,
        tensor_parsing=tensor_parsing,
        tensor_name="activation",
    )
    assert not ret

    ret, parsed_cfg_fc1 = GenericConfigAPIMapper().parse_config_and_api(
        cfg_fc1["LogTensorStats"],
        tensor=tensor,
        tensor_parsing=tensor_parsing,
        tensor_name="activation",
    )
    assert ret
    assert parsed_cfg_fc1 == {
        "tensor": "activation",
        "stats": ["mean", "std", "l1_norm", "l2_norm"],
        "freq": 1,
        "start_step": 100,
        "end_step": 500,
    }

    debug_api.end_debug()


if __name__ == "__main__":
    pass
