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
import re
from abc import abstractmethod
import pathlib

import yaml

from nvdlfw_inspect.logging import get_logger, print_rank_0


def feature_not_found_error(feature_name, namespace):
    return f"[NVDLFW INSPECT ERROR] Could not find feature '{feature_name}' in namespace '{namespace}' while loading config."


class BaseSpec:
    def __init__(self):
        pass

    @abstractmethod
    def verify_validity(self):
        pass

    @abstractmethod
    def initialize(self, cfg):
        pass


def spec_attribute_iterator(spec: BaseSpec):
    all_attrs = inspect.getmembers(spec, lambda a: not (inspect.isroutine(a)))
    for attr in all_attrs:
        if not (attr[0].startswith("__") and attr[0].endswith("__")):
            yield (attr[0], attr[1])


def set_all_spec_attributes(spec: BaseSpec, cfg):
    """Sets all attributes of a particular spec from the config file."""

    for name, _ in spec_attribute_iterator(spec):
        if name in cfg:
            setattr(spec, name, cfg[name])


class LayerSpec(BaseSpec):
    layer_numbers: set[int] = None
    layer_types: list[str] = None
    exact_layer_names: list[str] = None
    layer_name_regex_pattern: str = None
    layer_type_regex_pattern: str = None

    def verify_validity(self):
        assert any(
            [
                self.layer_types is not None,
                self.exact_layer_names is not None,
                self.layer_numbers is not None,
                self.layer_name_regex_pattern is not None,
            ]
        ), (
            "Atleast one of 'layer_numbers', 'layer_types', 'exact_layer_names' or 'layer_name_regex_pattern' must be provided."
        )
        if self.layer_types is not None:
            pattern = "|".join(map(re.escape, self.layer_types))
            self.layer_type_regex_pattern = re.compile(pattern)
        # TODO: verification must also be done against all of the provided layer names.
        # However, layernames can only be registered after modules are initialized.
        # But API is initialized and config read before modules are initialized.

    def initialize(self, cfg):
        set_all_spec_attributes(self, cfg)

        if "layer_numbers" in cfg:
            layer_numbers = self._parse_layer_numbers(cfg["layer_numbers"])
            self.layer_numbers = layer_numbers

        if self.layer_name_regex_pattern is not None:
            self.layer_name_regex_pattern = re.compile(self.layer_name_regex_pattern)

    def _parse_layer_numbers(self, raw_inp: list[int | str] | int | str) -> set[int]:
        """
        Parse layer_numbers argument provided in the config.

        Returns a set() for O(1) lookup during training.
        """

        if isinstance(raw_inp, list):
            if "all" in raw_inp:
                return [-1]
            layer_numbers = []
            for i in range(len(raw_inp)):
                if isinstance(raw_inp[i], str):
                    min_range, max_range = list(map(int, raw_inp[i].split("-")))
                    layer_numbers += list(range(min_range, max_range + 1))
                elif isinstance(raw_inp[i], int):
                    layer_numbers.append(raw_inp[i])
                else:
                    raise TypeError(
                        f"Expected all values in layer_numbers to be of type 'str' or 'int' but got: {type(raw_inp[i])}"
                    )
            return set(layer_numbers)
        elif isinstance(raw_inp, int):
            return set(raw_inp)
        elif isinstance(raw_inp, str) and raw_inp.lower() == "all":  # all layers
            return [-1]
        else:
            raise TypeError(
                f"Invalid type for argument 'layer_numbers'. Expected: {list[int | str] | int | str}, found: {type(raw_inp)}"
            )


class ConfigSpec(BaseSpec):
    def __init__(self):
        self.layers = LayerSpec()
        self.extension_config = {}  # extension name -> its config

    def verify_validity(self):
        self.layers.verify_validity()

    def initialize(self, cfg):
        assert "layers" in cfg, "[NVDLFW INSPECT ERROR] Provide layers field in config."
        self.enabled = cfg["enabled"]
        self.layers.initialize(cfg["layers"])
        self.verify_validity()

    def add_extension(self, ext: str, args: dict):
        self.extension_config[ext] = args

    def reset(self):
        self.layers = LayerSpec()
        self.extension_config = {}
        self.initialized = False


class ConfigManager:
    configs: list[ConfigSpec] = []  # noqa: RUF012
    constraints = {}  # noqa: RUF012
    layer_name_to_config_cache = {}  # noqa: RUF012

    @classmethod
    def _load_config(cls, config_input, registry_data):
        """Parse the config from either a file path or a Python dictionary and set spec attributes.

        Args:
            config_input: Either a filepath (str or Path) or a Python dictionary containing config
            registry_data: Registry data containing feature information
        """
        if isinstance(config_input, (str, pathlib.Path)):
            # Handle YAML file input
            try:
                with open(config_input) as f:
                    config = yaml.safe_load(f)
                    get_logger().log_message(
                        f"Reading config from {config_input}.", level=logging.INFO
                    )
            except FileNotFoundError:
                print_rank_0(
                    f"[NVDLFW INSPECT WARNING]: Could not find config file at {config_input}. Please make sure file path to config exists. "
                    "Running without config. "
                )
                return
        elif isinstance(config_input, dict):
            # Handle Python dictionary input
            config = config_input
            get_logger().log_message(
                "Reading config from Python dictionary.", level=logging.INFO
            )
        else:
            raise TypeError(
                f"Config input must be either a filepath (str or Path) or a dictionary, got {type(config_input)}"
            )

        for config_name in config:
            cfg = config[config_name]
            assert "enabled" in cfg, (
                f"[NVDLFW INSPECT ERROR] Missing required config field `enabled` under {config_name} context."
            )

            if cfg["enabled"]:
                config_spec = ConfigSpec()
                config_spec.initialize(cfg)

                for field in cfg:
                    if field in {"layers", "enabled"}:
                        continue
                    if field in registry_data:
                        for sub_field in cfg[field]:
                            assert sub_field in registry_data[field].features, (
                                feature_not_found_error(sub_field, field)
                            )
                        config_spec.add_extension(field, cfg[field])
                    else:
                        assert field in registry_data["base"].features, (
                            feature_not_found_error(field, "base")
                        )
                        config_spec.add_extension("base", {field: cfg[field]})

                cls.configs.append(config_spec)
            else:
                print_rank_0(
                    f"[NVDLFW INSPECT INFO] Skipping {config_name} config as it is not enabled."
                )

        get_logger().log_message(
            f"Loaded configs for {list(config.keys())}.", level=logging.INFO
        )
        assert len(cls.configs) > 0, (
            "[NVDLFW INSPECT ERROR] Could not load config from input. Ensure config format is correct."
        )

    @classmethod
    def get_config_for_layer(cls, layer_name):
        if layer_name in cls.layer_name_to_config_cache:
            return cls.layer_name_to_config_cache[layer_name]

        layer_config = {}
        for config in cls.configs:
            if is_layer_in_cfg(layer_name, config):
                if layer_config == {}:
                    layer_config = config.extension_config
                else:
                    raise ValueError(
                        f"[NVDLFW INSPECT ERROR] Found multiple debug configs targeting the same layer(s): {layer_name}."
                    )

        cls.layer_name_to_config_cache[layer_name] = layer_config
        return layer_config

    @classmethod
    def reset(cls):
        for config in cls.configs:
            config.reset()
        cls.configs = []
        cls.layer_name_to_config_cache.clear()


def is_layer_in_cfg(layer_name: str, cfg: ConfigSpec) -> bool:
    """
    Check if layer is part of ConfigSpec.
    """

    # Match exact layer names provided.
    if cfg.layers.exact_layer_names is not None:
        name_matched = layer_name in cfg.layers.exact_layer_names
        if name_matched:
            return True
        else:
            raise ValueError(
                "Could not find layer using exact layer name from config. "
                " Make sure exact layer name specified matches layer name provided. "
                f"Expected {layer_name}."
            )

    # Match against the regex pattern provided.
    if cfg.layers.layer_name_regex_pattern is not None:
        return cfg.layers.layer_name_regex_pattern.search(layer_name) is not None

    # Match layer numbers and/or layer types provided.

    # Consider all layer numbers by default
    layer_number_match = False
    if cfg.layers.layer_numbers is not None:
        # Ignore the number part of 'fc'.
        layer_num_matches = re.findall(r"(?<!fc)\d+", layer_name)
        if layer_num_matches is None or len(layer_num_matches) > 1:
            raise ValueError(
                f"Could not extract layer number from specified layer name {layer_name}. "
                f" Found {layer_num_matches}."
            )
        layer_num = int(layer_num_matches[0])
        layer_number_match = (
            layer_num in cfg.layers.layer_numbers or cfg.layers.layer_numbers == [-1]
        )

    # Consider all layer types by default
    layer_type_match = False
    if cfg.layers.layer_types is not None:
        layer_type_match = (
            cfg.layers.layer_type_regex_pattern.search(layer_name) is not None
            or "all" in cfg.layers.layer_types
        )

    if layer_number_match:
        # If no layer_types mentioned, consider entire layer in config.
        return True if cfg.layers.layer_types is None else layer_type_match

    if layer_type_match:
        # If no layer_numbers mentioned, consider all layer_numbers in config.
        return True if cfg.layers.layer_numbers is None else layer_number_match

    return False
