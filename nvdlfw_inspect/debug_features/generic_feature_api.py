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

import copy

from nvdlfw_inspect.base import BaseConfigAPIMapper, BaseNamespaceAPI
from nvdlfw_inspect.registry import Registry


class GenericConfigAPIMapper(BaseConfigAPIMapper):
    """
    Supported yaml config structure for the feature:
    1. tensors: [tensor1, tensor2]
       feature_param1: value
       feature_param2: value

    2. tensor_struct:
        - tensor: tensor1
          feature_param1: value
          feature_param2: value
        - tensor: tensor2
          feature_param2: value
          feature_param2: value
    """

    tensor_config_docstring = """
    Supported yaml config structure for the feature:
    1. tensors: [tensor1, tensor2]
       feature_param1: value
       feature_param2: value

    2. tensor_struct:
        - tensor: tensor1
          feature_param1: value
          feature_param2: value
        - tensor: tensor2
          feature_param2: value
          feature_param2: value
    """

    def parse_config_and_api(self, config, **kwargs):
        # Process the config and returns True if the config and api args match, along with processed config.
        processed_config = None
        config_copy = copy.deepcopy(config)
        tensor_parsing = kwargs.get("tensor_parsing", False)
        if tensor_parsing:
            processed_config = self._process_tensor_config(
                config_copy, kwargs["tensor_name"]
            )

        if not processed_config:
            return False, None

        if "enabled" in processed_config:
            processed_config.pop("enabled")
        return True, processed_config


@Registry.register_namespace_api(namespace="base")
class GenericFrameworkAPI(BaseNamespaceAPI):
    def __init__(self):
        BaseNamespaceAPI.__init__(self)
        self._cacheable_api_kwargs_map = {
            "log_tensor_stats": ["tensor_name"],
        }

    def input_assertions_hook(self, api_name, **kwargs):
        if api_name in {"log_tensor_stats"}:
            assert "tensor" in kwargs, f"A tensor must be input to {api_name}."
            assert "tensor_name" in kwargs, (
                f"tensor_name must be an input to {api_name}"
            )

    def routing_condition(
        self, api_name, config, layer_name, feature_obj, **kwargs
    ) -> tuple[bool, dict | None]:
        status, modified_config = feature_obj.parse_config_and_api(
            config, tensor_parsing=True, **kwargs
        )
        return status, modified_config

    def output_assertions_hook(self, api_name, ret, **kwargs):
        pass

    def is_multiple_feature_invocation_allowed(self, api_name):
        return False
