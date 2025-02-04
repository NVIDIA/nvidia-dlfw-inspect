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
from nvdlfw_inspect.logging import custom_assert
from nvdlfw_inspect.registry import Registry


class SampleConfigAPIMapper(BaseConfigAPIMapper):
    """
    Supported yaml config structure for the feature:
    1. gemms: [gemm1, gemm2]
       tensors: [tensor1, tensor2]
       tensor_feature_param1: value
       gemm_feature_param1: value

    2. gemms: [gemm1, gemm2]
       tensors_struct:
        - tensor: tensor1
          tensor_feature_param1: value
        - tensor: tensor2
          tensor_feature_param2: value
        gemm_feature_param1: value

    3. gemms_struct:
        - gemm: gemm1
          tensors: [tensor1, tensor2]
          tensor_feature_param1: value
          gemm_feature_param1: value
        - gemm: gemm2
          tensors_struct:
          - tensor: tensor1
            tensor_feature_param1: value
          - tensor: tensor2
            tensor_feature_param2: value
          gemm_feature_param1: value
    """

    def parse_config_and_api(self, config, **kwargs):
        # Process the config and returns True if the config and api args match, along with processed config.
        processed_config = None
        config_copy = copy.deepcopy(config)
        gemm_parsing = kwargs.get("gemm_parsing", False)
        tensor_parsing = kwargs.get("tensor_parsing", False)

        if gemm_parsing and tensor_parsing:
            # parse with GEMM and tensor
            processed_config = self._process_gemm_and_tensor_config(
                config_copy, **kwargs
            )

        if not processed_config:
            return False, None

        if "enabled" in processed_config:
            processed_config.pop("enabled")
        return True, processed_config

    def _process_gemm_and_tensor_config(self, config, **kwargs):
        """
        Return config specific to a particular tensor name and gemm that matches the api args.
        """

        if "gemms_struct" in config:
            for cfg in config["gemms_struct"]:
                if cfg["gemm"] == kwargs["gemm"]:
                    if kwargs["tensor_parsing"]:
                        cfg = self._process_tensor_config(cfg, kwargs["tensor_name"])
                        if not cfg:
                            return None
                    cfg_copy = copy.deepcopy(cfg)
                    config.pop("gemms_struct")
                    config.update(cfg_copy)
                    return config
        elif (
            "gemms" in config
            and kwargs["gemm"] in config["gemms"]
            and kwargs["tensor_parsing"]
        ):
            cfg = self._process_tensor_config(config, kwargs["tensor_name"])
            if not cfg:
                return None
            config["gemm"] = kwargs["gemm"]
            config.pop("gemms")
            return config
        return None


class SampleDefaultFeatures:
    def process_tensor(self, config, layer_name, **kwargs):
        """
        API to process a tensor. This must return a tensor.
        """
        return {"tensor": kwargs["tensor"]}


@Registry.register_namespace_api(namespace="sample_namespace")
class SampleFrameworkAPI(BaseNamespaceAPI):
    def __init__(self):
        BaseNamespaceAPI.__init__(self)
        self._default_api_impl = SampleDefaultFeatures()
        self._cacheable_api_kwargs_map = {
            "process_tensor": ["tensor_name", "gemm"],
        }

    def input_assertions_hook(self, api_name, **kwargs):
        """
        These args must be passed as kwargs in the API call for all TransformerEngine specific APIs.
        """
        required_kwargs = {"process_tensor": ["tensor", "gemm", "tensor_name"]}

        if api_name in required_kwargs:
            for kwarg in required_kwargs[api_name]:
                custom_assert(
                    kwarg in kwargs,
                    f"Cannot route API, too ambiguous. Provide {kwarg} in {api_name}.",
                )

    def routing_condition(
        self, api_name, config, layer_name, feature_obj, **kwargs
    ) -> tuple[bool, dict | None]:
        status, modified_config = feature_obj.parse_config_and_api(
            config, gemm_parsing=True, tensor_parsing=True, **kwargs
        )
        return status, modified_config

    def output_assertions_hook(self, api_name, ret, **kwargs):
        if api_name in {"process_tensor"}:
            custom_assert("tensor" in ret, f"This API {api_name} must return a tensor.")

    def is_multiple_feature_invocation_allowed(self, api_name):
        return False
