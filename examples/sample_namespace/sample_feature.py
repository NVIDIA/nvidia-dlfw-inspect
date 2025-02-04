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
from sample_feature_api import SampleConfigAPIMapper

import nvdlfw_inspect.api as debug_api
from nvdlfw_inspect.registry import Registry, api_method
from nvdlfw_inspect.utils import append_parent_docstring


@Registry.register_feature(namespace="sample_namespace")
@append_parent_docstring(parent=SampleConfigAPIMapper)
class FakeQuant(SampleConfigAPIMapper):
    """
    Fake Quantization feature in Pytorch with torch.fake_quantize_per_tensor_affine
    Returns a new tensor with the data in input fake quantized using scale, zero_point, quant_min and quant_max.

    APIs:
    1. sample_namespace.process_tensor
    - When using this api, you would need to pass args:
    -- layer_name: : this is matched with the layer description in the config file
    -- gemm: this is matched with one of the gemms in the config field, and passed as a kwarg. For example, gemm='gemm1'
    -- tensor_name: this is matched with one of the tensors in the config field for a given gemm, and passed as a kwarg. For example, tensor_name='tensor1'
    -- tensor: the tensor to process, and passed as a kwarg. For example, tensor={torch tensor}

    Config:
    To enable the feature in yaml config:
    sample_namespace:
      fake_quant:
        enabled: True
        feature_properties:
        ...

    Config fields:
    This feature works at a tensor level, you can set the following properties for each tensor:
    - scale: double or float32 scalar, quantization scalar
    - zero_point: int scalar or tensor, quantization zero_point
    - quant_min: int, lower bound of quantized domain
    - quant_max: int, upper bound of quantized domain
    """

    @api_method
    def process_tensor(self, config, layer_name, **kwargs):
        debug_api.log_message(
            f"Feature={self.__class__.__name__}, API=process_tensor, GEMM={kwargs['gemm']} TENSOR={kwargs['tensor_name']}: Called",
            layer_name=layer_name,
            level=logging.INFO,
        )

        for field in ["scale", "zero_point", "quant_min", "quant_max"]:
            if field not in config:
                debug_api.log_message(
                    f"Feature={self.__class__.__name__}, API=process_tensor: {field} missing for Tensor: {kwargs['tensor_name']} in the config yaml for FakeQuant feature which is a required field",
                    layer_name=layer_name,
                    level=logging.ERROR,
                )
                raise ValueError(
                    f"Feature={self.__class__.__name__}, API=process_tensor: {field} missing for Tensor: {kwargs['tensor_name']} in the config yaml for FakeQuant feature which is a required field"
                )

        q_tensor = torch.fake_quantize_per_tensor_affine(
            kwargs["tensor"],
            config["scale"],
            config["zero_point"],
            config["quant_min"],
            config["quant_max"],
        )
        return {"tensor": q_tensor}
