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

from nvdlfw_inspect.registry import Registry, api_method


@Registry.register_feature(namespace="framework")
class FeatureA:
    @api_method
    def process_tensor(self, config, layer_name, **kwargs):
        print("FeatureA - process_tensor")

    @api_method
    def another_api(self, config, layer_name, **kwargs):
        print("FeatureA - another_api")


@Registry.register_feature(namespace="framework")
class FeatureB:
    @api_method
    def process_tensor(self, config, layer_name, **kwargs):
        print("FeatureB - process_tensor")

    @api_method
    def another_api(self, config, layer_name, **kwargs):
        print("FeatureB - another_api")
