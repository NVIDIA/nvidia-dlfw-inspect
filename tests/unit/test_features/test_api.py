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

from nvdlfw_inspect.base import BaseNamespaceAPI
from nvdlfw_inspect.registry import Registry


@Registry.register_namespace_api(namespace="framework")
class API(BaseNamespaceAPI):
    def __init__(self):
        BaseNamespaceAPI.__init__(self)

    def input_assertions_hook(self, api_name, **kwargs):
        pass

    def routing_condition(self, api_name, config, layer_name, feature_obj, **kwargs):
        return True, None

    def output_assertions_hook(self, api_name, ret, **kwargs):
        pass

    def is_multiple_feature_invocation_allowed(self, api_name):
        pass


@Registry.register_namespace_api(namespace="framework_3")
class API3:
    def __init__(self):
        pass
