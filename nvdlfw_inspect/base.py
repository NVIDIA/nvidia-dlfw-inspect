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
import logging
import sys
from abc import ABC, abstractmethod
from functools import partial

import nvdlfw_inspect.api as debug_api
from nvdlfw_inspect.config_manager import ConfigManager
from nvdlfw_inspect.logging import custom_assert, get_logger
from nvdlfw_inspect.utils import APICacheIdentifier


class BaseConfigAPIMapper(ABC):
    @abstractmethod
    def parse_config_and_api(self, config, **kwargs):
        pass

    def _process_tensor_config(self, cfg, api_tensor_name):
        """
        Return config specific to a particular tensor name that matches the api args.
        """

        if "tensors_struct" in cfg:
            for item in cfg["tensors_struct"]:
                if item["tensor"] == api_tensor_name:
                    cfg_copy = copy.deepcopy(item)
                    cfg.pop("tensors_struct")
                    cfg.update(cfg_copy)
                    return cfg
            return None
        elif "tensors" in cfg:
            if api_tensor_name in cfg["tensors"]:
                cfg["tensor"] = api_tensor_name
                cfg.pop("tensors")
                return cfg
            return None

        debug_api.log_message(
            "Provide 'tensors_struct: List[Dict]' or 'tensors: List[str]' in the config yaml.",
            level=logging.ERROR,
        )
        raise ValueError(
            "Provide 'tensors_struct: List[Dict]' or 'tensors: List[str]' in the config yaml."
        )


class BaseNamespaceAPI(ABC):
    """
    Base namespace API class. Framework specific APIs should inherit from this class.
    - Implements default APIs that will be used when a config is not specified or
        when a layer is not selected in the config.
    - Implements an API router if multiple features use the same API to ensure
        the correct API is invoked given the kwargs and the config.
    - If an API is unique in the given namespace, it will call it by default.
    """

    name: str

    def __init__(self):
        # All features available under a namespace.
        self.namespace_features = {}

        # APIs to features that implement those APIs
        self.api_to_features_map = {}

        # Cache contains feature API to invoke along with feature config
        self._api_cache = {}

        # Pointer to a class container default APIs to use.
        # These APIs are used when config is not provided.
        self._default_api_impl = None

        # Maps API name to the API method
        self._default_api_map = None

        # kwargs required to cache APIs.
        self._cacheable_api_kwargs_map = {}

    def set_features(self, features):
        self.namespace_features = features

    def set_default_api_impl(self):
        if self._default_api_impl:
            self._default_api_map = {}
            for method_name in dir(self._default_api_impl):
                if not method_name.startswith("_"):
                    method = getattr(self._default_api_impl, method_name)
                    if callable(method):
                        self._default_api_map[method_name] = method

    def set_api_map(self, api_map):
        self.api_to_features_map = api_map

    @abstractmethod
    def input_assertions_hook(self, api_name, **kwargs):
        """
        Use this hook to define any specific assertions for the input of an API call.

        api_name: Name of the API being invoked.
        """
        pass

    @abstractmethod
    def routing_condition(
        self, api_name, config, layer_name, feature_obj, **kwargs
    ) -> tuple[bool, dict | None]:
        """
        Use this to define specific routing conditions for an API.

        api_name: Name of the API being invoked.
        config: Config of the feature that contains this API.
        """
        pass

    @abstractmethod
    def output_assertions_hook(self, api_name, ret, **kwargs):
        """
        Use this hook to define any specific assertions for the output of an API call.

        api_name: Name of the API being invoked.
        ret: Returned dictionary from the API
        """
        pass

    @abstractmethod
    def is_multiple_feature_invocation_allowed(self, api_name):
        """
        Check if API allowes executing multiple features for a single call
        """
        pass

    def route_api(self, layer_name, api_name, **kwargs):  # noqa: C901
        """
        Routes the API call based on the condition defined in routing_condition.

        Parameters:
        api_name (str): The name of the API to be routed.
        layer_name (str): The name of the layer to which the API call is being routed.
        **kwargs: Additional keyword arguments used to determine the routing condition.

        Returns:
        The result of the routed API call.

        Raises:
        AssertionError: If multiple features are enabled for the same API call.
        AttributeError: If the API method is not found on the feature instance.
        """
        self.input_assertions_hook(api_name, **kwargs)

        uid = APICacheIdentifier.get_unique_identifier(
            self._cacheable_api_kwargs_map, layer_name, api_name, **kwargs
        )

        if uid not in get_logger().logged_api_encountered:
            debug_api.log_message(
                f"Debug API call '{api_name}' encountered at the line: {APICacheIdentifier.get_call_details()}",
                level=logging.DEBUG,
            )
            get_logger().logged_api_encountered.add(uid)

        if uid is not None and uid in self._api_cache:
            if self._api_cache[uid] is not None:
                return self.call_feature(
                    self._api_cache[uid][0],
                    self._api_cache[uid][1],
                    layer_name,
                    **kwargs
                )
            return {}

        layer_config = ConfigManager.get_config_for_layer(layer_name).get(self.name, {})
        features_to_invoke = {}  # feature_name -> feature_config
        for feature in self.api_to_features_map[api_name]:
            cfg = layer_config.get(feature, {})
            if cfg:
                custom_assert(
                    "enabled" in cfg,
                    f"Missing required config field `enabled` under {feature} context.",
                )

                if cfg["enabled"]:
                    # Based on the Feature, routing condition will call the feature's config parser and can return the needed params from the config, given the arguments passed to the api.
                    # This will be useful for caching and making execution for debug features for the future iterations faster and avoid parsing config on each step.
                    status, optional_modified_config = self.routing_condition(
                        api_name,
                        cfg,
                        layer_name,
                        self.namespace_features[feature],
                        **kwargs,
                    )
                    if status:
                        if optional_modified_config:
                            features_to_invoke[feature] = optional_modified_config
                        else:
                            features_to_invoke[feature] = cfg

        if len(features_to_invoke) == 0:
            # Use default API if no features found or feature not in layer
            if self._default_api_map and api_name in self._default_api_map:
                self._api_cache[uid] = (self._default_api_map[api_name], {})
                return self._default_api_map[api_name]({}, layer_name, **kwargs)
            else:
                self._api_cache[uid] = None
                return {}

        if len(features_to_invoke) > 1:
            custom_assert(
                self.is_multiple_feature_invocation_allowed(api_name),
                f"Only 1 operation permitted per API call. Found {len(features_to_invoke)} ops {features_to_invoke.keys()} enabled for {kwargs}.",
            )
            try:
                multi_feature_out = []
                for feat_name, feat_config in features_to_invoke.items():
                    ret = self.call_feature(
                        getattr(self.namespace_features[feat_name], api_name),
                        feat_config,
                        layer_name,
                        **kwargs
                    )
                    multi_feature_out.append(ret)
                if uid not in get_logger().logged_api_executed:
                    debug_api.log_message(
                        f"Debug API call '{api_name}' found and multiple debug features were executed.",
                        level=logging.DEBUG,
                    )
                    get_logger().logged_api_executed.add(uid)
                for out in multi_feature_out:
                    self.output_assertions_hook(api_name, out, **kwargs)
                return self.handle_multi_feature_output(api_name, multi_feature_out, features_to_invoke, **kwargs)
                    
            except AttributeError as e:
                debug_api.log_message(
                    f"Could not run API {api_name} for multiple features {features_to_invoke.keys()} - got error: {e}. Exiting.",
                    level=logging.ERROR,
                )
                print(e, file=sys.stderr)
                sys.exit(1)

        try:
            feat_name = list(features_to_invoke.keys())[0]  # noqa: RUF015
            feat_config = features_to_invoke[feat_name]
            self._api_cache[uid] = (
                getattr(self.namespace_features[feat_name], api_name),
                feat_config,
            )
            if uid not in get_logger().logged_api_executed:
                debug_api.log_message(
                    f"Debug API call '{api_name}' found and the corresponding debug feature '{feat_name}' was called.",
                    level=logging.DEBUG,
                )
                get_logger().logged_api_executed.add(uid)
            ret = self.call_feature(
                getattr(self.namespace_features[feat_name], api_name),
                feat_config,
                layer_name,
                **kwargs
            )
            self.output_assertions_hook(api_name, ret, **kwargs)
            return ret  # noqa: TRY300
        except AttributeError as e:
            debug_api.log_message(
                f"Could not run API {api_name} in feature {features_to_invoke.keys()} - got error: {e}. Exiting.",
                level=logging.ERROR,
            )
            print(e, file=sys.stderr)
            sys.exit(1)
    
    def handle_multi_feature_output(self, api_name, multi_feature_outputs, features_to_invoke, **kwargs):
        # Basic scenario: all features should return the same output.
        custom_assert(
            all(x == multi_feature_outputs[0] for x in multi_feature_outputs),
            "Different Outputs when invoking multiple features per API call is not allowed. "
            + f"Found {len(features_to_invoke)} ops {features_to_invoke.keys()} enabled for {api_name}({kwargs}) returning outputs: {multi_feature_outputs}.",
        )
        return multi_feature_outputs[0]

    def call_feature(self, call, feat_config, layer_name, **kwargs):
        return call(
            feat_config, layer_name, **kwargs
        )

    def step(self):
        pass

    def end_debug(self):
        pass

    def __getattr__(self, api_name):
        return partial(self.route_api, api_name=api_name)
