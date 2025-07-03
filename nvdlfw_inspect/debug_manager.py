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
import os
from functools import partial

from nvdlfw_inspect.config_manager import ConfigManager
from nvdlfw_inspect.logging import (
    BaseLogger,
    MetricLogger,
    get_logger,
    make_default_statistics_logger,
    print_rank_0,
    wrap_tensorboard_writer,
)
from nvdlfw_inspect.registry import Registry
from nvdlfw_inspect.utils import import_and_exec_module


class DebugManager:
    def __init__(
        self,
        config_file,
        feature_dirs=None,
        statistics_logger: BaseLogger | None = None,
        log_dir=".",
        init_training_step=0,
        **kwargs,
    ):
        self._features_explainer = {}
        self._namespace_features_api_map = {}
        self._namespace_apis = {}

        self._initialize_loggers(statistics_logger, log_dir=log_dir, **kwargs)
        self._load_namespace_apis_and_features(feature_dirs)
        self.config_file = config_file

        self._trainer_iteration_count = init_training_step
        self._tensor_reduction_group = None
        self._api_caller_details = None

    def load_config(self):
        ConfigManager._load_config(self.config_file, registry_data=Registry.data)

    def _initialize_loggers(
        self, statistics_logger: BaseLogger | None, log_dir, **kwargs
    ):
        """
        Initializes loggers for tensor statistics.

        Args:
            statistics_logger (Union[BaseLogger, None]): The logger to use for logging tensor statistics.
                If None, a default logger will be created.
            log_dir (str): The directory where log files will be saved.
            **kwargs:
                tb_writer (TensorBoardWriter, optional): A TensorBoard writer to use for logging.
                default_logging_enabled (bool, optional): Whether to enable default logging to file.

        Notes:
            If a logger is provided, it must be an instance of BaseLogger.
            If a TensorBoard writer is provided, it will be used for logging.
            If default logging is enabled, a file logger will be created.
        """

        if statistics_logger is not None:
            if not (isinstance(statistics_logger, BaseLogger)):
                message = f"Loggers should be of type 'BaseLogger' but found {type(statistics_logger)}. Please follow reference implementation."
                get_logger().log_message(message, level=logging.INFO)
                raise AssertionError(message)
            get_logger().log_message(
                "Using the provided logger for logging tensor statistics.",
                level=logging.INFO,
            )
            MetricLogger.add_logger(statistics_logger)
        else:
            if "tb_writer" in kwargs and kwargs["tb_writer"] is not None:
                get_logger().log_message(
                    "TB Writer found. Logging to Tensorboard..", level=logging.INFO
                )
                MetricLogger.add_logger(wrap_tensorboard_writer(kwargs["tb_writer"]))

        if MetricLogger.enabled_loggers == [] or kwargs.get(
            "default_logging_enabled", False
        ):
            get_logger().log_message(
                f"Default logging to file enabled at {log_dir}", level=logging.INFO
            )
            MetricLogger.add_logger(make_default_statistics_logger())

    def _load_namespace_apis_and_features(self, feature_dirs: list[str] | str | None):  # noqa: C901
        """
        Loads framework APIs and features from feature dirs.

        Args:
            feature_dirs (list[str]): A list of directories containing feature and API files.

        Description:
            This method loads API classes from the specified feature dirs and stores them in the registry.
            See registry.py for more information on registering features and APIs.

        Notes:
            Generic features provided with debug tool are always loaded.
        """

        def _recursive_walk(path: str):
            for file in os.listdir(path):
                if os.path.isfile(os.path.join(path, file)) and file.endswith(".py"):
                    import_and_exec_module(os.path.join(path, file))
                elif os.path.isdir(os.path.join(path, file)):
                    _recursive_walk(os.path.join(path, file))

        generic_features_dir = os.path.join(os.path.dirname(__file__), "debug_features")

        if isinstance(feature_dirs, str):
            feature_dirs = [feature_dirs] if feature_dirs else []
        elif feature_dirs and not isinstance(feature_dirs, list):
            get_logger().log_message(
                f"Recieved type {type(feature_dirs)}. Expected type is either `List[str]` or `str`.",
                level=logging.ERROR,
            )
            raise ValueError(
                f"Recieved type {type(feature_dirs)}. Expected type is either `List[str]` or `str`."
            )

        feature_dirs = (
            [generic_features_dir, *feature_dirs]
            if feature_dirs
            else [generic_features_dir]
        )

        for feat_dir in feature_dirs:
            if not (os.path.exists(feat_dir)):
                message = f"Could not find features path: {feat_dir}."
                get_logger().log_message(message, level=logging.ERROR)
                raise AssertionError(message)
            _recursive_walk(feat_dir)

        Registry.validate_and_init()

        for namespace in Registry.data:
            for feature_name, feature_obj in Registry.data[namespace].features.items():
                self._features_explainer[f"{namespace}.{feature_name}"] = feature_obj

            self._namespace_features_api_map[namespace] = Registry.data[
                namespace
            ].feat_api_to_features
            self._namespace_apis[namespace] = Registry.data[namespace].api

    def get_extension_api(self, name):
        if name in self._namespace_features_api_map.get("base", {}):
            return partial(self._namespace_apis["base"].route_api, api_name=name)
        elif name not in self._namespace_apis:
            get_logger().log_message(
                f"API for {name} not found. Make sure {name}.api.py is implemented correctly.",
                level=logging.ERROR,
            )
            raise ValueError(
                f"API for {name} not found. Make sure {name}.api.py is implemented correctly."
            )
        return self._namespace_apis[name]

    def close(self):
        self._invoke_for_each_namespace("end_debug")
        ConfigManager.reset()
        Registry.reset()
        self._namespace_features_api_map.clear()
        self._namespace_apis.clear()
        MetricLogger.enabled_loggers.clear()

    def list_features(self):
        print_rank_0(list(self._features_explainer.keys()))

    def explain_features(self, features: list | str | None = None):
        method_helper_str = 'Can pass a list of features or feature string to this method. use list_features() method to get the feature names or pass "all" string to this method for all features'

        def print_feature_docstring(feature):
            if feature in self._features_explainer:
                print_rank_0("-----")
                print_rank_0(f"feature: {feature}")
                print_rank_0(self._features_explainer[feature].__doc__)
                print_rank_0("*****")
            else:
                print_rank_0(
                    f"feature: {feature} is not supported in this tool. {method_helper_str}"
                )

        if features is None:
            print_rank_0(f"No feature provided. {method_helper_str}")
        elif isinstance(features, list):
            for feature in features:
                print_feature_docstring(feature)
        elif isinstance(features, str):
            if features != "all":
                print_feature_docstring(features)
            else:
                for feature in self._features_explainer:
                    print_feature_docstring(feature)
        else:
            print_rank_0(
                f"Invalid type is provided to the features argument. {method_helper_str}"
            )

    def initialize_training_step(self, train_step: int):
        self._trainer_iteration_count = train_step

    def step(self):
        self._trainer_iteration_count += 1
        self._invoke_for_each_namespace("step")

    def _invoke_for_each_namespace(self, function_name):
        # It can be used to run such calls like step()
        # that may need to be handled by the feature namespaces.
        for namespace_name in Registry.data:
            namespace = Registry.data[namespace_name]
            getattr(namespace.api, function_name, lambda: None)()
