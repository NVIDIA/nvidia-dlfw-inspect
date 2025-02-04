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
from collections import defaultdict
from dataclasses import dataclass, field

from .base import BaseNamespaceAPI
from .logging import get_logger


@dataclass
class NamespaceData:
    # namespace name
    name: str
    # Namespace API class object
    api: type
    # All features in the namespace. feature_name -> object
    features: dict[str, object] = field(default_factory=dict)
    # All APIs defined in namespace to all features containing those APIs
    feat_api_to_features: dict[str, list[str]] = field(
        default_factory=lambda: defaultdict(list)
    )
    # List to store deferred features that will be registered after API registration
    deferred_features: list[type] = field(default_factory=list)


class Registry:
    data: dict[str, NamespaceData] = {}  # noqa: RUF012

    @classmethod
    def register_feature(cls, namespace: str):
        def register(cls_incoming: type):
            if namespace not in cls.data:
                cls.data[namespace] = NamespaceData(namespace, None)
            # If namespace API is not yet registered, defer the registration of its features
            if cls.data[namespace].api is None:
                cls.data[namespace].deferred_features.append(cls_incoming)
            else:
                feature_name = cls_incoming.__name__
                if feature_name not in cls.data[namespace].features:
                    # Why we need the check:
                    # If we are inheriting from another feature class that is already registered,
                    # then we dont want it to register again
                    cls.data[namespace].features[feature_name] = cls_incoming()
                    cls._register_feature_api(namespace, feature_name)
                    get_logger().log_message(
                        f"Registered feature {feature_name} in {namespace}.",
                        level=logging.DEBUG,
                    )
            return cls_incoming

        return register

    @classmethod
    def register_namespace_api(cls, namespace: str):
        def register(cls_incoming: BaseNamespaceAPI | type):
            if namespace not in cls.data:
                cls.data[namespace] = NamespaceData(namespace, cls_incoming())
            else:
                cls.data[namespace].api = cls_incoming()
                get_logger().log_message(
                    f"Registered {namespace} namespace.", level=logging.DEBUG
                )

            # Once namespace API is registered, register all deferred features
            for deferred_feature in cls.data[namespace].deferred_features:
                feature_name = deferred_feature.__name__
                if feature_name not in cls.data[namespace].features:
                    # Why we need the check:
                    # If we are inheriting from another feature class that is already registered,
                    # then we dont want it to register again
                    cls.data[namespace].features[feature_name] = deferred_feature()
                    cls._register_feature_api(namespace, feature_name)
                    get_logger().log_message(
                        f"Registered feature {feature_name} in {namespace}.",
                        level=logging.DEBUG,
                    )

            cls.data[namespace].deferred_features.clear()
            return cls_incoming

        return register

    @classmethod
    def _register_feature_api(cls, namespace, feature_name):
        feature_obj = cls.data[namespace].features[feature_name]

        for method_name in dir(feature_obj):
            if not method_name.startswith("_"):
                method = getattr(feature_obj, method_name)
                if getattr(method, "is_api", False) and callable(method):
                    cls.data[namespace].feat_api_to_features[method_name].append(
                        feature_name
                    )

    @classmethod
    def validate_and_init(cls):
        for namespace in cls.data:
            if cls.data[namespace].api is not None and not isinstance(
                cls.data[namespace].api, BaseNamespaceAPI
            ):
                raise TypeError(f"{namespace} API must inherit from BaseNamespaceAPI")
            if len(cls.data[namespace].deferred_features) != 0:
                raise NotImplementedError(
                    f"Found features {cls.data[namespace].deferred_features} for namespace {namespace} but API was not found."
                )

            cls.data[namespace].api.name = namespace
            cls.data[namespace].api.set_default_api_impl()
            cls.data[namespace].api.set_api_map(
                cls.data[namespace].feat_api_to_features
            )
            cls.data[namespace].api.set_features(cls.data[namespace].features)

    @classmethod
    def reset(cls):
        cls.data.clear()


def api_method(func):
    func.is_api = True
    return func
