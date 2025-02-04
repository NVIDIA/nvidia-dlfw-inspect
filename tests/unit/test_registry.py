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

import os
import pathlib

from nvdlfw_inspect.registry import Registry
from nvdlfw_inspect.utils import import_and_exec_module


def load_features():
    def _recursive_walk(path: str):
        for file in os.listdir(path):
            if os.path.isfile(os.path.join(path, file)) and file.endswith(".py"):
                import_and_exec_module(os.path.join(path, file))
            elif os.path.isdir(os.path.join(path, file)):
                _recursive_walk(os.path.join(path, file))

    for feat_dir in [pathlib.Path(__file__).resolve().parent / "test_features"]:
        assert os.path.exists(feat_dir), f"Could not find features path: {feat_dir}."
        _recursive_walk(feat_dir)


def test_namespace_api_registration():
    load_features()

    assert "framework" in Registry.data
    assert "framework_1" in Registry.data

    assert Registry.data["framework"].api.routing_condition(None, None, None, None)[0]
    Registry.reset()


def test_feature_registration():
    load_features()

    assert "FeatureA" in Registry.data["framework"].features
    assert "FeatureB" in Registry.data["framework"].features

    assert "FeatureC" in Registry.data["framework_1"].features
    assert "FeatureD" in Registry.data["framework_1"].features

    assert len(Registry.data["framework_2"].deferred_features) == 2
    Registry.reset()


def test_feature_api_registration():
    load_features()
    assert "process_tensor" in Registry.data["framework"].feat_api_to_features
    assert "another_api" in Registry.data["framework"].feat_api_to_features

    assert (
        "FeatureA" in Registry.data["framework"].feat_api_to_features["process_tensor"]
    )
    assert (
        "FeatureB" in Registry.data["framework"].feat_api_to_features["process_tensor"]
    )

    assert "process_tensor" in Registry.data["framework_1"].feat_api_to_features
    assert "another_api" in Registry.data["framework_1"].feat_api_to_features

    assert (
        "FeatureC"
        in Registry.data["framework_1"].feat_api_to_features["process_tensor"]
    )
    assert (
        "FeatureD"
        in Registry.data["framework_1"].feat_api_to_features["process_tensor"]
    )
    Registry.reset()


def test_registry_validation():
    load_features()
    try:
        Registry.validate_and_init()
    except TypeError:
        pass
    except NotImplementedError:
        pass

    Registry.reset()
