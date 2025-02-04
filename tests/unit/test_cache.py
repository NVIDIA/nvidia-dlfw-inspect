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

from nvdlfw_inspect.utils import APICacheIdentifier

FILE_NAME = __file__


def make_call():
    current_frame = inspect.currentframe()
    caller_frame = current_frame.f_back
    APICacheIdentifier.save_call_details(caller_frame)


def assert_equal(actual, expected):
    assert actual == expected, f"FAILURE: actual={actual}, expected={expected}"


def test_save_call_details():
    # The test will fail if the ordering of the next two line change.
    # Please don't add anything between make_call() and CALLER_LINE_NUM
    make_call()
    CALLER_LINE_NUM = inspect.currentframe().f_lineno - 1
    assert_equal(
        APICacheIdentifier.get_call_details(), f"{FILE_NAME}.{CALLER_LINE_NUM}"
    )


def test_get_unique_identifier_cacheable():
    # The test will fail if the ordering of the next two line change.
    # Please don't add anything between make_call() and CALLER_LINE_NUM
    make_call()
    CALLER_LINE_NUM = inspect.currentframe().f_lineno - 1

    cacheable_apis = {"api1": ["param1", "param2"], "api2": ["paramA"]}
    unique_id = APICacheIdentifier.get_unique_identifier(
        cacheable_apis, "layer1", "api1", param1="value1", param2="value2"
    )
    assert_equal(unique_id, f"{FILE_NAME}.{CALLER_LINE_NUM}.layer1.api1.value1.value2")


def test_get_unique_identifier_with_missing_params():
    # The test will fail if the ordering of the next two line change.
    # Please don't add anything between make_call() and CALLER_LINE_NUM
    make_call()
    CALLER_LINE_NUM = inspect.currentframe().f_lineno - 1

    cacheable_apis = {"api1": ["param1", "param2"]}
    unique_id = APICacheIdentifier.get_unique_identifier(
        cacheable_apis, "layer1", "api1", param1="value1"
    )
    assert_equal(unique_id, f"{FILE_NAME}.{CALLER_LINE_NUM}.layer1.api1.value1.0")


def test_get_unique_identifier_not_cacheable():
    make_call()
    cacheable_apis = {"api1": ["param1", "param2"]}
    unique_id = APICacheIdentifier.get_unique_identifier(
        cacheable_apis, "layer1", "api3", param1="value1"
    )
    assert_equal(unique_id, None)


def test_get_unique_identifier_no_kwargs():
    # The test will fail if the ordering of the next two line change.
    # Please don't add anything between make_call() and CALLER_LINE_NUM
    make_call()
    CALLER_LINE_NUM = inspect.currentframe().f_lineno - 1

    cacheable_apis = {"api1": ["param1", "param2"]}
    unique_id = APICacheIdentifier.get_unique_identifier(
        cacheable_apis, "layer1", "api1"
    )
    assert_equal(unique_id, f"{FILE_NAME}.{CALLER_LINE_NUM}.layer1.api1.0.0")


if __name__ == "__main__":
    test_get_unique_identifier_no_kwargs()
