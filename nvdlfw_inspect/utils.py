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

import importlib.util
import inspect
import logging
import os
import sys
from typing import Any

import torch
import torch.distributed as dist

from .logging import get_logger


def get_distributed_world_size(group: dist.ProcessGroup | None = None) -> int:
    """Return world size for the distributed group."""
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size(group=group)


def get_distributed_rank(group: dist.ProcessGroup | None = None) -> int:
    """Return my rank for the distributed group."""
    assert torch.distributed.is_initialized(), "torch.distributed is not initialized."
    return torch.distributed.get_rank(group=group)


def gather_along_first_dim(
    input_: torch.Tensor,
    process_group: dist.ProcessGroup,
    async_op: bool = False,
) -> tuple[torch.Tensor, Any]:
    """All-gather tensors and concatenate along first dimension."""
    world_size = get_distributed_world_size(process_group)
    if world_size == 1:
        return input_, None

    output_shape = list(input_.size())
    output_shape[0] *= world_size
    output = torch.empty(
        output_shape,
        dtype=input_.dtype,
        device=input_.device,
        memory_format=torch.contiguous_format,
    )
    src = input_.contiguous()
    dst = output

    handle = torch.distributed.all_gather_into_tensor(
        dst,
        src,
        group=process_group,
        async_op=async_op,
    )
    return output, handle


def gather_tensor_on_last_rank(
    input_: torch.Tensor,
    process_group: dist.ProcessGroup | None = None,
    async_op: bool = False,
) -> tuple[torch.Tensor, Any]:
    """Gather tensor and concatenate on last rank"""
    world_size = get_distributed_world_size(process_group)
    if world_size == 1:
        return input_, None

    rank = get_distributed_rank(process_group)
    gathered_tensor = (
        [torch.zeros_like(input_) for _ in range(world_size)]
        if rank == world_size - 1
        else None
    )

    handle = torch.distributed.gather(
        input_,
        gather_list=gathered_tensor if rank == world_size - 1 else None,
        dst=world_size - 1,
        async_op=async_op,
    )
    if rank == world_size - 1:
        gathered_tensor = torch.cat(gathered_tensor, dim=0)

    return gathered_tensor, handle


def import_and_exec_module(module_path: str):
    module_dir = os.path.dirname(module_path)
    sys.path.insert(0, module_dir)
    module_name = os.path.splitext(os.path.basename(module_path))[0]
    m_spec = importlib.util.spec_from_file_location(module_name, module_path)
    if m_spec is None:
        get_logger().log_message(
            f"Could not create a module spec for {module_path}", level=logging.ERROR
        )
        sys.exit(1)
    module = importlib.util.module_from_spec(m_spec)
    package_name = os.path.basename(module_dir)
    module.__package__ = package_name
    try:
        m_spec.loader.exec_module(module)
    except Exception as e:
        get_logger().log_message(
            f"An error occured while loading module {module_name}: {e}",
            level=logging.ERROR,
        )
        sys.exit(1)
    finally:
        sys.path.pop(0)


class APICacheIdentifier:
    _call_details: str

    @classmethod
    def save_call_details(cls):
        current_frame = inspect.currentframe()
        caller_frame = current_frame.f_back.f_back # 2 back since nvdlfw.api routes here
        cls._call_details = f"{caller_frame.f_code.co_filename}.{caller_frame.f_lineno}"
        del current_frame, caller_frame

    @classmethod
    def get_call_details(cls):
        return cls._call_details

    @classmethod
    def get_unique_identifier(cls, cacheable_apis, layer_name, api_name, **kwargs):
        _id = None
        if api_name in cacheable_apis:
            api_caller_details = cls._call_details
            _id = f"{layer_name}.{api_name}"
            for key in cacheable_apis[api_name]:
                _id += ".0" if key not in kwargs else f".{kwargs[key]}"
            _id = api_caller_details + "." + _id

        return _id


class SingleMessageLogger:
    logged_messages = set()  # noqa: RUF012

    @classmethod
    def log_message_once(
        cls, message, layer_name=None, level=logging.INFO, extra_cachable_args=None
    ):
        caller_frame = inspect.currentframe().f_back
        feature_call_details = (
            f"{caller_frame.f_code.co_filename}.{caller_frame.f_lineno}"
        )
        try:
            api_call_line = APICacheIdentifier.get_call_details()
        except Exception:
            api_call_line = None
        message_key = (
            api_call_line,
            feature_call_details,
            layer_name,
            extra_cachable_args,
        )
        if message_key not in cls.logged_messages:
            if layer_name:
                get_logger().log_message(f"LAYER={layer_name}: {message}", level=level)
            else:
                get_logger().log_message(message, level=level)
            cls.logged_messages.add(message_key)


def append_parent_docstring(parent):
    def decorator(cls):
        parent_doc = getattr(parent, "__doc__", "")
        if not parent_doc:
            parent_doc = f"No docstring found for parent class: {parent.__name__}"

        cls_doc = getattr(cls, "__doc__", "")
        cls.__doc__ = f"{cls_doc}\n\n{parent_doc}"
        return cls

    return decorator
