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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import nvdlfw_inspect.api as debug_api

torch.random.manual_seed(1234)

current_file_path = os.path.dirname(os.path.realpath(__file__))


def ddp_setup():
    init_process_group(backend="nccl")
    gpu_id = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(gpu_id)
    return gpu_id


class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, input_tensor):
        fc1_out = self.fc1(input_tensor)

        # Log output tensor stats for fc1 layer if enabled in the config YAML
        debug_api.log_tensor_stats(
            "fc1", tensor=fc1_out, tensor_name="activation", skip_reduction=True
        )
        relu_out = self.relu(fc1_out)

        # Log output tensor stats for ReLU layer if enabled in the config YAML
        debug_api.log_tensor_stats(
            "relu",
            tensor=relu_out,
            tensor_name="activation",
            reduction_group=torch.distributed.group.WORLD,
        )

        # Apply fake quantization on the output of the ReLU layer if enabled in the config YAML
        relu_quant = debug_api.sample_namespace.process_tensor(
            "relu", tensor=relu_out, gemm="fprop", tensor_name="relu_quant_act"
        )["tensor"]

        # Log Quantized ReLU tensor stats if enabled in the config YAML
        debug_api.log_tensor_stats(
            "relu",
            tensor=relu_quant,
            tensor_name="relu_quant_act",
            reduction_group=torch.distributed.group.WORLD,
        )
        fc2_out = self.fc2(relu_out)

        # Log output tensor stats for fc2 layer if enabled in the config YAML
        debug_api.log_tensor_stats("fc2", tensor=fc2_out, tensor_name="activation")

        return fc2_out


if __name__ == "__main__":
    # To run the script: torchrun --nproc_per_node={num_gpus} test_simple_model_train.py
    gpu_id = ddp_setup()
    debug_config_file = os.path.join(
        current_file_path, "configs/simple_model_train_sample.yaml"
    )
    assert os.path.exists(debug_config_file), (
        f"Debug Config File: {debug_config_file} not found"
    )
    debug_feature_dir = os.path.join(current_file_path, "sample_namespace")
    assert os.path.exists(debug_feature_dir), (
        f"Debug Feature Directory: {debug_feature_dir} not found"
    )

    debug_output_dir = "/tmp/nvdlfw_inspect"
    print(f"Debug Output Directory: {debug_output_dir}")
    debug_api.initialize(
        debug_config_file,
        feature_dirs=[debug_feature_dir],
        log_dir=debug_output_dir,
    )

    input_size = 8
    hidden_size = 16
    output_size = 4

    batch_size = 10

    training_iterations = 50

    gpu_id = int(os.environ["LOCAL_RANK"])
    model = SimpleModel(input_size, hidden_size, output_size)
    model = model.to(gpu_id)
    model = DDP(model, device_ids=[gpu_id])
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for step in range(training_iterations):
        rank = int(torch.distributed.get_rank())
        input_tensor = torch.randn(batch_size, input_size, requires_grad=True) * (
            rank + 1
        )
        input_tensor = input_tensor.to(gpu_id)
        target = torch.randn(batch_size, output_size).to(gpu_id)
        output = model(input_tensor)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the training iteration counter in nvdlfw_inspect
        debug_api.step()
        if gpu_id == 0:
            print(f"Training Step: {step}, Loss for gpu 0: {loss.item()}")

    destroy_process_group()
    print("Training End")
