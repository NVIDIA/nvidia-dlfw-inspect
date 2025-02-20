# Getting Started

This guide explains how to use the **Nvidia DLFW Inspect** tool to debug deep learning model convergence while training a simple model in PyTorch.

Before proceeding, ensure you have set up the tool by following the [installation guide](/docs/installation.md)

This document covers two scenarios:

1. Logging Tensor Statistics during model training in Distributed Data Parallel (DDP) setup
2. Manipulating tensors during training (e.g., FakeQuantization)

These functionalities require only minimal changes to your code.

## Example Training Code

Below is a sample training script. You can also find the complete working example with the debugging APIs: [here](/examples/test_simple_model_train.py)

```

import nvdlfw_inspect.api as debug_api

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
            "relu", tensor=relu_out, tensor_name="activation", reduction_group=torch.distributed.group.WORLD
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
    gpu_id = ddp_setup()
    debug_api.initialize(
        "./configs/simple_model_train_sample.yaml",
        feature_dirs=["./sample_namespace"],
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
```

To train this model using 2 GPUs, run the following command:

```
torchrun --nproc_per_node=2 test_simple_model_train.py
```

Next, we will go over nvdlfw_inspect specific code changes

### 1. Importing the tool

```
import nvdlfw_inspect.api as debug_api
```

### 2. Initializing the Tool.

In your main block, initialize the tool using:

```
debug_api.initialize(
        "nvdlfw_inspect_repo/examples/configs/simple_model_train_sample.yaml" ,
        feature_dirs=["nvdlfw_inspect_repo/examples/sample_namespace"],
        log_dir="/tmp/nvdlfw_inspect"
    )
```

This function must be called once on every rank in global context to initialize Nvidia-DLFW-Inspect.

**Parameters**

* **config_file** ( *str* , default=""): Path to the `config.yaml` file specifying features and layer names.
* **feature_dirs** ( *List[str] | str* ): Directories containing custom features to load. If empty, only [default features](/nvdlfw_inspect/debug_features) are available. In the example above feature directory for [FakeQuantization](/examples/sample_namespace) is provided
* **log_dir** ( *str* , default= "."): Directory where logs and statistics will be stored. The tool creates two subdirectories in the `log_dir` path: `nvdlfw_inspect_logs` and `nvdlfw_inspect_statistics_logs`
* **init_training_step** ( *int* , default=0): Set the training step counter.
* **statistics_logger** ( *Union[BaseLogger, None]* , default=None): Custom Logger for logging tensor statistics. Should adhere to `BaseLogger` from the Nvidia-DLFramework-Inspect package.
* **tb_writer** ( *TensorBoardWriter* , default=None): TensorBoard writer for logging.

### 3. Debug Config Yaml

Below is an example configuration file used in ths guide: [debug config](/examples/configs/simple_model_train_sample.yaml)

```
sample_model_train_log_stats:
  enabled: True
  layers:
    layer_name_regex_pattern: .*(fc1|fc2|relu)
  LogTensorStats:
    enabled: True
    tensors_struct:
        - tensor: activation
          stats: [max, min]
          freq: 4
          start_step: 20
          end_step: 40
        - tensor: relu_quant_act
          stats: [mean, max, min]
          freq: 2
          start_end_list: [[6, 10], [20, 24], [35, 42]]
  sample_namespace:
    FakeQuant:
      enabled: True
      gemms_struct:
        - gemm: fprop
          tensors_struct:
            - tensor: relu_quant_act
              scale: 1
              zero_point: 0
              quant_min: 0
              quant_max: 255
```

**Brief summary of features**

1. Statistics Logging ([LogTensorStats](/nvdlfw_inspect/debug_features/log_tensor_stats.py)):
   1. This is one of default features available with this tool for logging stats. You can find more details about the feature by running: `debug_api.explain_features("base.LogTensorStats")`
   2. Max and Min stats will be logged for activation tensors of FC1, FC2, and ReLU layers if the `log_tensor_stats` api is called for each of the tensors.
      1. The logging will start between the training iterations 20 and 40, on every 4th iteration
   3. Mean, Max, Min stats will be logged for the `relu_quant_act` tensor if the `log_tensor_stats` api is called for this tensor
      1. This logging will only happen during the training iterations specified in the `start_end_list` and at a frequency specified.
2. Tensor Quantization ([FakeQuant](/examples/sample_namespace/sample_feature.py))
   1. This example shows how to use custom features from your own [feature namespace](/examples/sample_namespace). In this example, `FakeQuant` debug feature internally calls `torch.fake_quantize_per_tensor_affine`. You can find more details about the feature by running: `debug_api.explain_features("sample_namespace.FakeQuant")`

### 4. debug_api.step()

The tool maintains an internal training step counter which can be used by different features. In this example, **log_tensor_stats** uses this to determine if the tensor statistics need to be logged for the current training step. This should be added in the main training loop after each forward-backward pass. If resuming training from a checkpoint, you can set the training step counter in two ways.
First would be to pass it during the initialization:
```
debug_api.initialize(
        ...,
        init_training_step={TRAIN_STEP}
    )
```
The other way is to call the following method after initialization:
`debug_api.update_training_step({TRAIN_STEP})`

## Logs

After running the code, you can find two log directories as described before.

1. `{log_dir}/nvdlfw_inspect_logs/nvdlfw_inspect_globalrank-0.log`. Each GPU will have its own log file. Let's briefly review the log

```
INFO - Default logging to file enabled at .
INFO - Reading config from examples/configs/simple_model_train_sample.yaml.
INFO - Loaded configs for dict_keys(['sample_model_train_log_stats']).
INFO - LAYER=fc1: Feature=LogTensorStats, API=log_tensor_stats, TENSOR=activation: Called
INFO - LAYER=relu: Feature=LogTensorStats, API=log_tensor_stats, TENSOR=activation: Called
INFO - LAYER=relu: Feature=FakeQuant, API=process_tensor, GEMM=fprop TENSOR=relu_quant_act: Called
INFO - LAYER=fc2: Feature=LogTensorStats, API=log_tensor_stats, TENSOR=activation: Called
WARNING - LAYER=fc2: `reduction_group` not found and `skip_reduction` is False. Tensor will be only reduced along DP group if initialzed. If this not the desired behavior, pass `reduction_group` when using `log_tensor_stats` feature API. Per-GPU stats are logged in `nvdlfw_inspect_statistics_logs`
```

This log provides detailed information describing the tool's behavior: which configs it processed, and the APIs that were called. This could be useful to check if the tool is doing what you expect.
The default logging level is set to INFO. If you want to get more debugging messages, you can set the environment variable `NVDLFW_INSPECT_LOG_LEVEL=DEBUG`

2. `{log_dir}/nvdlfw_inspect_statistics_logs/nvdlfw_inspect_globalrank-0.log.` Each GPU will have its own statistics log file. Let's briefly review the log

```
INFO - relu_relu_quant_act_mean 				 iteration=000006 				 value=0.1625
INFO - relu_relu_quant_act_max 				 iteration=000006 				 value=2.0000
INFO - relu_relu_quant_act_min 				 iteration=000006 				 value=0.0000
INFO - relu_relu_quant_act_mean 				 iteration=000008 				 value=0.2062
INFO - relu_relu_quant_act_max 				 iteration=000008 				 value=1.0000
INFO - relu_relu_quant_act_min 				 iteration=000008 				 value=0.0000
INFO - fc1_activation_max 				 iteration=000020 				 value=1.4388
INFO - fc1_activation_min 				 iteration=000020 				 value=-1.4398
INFO - relu_activation_max 				 iteration=000020 				 value=1.4388
INFO - relu_activation_min 				 iteration=000020 				 value=0.0000
INFO - relu_relu_quant_act_mean 				 iteration=000020 				 value=0.1562
...
```

## Features

### Statistics Logging

In the code, you can find four api calls to statistics logging. There are slight differences in their operation for DDP training. Below, we briefly review some of them

1. `debug_api.log_tensor_stats( "fc1", tensor=fc1_out, tensor_name="activation", skip_reduction=True)`
   In DDP, each GPU will log the local statistics of the tensor since `skip_reduction` is set to `True`.
2. `debug_api.log_tensor_stats("relu", tensor=relu_quant, tensor_name="relu_quant_act", reduction_group=torch.distributed.group.WORLD)`
   In DDP, each GPU will first gather the tensor from all other GPUs and then each GPU will calculate the stats on the gathered tensor and log it. In this case, each rank will log the same value.
3. `debug_api.log_tensor_stats("fc2", tensor=fc2_out, tensor_name="activation")`
   In DDP, the GPU with the last rank will gather the tensor from all other GPUs, then the last rank GPU will calculate the desired statistics on the gathered tensor and log it. In the case of 2 GPUs, only one rank will log the tensor statistics.

### Tensor Processing

The code provides one example showing how a tensor can be modified for testing convergence and exploring other algorithms. Here, we quantize the tensor by calling the following API:

```
relu_quant = debug_api.sample_namespace.process_tensor(
            "relu", tensor=relu_out, gemm="fprop", tensor_name="relu_quant_act"
        )["tensor"]
```

This API returns the modified tensor in a dictionary with the key `tensor`
