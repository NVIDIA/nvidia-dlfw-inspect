# General Logging

The tool provides a basic logging mechanism to track the state of execution. Specify the `log_dir` during tool initialization to set the directory for log files:
```python
debug_api.initialize(..., log_dir=path_to_dir)
```
Logs are collected in the dictory: `{log_dir}/nvdlfw_inspect_logs`. You can control logging with several environment variables:
1. Set the logging level using the environment variable: `NVDLFW_INSPECT_LOG_LEVEL={DEBUG, INFO, WARNING, ERROR, etc}`. The default level is `INFO`.
2. By default, the logger will overwrite existing logs. To disable overwriting, set the environment variable: `NVDLFW_INSPECT_ENABLE_LOG_OVERWRITE=0`.
3. To add timestamps to log files, enable the environment variable: `NVDLFW_INSPECT_ENABLE_LOG_TIMESTAMP=1`.

After initializing the nvdlfw inspect tool, you can add debug logs using the following API:
```python
debug_api.log_message("MESSAGE", level=logging.{LEVEL})
```

# Tensor Statistics Logging

To enable logging of tensor statistics, users can add the following features to the config:
```yaml
LogTensorStats: # generic tensor stats including [max, min, mean, std, l1_norm, l2_norm]
    ...
transformer_engine:
    LogTensorStats: # tensor stats specific for transformer engine. Includes generic + [cur_amax, dynamic_range]
        ...
    LogFp8TensorStats: # tensor stats for FP8 tensors. Includes [overflows, underflows]
        ...
```

There are current 3 ways to log and visualize the tensor statistics:
 1. Users can pass a logger while initializing the nvdlfw inspect tool. The logger must inherit from `BaseLogger` reference implementation provided in logging.py.
```python
debug_api.initialize(..., statistics_logger=your_logger_here)
```

2. Users can also choose to pass a tensorboard writer while initializing the nvdlfw inspect tool.
```python
debug_api.initialize(..., tb_writer=your_writer_here)
```

3. If no loggers are passed during initialization and the logging features are enabled, the nvdlfw inspect tool will log tensor statistics to a file. The file will be under `debug_statistic_logs` directory.

To change the logging directory, pass the directory using the `log_dir` kwarg.
```python
debug_api.initialize(..., log_dir=path_to_dir)
```

## Logging with Model Parallelism

The nvdlfw inspect tool is initialized once per GPU rank. This means that each rank will compute its own tensor statistic and the final logged tensor statistic would require reduction across tensor parallel and data parallel groups before logging.

To pass a tensor reduction group to the nvdlfw inspect tool, use the following API:
```python
debug_api.set_tensor_reduction_group(group)
```
The statistics will be reduced across this group before being logged.

However, some statistics might not require reduction. Users can use the `skip_reduction` kwarg in the feature API call to skip reduction:
```python
debug_api.log_tensor_stats(layer_name, tensor=tensor, tensor_name="name", skip_reduction=True)
```

Finally, the nvdlfw inspect tool will only log on the ranks that contain a tensorboard writer. This is not an issue when using just tensor and data parallel, however, it can be an issue when using pipeline parallel. There are 2 ways to solve this:

1. Log statistics on every rank to file. This will compute and log statistics using the default logger to file and the statistics can be gathered during post-process. This is also the most optimized way since it avoids a reduction step. To enable default statistic logging to file, use:
```python
debug_api.initialize(default_logging_enabled=True)
# AND
debug_api.log_tensor_stats(..., skip_reduction=True)
```

2. Pass a logger to the nvdlfw inspect tool on every rank in the last pipeline parallel group and pass the tensor and data parallel group as the reduction group.
Here is an example of this using a tensorboard writer (already has been setup in Megatron-LM):
```python
tb_writer = None
# Check if rank is in last pipeline parallel group
if parallel_state.is_in_last_pipeline_model_parallel_group() and log_dir != None:
        tb_writer = create_tensorboard_writer(
            os.path.join(log_dir, f"pp_rank_{torch.distributed.get_rank()}")
        )
# pass tensorboard writer only when rank is in last pipeline parallel stage.
debug_api.initialize(config,
                            extensions=extensions,
                            log_dir=log_dir,
                            tb_writer=tb_writer)

# set the group
debug_api.set_tensor_reduction_group(tensor_and_data_parallel_group)
```

## Train Step Counter

The logging feature APIs require an iteration to be passed as a kwarg to log the tensor statistics at that iteration. However, if the iteration count is not available from where the APIs are being invoked, the nvdlfw inspect tool has an internal step counter. Users have to invoke the step counter API to increase the train step from within the training loop:
```python
for i in range(num_epochs):
    ...
    debug_api.step()
```
