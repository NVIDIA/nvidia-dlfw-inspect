# Defining Custom Features
In the context of this tool, a feature has a set of APIs that provide a specific functionality for a namespace. Features are defined as classes and are registered to a corresponding namespace.

This document provides a step-by-step guide on how to add a new feature: please check [FakeQuant](examples/sample_namespace/sample_feature.py) for the full implementation.

## Prerequisites

All features must belong to a namespace. The namespace must be defined along with its API, default features and ConfigAPIMapper.
We provide 2 predefined namespaces that you can attach features too if required - [GenericFrameworkAPI](nvdlfw_inspect/debug_features/generic_feature_api.py) and `TransformerEngineAPI` (This is available on TransformerEngine [Github](https://github.com/NVIDIA/TransformerEngine)) . GenericFrameworkAPI is used for features that can be used within any framework, whereas, TransformerEngineAPI is used for TransformerEngine specific features only.

To define new namespaces, refer to [docs/namespaces.md](docs/namespaces.md) doc before creating features.

## Overview

The process involves the following key components:
1. **Feature Class**: Define and register features.
2. **Feature APIs**: Implement APIs as part of this feature.


### Feature Class

Define the feature logic in a separate class and implement the required API methods.

- The following example defines a new feature called `FakeQuant` in the `sample_namespace` namespace. Looking at [examples/sample_namespace](examples/sample_namespace), we see that it just has 1 API - `process_tensor` defined. Hence, this feature can override that API and implement custom logic. A feature class can have more than 1 API based on the use-case (Please check the features in [TransformerEngine](https://github.com/NVIDIA/TransformerEngine) as well)
- Feature classes also must inherit from the corresponding ConfigAPIMapper to parse the config from the yaml required for APIs to execute.
- The `process_tensor` API takes as input the parsed config, layer names and other kwargs passed during API call.
- Finally, the `process_tensor` API requires the decorator `@api_method` so that its registered as an external API.

#### Example: FakeQuant

```python
from sample_feature_api import SampleConfigAPIMapper
import nvdlfw_inspect.api as debug_api
from nvdlfw_inspect.registry import Registry, api_method

@Registry.register_feature(namespace="sample_namespace")
class FakeQuant(SampleConfigAPIMapper):
    """
    Fake Quantization feature in Pytorch with torch.fake_quantize_per_tensor_affine
    Returns a new tensor with the data in input fake quantized using scale, zero_point, quant_min and quant_max.

    APIs:
    1. sample_namespace.process_tensor
    - When using this api, you would need to pass args:
    -- layer_name: : this is matched with the layer description in the config file
    -- gemm: this is matched with one of the gemms in the config field, and passed as a kwarg. For example, gemm='gemm1'
    -- tensor_name: this is matched with one of the tensors in the config field for a given gemm, and passed as a kwarg. For example, tensor_name='tensor1'
    -- tensor: the tensor to process, and passed as a kwarg. For example, tensor={torch tensor}

    Config:
    To enable the feature in yaml config:
    sample_namespace:
      fake_quant:
        enabled: True
        feature_properties:
        ...

    Config fields:
    This feature works at a tensor level, you can set the following properties for each tensor:
    - scale: double or float32 scalar, quantization scalar
    - zero_point: int scalar or tensor, quantization zero_point
    - quant_min: int, lower bound of quantized domain
    - quant_max: int, upper bound of quantized domain

    Gemm and Tensor structure is described below:
    """
    @classmethod
    def _set_class_doc(cls):
        cls.__doc__ = (f"\n{cls.__doc__}\n  gemm and tensor structure in yaml will be as follows: {SampleConfigAPIMapper.gemm_and_tensor_config_docstring}")

    @api_method
    def process_tensor(self, config, layer_name, **kwargs):
        self.log_message_once(f"Feature={self.__class__.__name__}, API=process_tensor: {kwargs['gemm']}, {kwargs['tensor_name']}: FakeQuant", layer_name)

        for field in ["scale", "zero_point", "quant_min", "quant_max"]:
            if field not in config:
                raise ValueError(f"[NVDLFW INSPECT ERROR] Feature={self.__class__.__name__}, API=process_tensor: {field} missing for Tensor: {kwargs['tensor_name']} in the config yaml for FakeQuant feature which is a required field")

        q_tensor = torch.fake_quantize_per_tensor_affine(kwargs["tensor"], config['scale'], config['zero_point'], config['quant_min'], config['quant_max'])
        return {"tensor": q_tensor}

FakeQuant._set_class_doc()
```

#### Invoking FakeQuant

To enable the feature, define a config as shown below with the required feature properties:
```yaml
fake_quant_config:
  enabled: True
  layers:
    layer_name_regex_pattern: .*(attention).*
  sample_namespace: # This is required since its part of the `sample_namespace`
    FakeQuant: # This name should match the class name
      enabled: True
      tensors: [linear_proj_in]
      zero_point: ...
      scale: ...
```
The above config will enable fake quant for the specified tensors in the specified layers.

Then, invoke the API within the layer as follows:
```python
class AttentionLayer:
    def forward(self, inp):
        hidden = self.linear_qkv(inp)

        # This API will only call process_tensor API of the FakeQuant feature since the tensor and layer names passed as args
        # match the names defined in the config.
        hidden = debug_api.sample_namespace.process_tensor(layer_name="layer1.attention", tensor=hidden, tensor_name="linear_proj_in")["tensor"]

        hidden = self.linear_proj(hidden)
        ...
        return out
```

#### Example: LogTensorStats

Here is another example for a feature for logging tensor statistics using a different API called `log_tensor_stats`. It is defined and invoked in the same way as before.

```python
@Registry.register_feature(namespace="base")
class LogTensorStats(GenericConfigAPIMapper):
    return_tensor = False

    def _check_log_frequency(self, config, **kwargs):
        if config.get("freq", None) is None:
            self.log_message_once("Frequency of logging is not provided. Using freq = 1 train step as default.")
            freq = 1
        else:
            freq = int(config["freq"])

        iteration = self._get_current_iteration(**kwargs)
        if not enable_logging_at_current_step(iteration, freq, config.get("start_step", 0), config.get("end_step", -1), config.get("start_end_list", None)):
            return False
        return True

    def _check_params(self, config, layer_name, **kwargs):
        if not self._check_log_frequency(config, **kwargs):
            return False
        return True

    def _check_and_gather_tensor(self, config, layer_name, **kwargs):
        skip_reduction = kwargs.get("skip_reduction", False)
        if skip_reduction:
            return kwargs["tensor"], None

        # override global group
        reduction_group = debug_api.get_tensor_reduction_group()
        if kwargs.get("reduction_group", None) is not None:
            reduction_group = kwargs["reduction_group"]

        if not reduction_group:
            self.log_message_once("`reduction_group` not found and `skip_reduction` is False. " + \
                                  "Tensor will be only reduced along DP group if initialzed. " + \
                                  "If this not the desired behavior, pass `reduction_group` when using `log_tensor_stats` feature API. " + \
                                  "Per-GPU stats are logged in `nvdlfw_inspect_statistics_logs`",
                                  layer_name=layer_name,
                                  level=logging.WARNING)
            return gather_tensor_on_last_rank(kwargs["tensor"])

        return gather_along_first_dim(kwargs["tensor"], process_group=reduction_group)

    def _get_current_iteration(self, **kwargs):
        if "iteration" in kwargs:
            iteration = kwargs["iteration"]
        else:
            iteration = debug_api.DEBUG_MANAGER._trainer_iteration_count
        return iteration

    @api_method
    def log_tensor_stats(self, config, layer_name, **kwargs):
        if not self._check_params(config, layer_name, **kwargs):
            if self._is_tensor_return_enabled():
                return {}, None
            else:
                return {}

        gathered_tensor, _ = self._check_and_gather_tensor(config, layer_name, **kwargs)
        iteration = self._get_current_iteration(**kwargs)

        stats = {}
        non_supported_stats_list = []
        if torch.is_tensor(gathered_tensor):
            for stat in config.get('stats', []):
                if stat.lower() not in self._get_supported_stats_list():
                    non_supported_stats_list.append(stat.lower())
                    continue
                ...
```

### Logging during Training

We provide a `SingleMessageLogger` class accessible through the nvinspect API so that users can log messages from within feature classes. The `log_message` API can be invoked from within the feature class during train steps and will only log the given message once. It uses API call caching to determine if the API `log_message` has already been invoked from a particular line in the file.

```python
class LogTensorStats(GenericConfigAPIMapper):
    def custom_api(self, *):
        debug_api.log_message(...)
```
