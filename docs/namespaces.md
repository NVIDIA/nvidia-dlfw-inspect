# Adding Custom Namespace/Framework Specific APIs

This document provides a step-by-step guide on how to add custom namespace/framework APIs by using an example implementations: `SampleFrameworkAPI`.
The example implementation can be found in [examples/sample_namespace/sample_feature_api.py](/examples/sample_namespace/sample_feature_api.py).

## Overview

The process involves the following key components:
1. **Custom Configuration API Mapper**: To parse and process configuration files.
2. **Default Features Implementation**: To define default behaviors for APIs for when config files are not provided.
3. **Custom Namespace/Framework API**: To define and register the namespace API with specific hooks and conditions.

### 1. Custom Configuration API Mapper

The ConfigAPIMapper has 3 main functionalities:
1. Responsible for parsing configuration files and extracting relevant settings based on the API requirements.
2. Checks whether the feature should be invoked by comparing the API arguments (in kwargs) with the user desired features in the config file.
3. If feature should be invoked, returns status=True along with the parsed config for the feature. Otherwise, returns False and empty config.

All features defined in this namespace must inherit from the ConfigAPIMapper to parse their configs during an API invocation.
Below is an example of how to create such a mapper:

#### Example: SampleConfigAPIMapper

```python
from nvdlfw_inspect.base import BaseConfigAPIMapper

# must inherit from BaseConfigAPIMapper
class SampleConfigAPIMapper(BaseConfigAPIMapper):
    gemm_and_tensor_config_docstring = '''
    Supported yaml config structure for this config mapper:
    1. gemms: [gemm1, gemm2]
       tensors: [tensor1, tensor2]
       tensor_feature_param1: value
       gemm_feature_param1: value

    2. gemms: [gemm1, gemm2]
       tensors_struct:
        - tensor: tensor1
          tensor_feature_param1: value
        - tensor: tensor2
          tensor_feature_param2: value
        gemm_feature_param1: value

    3. gemms_struct:
        - gemm: gemm1
          tensors: [tensor1, tensor2]
          tensor_feature_param1: value
          gemm_feature_param1: value
        - gemm: gemm2
          tensors_struct:
          - tensor: tensor1
            tensor_feature_param1: value
          - tensor: tensor2
            tensor_feature_param2: value
          gemm_feature_param1: value
    '''
    def parse_config_and_api(self, config, **kwargs):
        """
        Defines custom logic to parse a config provided by the user.

        If the user provides a config yaml defined as such:
        config_name:
            enabled: True
            layers:
                (...)
            LogTensorStats:
                enabled: True
                stats: [max, min, mean, std, l1_norm]
                tensors: [activation]
                (...)

        The config input to this method for feature LogTensorStats will be:
        {
            "enabled": True
            "stats": [max, min, mean, std, l1_norm]
            "tensors": [activation]
        }

        Kwargs
        ------
        User can pass kwargs as required. In this case, we use `gemm_parsing` and `tensor_parsing` to indicate the
        various levels of the config (a GEMM config can contain tensor configs).
        """
        processed_config = None
        config_copy = copy.deepcopy(config)
        gemm_parsing = kwargs.get('gemm_parsing', False)
        tensor_parsing = kwargs.get('tensor_parsing', False)

        if gemm_parsing and tensor_parsing:
            processed_config = self._process_gemm_and_tensor_config(config_copy, **kwargs)

        if not processed_config:
            return False, None

        if "enabled" in processed_config:
            processed_config.pop("enabled")
        return True, processed_config
```

- The `SampleConfigAPIMapper` overrides the `parse_config_and_api` method of the parent class and adds custom logic to parse the config.
- In this case, the config file contains _gemms_, _tensor_, _gemms\_struct_ or _tensors_struct_ along with other parameters.
- The base class provides a helper function to parse tensor configs - see `BaseConfigAPIMapper` in [nvdlfw_inspect/base.py](/nvdlfw_inspect/base.py).
- If your use-case is adding features that work only on tensor-level granularity, you can reuse the `GenericConfigAPIMapper` provided in [nvdlfw_inspect/debug_features/generic_feature_api.py](/nvdlfw_inspect/debug_features/generic_feature_api.py) that already implements parsing config based on tensors.


### 2. Default Features Implementation

Define default behaviors for APIs in a separate class. These default APIs are called either when a layer is not selected in a config or
a config is not provided by the user.

Users must define methods for all APIs exposed by their various features.


#### Example: SampleDefaultFeatures

```python
class SampleDefaultFeatures:
    def process_tensor(self, config, layer_name, **kwargs):
        """
        API to process a tensor. This must return a tensor.
        """
        return {"tensor": kwargs["tensor"]}
```

### 3. Custom Namespace/Framework API

- Create the namespace API by inheriting from `BaseNamespaceAPI`, and implement the required hooks and conditions for routing API calls.
- Register the namespace API with a unique global name.

#### Example: SampleFrameworkAPI

```python
from nvdlfw_inspect.base import BaseNamespaceAPI
from nvdlfw_inspect.registry import Registry

@Registry.register_namespace_api(namespace="sample_namespace")
class SampleFrameworkAPI (BaseNamespaceAPI):
    def __init__(self):
        BaseNamespaceAPI.__init__(self)
        # NOTE: Pointer to class containing default implementations of APIs
        self._default_api_impl = SampleDefaultFeatures()

        # NOTE: Map of API arguments that can be used for API caching.
        # In this case, the values of tensor_name and gemm in the process_tensor API can be cached.
        self._cacheable_api_kwargs_map = {
            "process_tensor": ["tensor_name", "gemm"],
        }

    def input_assertions_hook(self, api_name, **kwargs):
        required_kwargs = {
            "process_tensor": ["tensor", "gemm", "tensor_name"]
        }

        if api_name in required_kwargs:
            for kwarg in required_kwargs[api_name]:
                assert kwarg in kwargs, f"[NVDLFW INSPECT ERROR] Cannot route API, too ambiguous. Provide {kwarg} in {api_name}."

    def routing_condition(self, api_name, config, layer_name, feature_obj, **kwargs):
        status, modified_config = feature_obj.parse_config_and_api(config, gemm_parsing=True, tensor_parsing=True, **kwargs)
        return status, modified_config

    def output_assertions_hook(self, api_name, ret, **kwargs):
        if api_name in {"process_tensor"}:
            assert "tensor" in ret, f"This API {api_name} must return a tensor."

    def is_multiple_feature_invocation_allowed(self, api_name):
        return False
```

The namespace API must override the following methods.

1. `input_assertions_hook`: This function checks if the necessary keyword arguments for a specific API call are provided, and raises an assertion error if any are missing.
2. `output_assertions_hook`: Similar to input assertion hook but for outputs.
3. `routing_condition`: This function takes as input the name of the API and corresponding feature object, and checks whether that API should be invoked based on the user-defined config. The user-defined config is parsed using the ConfigAPIMapper as shown previously. If the API should be invoked, it returns status=True and the parsed feature-only config.
4. `is_multiple_feature_invocation_allowed`: In most cases, this should return `False` as we expect each API call for a given layer name to map exactly to 1 feature class. However, in some cases, that might not be possible (we encountered this for some features in TransformerEngine). If returned `True`, only 1 case is supported where all invoked features return the same value, otherwise it exits with an error. In the future, we plan to move this functionality to the feature namespace for handling more complex scenarios.

## Using Custom Namespaces

All APIs defined in the namespace can be called as follows:
```python
debug_api.sample_namespace.process_tensor(...)
```

If the user provides a config or the layer is selected as a part of the config, the above API call loops over all feature objects that implement
it and, for each one, checks whether the feature should be invoked using the `parse_config_and_api` method. Otherwise, the default version of the API is invoked.
