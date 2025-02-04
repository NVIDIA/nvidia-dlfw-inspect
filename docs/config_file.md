# Config File Structure

NVDLFW uses `config.yaml` files to track, enable and disable various features such as defining which statistics to log for particular tensors. Below, we outline how to structure the `config.yaml` file.

## Overview

A config file can have one or more sections, each containing settings for specific layers and features:

```yaml
section_name_1:
  enabled: True # experiment 1 is enabled
  layers:
    # Specify layers here...
  namespace: # Features that are tied to a particular namespace e.g. transformer_engine
    Feature1Name:
      enabled: ...
      # Feature details...
    Feature2Name:
      enabled: ...
      # Feature details...

section_name_2:
  enabled: False # experiment 2 is disabled
  layers:
    # Specify layers here...
  Feature1Name: # If feature has no namespace, then it is in default namespace.
    enabled: ...
    # Feature details...
```

Each section can have any name and must contain:
1. An enabled field that specifies whether the features in that section will be active.
2. A layer's field specifying which layers the section applies to. Each layer can belong to only one section.
3. Additional fields describing features for those layers.

## Layer Specification
Debug layers can be identified either by a the name parameter:

```python
linear = transformer_engine.debug.pytorch.Linear(in_features, out_features, debug_name="linear1")
```

or found in the nvdlfw logs when using inferring layer names from the pytorch graph:

```
2025-01-31 05:11:57,884 - INFO - Assigned layer name: model.module.module.module.decoder.layers.1
2025-01-31 05:11:57,884 - INFO - Assigned layer name: model.module.module.module.decoder.layers.1.input_layernorm
2025-01-31 05:11:57,885 - INFO - Assigned layer name: model.module.module.module.decoder.layers.1.self_attention
```

This name is used in the config file to identify the layer. To specify the layers field, you can use one of the following methods:

1. `layer_name_regex_pattern`: Use a regex to match layer names. **Recommened**
2. `layer_types`: Provide a list of strings, where a layer will be selected if any string matches part of its name.

Examples:
``` yaml
# Example 1: Using regex to select layers
my_section:
  enabled: ...
  layers:
    layer_name_regex_pattern: 'self_attn.*'

my_section:
  enabled: ...
  layers:
    layer_types: ['fc1', 'layernorm_linear']
```

## Features Specification

Users can select multiple features that they want to use in a section and add more features in other sections, all in 1 `config.yaml`.

To further illustrate this, lets consider running an experiment in TransformerEngine that uses per-tensor current scaling for FP8 casting of the gradient tensor and collects the current absolute maximum of the gradient before casting and the underflow percentage of the casted gradient tensor.

```yaml
experiment_1:
  enabled: True
  layers:
    layer_name_regex_pattern: .*(fc1|fc2) # Regex pattern selecting all layers ending with fc1 or fc2
  transformer_engine: # Enable transformer_engine specific features
    LogTensorStats: # Log current amax of tensor before casting
      enabled: True
      stats: [cur_amax]
      tensors: [gradient]
      freq: 10
    LogFp8TensorStats: # Log underflow percentage after casting
      enabled: True
      stats: [underflows]
      tensors: [gradient]
      freq: 10
    PerTensorScaling: # per tensor current scaling for gradient tensor
        enabled: True
        gemms_struct:
          - gemm: dgrad
            tensors: [gradient]
          - gemm: wgrad
            tensors: [gradient]
```

To take it a step further, lets run an experiment that runs the Fprop GEMM in self-attention layers in high precision.

```yaml
experiment_1:
  enabled: True
  layers:
    layer_name_regex_pattern: .*(fc1|fc2)
  ... FROM PREVIOUS CONFIG

experiment_2:
  enabled: True
  layer_name_regex_pattern: .*(self_attention).*
  transformer_engine:
    DisableFp8Gemm: # Disable Fp8 GEMM feature
      enabled: True
      gemms: [fprop]
```

### Gemms_struct and tensors_struct

Sometimes a feature is parameterized by a list of tensors or by a list of GEMMs. There are multiple ways of describing this parametrization.

We can pass lists, as below:

```yaml
Feature:
  enabled: ...
  gemms: [gemm1, gemm2]
  tensors: [tensor1, tensor2]
  ...
```

We can use struct for tensors:

```yaml
Feature:
  gemms: [gemm1, gemm2]
  tensors_struct:
  - tensor: tensor1
    feature_param1: value
  - tensor: tensor2
    feature_param1: value
  gemm_feature_param1: value
```

**NOTE:** If we want to use structs both for tensors and GEMMs, tensors_struct should be inside GEMMs_struct.

Similarly, we can use struct for GEMMs.

```yaml
Feature:
  enabled: ...
  gemms_struct:
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
```

By organizing your `config.yaml` properly, you can easily manage debugging features, ensuring a more streamlined and customizable debugging experience.