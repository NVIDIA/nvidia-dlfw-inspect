# Nvidia-DL-Framework-Inspect

This tool facilitates debugging convergence issues and testing new algorithms/recipes for training LLMs using Nvidia libraries: Transformer Engine, Megatron-LM, and NeMo. It is compatible with any PyTorch model and/or any combination of our libraries (e.g., Megatron-LM/NeMo + Transformer Engine or Transformer Engine/Megatron/Nemo standalone). 

Users can leverage configuration files to enable or disable specific features by selecting layers based on their names. This allows for a more tailored debugging experience, as users can add framework-specific features and only use the features that are necessary, without having to load all debug features.

Additionally, APIs are provided for processing tensors and collecting tensor statistics. These APIs can be easily customized based on user specifications, providing a flexible and adaptable debugging solution.

## Getting Started

### Installation

Please check the installation [doc](/docs/installation.md)

## Initialization

To initialize, use the `initialize` API in a global context. For multi-GPUs, initialize once on every rank.

```python
# import the tool
import nvdlfw_inspect.api as debug_api

# Initialize the debug API
debug_api.initialize()
```

### Configuration Files

The config file contains all the features that will be enabled in the run. The config is structured as follows:

```yaml
config_name: # Config name. Used to differentiate between configs in the same file.
  enabled: True
  layers: # This field is used for layer selection.
    layer_name_regex_pattern: .*(fc1|qkv) # In this config, all layers ending with fc1 or qkv are selected.
  LogTensorStats: # Enabling a feature class called `LogTensorStats`.
    enabled: True
    type: [mean, std, l1_norm] # Type of statistics to log
    tensors: [activation, weight, gradient] # statistics collected from these tensors
```

To initialize a configuration file, use the `config_file` argument when initializing debug.

```python
debug_api.initialize(config_file="debug_config.yaml")
```

To learn more about the structure of the config file, check out [docs/config_file.md](/docs/config_file.md) or look through some example configs in the `examples/configs/` directory.

### Loading Specific Features

Specific debug features can be loaded using `feature_dirs` argument while initializing debug.
This will register all features and framework APIs found under the directory. By default, debug tool will just load generic features found in `debug_features/`.

```python
# Example to load transformer_engine specific features
debug_api.initialize(config_file="debug_config.yaml", feature_dirs=["/path/to/transformer_engine/debug/features"])
```

### Available Features

To list features, run the following after initialization. This gives the list of features that can be enabled through the configuration file

```python
debug_api.list_features()
```

To know more about each of these features, run the following after initialization.
```python
debug_api.explain_features(features)
```
the argument can be a string or list with the feature names. To print all the features, you can pass the string 'all'.

### API Usage

All APIs are defined under a namespace. Generic features that are framework-agnostic contain generic APIs that can be accessed as:

```python
debug_api.log_tensor_stats(layer_name, tensor=weight, tensor_name="weight")
```

Whereas, framework specific features contain framework specific APIs that can be accessed as:

```python
# Even though the API name is the same, it uses transformer_engine specific API to log transformer_engine specific tensor statistics.
debug_api.transformer_engine.log_tensor_stats(layer_name, tensor=weight, tensor_name="weight")
```

## Documentation
Please read the following docs for more information on adding new features, namespaces and APIs.

- Getting Started: [docs/getting_started.md](/docs/getting_started.md)
- Logging: [docs/logging.md](/docs/logging.md)
- Config: [docs/config_file.md](/docs/config_file.md)
- Debug Features: [docs/features.md](/docs/features.md)
- Namespaces: [docs/namespaces.md](/docs/namespaces.md)
