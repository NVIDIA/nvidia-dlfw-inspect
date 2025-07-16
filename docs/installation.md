# Installation Guide for `nvdlfw_inspect`

This document provides step-by-step instructions to set up and install the `nvdlfw_inspect` tool. The recommended setup is to use the **Nvidia PyTorch Docker container** as the base environment. Additional instructions include installing directly via GitHub, setting up a development environment, running tests, and building the wheel.

---

## Prerequisites

`nvdlfw_inspect` requires the following dependencies:
- Python >= 3.8
- PyTorch >= 2.4.0
- NumPy == 1.26.4
- PyYAML >= 6.0.0

**Note:** When using NVIDIA PyTorch Docker containers, PyTorch and NumPy are typically pre-installed. The `pip install` command will automatically skip reinstalling packages that already meet the version requirements.


## Installation Options

## Installing via Pip

NVDLFW-Inspect has very minimal dependecies and can be directly installed in an environment that already has torch and numpy (we don't enforce the versions for these libraries but are tested for `torch>=2.4.0` and `numpy==1.26.4`). This method is also applicable if you're running inside a Nvidia PyTorch container.

### 1. Installing Directly via GitHub

You can install `nvdlfw_inspect` directly from its GitHub repository:

`pip install git+https://github.com/NVIDIA/nvidia-dlfw-inspect`

### 2. Building and Installing for the source

You can also clone the repostory, run `pip install .` / `make install` or run in the dev mode `pip install .[dev]`. You can add the repository path to `$PYTHONPATH` env variable to access the tool if you dont want to install it.

The installation can be verified by running `python3 -c "import nvdlfw_inspect"`.

## Installing via Docker

1. **Pull the NVIDIA PyTorch Docker Image**:

`docker pull nvcr.io/nvidia/pytorch:24.12-py3` (or any other version).

2. **Build the Docker Image**:
Run the following command from the root of the project directory (where the `Makefile` is located):

`make docker-build`

3. **Run the Docker Container**:
Launch an interactive container with GPU support:

`make docker-run`

## Development Setup

For local development (when not using a nvidia-pytorch container), you can set up the local development environment by running: `make dev-local` and then source `source dev_local_venv/bin/activate`

To contribute or modify the tool, you can also set up a development environment: `make dev-env` and activate the virtual environment `source dev_venv/bin/activate`. This can only be used for running the pre-commit checks: `make check`

## Running Tests

To ensure everything is working as expected, you can run tests using `pytest`: `make test`

## Building a Wheel

To create a distributable `.whl` file for `nvdlfw_inspect`:

1. Run the build script: `./scripts/build_wheels.sh`

The resulting wheel file will be located in the `dist/` directory.
