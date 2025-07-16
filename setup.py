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

import pathlib

import setuptools

README = (pathlib.Path(__file__).parent / "README.md").read_text()

setuptools.setup(
    name="nvdlfw_inspect",
    author="NVIDIA Corporation",
    description="Facilitates debugging convergence issues and testing new algorithms/recipes for training LLMs using Nvidia libraries.",
    long_description=README,
    long_description_content_type="text/markdown",
    version="0.1.0",
    url="https://github.com/NVIDIA/nvidia-dlfw-inspect",
    license="Apache2",
    packages=["nvdlfw_inspect"],
    package_dir={"nvdlfw_inspect": "nvdlfw_inspect"},
    package_data={
        "nvdlfw_inspect": ["debug_features/*"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pyyaml>=6.0.0",  # Required for config file parsing
        # Note: torch and numpy are expected to be pre-installed in most environments
        # (e.g., NVIDIA PyTorch containers). We specify them here with flexible
        # version constraints to ensure compatibility without forcing reinstallation.
        "torch>=2.4.0",
        "numpy==1.26.4",
    ],
    extras_require={
        "dev": ["pre-commit==4.1.0", "pytest==8.1.1", "ruff==0.9.3"],
    },
)
