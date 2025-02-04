FROM nvcr.io/nvidia/pytorch:24.12-py3

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=off
ENV PIP_DISABLE_PIP_VERSION_CHECK=on

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl jq rsync sshpass uuid-runtime jq vim wget make -y > /dev/null \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools pre-commit pytest

WORKDIR /workspace

COPY . /workspace/nvdlfw_inspect
RUN cd nvdlfw_inspect && make install