#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
cd "$PROJECT_ROOT"

VENV_DIR="build_wheels_venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    python3 -m venv $VENV_DIR
fi

source $VENV_DIR/bin/activate

pip install --upgrade pip setuptools wheel build

python3 -m build --wheel

deactivate

echo "Cleaning up: Removing virtual environment at $VENV_DIR..."
rm -rf "$VENV_DIR"

echo "Wheel built successfully! Check the dist/ directory."