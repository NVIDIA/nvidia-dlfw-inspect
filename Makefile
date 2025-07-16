SHELL := /bin/bash

# Virtual environment paths
DEV_VENV = dev_venv
DEV_LOCAL_VENV = dev_local_venv

# Python and pip commands for each venv
DEV_PYTHON = $(DEV_VENV)/bin/python3
DEV_PIP = $(DEV_VENV)/bin/pip
DEV_LOCAL_PYTHON = $(DEV_LOCAL_VENV)/bin/python3
DEV_LOCAL_PIP = $(DEV_LOCAL_VENV)/bin/pip

DOCKER_IMAGE_NAME = nvdlfw_inspect

.PHONY: install
install: ## Install nvdlfw_inspect
	@pip install .
	@python3 -c "import torch; import numpy; import yaml" || { echo >&2 "Error: 'torch' and/or 'numpy', and/or 'yaml' are not installed in your environment. Please ensure they are pre-installed."; exit 1; }

.PHONY: docker-build
docker-build: ## Build Docker Images
	@echo "Building docker image: $(DOCKER_IMAGE_NAME)"
	@docker build . -t $(DOCKER_IMAGE_NAME)

.PHONY: docker-run
docker-run: ## Launch Docker Container
	@echo "Running docker container"
	@docker run -it --gpus all -v .:/workspace/nvdlfw_inspect $(DOCKER_IMAGE_NAME) bash

.PHONY: dev-local
dev-local: $(DEV_LOCAL_VENV) ## Create local dev enviroment for running the tool and pre-commit checks
	@echo "Installing dependencies dev-local environment..."
	$(DEV_LOCAL_PIP) install -e .[dev]
	$(DEV_LOCAL_PIP) install -r requirements-dev-local.txt
	@echo "To activate the dev-local environment, run: source $(DEV_LOCAL_VENV)/bin/activate"

$(DEV_LOCAL_VENV): requirements-dev-local.txt
	@echo "Creating virtual environment $(DEV_LOCAL_VENV)..."
	python3 -m venv $(DEV_LOCAL_VENV)
	$(DEV_LOCAL_PIP) install --upgrade pip
	@touch $(DEV_LOCAL_VENV)

.PHONY: test
test: ## Run tests
	@pytest -vvv -s

.PHONY: dev-env
dev-env: $(DEV_VENV) ## Create dev enviroment for pre-commit checks
	@echo "Installing dependencies for dev environment..."
	$(DEV_PIP) install -e .[dev]
	@$(DEV_PYTHON) -c "import torch; import numpy" || { \
	    echo >&2 "Warning: 'torch' and/or 'numpy' are not installed in your virtual environment and will not run the tests"; \
	}
	@echo "To activate the dev environment, run: source $(DEV_VENV)/bin/activate"

$(DEV_VENV):
	@echo "Creating virtual environment $(DEV_VENV)..."
	python3 -m venv $(DEV_VENV)
	$(DEV_PIP) install --upgrade pip
	@touch $(DEV_VENV)

.PHONY: check
check: ## Run pre-commit checks
	@pre-commit run -a

.PHONY: clean-build
clean-build: ## Remove .egg-info, build/ and dist/ directories
	@echo "Cleaning up build artifacts: .egg-info, build/, and dist/ directories"
	-rm -r build/ *.egg-info dist/ || echo "Warning: Failed to remove some build artifacts. Files may not exist or you may lack sufficient permissions."

.PHONY: clean-venv
clean-venv: ## Remove all the venvs
	@echo "Removing all virtual environments: $(DEV_VENV), $(DEV_LOCAL_VENV)"
	-rm -r $(DEV_VENV) $(DEV_LOCAL_VENV) || echo "Warning: Failed to remove some virtual environment files. Directories may not exist or you may lack sufficient permissions."

.PHONY: clean-docker
clean-docker: ## Remove the docker image
	@echo "Removing Docker image..."
	-docker rmi -f $(DOCKER_IMAGE_NAME) || echo "Docker image $(DOCKER_IMAGE_NAME) does not exist."
	@echo "Cleanup completed."

.PHONY: clean
clean: clean-venv clean-build clean-docker ## Remove all the artifacts: clean-build, clean-venv and clean-docker
	@echo "All cleanup tasks completed."

.PHONY: help
help: ## Display help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
