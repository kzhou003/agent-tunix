# Makefile for agent-tunix
# GRPO Training for Gemma3 using Google Tunix

.PHONY: help install install-dev build clean update train evaluate upload release test lint format

# Default target
.DEFAULT_GOAL := help

# Load environment variables from .env file
ifneq (,$(wildcard ./.env))
    include .env
    export
endif

# Variables
PYTHON := python
UV := uv
VERSION := $(shell grep 'version = ' pyproject.toml | head -1 | cut -d'"' -f2)
PACKAGE_NAME := agent-tunix
WHEEL_DIR := dist
HF_REPO ?= your-username/agent-tunix-gemma3-270m
GITHUB_REPO ?= your-username/agent-tunix

# Python runner that handles venv isolation from conda/system packages
PYTHON_RUN := env -i PATH="$(PATH)" HOME="$(HOME)" .venv/bin/python

# Colors for help
BLUE := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RESET := \033[0m

##@ General

help: ## Display this help message
	@echo "$(GREEN)agent-tunix$(RESET) - GRPO Training for Gemma3 using Tunix"
	@echo ""
	@echo "$(YELLOW)Usage:$(RESET)"
	@echo "  make $(BLUE)<target>$(RESET)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { printf "  $(BLUE)%-15s$(RESET) %s\n", $$1, $$2 } /^##@/ { printf "\n$(YELLOW)%s$(RESET)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Installation

install: ## Install package and dependencies
	$(UV) pip install -e .
	@echo "$(GREEN)Installation complete!$(RESET)"

install-dev: ## Install package with development dependencies
	$(UV) pip install -e ".[dev]"
	@echo "$(GREEN)Development installation complete!$(RESET)"

update: ## Update all dependencies to latest versions
	$(UV) pip install --upgrade -e .
	@echo "$(GREEN)Dependencies updated!$(RESET)"

##@ Development

lint: ## Run linting checks
	@if [ -f ".venv/bin/python" ]; then \
		$(PYTHON_RUN) -m ruff check src/ && $(PYTHON_RUN) -m mypy src/; \
	else \
		$(PYTHON) -m ruff check src/ && $(PYTHON) -m mypy src/; \
	fi

format: ## Format code with black and ruff
	@if [ -f ".venv/bin/python" ]; then \
		$(PYTHON_RUN) -m black src/ && $(PYTHON_RUN) -m ruff check --fix src/; \
	else \
		$(PYTHON) -m black src/ && $(PYTHON) -m ruff check --fix src/; \
	fi

test: ## Run tests
	@if [ -f ".venv/bin/python" ]; then \
		$(PYTHON_RUN) -m pytest tests/ -v; \
	else \
		$(PYTHON) -m pytest tests/ -v; \
	fi

clean: ## Clean build artifacts and caches
	rm -rf $(WHEEL_DIR)/
	rm -rf build/
	rm -rf *.egg-info
	rm -rf src/*.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)Cleaned build artifacts!$(RESET)"

clean-checkpoints: ## Clean all checkpoints and data cache
	rm -rf checkpoints/ckpts/*
	rm -rf checkpoints/intermediate/*
	rm -rf checkpoints/tensorboard/*
	rm -rf data/train/*
	rm -rf data/test/*
	rm -rf wandb/
	@echo "$(GREEN)Cleaned checkpoints and data cache!$(RESET)"

##@ Build & Release

build: clean ## Build wheel and source distribution
	$(UV) pip install build
	@if [ -f ".venv/bin/python" ]; then \
		$(PYTHON_RUN) -m build; \
	else \
		$(PYTHON) -m build; \
	fi
	@echo "$(GREEN)Build complete! Wheels in $(WHEEL_DIR)/$(RESET)"
	@ls -la $(WHEEL_DIR)/

release: build ## Release package to GitHub releases
	@echo "$(YELLOW)Creating GitHub release v$(VERSION)...$(RESET)"
	@if [ -z "$(GITHUB_TOKEN)" ]; then \
		echo "$(YELLOW)Warning: GITHUB_TOKEN not set. Using gh CLI authentication.$(RESET)"; \
	fi
	gh release create v$(VERSION) \
		--title "Release v$(VERSION)" \
		--notes "Release v$(VERSION) of $(PACKAGE_NAME)" \
		$(WHEEL_DIR)/*.whl \
		$(WHEEL_DIR)/*.tar.gz
	@echo "$(GREEN)Released v$(VERSION) to GitHub!$(RESET)"

publish-pypi: build ## Publish package to PyPI
	$(UV) pip install twine
	$(PYTHON) -m twine upload $(WHEEL_DIR)/*
	@echo "$(GREEN)Published to PyPI!$(RESET)"

##@ Training

train: ## Run GRPO training with default configuration
	@echo "$(YELLOW)Starting GRPO training...$(RESET)"
	@if [ -f ".venv/bin/python" ]; then \
		$(PYTHON_RUN) run_training.py; \
	else \
		$(PYTHON) run_training.py; \
	fi

train-quick: ## Run quick training test (10 steps)
	@echo "$(YELLOW)Running quick training test...$(RESET)"
	@if [ -f ".venv/bin/python" ]; then \
		$(PYTHON_RUN) run_training.py +experiment=quick_test; \
	else \
		$(PYTHON) run_training.py +experiment=quick_test; \
	fi

train-show-config: ## Show training configuration
	@echo "$(YELLOW)Training configuration:$(RESET)"
	@if [ -f ".venv/bin/python" ]; then \
		$(PYTHON_RUN) run_training.py --cfg job; \
	else \
		$(PYTHON) run_training.py --cfg job; \
	fi

train-show-defaults: ## Show configuration defaults tree
	@echo "$(YELLOW)Configuration defaults tree:$(RESET)"
	@if [ -f ".venv/bin/python" ]; then \
		$(PYTHON_RUN) run_training.py --info defaults-tree; \
	else \
		$(PYTHON) run_training.py --info defaults-tree; \
	fi

train-sweep: ## Run hyperparameter sweep (models example)
	@echo "$(YELLOW)Running hyperparameter sweep...$(RESET)"
	@if [ -f ".venv/bin/python" ]; then \
		$(PYTHON_RUN) run_training.py --multirun model=gemma3_270m,gemma3_1b 2>&1 | head -100; \
	else \
		$(PYTHON) run_training.py --multirun model=gemma3_270m,gemma3_1b 2>&1 | head -100; \
	fi

##@ Evaluation

evaluate: ## Evaluate trained model
	@echo "$(YELLOW)Running evaluation...$(RESET)"
	@if [ -f ".venv/bin/python" ]; then \
		$(PYTHON_RUN) evaluate.py; \
	else \
		$(PYTHON) evaluate.py; \
	fi

evaluate-show-config: ## Show evaluation configuration
	@echo "$(YELLOW)Evaluation configuration:$(RESET)"
	@if [ -f ".venv/bin/python" ]; then \
		$(PYTHON_RUN) evaluate.py --cfg job; \
	else \
		$(PYTHON) evaluate.py --cfg job; \
	fi

##@ Utilities

check-gpu: ## Check GPU availability and memory
	@if [ -f ".venv/bin/python" ]; then \
		$(PYTHON_RUN) -m agent_tunix.utils check-gpu; \
	else \
		$(PYTHON) -m agent_tunix.utils check-gpu; \
	fi

show-config: ## Show default training configuration
	@if [ -f ".venv/bin/python" ]; then \
		$(PYTHON_RUN) -m agent_tunix.utils show-config; \
	else \
		$(PYTHON) -m agent_tunix.utils show-config; \
	fi

tensorboard: ## Launch TensorBoard to view training logs
	tensorboard --logdir=./checkpoints/tensorboard/ --port=6006

version: ## Show package version
	@echo "$(PACKAGE_NAME) v$(VERSION)"
