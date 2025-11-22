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
	$(PYTHON) -m ruff check src/
	$(PYTHON) -m mypy src/

format: ## Format code with black and ruff
	$(PYTHON) -m black src/
	$(PYTHON) -m ruff check --fix src/

test: ## Run tests
	$(PYTHON) -m pytest tests/ -v

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
	$(PYTHON) -m build
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

train: ## Run GRPO training with default config
	@echo "$(YELLOW)Starting GRPO training...$(RESET)"
	$(PYTHON) run_training.py

train-quick: ## Run quick training test (10 steps)
	@echo "$(YELLOW)Running quick training test...$(RESET)"
	$(PYTHON) -c "\
import os; \
os.environ['HF_HUB_DISABLE_XET'] = '1'; \
os.environ['WANDB_MODE'] = 'disabled'; \
from agent_tunix.config import GRPOTrainingConfig, ModelConfig, TrainingConfig, GRPOConfig, GenerationConfig; \
from agent_tunix.train import train; \
config = GRPOTrainingConfig( \
    model=ModelConfig(model_size='270m', lora_rank=8, mesh_shape=((1,1), ('fsdp','tp'))), \
    generation=GenerationConfig(max_prompt_length=64, max_generation_steps=64), \
    grpo=GRPOConfig(num_generations=2), \
    training=TrainingConfig(num_batches=10, micro_batch_size=1), \
); \
train(config)"

##@ Evaluation

evaluate: ## Evaluate trained model
	@echo "$(YELLOW)Running evaluation...$(RESET)"
	$(PYTHON) -c "\
import os; \
os.environ['HF_HUB_DISABLE_XET'] = '1'; \
os.environ['WANDB_MODE'] = 'disabled'; \
from agent_tunix.config import GRPOTrainingConfig, ModelConfig; \
from agent_tunix.data import prepare_datasets; \
from agent_tunix.models import setup_models, load_trained_checkpoint; \
from agent_tunix.evaluate import create_sampler, evaluate; \
config = GRPOTrainingConfig(model=ModelConfig(model_size='270m', mesh_shape=((1,1), ('fsdp','tp')))); \
policy, ref, mesh, model_cfg, tokenizer = setup_models(config.model, './checkpoints/intermediate/'); \
load_trained_checkpoint(policy, './checkpoints/ckpts/'); \
_, _, test_ds = prepare_datasets('./data/train', './data/test', num_test_batches=25, micro_batch_size=1); \
sampler = create_sampler(policy, tokenizer, model_cfg, 256, 256); \
results = evaluate(test_ds, sampler, temperature=0.7); \
print(f'Accuracy: {results[\"accuracy\"]:.2f}%, Format: {results[\"format_accuracy\"]:.2f}%')"

##@ Model Upload

upload: ## Upload trained model to HuggingFace Hub
	@echo "$(YELLOW)Uploading model to HuggingFace Hub...$(RESET)"
	@if [ -z "$(HF_TOKEN)" ]; then \
		echo "$(YELLOW)Error: HF_TOKEN environment variable not set$(RESET)"; \
		echo "Set it with: export HF_TOKEN=your_token"; \
		exit 1; \
	fi
	$(PYTHON) -c "\
import os; \
from huggingface_hub import HfApi, create_repo; \
api = HfApi(); \
repo_id = '$(HF_REPO)'; \
try: \
    create_repo(repo_id, exist_ok=True, repo_type='model'); \
except Exception as e: \
    print(f'Repo exists or error: {e}'); \
api.upload_folder( \
    folder_path='./checkpoints/ckpts/', \
    repo_id=repo_id, \
    repo_type='model', \
    commit_message='Upload trained GRPO model' \
); \
print(f'Model uploaded to https://huggingface.co/{repo_id}')"
	@echo "$(GREEN)Model uploaded to HuggingFace!$(RESET)"

upload-config: ## Upload model config and training scripts to HuggingFace
	@echo "$(YELLOW)Uploading config to HuggingFace Hub...$(RESET)"
	$(PYTHON) -c "\
import os; \
from huggingface_hub import HfApi; \
api = HfApi(); \
repo_id = '$(HF_REPO)'; \
files = ['run_training.py', 'README.md', 'pyproject.toml']; \
for f in files: \
    if os.path.exists(f): \
        api.upload_file(path_or_fileobj=f, path_in_repo=f, repo_id=repo_id, repo_type='model'); \
        print(f'Uploaded {f}')"
	@echo "$(GREEN)Config uploaded to HuggingFace!$(RESET)"

##@ Utilities

check-gpu: ## Check GPU availability and memory
	$(PYTHON) -c "\
import jax; \
print(f'JAX devices: {jax.devices()}'); \
print(f'Backend: {jax.default_backend()}')" 2>/dev/null || echo "JAX not available"
	nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null || echo "nvidia-smi not available"

show-config: ## Show current training configuration
	$(PYTHON) -c "\
from agent_tunix.config import GRPOTrainingConfig; \
config = GRPOTrainingConfig(); \
print('=== Default Training Configuration ==='); \
print(f'Model size: {config.model.model_size}'); \
print(f'LoRA rank: {config.model.lora_rank}'); \
print(f'Max steps: {config.max_steps}'); \
print(f'Learning rate: {config.optimizer.learning_rate}'); \
print(f'Checkpoint dir: {config.training.checkpoint_dir}'); \
print(f'Data dir: {config.training.train_data_dir}')"

tensorboard: ## Launch TensorBoard to view training logs
	tensorboard --logdir=./checkpoints/tensorboard/ --port=6006

version: ## Show package version
	@echo "$(PACKAGE_NAME) v$(VERSION)"
