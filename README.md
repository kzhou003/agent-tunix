# Agent-Tunix

GRPO (Group Relative Policy Optimization) training for Gemma3-270m using Google Tunix.

## Overview

This package provides a complete pipeline for training language models using GRPO, a reinforcement learning algorithm designed to enhance the reasoning abilities of LLMs. GRPO is a variant of PPO that reduces memory usage by eliminating the need for a separate value function model.

### Key Features

- **GRPO Training**: Implements Group Relative Policy Optimization for improved reasoning
- **LoRA Fine-tuning**: Memory-efficient training using Low-Rank Adaptation
- **GSM8K Benchmark**: Train on grade school math word problems
- **Multiple Data Sources**: Support for HuggingFace, TFDS, and Kaggle datasets
- **Reward Functions**: Configurable reward functions for format and answer correctness
- **GPU/TPU Support**: Optimized for distributed training on GPU/TPU clusters

## Installation

### Prerequisites

#### Software Requirements
- Python 3.11+
- Kaggle account (for model weights access)

#### GPU Requirements
- NVIDIA GPU with 11GB+ VRAM (tested on RTX 2080 Ti with 11GB)
- CUDA Toolkit 11.5+
- NVIDIA Driver 470+

#### Detailed CUDA & Driver Requirements

This project requires compatible CUDA and driver versions to run JAX, TensorFlow, and other deep learning libraries.

**Minimum Specifications:**
- **NVIDIA Driver**: 470.x or higher
- **CUDA Toolkit**: 11.5 or higher
- **cuDNN**: 8.x or higher (automatically included with JAX/TensorFlow)

**Recommended Setup:**
- **NVIDIA Driver**: 580.x (latest stable)
- **CUDA Toolkit**: 12.x or 13.0
- **CUDA Compute Capability**: 7.0+ (RTX 20 series and newer, A100, H100, etc.)

**Verification:**

Check your driver version:
```bash
nvidia-smi
```

Check CUDA Toolkit version:
```bash
nvcc --version
```

Verify JAX can access GPU:
```bash
python -c "import jax; print(jax.devices())"
```

**Supported GPUs:**
- NVIDIA RTX 20 series (2060, 2070, 2080, 2080 Ti)
- NVIDIA RTX 30 series (3060, 3070, 3080, 3090)
- NVIDIA RTX 40 series (4090)
- NVIDIA A-series (A5000, A6000, A100)
- NVIDIA H-series (H100)

**Memory Requirements:**
- 270M model: 11GB+ VRAM (with LoRA and micro-batch size 1)
- 1B model: 14GB+ VRAM
- 4B model: 20GB+ VRAM
- 12B model: 40GB+ VRAM
- 27B model: 80GB+ VRAM

### Install from source

```bash
git clone https://github.com/yourusername/agent-tunix.git
cd agent-tunix
uv pip install -e .
```

**Note:** JAX with CUDA support will be installed as a dependency. If you encounter GPU detection issues, you may need to set environment variables:

```bash
export CUDA_HOME=/usr/local/cuda-13.0
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
```

Replace `cuda-13.0` with your installed CUDA version.

## Quick Start

All configuration is managed via YAML files in `conf/` with command-line overrides for flexible configuration.

### Training with Default Configuration

```bash
# Run training with default configuration
python run_training.py

# View the resolved configuration before running
python run_training.py --cfg job

# Show configuration defaults tree
python run_training.py --info defaults-tree
```

### Training with Custom Configuration

```bash
# Change model size
python run_training.py model=gemma3_1b

# Override multiple values
python run_training.py model=gemma3_1b optimizer.learning_rate=1e-5 training.num_batches=50

# Use a preset experiment
python run_training.py +experiment=quick_test
```

### Hyperparameter Sweeps

```bash
# Sweep over multiple models
python run_training.py --multirun model=gemma3_270m,gemma3_1b

# Sweep over learning rates
python run_training.py --multirun optimizer.learning_rate=1e-6,3e-6,1e-5
```

### Model Evaluation

```bash
# Evaluate with default configuration
python evaluate.py

# Evaluate with custom checkpoint
python evaluate.py checkpoint_dir=./checkpoints/ckpts/ inference_config=standard
```

### Programmatic Usage

```python
from agent_tunix import train

# Train using Hydra configuration
# This requires Hydra to be initialized (usually via command line)
# For programmatic use, directly call the training function:
train()
```

See [HYDRA_USAGE.md](HYDRA_USAGE.md) for complete Hydra documentation and advanced usage examples.

## Configuration

All configuration is managed through YAML files located in the `conf/` directory. Configuration is organized into logical groups:

### Configuration Groups

- **`model/`** - Model architecture configurations (gemma3_270m, gemma3_1b, etc.)
- **`optimizer/`** - Optimizer settings (adamw)
- **`scheduler/`** - Learning rate scheduler configurations
- **`grpo/`** - GRPO algorithm parameters
- **`generation/`** - Text generation settings
- **`training/`** - Training hyperparameters
- **`experiment/`** - Preset experiment configurations

### Common Parameters

#### Model Configuration
- `model_size`: Model size (270m, 1b, 4b, 12b, 27b)
- `lora_rank`: LoRA rank for parameter-efficient training
- `lora_alpha`: LoRA scaling factor

#### Optimizer Configuration
- `learning_rate`: Peak learning rate (default: 3e-6)
- `warmup_ratio`: Warmup as fraction of total steps (default: 0.1)
- `max_grad_norm`: Gradient clipping threshold (default: 0.1)

#### GRPO Configuration
- `num_generations`: Number of responses per prompt (default: 4)
- `num_iterations`: Iterations per batch (default: 1)
- `beta`: KL divergence penalty coefficient (default: 0.08)
- `epsilon`: PPO clipping epsilon (default: 0.2)

#### Training Configuration
- `micro_batch_size`: Batch size per device (default: 4)
- `num_batches`: Number of training batches (default: 3738)
- `num_epochs`: Number of training epochs (default: 1)

For complete configuration details and examples, see [HYDRA_USAGE.md](HYDRA_USAGE.md).

## Reward Functions

The package includes four reward functions:

1. **`match_format_exactly`**: Rewards exact format compliance (3 points)
2. **`match_format_approximately`**: Rewards partial format matching
3. **`check_answer`**: Rewards correct/partially correct answers
4. **`check_numbers`**: Extracts and validates numerical answers

## Project Structure

```
agent-tunix/
├── checkpoints/             # Model checkpoints (gitignored)
│   ├── ckpts/               # Training checkpoints
│   └── intermediate/        # Intermediate model state
├── data/                    # Dataset cache (gitignored)
│   ├── train/
│   └── test/
├── src/
│   └── agent_tunix/
│       ├── __init__.py      # Package exports
│       ├── cli.py           # Command-line interface
│       ├── config.py        # Configuration dataclasses
│       ├── data.py          # Data loading and preprocessing
│       ├── evaluate.py      # Evaluation utilities
│       ├── models.py        # Model loading and LoRA
│       ├── rewards.py       # Reward functions
│       └── train.py         # Training loop
├── run_training.py          # Main training script
├── pyproject.toml           # Package configuration
└── README.md                # This file
```

## References

- [GRPO Paper](https://arxiv.org/pdf/2402.03300) - Group Relative Policy Optimization
- [Google Tunix](https://github.com/google/tunix) - Training framework
- [GSM8K Dataset](https://huggingface.co/datasets/openai/gsm8k) - Math reasoning benchmark
- [Gemma Models](https://deepmind.google/models/gemma/) - Base model family

## License

Apache 2.0
