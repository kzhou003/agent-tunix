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

- Python 3.11+
- NVIDIA GPU with 11GB+ VRAM (for single GPU training)
- Kaggle account (for model weights access)

### Install from source

```bash
git clone https://github.com/yourusername/agent-tunix.git
cd agent-tunix
uv pip install -e .
```

## Quick Start

### Training

```bash
cd agent-tunix
python run_training.py
```

### Programmatic Usage

```python
from agent_tunix import GRPOTrainingConfig, train
from agent_tunix.config import ModelConfig, TrainingConfig

config = GRPOTrainingConfig(
    model=ModelConfig(
        model_size="270m",
        lora_rank=16,
        mesh_shape=((1, 1), ("fsdp", "tp")),  # Single GPU
    ),
    training=TrainingConfig(
        num_batches=100,
        micro_batch_size=1,
    ),
)

train(config)
```

## Configuration

### Model Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_size` | `"270m"` | Model size (`"270m"`, `"1b"`, `"4b"`, `"12b"`, `"27b"`) |
| `lora_rank` | `32` | LoRA rank for adaptation |
| `lora_alpha` | `32.0` | LoRA scaling factor |

### GRPO Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_generations` | `4` | Number of responses per prompt (G in paper) |
| `num_iterations` | `1` | Iterations per batch (mu in paper) |
| `beta` | `0.08` | KL divergence penalty coefficient |
| `epsilon` | `0.2` | Clipping epsilon for stable updates |

### Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | `3e-6` | Peak learning rate |
| `warmup_ratio` | `0.1` | Warmup steps as fraction of total |
| `max_grad_norm` | `0.1` | Gradient clipping threshold |
| `micro_batch_size` | `4` | Batch size per device |

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
