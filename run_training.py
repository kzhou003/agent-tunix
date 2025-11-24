"""Run GRPO training with HuggingFace and Wandb enabled.

This script is a convenience wrapper that runs training using Hydra configuration.
It uses a curated default configuration optimized for single GPU training.

For more control over training parameters, use command-line overrides:
  python run_training.py model=gemma3_1b optimizer.learning_rate=1e-5
  python run_training.py --multirun model=gemma3_270m,gemma3_1b

For quick testing (10 steps, reduced config):
  python run_training.py +experiment=quick_test

To see the resolved configuration:
  python run_training.py --cfg job

For full Hydra help:
  python run_training.py --help
"""

import sys
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from agent_tunix.train import train


if __name__ == "__main__":
    # Run Hydra training directly
    # Use command-line arguments to override defaults:
    # python run_training.py model=gemma3_1b training.num_batches=50
    train()
