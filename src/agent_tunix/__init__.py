"""Agent-Tunix: GRPO Training for Gemma3-270m using Google Tunix.

Configuration is managed via Hydra. Use:
    python run_training.py              # Train with defaults
    python evaluate_hydra.py            # Evaluate model

For more details, see HYDRA_USAGE.md
"""

__version__ = "0.1.0"

# Core functions - still available but primarily use Hydra
from .train import train
from .evaluate import evaluate, create_sampler, evaluate_with_config

__all__ = [
    "train",
    "evaluate",
    "create_sampler",
    "evaluate_with_config",
]
