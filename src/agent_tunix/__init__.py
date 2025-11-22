"""Agent-Tunix: GRPO Training for Gemma3-270m using Google Tunix."""

__version__ = "0.1.0"

from .config import GRPOTrainingConfig, ModelConfig
from .train import train
from .evaluate import evaluate

__all__ = [
    "GRPOTrainingConfig",
    "ModelConfig",
    "train",
    "evaluate",
]
