"""Configuration for GRPO training with Gemma3-270m."""

import os
from dataclasses import dataclass, field
from typing import Optional


def _abspath(path: str) -> str:
    """Convert relative path to absolute path."""
    return os.path.abspath(os.path.expanduser(path))


@dataclass
class ModelConfig:
    """Model configuration for Gemma3-270m."""

    model_family: str = "gemma3"
    model_size: str = "270m"

    # LoRA configuration
    lora_rank: int = 32
    lora_alpha: float = 32.0
    lora_module_path: str = (
        ".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|"
        ".*attn_vec_einsum"
    )

    # Sharding configuration for TPU
    mesh_shape: tuple = ((1, 4), ("fsdp", "tp"))


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    max_prompt_length: int = 256
    max_generation_steps: int = 512
    temperature: float = 0.9
    top_p: float = 1.0
    top_k: int = 50
    eos_tokens: list = field(default_factory=lambda: [1, 106])


@dataclass
class GRPOConfig:
    """GRPO algorithm configuration."""

    # Number of response generations per prompt (G in paper)
    num_generations: int = 4
    # Number of iterations per batch (mu in paper)
    num_iterations: int = 1
    # KL divergence penalty coefficient (beta)
    beta: float = 0.08
    # Epsilon for clipping
    epsilon: float = 0.2


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""

    learning_rate: float = 3e-6
    beta1: float = 0.9
    beta2: float = 0.99
    weight_decay: float = 0.1
    warmup_ratio: float = 0.1
    max_grad_norm: float = 0.1


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Data
    train_data_dir: str = "./data/train"
    test_data_dir: str = "./data/test"
    data_source: str = "huggingface"  # tfds, kaggle, or huggingface
    train_fraction: float = 1.0

    # Batch sizes
    micro_batch_size: int = 4
    num_batches: int = 3738
    num_test_batches: int = 100
    num_epochs: int = 1

    # Evaluation
    eval_every_n_steps: int = 10

    # Checkpointing
    checkpoint_dir: str = "./checkpoints/ckpts/"
    intermediate_checkpoint_dir: str = "./checkpoints/intermediate/"
    save_interval_steps: int = 500
    max_checkpoints_to_keep: int = 4

    # Logging
    log_dir: str = "./checkpoints/tensorboard/"
    flush_every_n_steps: int = 20
    wandb_project: Optional[str] = "tunix-grpo"

    def __post_init__(self):
        """Convert relative paths to absolute paths."""
        self.train_data_dir = _abspath(self.train_data_dir)
        self.test_data_dir = _abspath(self.test_data_dir)
        self.checkpoint_dir = _abspath(self.checkpoint_dir)
        self.intermediate_checkpoint_dir = _abspath(self.intermediate_checkpoint_dir)
        self.log_dir = _abspath(self.log_dir)


@dataclass
class GRPOTrainingConfig:
    """Complete configuration for GRPO training."""

    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @property
    def max_steps(self) -> int:
        """Calculate maximum training steps."""
        return int(
            self.training.num_batches
            * self.grpo.num_iterations
            * self.training.train_fraction
            * self.training.num_epochs
        )

    @property
    def warmup_steps(self) -> int:
        """Calculate warmup steps."""
        return int(self.optimizer.warmup_ratio * self.max_steps)


# Inference generation configs
INFERENCE_CONFIGS = {
    "greedy": {"temperature": 1e-4, "top_k": 1, "top_p": 1.0},
    "standard": {"temperature": 0.7, "top_k": 50, "top_p": 0.95},
    "liberal": {"temperature": 0.85, "top_k": 2000, "top_p": 1.0},
}


def get_default_config() -> GRPOTrainingConfig:
    """Get default configuration for Gemma3-270m GRPO training."""
    return GRPOTrainingConfig()
