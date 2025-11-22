"""Run GRPO training with HuggingFace and Wandb enabled."""

import os
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from agent_tunix.config import (
    GRPOTrainingConfig,
    ModelConfig,
    TrainingConfig,
    GRPOConfig,
    GenerationConfig,
    OptimizerConfig,
)
from agent_tunix.train import train


def main():
    config = GRPOTrainingConfig(
        model=ModelConfig(
            model_size="270m",
            lora_rank=16,  # Reduced for memory
            lora_alpha=16.0,
            mesh_shape=((1, 1), ("fsdp", "tp")),  # Single GPU
        ),
        generation=GenerationConfig(
            max_prompt_length=128,  # Reduced for memory
            max_generation_steps=128,  # Reduced for memory
        ),
        grpo=GRPOConfig(
            num_generations=2,  # Reduced for memory
            num_iterations=1,
            beta=0.08,
            epsilon=0.2,
        ),
        optimizer=OptimizerConfig(
            learning_rate=3e-6,
            warmup_ratio=0.1,
            max_grad_norm=0.1,
        ),
        training=TrainingConfig(
            train_data_dir="./data/train",
            test_data_dir="./data/test",
            data_source="huggingface",
            num_batches=100,  # Adjust for longer training
            num_test_batches=25,
            micro_batch_size=1,  # Reduced for memory
            num_epochs=1,
            checkpoint_dir="./checkpoints/ckpts/",
            intermediate_checkpoint_dir="./checkpoints/intermediate/",
            save_interval_steps=50,
            wandb_project="tunix-grpo-270m",
        ),
    )

    print("=" * 60)
    print("GRPO Training Configuration")
    print("=" * 60)
    print(f"Model: Gemma3-{config.model.model_size}")
    print(f"LoRA rank: {config.model.lora_rank}")
    print(f"Batch size: {config.training.micro_batch_size}")
    print(f"Num batches: {config.training.num_batches}")
    print(f"Max steps: {config.max_steps}")
    print(f"Learning rate: {config.optimizer.learning_rate}")
    print(f"Wandb project: {config.training.wandb_project}")
    print("=" * 60)

    train(config)


if __name__ == "__main__":
    main()
