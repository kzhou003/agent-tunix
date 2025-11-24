"""Training module for GRPO with Gemma3 using Hydra configuration."""

import logging
import os
from pathlib import Path

import hydra
import optax
from omegaconf import DictConfig, OmegaConf
from orbax import checkpoint as ocp

from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.grpo.grpo_learner import GRPOConfig, GRPOLearner
from tunix.rl.rollout import base_rollout
from tunix.sft import metrics_logger

from .data import prepare_datasets
from .models import setup_models
from .rewards import get_reward_functions

log = logging.getLogger(__name__)


def create_optimizer(cfg: DictConfig, max_steps: int):
    """Create optimizer with warmup and cosine decay schedule.

    Args:
        cfg: Configuration dictionary with optimizer settings
        max_steps: Total training steps for scheduler

    Returns:
        Optax optimizer chain
    """
    warmup_steps = int(cfg.optimizer.warmup_ratio * max_steps)

    optimizer = optax.adamw(
        learning_rate=optax.schedules.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=cfg.optimizer.learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=max_steps,
            end_value=0.0,
        ),
        b1=cfg.optimizer.beta1,
        b2=cfg.optimizer.beta2,
        weight_decay=cfg.optimizer.weight_decay,
    )

    if cfg.optimizer.max_grad_norm is not None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(max_norm=cfg.optimizer.max_grad_norm),
            optimizer,
        )

    return optimizer


def create_cluster_config(
    cfg: DictConfig,
    mesh,
    optimizer,
    max_steps: int,
):
    """Create RL cluster configuration.

    Args:
        cfg: Configuration dictionary
        mesh: JAX mesh for distributed training
        optimizer: Optax optimizer
        max_steps: Total training steps

    Returns:
        RLCluster configuration
    """
    # Checkpointing options
    checkpointing_options = ocp.CheckpointManagerOptions(
        save_interval_steps=cfg.training.save_interval_steps,
        max_to_keep=cfg.training.max_checkpoints_to_keep,
    )

    # Metrics logging options
    metrics_logging_options = metrics_logger.MetricsLoggerOptions(
        log_dir=cfg.training.log_dir,
        flush_every_n_steps=cfg.training.flush_every_n_steps,
    )

    # Training config
    training_config = rl_cluster_lib.RLTrainingConfig(
        actor_optimizer=optimizer,
        eval_every_n_steps=cfg.training.eval_every_n_steps,
        max_steps=max_steps,
        mini_batch_size=cfg.training.micro_batch_size,
        train_micro_batch_size=cfg.training.micro_batch_size,
        metrics_logging_options=metrics_logging_options,
        checkpoint_root_directory=cfg.training.checkpoint_dir,
        checkpointing_options=checkpointing_options,
    )

    # Rollout config
    rollout_config = base_rollout.RolloutConfig(
        max_tokens_to_generate=cfg.generation.max_generation_steps,
        max_prompt_length=cfg.generation.max_prompt_length,
        kv_cache_size=(
            cfg.generation.max_prompt_length
            + cfg.generation.max_generation_steps
            + 256
        ),
        temperature=cfg.generation.temperature,
        top_p=cfg.generation.top_p,
        top_k=cfg.generation.top_k,
        eos_tokens=cfg.generation.eos_tokens,
    )

    # Cluster config
    cluster_config = rl_cluster_lib.ClusterConfig(
        role_to_mesh={
            rl_cluster_lib.Role.ACTOR: mesh,
            rl_cluster_lib.Role.REFERENCE: mesh,
            rl_cluster_lib.Role.ROLLOUT: mesh,
        },
        rollout_engine='vanilla',
        offload_to_cpu=False,
        training_config=training_config,
        rollout_config=rollout_config,
    )

    return cluster_config


def calculate_max_steps(cfg: DictConfig) -> int:
    """Calculate maximum training steps.

    Args:
        cfg: Configuration dictionary

    Returns:
        Maximum number of training steps
    """
    return int(
        cfg.training.num_batches
        * cfg.grpo.num_iterations
        * cfg.training.train_fraction
        * cfg.training.num_epochs
    )


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def train(cfg: DictConfig) -> None:
    """Run GRPO training with Hydra configuration.

    Args:
        cfg: Hydra configuration dictionary
    """
    # Load environment variables from .env file if it exists
    env_file = Path(__file__).parent.parent.parent / ".env"
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(env_file)

    # Log configuration
    log.info("=" * 60)
    log.info("GRPO Training Configuration")
    log.info("=" * 60)
    log.info(f"\n{OmegaConf.to_yaml(cfg)}")
    log.info("=" * 60)

    # Convert checkpoint paths to absolute paths for Orbax compatibility
    cfg.training.checkpoint_dir = os.path.abspath(cfg.training.checkpoint_dir)
    cfg.training.intermediate_checkpoint_dir = os.path.abspath(cfg.training.intermediate_checkpoint_dir)

    # Set random seed
    import jax
    import numpy as np

    np.random.seed(cfg.seed)
    jax.config.update("jax_default_prng_impl", "threefry2x32")

    # Calculate max steps
    max_steps = calculate_max_steps(cfg)
    log.info(f"Max steps: {max_steps}")

    # Setup models
    log.info("Setting up models...")
    policy_model, ref_model, mesh, model_config, tokenizer = setup_models(
        cfg.model,
        cfg.training.intermediate_checkpoint_dir,
    )

    # Prepare datasets
    log.info("Preparing datasets...")
    train_dataset, val_dataset, test_dataset = prepare_datasets(
        train_data_dir=cfg.training.train_data_dir,
        test_data_dir=cfg.training.test_data_dir,
        source=cfg.training.data_source,
        micro_batch_size=cfg.training.micro_batch_size,
        num_batches=cfg.training.num_batches,
        num_test_batches=cfg.training.num_test_batches,
        train_fraction=cfg.training.train_fraction,
        num_epochs=cfg.training.num_epochs,
    )

    log.info(f"Dataset sizes: train={len(train_dataset)}, test={len(test_dataset)}")

    # Create optimizer
    log.info("Creating optimizer...")
    optimizer = create_optimizer(cfg, max_steps)

    # Create cluster config
    log.info("Creating cluster config...")
    cluster_config = create_cluster_config(cfg, mesh, optimizer, max_steps)

    # GRPO config
    grpo_config = GRPOConfig(
        num_generations=cfg.grpo.num_generations,
        num_iterations=cfg.grpo.num_iterations,
        beta=cfg.grpo.beta,
        epsilon=cfg.grpo.epsilon,
    )

    # Setup RL cluster
    log.info("Setting up RL cluster...")
    rl_cluster = rl_cluster_lib.RLCluster(
        actor=policy_model,
        reference=ref_model,
        tokenizer=tokenizer,
        cluster_config=cluster_config,
    )

    # Create GRPO trainer
    log.info("Creating GRPO trainer...")
    grpo_trainer = GRPOLearner(
        rl_cluster=rl_cluster,
        reward_fns=get_reward_functions(),
        algo_config=grpo_config,
    )

    # Run training
    log.info(f"Starting training for {max_steps} steps...")
    with mesh:
        grpo_trainer.train(train_dataset)

    log.info("Training complete!")


if __name__ == "__main__":
    train()
