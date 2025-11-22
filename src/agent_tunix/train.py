"""Training module for GRPO with Gemma3."""

import optax
from orbax import checkpoint as ocp

from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.grpo.grpo_learner import GRPOConfig, GRPOLearner
from tunix.rl.rollout import base_rollout
from tunix.sft import metrics_logger

from .config import GRPOTrainingConfig, get_default_config
from .data import prepare_datasets
from .models import setup_models
from .rewards import get_reward_functions


def create_optimizer(config: GRPOTrainingConfig):
    """Create optimizer with warmup and cosine decay schedule."""
    optimizer = optax.adamw(
        learning_rate=optax.schedules.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=config.optimizer.learning_rate,
            warmup_steps=config.warmup_steps,
            decay_steps=config.max_steps,
            end_value=0.0,
        ),
        b1=config.optimizer.beta1,
        b2=config.optimizer.beta2,
        weight_decay=config.optimizer.weight_decay,
    )

    if config.optimizer.max_grad_norm is not None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(max_norm=config.optimizer.max_grad_norm),
            optimizer,
        )

    return optimizer


def create_cluster_config(
    config: GRPOTrainingConfig,
    mesh,
    optimizer,
):
    """Create RL cluster configuration."""
    # Checkpointing options
    checkpointing_options = ocp.CheckpointManagerOptions(
        save_interval_steps=config.training.save_interval_steps,
        max_to_keep=config.training.max_checkpoints_to_keep,
    )

    # Metrics logging options
    metrics_logging_options = metrics_logger.MetricsLoggerOptions(
        log_dir=config.training.log_dir,
        flush_every_n_steps=config.training.flush_every_n_steps,
    )

    # Training config
    training_config = rl_cluster_lib.RLTrainingConfig(
        actor_optimizer=optimizer,
        eval_every_n_steps=config.training.eval_every_n_steps,
        max_steps=config.max_steps,
        mini_batch_size=config.training.micro_batch_size,
        train_micro_batch_size=config.training.micro_batch_size,
        metrics_logging_options=metrics_logging_options,
        checkpoint_root_directory=config.training.checkpoint_dir,
        checkpointing_options=checkpointing_options,
    )

    # Rollout config
    rollout_config = base_rollout.RolloutConfig(
        max_tokens_to_generate=config.generation.max_generation_steps,
        max_prompt_length=config.generation.max_prompt_length,
        kv_cache_size=(
            config.generation.max_prompt_length
            + config.generation.max_generation_steps
            + 256
        ),
        temperature=config.generation.temperature,
        top_p=config.generation.top_p,
        top_k=config.generation.top_k,
        eos_tokens=config.generation.eos_tokens,
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


def train(config: GRPOTrainingConfig | None = None):
    """Run GRPO training.

    Args:
        config: Training configuration. Uses default if None.
    """
    if config is None:
        config = get_default_config()

    print("Setting up models...")
    policy_model, ref_model, mesh, model_config, tokenizer = setup_models(
        config.model,
        config.training.intermediate_checkpoint_dir,
    )

    print("Preparing datasets...")
    train_dataset, val_dataset, test_dataset = prepare_datasets(
        train_data_dir=config.training.train_data_dir,
        test_data_dir=config.training.test_data_dir,
        source=config.training.data_source,
        micro_batch_size=config.training.micro_batch_size,
        num_batches=config.training.num_batches,
        num_test_batches=config.training.num_test_batches,
        train_fraction=config.training.train_fraction,
        num_epochs=config.training.num_epochs,
    )

    print(f"Dataset sizes: train={len(train_dataset)}, test={len(test_dataset)}")

    print("Creating optimizer...")
    optimizer = create_optimizer(config)

    print("Creating cluster config...")
    cluster_config = create_cluster_config(config, mesh, optimizer)

    # GRPO config
    grpo_config = GRPOConfig(
        num_generations=config.grpo.num_generations,
        num_iterations=config.grpo.num_iterations,
        beta=config.grpo.beta,
        epsilon=config.grpo.epsilon,
    )

    print("Setting up RL cluster...")
    rl_cluster = rl_cluster_lib.RLCluster(
        actor=policy_model,
        reference=ref_model,
        tokenizer=tokenizer,
        cluster_config=cluster_config,
    )

    print("Creating GRPO trainer...")
    grpo_trainer = GRPOLearner(
        rl_cluster=rl_cluster,
        reward_fns=get_reward_functions(),
        algo_config=grpo_config,
    )

    print(f"Starting training for {config.max_steps} steps...")
    with mesh:
        grpo_trainer.train(train_dataset)

    print("Training complete!")
    return policy_model, ref_model, mesh, model_config, tokenizer


if __name__ == "__main__":
    train()
