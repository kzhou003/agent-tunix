"""Model loading and LoRA utilities for Gemma3."""

import gc
import os
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
from orbax import checkpoint as ocp
import qwix

from .config import ModelConfig


def get_model_config_and_checkpoint(model_size: str = "270m"):
    """Get model configuration and checkpoint path based on model size.

    Args:
        model_size: Model size ("270m", "1b", "4b", "12b", "27b")

    Returns:
        Tuple of (model_config, checkpoint_path)
    """
    from tunix.models.gemma3 import model, params

    # Map of supported model sizes to their configs and checkpoints
    model_configs = {
        "270m": (model.ModelConfig.gemma3_270m, params.GEMMA3_270M_IT),
        "1b": (model.ModelConfig.gemma3_1b, params.GEMMA3_1B_IT),
        "4b": (model.ModelConfig.gemma3_4b, params.GEMMA3_4B_IT),
        "12b": (model.ModelConfig.gemma3_12b, params.GEMMA3_12B_IT),
        "27b": (model.ModelConfig.gemma3_27b, params.GEMMA3_27B_IT),
    }

    if model_size not in model_configs:
        raise ValueError(
            f"Unknown model size: {model_size}. "
            f"Available sizes: {list(model_configs.keys())}"
        )

    config_fn, ckpt_path = model_configs[model_size]
    config = config_fn()

    return config, ckpt_path


def save_intermediate_checkpoint(
    model_instance: nnx.Module,
    checkpoint_dir: str,
) -> None:
    """Save model checkpoint to intermediate directory.

    This is a workaround to re-save the pre-trained model checkpoint
    into a format compatible with Flax NNX.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpointer = ocp.StandardCheckpointer()
    _, state = nnx.split(model_instance)
    checkpointer.save(os.path.join(checkpoint_dir, "state"), state)
    checkpointer.wait_until_finished()


def load_reference_model(
    checkpoint_path: str,
    model_config: Any,
    original_ckpt_path: str,
    mesh_shape: tuple = ((1, 4), ("fsdp", "tp")),
):
    """Load the reference model from checkpoint.

    Args:
        checkpoint_path: Path to the saved intermediate checkpoint.
        model_config: Model configuration object.
        original_ckpt_path: Original model checkpoint path (for creating abstract model).
        mesh_shape: TPU mesh configuration.

    Returns:
        Tuple of (model, mesh, model_config).
    """
    from tunix.models.gemma3 import params, model as model_lib

    mesh = jax.make_mesh(*mesh_shape)

    abs_model: nnx.Module = nnx.eval_shape(
        lambda: params.create_model_from_checkpoint(original_ckpt_path, model_config)
    )

    abs_state = nnx.state(abs_model)
    abs_state = jax.tree.map(
        lambda a, s: jax.ShapeDtypeStruct(a.shape, jnp.bfloat16, sharding=s),
        abs_state,
        nnx.get_named_sharding(abs_state, mesh),
    )

    checkpointer = ocp.StandardCheckpointer()
    restored_params = checkpointer.restore(checkpoint_path, target=abs_state)

    graph_def, _ = nnx.split(abs_model)
    loaded_model = nnx.merge(graph_def, restored_params)

    return loaded_model, mesh, model_config


def apply_lora(
    base_model: nnx.Module,
    mesh: Any,
    rank: int = 32,
    alpha: float = 32.0,
    module_path: str = (
        ".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|"
        ".*attn_vec_einsum"
    ),
) -> nnx.Module:
    """Apply LoRA to the base model.

    Args:
        base_model: The base model to apply LoRA to.
        mesh: JAX mesh for sharding.
        rank: LoRA rank.
        alpha: LoRA alpha scaling factor.
        module_path: Regex pattern for modules to apply LoRA to.

    Returns:
        Model with LoRA layers applied.
    """
    lora_provider = qwix.LoraProvider(
        module_path=module_path,
        rank=rank,
        alpha=alpha,
    )

    model_input = base_model.get_model_input()
    lora_model = qwix.apply_lora_to_model(
        base_model, lora_provider, **model_input
    )

    with mesh:
        state = nnx.state(lora_model)
        pspecs = nnx.get_partition_spec(state)
        sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
        nnx.update(lora_model, sharded_state)

    return lora_model


def setup_models(
    config: ModelConfig,
    intermediate_checkpoint_dir: str,
) -> tuple:
    """Set up reference and policy models for GRPO training.

    Args:
        config: Model configuration.
        intermediate_checkpoint_dir: Directory for intermediate checkpoints.

    Returns:
        Tuple of (policy_model, reference_model, mesh, model_config, tokenizer).
    """
    from tunix.models.gemma3 import params, model as model_lib

    # Clean up any existing checkpoints
    import shutil
    if os.path.exists(intermediate_checkpoint_dir):
        shutil.rmtree(intermediate_checkpoint_dir)
    os.makedirs(intermediate_checkpoint_dir, exist_ok=True)

    # Get model config based on size
    model_config, original_ckpt_path = get_model_config_and_checkpoint(config.model_size)

    # Create and save intermediate checkpoint
    base_model = params.create_model_from_checkpoint(original_ckpt_path, model_config)
    tokenizer = params.create_tokenizer()

    save_intermediate_checkpoint(base_model, intermediate_checkpoint_dir)

    # Clean up intermediate model
    del base_model
    gc.collect()

    # Load reference model
    ref_model, mesh, _ = load_reference_model(
        checkpoint_path=os.path.join(intermediate_checkpoint_dir, "state"),
        model_config=model_config,
        original_ckpt_path=original_ckpt_path,
        mesh_shape=config.mesh_shape,
    )

    # Create policy model with LoRA
    policy_model = apply_lora(
        ref_model,
        mesh,
        rank=config.lora_rank,
        alpha=config.lora_alpha,
        module_path=config.lora_module_path,
    )

    return policy_model, ref_model, mesh, model_config, tokenizer


def load_trained_checkpoint(
    policy_model: nnx.Module,
    checkpoint_dir: str,
    step: int | None = None,
) -> None:
    """Load trained LoRA parameters into policy model.

    Args:
        policy_model: The policy model to update.
        checkpoint_dir: Root checkpoint directory.
        step: Specific step to load (None for latest).
    """
    import re

    actor_ckpt_dir = os.path.join(checkpoint_dir, "actor")

    if step is None:
        # Find latest checkpoint
        latest_step = -1
        if os.path.exists(actor_ckpt_dir):
            for item in os.listdir(actor_ckpt_dir):
                if os.path.isdir(os.path.join(actor_ckpt_dir, item)) and re.match(r'^\d+$', item):
                    s = int(item)
                    if s > latest_step:
                        latest_step = s

        if latest_step == -1:
            raise FileNotFoundError(f"No checkpoints found in {actor_ckpt_dir}")
        step = latest_step

    print(f"Loading checkpoint from step: {step}")

    trained_ckpt_path = os.path.join(actor_ckpt_dir, str(step), "model_params")

    abs_params = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
        nnx.state(policy_model, nnx.LoRAParam),
    )

    checkpointer = ocp.StandardCheckpointer()
    trained_lora_params = checkpointer.restore(trained_ckpt_path, target=abs_params)

    nnx.update(
        policy_model,
        jax.tree.map(
            lambda a, b: b,
            nnx.state(policy_model, nnx.LoRAParam),
            trained_lora_params,
        ),
    )
