"""Model evaluation script.

Usage:
    python evaluate.py --cfg job                     # Show configuration
    python evaluate.py                               # Run evaluation
    python evaluate.py checkpoint_dir=./checkpoints/ckpts/ inference_config=standard
"""

import logging
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from agent_tunix.data import prepare_datasets
from agent_tunix.evaluate import create_sampler, evaluate_with_config
from agent_tunix.models import setup_models, load_trained_checkpoint

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def evaluate(cfg: DictConfig) -> None:
    """Evaluate a trained GRPO model.

    Args:
        cfg: Configuration dictionary
    """
    # Load environment variables from .env file if it exists
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(env_file)

    # Log configuration
    log.info("=" * 60)
    log.info("Model Evaluation Configuration")
    log.info("=" * 60)
    log.info(f"\n{OmegaConf.to_yaml(cfg)}")
    log.info("=" * 60)

    # Convert checkpoint paths to absolute paths for Orbax compatibility
    if "checkpoint_dir" in cfg:
        cfg.checkpoint_dir = os.path.abspath(cfg.checkpoint_dir)
    if "intermediate_checkpoint_dir" in cfg.training:
        cfg.training.intermediate_checkpoint_dir = os.path.abspath(cfg.training.intermediate_checkpoint_dir)

    # Get evaluation-specific config
    checkpoint_dir = cfg.get("checkpoint_dir", "./checkpoints/ckpts/")
    inference_config = cfg.get("inference_config", "greedy")
    step = cfg.get("step", None)
    num_passes = cfg.get("num_passes", 1)
    verbose = cfg.get("verbose", True)

    log.info("Setting up models...")
    policy_model, ref_model, mesh, model_config_obj, tokenizer = setup_models(
        cfg.model,
        cfg.training.intermediate_checkpoint_dir,
    )

    log.info(f"Loading checkpoint from {checkpoint_dir}...")
    load_trained_checkpoint(policy_model, checkpoint_dir, step)

    log.info("Preparing test dataset...")
    _, _, test_dataset = prepare_datasets(
        train_data_dir=cfg.training.train_data_dir,
        test_data_dir=cfg.training.test_data_dir,
        source=cfg.training.data_source,
        micro_batch_size=cfg.training.micro_batch_size,
        num_test_batches=cfg.training.num_test_batches,
    )

    log.info("Creating sampler...")
    sampler = create_sampler(
        policy_model,
        tokenizer,
        model_config_obj,
        cfg.generation.max_prompt_length,
        cfg.generation.max_generation_steps,
    )

    log.info(f"Evaluating with {inference_config} config (num_passes={num_passes})...")
    results = evaluate_with_config(
        test_dataset,
        sampler,
        config_name=inference_config,
        num_passes=num_passes,
        verbose=verbose,
    )

    log.info("\n" + "=" * 60)
    log.info("Evaluation Results")
    log.info("=" * 60)
    log.info(f"Correct: {results['correct']}/{results['total']}")
    log.info(f"Accuracy: {results['accuracy']:.2f}%")
    log.info(f"Partial Accuracy: {results['partial_accuracy']:.2f}%")
    log.info(f"Format Accuracy: {results['format_accuracy']:.2f}%")
    log.info("=" * 60)

    return results


if __name__ == "__main__":
    evaluate()
