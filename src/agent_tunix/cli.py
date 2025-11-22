"""Command-line interface for agent-tunix."""

import argparse
import sys


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="GRPO Training for Gemma3 using Google Tunix"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model with GRPO")
    train_parser.add_argument(
        "--model-size",
        type=str,
        default="270m",
        choices=["270m", "1b", "4b", "12b", "27b"],
        help="Model size to train (default: 270m)",
    )
    train_parser.add_argument(
        "--data-source",
        type=str,
        default="huggingface",
        choices=["huggingface", "tfds", "kaggle"],
        help="Data source for GSM8K dataset (default: huggingface)",
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-6,
        help="Learning rate (default: 3e-6)",
    )
    train_parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1)",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Micro batch size (default: 4)",
    )
    train_parser.add_argument(
        "--lora-rank",
        type=int,
        default=32,
        help="LoRA rank (default: 32)",
    )
    train_parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="/tmp/content/ckpts/",
        help="Directory to save checkpoints",
    )
    train_parser.add_argument(
        "--wandb-project",
        type=str,
        default="tunix-grpo",
        help="Weights & Biases project name",
    )

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Directory containing checkpoints",
    )
    eval_parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Checkpoint step to evaluate (default: latest)",
    )
    eval_parser.add_argument(
        "--inference-config",
        type=str,
        default="greedy",
        choices=["greedy", "standard", "liberal"],
        help="Inference configuration (default: greedy)",
    )
    eval_parser.add_argument(
        "--model-size",
        type=str,
        default="270m",
        choices=["270m", "1b", "4b", "12b", "27b"],
        help="Model size (default: 270m)",
    )

    return parser.parse_args()


def cmd_train(args):
    """Run training command."""
    from .config import GRPOTrainingConfig, ModelConfig, OptimizerConfig, TrainingConfig
    from .train import train

    # Build config from CLI args
    model_config = ModelConfig(
        model_size=args.model_size,
        lora_rank=args.lora_rank,
    )

    optimizer_config = OptimizerConfig(
        learning_rate=args.learning_rate,
    )

    training_config = TrainingConfig(
        data_source=args.data_source,
        num_epochs=args.num_epochs,
        micro_batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        wandb_project=args.wandb_project,
    )

    config = GRPOTrainingConfig(
        model=model_config,
        optimizer=optimizer_config,
        training=training_config,
    )

    print("Starting GRPO training with config:")
    print(f"  Model size: {args.model_size}")
    print(f"  Data source: {args.data_source}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LoRA rank: {args.lora_rank}")

    train(config)


def cmd_evaluate(args):
    """Run evaluation command."""
    from .config import GRPOTrainingConfig, ModelConfig, INFERENCE_CONFIGS
    from .data import prepare_datasets
    from .models import setup_models, load_trained_checkpoint
    from .evaluate import create_sampler, evaluate_with_config

    model_config = ModelConfig(model_size=args.model_size)
    config = GRPOTrainingConfig(model=model_config)

    print("Setting up models...")
    policy_model, ref_model, mesh, model_config_obj, tokenizer = setup_models(
        config.model,
        config.training.intermediate_checkpoint_dir,
    )

    print(f"Loading checkpoint from {args.checkpoint_dir}...")
    load_trained_checkpoint(policy_model, args.checkpoint_dir, args.step)

    print("Preparing test dataset...")
    _, _, test_dataset = prepare_datasets(
        train_data_dir=config.training.train_data_dir,
        test_data_dir=config.training.test_data_dir,
        source=config.training.data_source,
        micro_batch_size=config.training.micro_batch_size,
        num_test_batches=config.training.num_test_batches,
    )

    print("Creating sampler...")
    sampler = create_sampler(
        policy_model,
        tokenizer,
        model_config_obj,
        config.generation.max_prompt_length,
        config.generation.max_generation_steps,
    )

    print(f"Evaluating with {args.inference_config} config...")
    results = evaluate_with_config(
        test_dataset,
        sampler,
        config_name=args.inference_config,
    )

    print("\nEvaluation Results:")
    print(f"  Correct: {results['correct']}/{results['total']}")
    print(f"  Accuracy: {results['accuracy']:.2f}%")
    print(f"  Partial Accuracy: {results['partial_accuracy']:.2f}%")
    print(f"  Format Accuracy: {results['format_accuracy']:.2f}%")


def main():
    """Main entry point."""
    args = parse_args()

    if args.command is None:
        print("Please specify a command. Use --help for available commands.")
        sys.exit(1)

    if args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
