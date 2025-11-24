Training API
============

.. py:module:: agent_tunix.train
   :noindex:

This module contains the main training entry point and training utilities.

Main Entry Point
----------------

.. autofunction:: train
   :members:

Training Utilities
------------------

.. autofunction:: create_optimizer
   :members:

.. autofunction:: create_cluster_config
   :members:

.. autofunction:: calculate_max_steps
   :members:

Configuration Classes
---------------------

The training module works with Hydra's DictConfig. Configuration structure::

    training:
      micro_batch_size: 4        # Batch size per device
      num_batches: 3738          # Total training batches
      num_epochs: 1              # Number of epochs
      checkpoint_dir: ./checkpoints/ckpts/
      save_interval_steps: 100   # Save checkpoint every N steps
      eval_interval_steps: 500   # Evaluate every N steps

    optimizer:
      learning_rate: 3e-6        # Peak learning rate
      warmup_ratio: 0.1          # Warmup as % of total
      max_grad_norm: 0.1         # Gradient clipping

    grpo:
      num_generations: 4         # Responses per prompt
      beta: 0.08                 # KL divergence weight
      epsilon: 0.2               # PPO clipping

    generation:
      max_prompt_length: 256
      max_generation_steps: 512
      temperature: 0.9
      top_k: 50
      top_p: 1.0

Training Flow
-------------

The training process follows these steps:

1. **Configuration Loading**: Hydra loads and composes configuration from YAML files
2. **Seed Management**: Sets random seeds for reproducibility
3. **Dataset Preparation**: Loads and tokenizes training data (GSM8K by default)
4. **Model Setup**:
   - Loads base model from Hugging Face or Kaggle
   - Applies LoRA to specified layers
   - Initializes reference and policy models
5. **Training Loop**:
   - Generate multiple responses per prompt
   - Compute rewards (correctness + format matching)
   - Calculate policy gradients with KL penalty
   - Update model weights
   - Log metrics to Weights & Biases
6. **Checkpointing**: Saves model weights and configuration periodically
7. **Evaluation**: Periodically evaluates on validation set

GRPO Algorithm
--------------

The Group Relative Policy Optimization algorithm:

1. For each prompt, generate K responses (num_generations)
2. Compute rewards for each response
3. Normalize rewards relative to group (Group Relative)
4. Compute policy gradients with PPO clipping
5. Apply KL divergence penalty to stay close to reference model
6. Update weights using optimizer with gradient clipping

Key hyperparameters:

- **num_generations**: Number of candidate responses to generate per prompt
- **beta**: Weight of KL divergence penalty (controls deviation from reference)
- **epsilon**: PPO clipping range for gradient updates

Example Usage
-------------

Basic training with defaults::

    from agent_tunix.train import train
    from hydra import compose, initialize
    from omegaconf import DictConfig

    if __name__ == "__main__":
        train()  # Hydra decorator handles config loading

Command-line examples::

    # Default configuration
    python run_training.py

    # Override learning rate
    python run_training.py optimizer.learning_rate=1e-5

    # Use different model
    python run_training.py model=gemma3_1b

    # Multiple overrides
    python run_training.py model=gemma3_1b training.micro_batch_size=2 optimizer.learning_rate=3e-6

    # Use experiment preset
    python run_training.py +experiment=quick_test

    # Parameter sweep
    python run_training.py --multirun optimizer.learning_rate=1e-7,1e-6,1e-5,1e-4

Tips and Best Practices
-----------------------

1. **Start with quick_test**: Use ``+experiment=quick_test`` for 10-step validation runs
2. **Monitor GPU memory**: Use ``nvidia-smi`` during training
3. **Check configuration**: Use ``python run_training.py --cfg job`` before running
4. **Save checkpoints frequently**: Lower ``save_interval_steps`` for important runs
5. **Enable logging**: W&B is enabled by default, set ``wandb_disabled=true`` to disable
6. **Use gradual unfreezing**: Start with smaller learning rates and increase if loss plateaus

Next Steps
----------

- :doc:`../guide/training` - Detailed training guide
- :doc:`../guide/hyperparameter_tuning` - Hyperparameter tuning strategies
- :doc:`evaluate` - Evaluation API reference
