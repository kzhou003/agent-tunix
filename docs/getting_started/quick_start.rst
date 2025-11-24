Quick Start
===========

Basic Training
--------------

Run training with default configuration::

    python run_training.py

This will:
- Load the Gemma3 270M model with LoRA
- Use default GRPO training parameters
- Save checkpoints to ``./checkpoints/ckpts/``

View Configuration
------------------

Before running, check the resolved configuration::

    python run_training.py --cfg job

This shows all configuration values that will be used.

Quick Testing
-------------

Run a quick test with reduced configuration (10 steps)::

    python run_training.py +experiment=quick_test

This is useful for:
- Testing setup without long training
- Validating data pipeline
- Debugging configuration issues

Custom Configuration
--------------------

Override any configuration value from the command line::

    # Different model size
    python run_training.py model=gemma3_1b

    # Custom learning rate
    python run_training.py optimizer.learning_rate=1e-5

    # Multiple overrides
    python run_training.py model=gemma3_1b optimizer.learning_rate=1e-5 training.num_batches=100

Configuration values use dot notation for nested values::

    python run_training.py optimizer.warmup_ratio=0.05
    python run_training.py training.micro_batch_size=2
    python run_training.py generation.max_generation_steps=256

Using Experiment Presets
------------------------

Create and use experiment presets for common scenarios::

    # Quick test experiment
    python run_training.py +experiment=quick_test

    # Full training experiment
    python run_training.py +experiment=full_training

See :doc:`Experiments Guide </guide/experiments>` for creating custom presets.

Model Evaluation
----------------

Evaluate a trained model::

    # With default configuration
    python evaluate.py

    # With custom checkpoint
    python evaluate.py checkpoint_dir=./checkpoints/ckpts/

    # Different inference strategy
    python evaluate.py inference_config=standard

Inference configurations: ``greedy``, ``standard``, ``liberal``

Utilities
---------

Check GPU availability::

    python -m agent_tunix.utils check-gpu

Show default configuration::

    python -m agent_tunix.utils show-config

Makefile Shortcuts
------------------

Quick commands using Make::

    make train                # Default training
    make train-quick          # Quick test
    make train-show-config    # Show configuration
    make evaluate             # Evaluate model
    make check-gpu            # Check GPU
    make show-config          # Show defaults

Hyperparameter Sweeps
---------------------

Run multiple experiments with different configurations::

    # Sweep over models
    python run_training.py --multirun model=gemma3_270m,gemma3_1b

    # Sweep over learning rates
    python run_training.py --multirun optimizer.learning_rate=1e-6,3e-6,1e-5

    # Multiple parameter sweep
    python run_training.py --multirun model=gemma3_270m,gemma3_1b optimizer.learning_rate=3e-6,1e-5

Each configuration runs sequentially, with results saved to separate output directories.

Output Structure
----------------

Training creates the following structure::

    outputs/
    └── tunix-grpo/
        └── YYYY-MM-DD/
            └── HH-MM-SS/
                ├── .hydra/
                │   ├── config.yaml           # Resolved configuration
                │   ├── overrides.yaml        # Command-line overrides
                │   └── launcher.yaml         # Launcher configuration
                └── checkpoints/
                    └── train.log             # Training logs

Configurations are automatically saved for reproducibility.

Next Steps
----------

- :doc:`Full Training Guide </guide/training>`
- :doc:`Configuration Guide </getting_started/configuration>`
- :doc:`Hyperparameter Tuning </guide/hyperparameter_tuning>`

Getting Help
------------

- Check logs in output directory for detailed information
- View configuration: ``python run_training.py --cfg job``
- See defaults tree: ``python run_training.py --info defaults-tree``
- Full help: ``python run_training.py --help``
