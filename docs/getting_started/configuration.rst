Configuration Guide
====================

Configuration Overview
----------------------

Agent-Tunix uses YAML-based configuration organized in the ``conf/`` directory. All settings can be overridden from the command line without code changes.

Configuration Structure
-----------------------

The configuration is organized into groups::

    conf/
    ├── config.yaml              # Main config with defaults list
    ├── model/                   # Model architectures
    ├── optimizer/               # Optimizer settings
    ├── scheduler/               # Learning rate schedulers
    ├── grpo/                    # GRPO algorithm parameters
    ├── generation/              # Text generation settings
    ├── training/                # Training hyperparameters
    ├── evaluation/              # Evaluation settings
    └── experiment/              # Experiment presets

Main Configuration
------------------

The ``conf/config.yaml`` file specifies which configs to load::

    defaults:
      - model: gemma3_270m
      - optimizer: adamw
      - scheduler: warmup_cosine
      - grpo: default
      - generation: default
      - training: default
      - _self_

The ``_self_`` entry controls merge order (configs loaded after override earlier ones).

Configuration Groups
--------------------

**Model Configuration**

Located in ``conf/model/``. Available models:

- ``gemma3_270m.yaml`` - 270M parameter model
- ``gemma3_1b.yaml`` - 1B parameter model

Override::

    python run_training.py model=gemma3_1b

**Optimizer Configuration**

Located in ``conf/optimizer/``. Current options:

- ``adamw.yaml`` - AdamW with warmup cosine decay

Key parameters::

    learning_rate: 3e-6
    warmup_ratio: 0.1
    max_grad_norm: 0.1

Override::

    python run_training.py optimizer.learning_rate=1e-5

**GRPO Configuration**

Located in ``conf/grpo/``. Algorithm parameters::

    num_generations: 4      # Responses per prompt
    num_iterations: 1       # Iterations per batch
    beta: 0.08              # KL divergence penalty
    epsilon: 0.2            # PPO clipping

**Training Configuration**

Located in ``conf/training/``. Key settings::

    micro_batch_size: 4
    num_batches: 3738
    num_epochs: 1
    checkpoint_dir: ./checkpoints/ckpts/

**Generation Configuration**

Located in ``conf/generation/``. Text generation settings::

    max_prompt_length: 256
    max_generation_steps: 512
    temperature: 0.9
    top_p: 1.0
    top_k: 50

Command-Line Overrides
----------------------

Use dot notation to override nested values::

    # Single override
    python run_training.py optimizer.learning_rate=5e-6

    # Multiple overrides
    python run_training.py model=gemma3_1b optimizer.learning_rate=1e-5 training.num_batches=50

    # Add new value (+ prefix)
    python run_training.py +optimizer.schedule_type=linear

    # Force override (++ prefix)
    python run_training.py ++optimizer.warmup_steps=100

    # Delete value (~ prefix)
    python run_training.py ~training.gradient_clip

    # List values
    python run_training.py 'generation.eos_tokens=[1,106,107]'

    # Dictionary values
    python run_training.py '+model.custom_params={hidden_size:512,num_layers:8}'

Experiment Presets
------------------

Experiment files in ``conf/experiment/`` combine multiple settings::

    # Quick testing (10 steps, reduced config)
    python run_training.py +experiment=quick_test

    # Full production training
    python run_training.py +experiment=full_training

Create custom experiments by adding YAML files to ``conf/experiment/``.

Configuration Inspection
------------------------

View resolved configuration::

    # Show resolved configuration
    python run_training.py --cfg job

    # Show with interpolations resolved
    python run_training.py --cfg job --resolve

    # Show defaults composition tree
    python run_training.py --info defaults-tree

    # List all available configuration groups
    python run_training.py --info config-groups

    # Dry run (test config without training)
    python run_training.py --dry-run

Advanced: Creating Custom Configurations
-----------------------------------------

Create a new model configuration in ``conf/model/custom.yaml``::

    model_family: gemma3
    model_size: 4b
    lora_rank: 16
    lora_alpha: 16.0
    lora_module_path: ".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|.*attn_vec_einsum"
    mesh_shape: [[1, 4], ["fsdp", "tp"]]

Use it::

    python run_training.py model=custom

Create an experiment preset in ``conf/experiment/ablation.yaml``::

    # @package _global_

    defaults:
      - override /model: gemma3_1b
      - override /optimizer: adamw

    training:
      num_batches: 100
      micro_batch_size: 2

    optimizer:
      learning_rate: 1e-5

    experiment_name: ablation_study
    tags: [ablation, learning_rate]

Use it::

    python run_training.py +experiment=ablation

Configuration Best Practices
-----------------------------

1. **Check before running**: Always view config with ``--cfg job``
2. **Use experiments**: Create presets for common configurations
3. **Semantic naming**: Use descriptive names for custom configs
4. **Document changes**: Add comments in custom YAML files
5. **Version control**: Keep ``conf/`` in version control
6. **Avoid magic numbers**: Use configuration groups instead

Memory Requirements by Model
-----------------------------

Configuration recommendations for different GPUs:

**RTX 2080 Ti (11GB)**

- Model: gemma3_270m
- Batch size: 1
- LoRA rank: 8-16

::

    python run_training.py model=gemma3_270m training.micro_batch_size=1

**RTX A6000 (48GB)**

- Model: gemma3_1b
- Batch size: 4
- LoRA rank: 32

::

    python run_training.py model=gemma3_1b training.micro_batch_size=4

**Multiple GPUs**

Modify ``mesh_shape`` in model config::

    python run_training.py model.mesh_shape=[[1,4],["fsdp","tp"]]

Next Steps
----------

- :doc:`Training Guide </guide/training>`
- :doc:`Hyperparameter Tuning </guide/hyperparameter_tuning>`
- :doc:`API Reference </api/train>`
