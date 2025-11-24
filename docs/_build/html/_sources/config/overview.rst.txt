Configuration Overview
======================

Agent-Tunix uses Hydra, a framework for configuring complex applications. All training parameters are managed through YAML configuration files organized in the ``conf/`` directory.

Directory Structure
-------------------

::

    conf/
    ├── config.yaml                  # Main configuration
    ├── model/                       # Model architecture configs
    │   ├── gemma3_270m.yaml
    │   ├── gemma3_1b.yaml
    │   └── gemma3_4b.yaml
    ├── optimizer/                   # Optimizer settings
    │   └── adamw.yaml
    ├── scheduler/                   # LR scheduler configs
    │   └── warmup_cosine.yaml
    ├── grpo/                        # GRPO algorithm params
    │   └── default.yaml
    ├── generation/                  # Text generation settings
    │   └── default.yaml
    ├── training/                    # Training hyperparams
    │   └── default.yaml
    ├── evaluation/                  # Evaluation settings
    │   └── default.yaml
    └── experiment/                  # Experiment presets
        ├── quick_test.yaml
        └── full_training.yaml

Main Configuration (config.yaml)
--------------------------------

The main configuration file specifies defaults and merge order::

    defaults:
      - model: gemma3_270m         # Default model
      - optimizer: adamw           # Default optimizer
      - scheduler: warmup_cosine   # Default scheduler
      - grpo: default              # Default GRPO params
      - generation: default        # Default generation
      - training: default          # Default training
      - _self_                     # This file (applied last)

The ``_self_`` entry controls when this file merges (typically last).

Configuration Groups
--------------------

**Model Group** (``conf/model/``)

Specifies model architecture and LoRA configuration.

**Optimizer Group** (``conf/optimizer/``)

Specifies optimizer algorithm and settings (learning rate, warmup, etc.).

**Scheduler Group** (``conf/scheduler/``)

Specifies learning rate schedule strategy.

**GRPO Group** (``conf/grpo/``)

Specifies GRPO algorithm parameters (generations, beta, epsilon).

**Generation Group** (``conf/generation/``)

Specifies text generation settings (temperature, sampling, lengths).

**Training Group** (``conf/training/``)

Specifies training hyperparameters (batch sizes, epochs, checkpointing).

**Evaluation Group** (``conf/evaluation/``)

Specifies evaluation settings (checkpoint, inference mode, metrics).

**Experiment Group** (``conf/experiment/``)

Presets combining multiple config settings for specific scenarios.

Using Configurations
---------------------

**Default Configuration**

Run with all defaults::

    python run_training.py

**Override Single Parameter**

Use dot notation::

    python run_training.py optimizer.learning_rate=1e-5

**Override Multiple Parameters**

::

    python run_training.py \
        model=gemma3_1b \
        optimizer.learning_rate=1e-5 \
        training.micro_batch_size=2

**Select Different Config Group**

::

    python run_training.py model=gemma3_1b

**Use Experiment Preset**

::

    python run_training.py +experiment=quick_test

**View Resolved Configuration**

::

    python run_training.py --cfg job

Shows all interpolations resolved and defaults applied.

**Dry Run**

Test configuration without training::

    python run_training.py --dry-run

Configuration Composition
--------------------------

Hydra merges configurations in this order:

1. Load defaults (from config.yaml defaults list)
2. Apply command-line overrides
3. Merge with _self_

Example::

    # Starting point: conf/config.yaml defaults
    - model: gemma3_270m

    # Override: command-line argument
    python run_training.py model=gemma3_1b

    # Result: model=gemma3_1b replaces model=gemma3_270m

Nested Overrides
~~~~~~~~~~~~~~~~

Override nested values with dot notation::

    python run_training.py optimizer.learning_rate=1e-5
    python run_training.py model.lora_rank=64

Add new fields with +::

    python run_training.py +training.custom_param=value

Force override with ++::

    python run_training.py ++model.custom_field=value

Delete fields with ~::

    python run_training.py ~training.gradient_clip

Interpolations
--------------

Reference other config values using ${path}::

    # In YAML files
    model_name: gemma3
    full_name: ${model_name}_1b

    # Results in: full_name = gemma3_1b

This enables:

- Reducing duplication
- Consistency across configs
- Semantic relationships

Example::

    # In conf/training/default.yaml
    num_batches: 3738
    total_steps: ${training.num_batches}  # References above

Creating Custom Configurations
-------------------------------

Create custom model config in ``conf/model/custom.yaml``::

    model_family: gemma3
    model_size: 4b
    lora_rank: 64
    lora_alpha: 64.0
    lora_module_path: ".*q_einsum|.*kv_einsum"
    mesh_shape: [[1, 4], ["fsdp", "tp"]]

Use it::

    python run_training.py model=custom

Create custom experiment in ``conf/experiment/my_exp.yaml``::

    # @package _global_
    defaults:
      - override /model: gemma3_1b
      - override /optimizer: adamw

    training:
      num_batches: 100
      micro_batch_size: 2

    optimizer:
      learning_rate: 1e-5

Use it::

    python run_training.py +experiment=my_exp

Configuration Validation
------------------------

**Check syntax**::

    python run_training.py --cfg job

**View composition tree**::

    python run_training.py --info defaults-tree

**List available configs**::

    python run_training.py --info config-groups

Best Practices
--------------

1. **Check before running**::

       python run_training.py --cfg job > /tmp/config.yaml

2. **Use semantic names**::

       # Good
       python run_training.py +experiment=ablation_lr_sweep

       # Avoid
       python run_training.py +experiment=exp1

3. **Document custom configs**::

       # In your custom.yaml, add comments
       # Custom model for experiment X
       model_family: gemma3

4. **Store successful configs**::

       # Keep in version control
       git add conf/

5. **Use experiments for reproducibility**::

       # Instead of remembering many overrides
       python run_training.py +experiment=best_config

6. **Avoid magic numbers**::

       # Create config group instead
       conf/learning_rates/high.yaml
       conf/learning_rates/low.yaml

Configuration File Format
--------------------------

YAML files use standard YAML syntax::

    # Comments
    model:
      name: gemma3
      size: 1b                      # Inline comment
      lora_rank: 32

    # Lists
    mesh_shape: [[1, 4], ["fsdp", "tp"]]

    # Strings
    model_path: "/path/to/model"

    # Numbers
    learning_rate: 1e-5

    # Booleans
    use_fp16: false

Hydra-Specific Syntax
~~~~~~~~~~~~~~~~~~~~~

``# @package _global_``
    Merge experiment into global namespace (required for experiments)

``${path.to.value}``
    Interpolate other config values

``???``
    Required value (error if not provided)

Environment Variables
---------------------

Reference environment variables::

    # In YAML
    model_path: ${oc.env:MODEL_PATH,/default/path}

Override all configs from environment::

    export HYDRA_OVERRIDE_MODEL=gemma3_1b
    python run_training.py

Advanced: Config Structured

With plugins, can use Python dataclasses for type-safe configs::

    @dataclass
    class ModelConfig:
        lora_rank: int = 32
        lora_alpha: float = 32.0

(Currently optional in Agent-Tunix)

Common Patterns
---------------

**Learning Rate Search**

Create ``conf/experiment/lr_sweep.yaml``::

    # @package _global_
    training:
      num_batches: 100

Then run::

    python run_training.py +experiment=lr_sweep \
        --multirun optimizer.learning_rate=1e-6,1e-5,1e-4

**Memory-Constrained Training**

::

    python run_training.py \
        model=gemma3_270m \
        model.lora_rank=8 \
        training.micro_batch_size=1 \
        grpo.num_generations=2

**Production Training**

::

    python run_training.py \
        +experiment=full_training \
        model=gemma3_1b \
        optimizer.learning_rate=3e-6

Next Steps
----------

- :doc:`model` - Model configuration reference
- :doc:`optimizer` - Optimizer configuration reference
- :doc:`training` - Training configuration reference
- :doc:`../getting_started/configuration` - Detailed configuration guide
