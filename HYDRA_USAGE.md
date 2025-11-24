# Hydra Configuration Management Guide

This project uses **Hydra** for configuration management instead of argparse. Hydra provides a more robust, composable, and scalable approach to configuration management for ML training pipelines.

## Quick Start

### Basic Training
```bash
# Run with default configuration
python run_training.py

# View resolved configuration before running
python run_training.py --cfg job

# View configuration tree
python run_training.py --info defaults-tree
```

### Override Configuration Values
```bash
# Change model size
python run_training.py model=gemma3_1b

# Override nested values with dot notation
python run_training.py optimizer.learning_rate=5e-6 training.num_batches=50

# Combine multiple overrides
python run_training.py model=gemma3_1b optimizer.learning_rate=1e-5 training.epochs=2
```

### Experiment Presets
```bash
# Quick testing (10 steps, reduced config, WANDB disabled)
python run_training.py +experiment=quick_test

# Full production training
python run_training.py +experiment=full_training
```

### Hyperparameter Sweeps
```bash
# Sweep over multiple values
python run_training.py --multirun model=gemma3_270m,gemma3_1b

# Sweep learning rates
python run_training.py --multirun optimizer.learning_rate=1e-6,3e-6,1e-5

# Combine multiple sweeps (creates all combinations)
python run_training.py --multirun model=gemma3_270m,gemma3_1b optimizer.learning_rate=3e-6,1e-5

# Sweep over ranges
python run_training.py --multirun 'training.num_batches=range(100,1000,100)'
```

## Directory Structure

```
agent-tunix/
├── conf/
│   ├── config.yaml                    # Main config with defaults
│   ├── model/
│   │   ├── gemma3_270m.yaml          # 270M model config
│   │   └── gemma3_1b.yaml            # 1B model config
│   ├── optimizer/
│   │   └── adamw.yaml                # AdamW optimizer config
│   ├── scheduler/
│   │   └── warmup_cosine.yaml        # Warmup cosine scheduler
│   ├── grpo/
│   │   └── default.yaml              # GRPO algorithm config
│   ├── generation/
│   │   └── default.yaml              # Text generation config
│   ├── training/
│   │   └── default.yaml              # Training hyperparameters
│   └── experiment/
│       ├── quick_test.yaml           # Quick testing preset
│       └── full_training.yaml        # Full training preset
├── run_training.py                    # Simple entry point
└── src/agent_tunix/train.py          # Hydra-decorated training function
```

## Configuration Details

### Main Configuration (conf/config.yaml)

The main config defines the defaults list and project-level settings:

```yaml
defaults:
  - model: gemma3_270m
  - optimizer: adamw
  - scheduler: warmup_cosine
  - grpo: default
  - generation: default
  - training: default
  - _self_
```

The `defaults` list controls which configs to load. The `_self_` entry determines when the main config's values are merged (at the end, so they override group defaults).

### Model Configuration (conf/model/gemma3_270m.yaml)

```yaml
model_family: gemma3
model_size: 270m
lora_rank: 32
lora_alpha: 32.0
lora_module_path: ".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|.*attn_vec_einsum"
mesh_shape: [[1, 4], ["fsdp", "tp"]]
```

### Training Configuration (conf/training/default.yaml)

Contains all training hyperparameters:
- Data paths and sources
- Batch sizes
- Checkpointing settings
- Logging configuration

## Advanced Usage

### Creating Custom Experiment Files

Create a new experiment config at `conf/experiment/my_experiment.yaml`:

```yaml
# @package _global_

defaults:
  - override /model: gemma3_1b
  - override /optimizer: adamw

# Override specific values
optimizer:
  learning_rate: 1e-5

training:
  num_batches: 100
  num_epochs: 2

# Metadata
experiment_name: my_experiment
tags: [custom, testing]
```

Use it with:
```bash
python run_training.py +experiment=my_experiment
```

### Creating New Config Groups

To add a new optimizer configuration, create `conf/optimizer/sgd.yaml`:

```yaml
# SGD optimizer configuration
learning_rate: 0.01
momentum: 0.9
nesterov: true
```

Then use it:
```bash
python run_training.py optimizer=sgd
```

### Override Grammar

Hydra supports special prefixes for overrides:

```bash
# Standard override
python run_training.py optimizer.learning_rate=1e-5

# Add new key (+ prefix)
python run_training.py +optimizer.schedule_type=linear

# Force override (++ prefix)
python run_training.py ++optimizer.warmup_steps=100

# Delete value (~ prefix)
python run_training.py ~training.gradient_clip

# List values
python run_training.py 'grpo.eos_tokens=[1,106,107]'

# Dict values
python run_training.py '+model.custom_params={hidden_size:512,num_layers:8}'
```

### Interpolation

You can reference other config values using ${key} syntax in YAML files:

```yaml
# conf/config.yaml
project_name: my_project
output_dir: outputs/${project_name}
checkpoint_dir: ${output_dir}/checkpoints
```

## Output Management

When you run training, Hydra automatically:

1. **Creates output directory**: `outputs/project_name/YYYY-MM-DD/HH-MM-SS/`
2. **Saves config snapshot**: `.hydra/config.yaml` in the output directory
3. **Saves working directory**: `.hydra/.hydra.yaml`
4. **Saves overrides**: `.hydra/overrides.yaml`
5. **Saves launcher info**: `.hydra/launcher.yaml`

This enables perfect reproducibility - you can see exactly which config was used for each run.

## Debugging and Inspection

### View Resolved Configuration

Before running, see the fully resolved config:
```bash
python run_training.py --cfg job
```

### View Configuration with Overrides Highlighted

```bash
python run_training.py --cfg job --resolve
```

### View Defaults Tree

See which configs are being loaded and in what order:
```bash
python run_training.py --info defaults-tree
```

### View Package Structure

```bash
python run_training.py --info config-groups
```

### Dry Run (Don't Actually Train)

```bash
python run_training.py --dry-run
```

## Integration with Code

In `src/agent_tunix/train.py`, the training function uses the `@hydra.main` decorator:

```python
@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def train(cfg: DictConfig) -> None:
    # cfg is the resolved configuration dictionary
    # Access values with dot notation
    model_size = cfg.model.model_size
    learning_rate = cfg.optimizer.learning_rate

    # ... training code ...
```

## Comparison: Before vs After

### Before (argparse)

```python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model-size', default='270m')
parser.add_argument('--learning-rate', default=3e-6)
parser.add_argument('--num-batches', default=100)
# ... 20+ more arguments ...
args = parser.parse_args()

train(args.model_size, args.learning_rate, args.num_batches, ...)
```

**Limitations**:
- Scattered defaults across code
- Difficult to manage complex nested configs
- Manual environment variable handling
- No built-in config composition
- Hard to track which config was used for reproduction

### After (Hydra)

```python
@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def train(cfg: DictConfig) -> None:
    # All config in YAML files
    # Use dot notation: cfg.model.model_size, cfg.optimizer.learning_rate
    # Automatic config snapshot for reproducibility
    # Built-in multi-run support
    # Easy config composition
```

## Makefile Integration

The Makefile provides convenient shortcuts:

```bash
make train                 # Run with defaults
make train-quick           # Run quick test
make train-show-config     # View resolved config
make train-sweep           # Run hyperparameter sweep
```

## Tips and Best Practices

1. **Use config groups for related settings**: Each `conf/model/*.yaml` handles one model configuration
2. **Leverage experiments**: Create preset combinations for common scenarios
3. **Use interpolation**: Avoid repeating values, reference them with `${key}`
4. **Check outputs**: Always inspect `.hydra/` directory for reproducibility info
5. **Use --dry-run**: Before sweeps, verify the config is correct
6. **Version control configs**: Keep conf/ in git to track config evolution
7. **Use descriptive names**: Name experiments clearly (e.g., `ablation_lr_vs_batch_size.yaml`)

## Resources

- [Hydra Official Documentation](https://hydra.cc/)
- [Hydra Tutorials](https://hydra.cc/docs/tutorials/structured_config/intro/)
- [Override Grammar](https://hydra.cc/docs/advanced/override_grammar/basic/)
- [Defaults List](https://hydra.cc/docs/advanced/defaults_list/)

## Migration from argparse

The old argparse-based CLI (`agent-tunix` command) is still available but deprecated. To use Hydra with the CLI:

```bash
# Deprecated (still works)
agent-tunix train --model-size 270m --learning-rate 1e-5

# New way (recommended)
python run_training.py model=gemma3_270m optimizer.learning_rate=1e-5

# Or for other commands
agent-tunix evaluate --checkpoint-dir ./checkpoints/ckpts/
agent-tunix check-gpu
agent-tunix show-config
```

For full Hydra help:
```bash
python run_training.py --help
```
