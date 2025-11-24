# Quick Reference: Hydra Commands

## Training

```bash
# Default config
python run_training.py

# Quick test (10 steps)
python run_training.py +experiment=quick_test

# Full production training
python run_training.py +experiment=full_training

# Custom model
python run_training.py model=gemma3_1b

# Custom learning rate
python run_training.py optimizer.learning_rate=1e-5

# Multiple overrides
python run_training.py model=gemma3_1b optimizer.learning_rate=1e-5 training.num_batches=100

# Show config before running
python run_training.py --cfg job

# Show defaults tree
python run_training.py --info defaults-tree

# Dry run (no actual training)
python run_training.py --dry-run
```

## Hyperparameter Sweeps

```bash
# Sweep models
python run_training.py --multirun model=gemma3_270m,gemma3_1b

# Sweep learning rates
python run_training.py --multirun optimizer.learning_rate=1e-6,3e-6,1e-5

# Sweep multiple parameters
python run_training.py --multirun model=gemma3_270m,gemma3_1b optimizer.learning_rate=3e-6,1e-5
```

## Evaluation

```bash
# Default evaluation
python evaluate.py

# Custom checkpoint
python evaluate.py checkpoint_dir=./checkpoints/ckpts/

# Different inference config
python evaluate.py inference_config=standard

# Show config
python evaluate.py --cfg job
```

## Utilities

```bash
# Check GPU
python -m agent_tunix.utils check-gpu

# Show default config
python -m agent_tunix.utils show-config
```

## Makefile Shortcuts

```bash
make train                 # Default training
make train-quick           # Quick test
make train-show-config     # Show resolved config
make train-show-defaults   # Show defaults tree
make train-sweep           # Example sweep

make evaluate              # Evaluate
make evaluate-show-config  # Show eval config

make check-gpu             # Check GPU
make show-config           # Show defaults
```

## Override Grammar

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

## Configuration Files

- `conf/config.yaml` - Main config
- `conf/model/*.yaml` - Model configs
- `conf/optimizer/*.yaml` - Optimizer configs
- `conf/scheduler/*.yaml` - Scheduler configs
- `conf/grpo/*.yaml` - GRPO configs
- `conf/generation/*.yaml` - Generation configs
- `conf/training/*.yaml` - Training configs
- `conf/experiment/*.yaml` - Experiment presets
- `conf/evaluation/*.yaml` - Evaluation settings

## Common Workflows

### Quick Testing
```bash
python run_training.py +experiment=quick_test
```

### Production Training
```bash
python run_training.py +experiment=full_training
```

### Model Comparison
```bash
python run_training.py --multirun model=gemma3_270m,gemma3_1b
```

### Learning Rate Ablation
```bash
python run_training.py --multirun optimizer.learning_rate=1e-7,1e-6,1e-5,1e-4
```

### Evaluate Multiple Checkpoints
```bash
# Evaluate with greedy
python evaluate.py inference_config=greedy

# Evaluate with standard
python evaluate.py inference_config=standard

# Evaluate with liberal
python evaluate.py inference_config=liberal
```

## Tips

1. **Always check config first**: `python run_training.py --cfg job`
2. **Dry run before sweep**: `python run_training.py --dry-run`
3. **Use experiments**: Create preset configs in `conf/experiment/`
4. **Defaults tree**: `--info defaults-tree` shows config composition
5. **Help**: `python run_training.py --help` for Hydra help
6. **Output tracking**: Configs automatically saved in `outputs/` directory

## Debugging

```bash
# Show resolved config
python run_training.py --cfg job --resolve

# Verbose logging
python run_training.py hydra.verbose=true

# Show all available configs
python run_training.py --info config-groups

# Test config composition
python run_training.py --cfg job model=gemma3_1b optimizer.learning_rate=1e-5
```
