Frequently Asked Questions
==========================

General Questions
-----------------

**What is Agent-Tunix?**

Agent-Tunix is a framework for training language models using GRPO (Group Relative Policy Optimization) with parameter-efficient fine-tuning via LoRA. It's designed for reinforcement learning from reward feedback.

**What models does Agent-Tunix support?**

Currently supports Google Gemma3 family:

- Gemma3 270M (lightweight)
- Gemma3 1B (standard)
- Gemma3 4B (large)

Can be extended to other models by adding configuration files.

**Do I need multiple GPUs to use Agent-Tunix?**

No. The framework works on a single GPU. For smaller models (270M), even 11GB GPUs (like RTX 2080 Ti) can work. Multi-GPU support is optional for faster training.

**Can I run on CPU?**

Not recommended. The framework is optimized for GPUs. You can set ``device=cpu`` but training will be very slow.

**What's the difference between GRPO and other RL methods?**

GRPO (Group Relative Policy Optimization):

- Generates K responses per prompt
- Normalizes rewards relative to the group
- More sample-efficient than standard PPO
- Designed for discrete sequence generation

**Do I need to understand GRPO to use Agent-Tunix?**

No. You can use the framework with default settings without understanding GRPO details. The configuration handles most complexity.

Setup and Installation
----------------------

**I'm getting CUDA errors. How do I fix this?**

First, verify CUDA is installed::

    python -m agent_tunix.utils check-gpu

If not installed, follow the :doc:`../getting_started/installation` guide.

**How much VRAM do I need?**

Minimum requirements by model:

- Gemma3 270M: 11GB (batch size 1)
- Gemma3 1B: 24GB (batch size 2)
- Gemma3 4B: 48GB (batch size 4)

See :doc:`../getting_started/configuration` for memory optimization tips.

**How do I install with a different CUDA version?**

See the :doc:`../getting_started/installation` guide's CUDA section for instructions.

**Can I use WSL2 on Windows?**

Yes. Install CUDA in WSL2 and follow normal installation steps. GPU passthrough works on WSL2.

Configuration
-------------

**How do I override configuration values?**

Use dot notation on command line::

    python run_training.py optimizer.learning_rate=1e-5
    python run_training.py model=gemma3_1b training.micro_batch_size=2

See :doc:`../getting_started/configuration` for more examples.

**What's the difference between parameters and overrides?**

- **Parameters**: Settings defined in YAML config files
- **Overrides**: Command-line changes to parameters

Example::

    # Parameter in conf/optimizer/adamw.yaml
    learning_rate: 3e-6

    # Override from command line
    python run_training.py optimizer.learning_rate=1e-5

**Can I use multiple experiments?**

Yes. Use the ``+experiment=`` syntax::

    python run_training.py +experiment=quick_test

Create custom experiments in ``conf/experiment/``.

**What's the difference between +experiment and --multirun?**

- **+experiment**: Load preset configuration combining multiple settings
- **--multirun**: Run multiple training jobs with different parameter combinations

Example::

    # Single run with preset
    python run_training.py +experiment=quick_test

    # Multiple runs with parameter sweep
    python run_training.py --multirun optimizer.learning_rate=1e-6,1e-5,1e-4

**Where are configuration files located?**

In ``conf/`` directory structure::

    conf/
    ├── config.yaml              # Main config
    ├── model/                   # Model configs
    ├── optimizer/               # Optimizer configs
    ├── scheduler/               # Scheduler configs
    ├── grpo/                    # GRPO algorithm configs
    ├── generation/              # Generation configs
    ├── training/                # Training configs
    ├── evaluation/              # Evaluation configs
    └── experiment/              # Experiment presets

Training
--------

**How long does training take?**

Depends on configuration:

- Quick test: ~5 minutes (10 steps)
- Full training: 1-24 hours (depends on GPU and data size)

Check logs for training speed::

    tail -f outputs/tunix-grpo/YYYY-MM-DD/HH-MM-SS/train.log

**How often should I save checkpoints?**

Default is every 100 steps. Adjust::

    python run_training.py training.save_interval_steps=50

More frequent saves → more disk usage but better checkpoint coverage.

**Can I resume training from a checkpoint?**

Yes. Provide checkpoint directory::

    python run_training.py checkpoint_dir=./checkpoints/ckpts/

Automatically loads latest checkpoint.

**How do I know if my learning rate is good?**

Monitor training loss:

- **Decreasing smoothly**: Good learning rate
- **Increasing (diverging)**: Learning rate too high
- **Decreasing very slowly**: Learning rate too low
- **Noisy**: Consider gradient clipping

Use Weights & Biases or TensorBoard to visualize::

    make tensorboard
    # Open http://localhost:6006

**What's a good batch size?**

Depends on GPU memory:

- 11GB GPU: 1
- 24GB GPU: 2-4
- 48GB GPU: 4-8
- 80GB+ GPU: 8-16

Larger batches = more stable gradients but slower per-step updates.

**Should I use more generations per prompt?**

More generations = better reward signal but slower training.

Default is 4. Try 2-8:

- 2: Fast training, sparse reward signal
- 4: Good balance (default)
- 8: Better training but 2x slower

**How do I debug training issues?**

Enable verbose logging::

    python run_training.py training.log_level=DEBUG

Check logs::

    tail -f outputs/tunix-grpo/YYYY-MM-DD/HH-MM-SS/train.log

**Can I use my own data?**

Yes. Create a custom data loading function in ``src/agent_tunix/data.py``.

See :doc:`../api/data` for guidelines.

**How do custom rewards work?**

Reward functions evaluate responses and return scores (0.0-1.0):

- 1.0 = perfect response
- 0.5 = partial credit
- 0.0 = incorrect

See :doc:`../advanced/custom_rewards` for examples.

**Can I use multiple GPUs?**

Yes. Configure mesh shape::

    python run_training.py model.mesh_shape=[[4,1],["fsdp","tp"]]

See :doc:`../advanced/distributed_training`.

Evaluation
----------

**How do I evaluate my trained model?**

::

    python evaluate.py

Uses latest checkpoint by default.

**Can I evaluate on a specific checkpoint?**

Yes::

    python evaluate.py step=1000

List available checkpoints::

    ls checkpoints/ckpts/actor/

**What metrics are computed?**

- **Accuracy**: Exact match percentage
- **Partial Accuracy**: Within 10% of correct answer
- **Format Accuracy**: Response matches expected format

**How do different inference strategies compare?**

Three strategies available:

- **Greedy**: Deterministic, fastest, reproducible
- **Standard**: Balanced sampling, moderate diversity
- **Liberal**: High diversity, creative outputs

Try all three::

    for config in greedy standard liberal; do
        python evaluate.py inference_config=$config
    done

**Can I get confidence estimates?**

Yes, use multiple passes::

    python evaluate.py num_passes=5

Runs 5 generations per question and compares consistency.

**What if evaluation takes too long?**

Solutions:

1. Use greedy inference (fastest)::

       python evaluate.py inference_config=greedy

2. Evaluate fewer samples
3. Use earlier checkpoint::

       python evaluate.py step=100

Hyperparameter Tuning
---------------------

**How do I find good hyperparameters?**

Use the workflow in :doc:`../guide/hyperparameter_tuning`:

1. **Quick learning rate search** (1 hour)
2. **Batch size tuning** (2 hours)
3. **Model size testing** (varies)
4. **Final training** with best parameters

**Should I tune one parameter at a time?**

Yes, generally. Tune in this order:

1. Learning rate
2. Batch size
3. LoRA rank
4. Number of generations
5. KL beta

Tuning one at a time is easier to interpret.

**What's a good starting learning rate?**

Default is 3e-6. Good range to try:

- Too high: 1e-4 (likely diverges)
- Good: 1e-6 to 1e-5
- Too low: 1e-7 (very slow)

Start with 1e-5 and adjust based on training loss.

**When should I use warmup?**

Almost always for stable training. Default warmup_ratio=0.1 (10% of training).

Reduces warmup for very short training::

    python run_training.py +experiment=quick_test optimizer.warmup_ratio=0.0

Advanced Topics
---------------

**Can I use other models besides Gemma3?**

Yes. Create configuration in ``conf/model/``. You'll need to:

1. Configure model architecture
2. Set LoRA module paths
3. Update tokenizer path

**Can I fine-tune a model I already trained?**

Yes. Resume from checkpoint::

    python run_training.py checkpoint_dir=./checkpoints/ckpts/

**How do I implement distributed training?**

See :doc:`../advanced/distributed_training`.

Requires setting up NCCL and configuring mesh shape.

**Can I use quantization to reduce memory?**

Not currently. LoRA already reduces parameters significantly.

Could be added as feature.

**How do I profile training?**

Enable profiling::

    python run_training.py training.profile=true

Check output directory for profiling results.

**Can I use Weights & Biases?**

Yes, enabled by default. Disable::

    python run_training.py wandb_disabled=true

Set up credentials::

    wandb login

**Can I export the model for inference?**

Yes. Save LoRA weights from checkpoint.

See evaluation guide for inference options.

Performance and Optimization
----------------------------

**How do I speed up training?**

1. Use larger batch size (if memory allows)
2. Reduce sequence length
3. Use fewer generations per prompt
4. Reduce gradient accumulation steps

**How do I reduce memory usage?**

See :doc:`../guide/training` Memory Optimization section:

1. Reduce batch size
2. Use smaller model
3. Reduce LoRA rank
4. Shorten sequences
5. Reduce generations

**Can I use mixed precision (fp16, bf16)?**

Not currently enabled by default. Could be added.

**Should I save every checkpoint?**

Only save important ones to save disk space::

    python run_training.py training.save_interval_steps=1000

Or use checkpointing (saves every N steps to limited slots).

Troubleshooting
---------------

**Training stops with NaN loss**

See :doc:`../advanced/troubleshooting` NaN Loss section.

Usually caused by:

- Learning rate too high
- Gradient overflow
- Bad data example

**Training is very slow**

Check GPU utilization::

    watch -n 1 nvidia-smi

Solutions in :doc:`../advanced/troubleshooting` Slow Training section.

**I can't find my checkpoint**

Find all checkpoints::

    find . -path "*/checkpoints/*" -name "*.pt"

Or use specific path::

    python evaluate.py checkpoint_dir=/full/path/to/checkpoints/

**My model accuracy is poor**

See :doc:`../advanced/troubleshooting` Model Not Improving section.

Usually need:

- More training data
- Better reward function
- Longer training
- Tuned hyperparameters

**I'm getting out of memory errors**

See Memory requirements section and follow incremental reduction guide in troubleshooting.

Still Can't Find Answer?
------------------------

Check:

1. :doc:`../advanced/troubleshooting` - Detailed troubleshooting guide
2. :doc:`../guide/training` - Training guide
3. :doc:`../getting_started/configuration` - Configuration reference
4. API reference docs for specific modules

Next Steps
----------

- :doc:`../guide/training` - Training guide
- :doc:`../advanced/troubleshooting` - Troubleshooting guide
- :doc:`../getting_started/configuration` - Configuration reference
