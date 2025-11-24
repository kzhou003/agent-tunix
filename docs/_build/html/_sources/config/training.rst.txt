Training Configuration Reference
=================================

This section details training configuration options for controlling the training process.

Training Configuration File
---------------------------

Located at: ``conf/training/default.yaml``

::

    micro_batch_size: 4
    num_batches: 3738
    num_epochs: 1
    checkpoint_dir: ./checkpoints/ckpts/
    save_interval_steps: 100
    eval_interval_steps: 500
    log_interval_steps: 10
    seed: 42
    device: cuda

Training Configuration Parameters
---------------------------------

**micro_batch_size**

Type: ``integer``

Range: ``1`` to ``16`` (depending on GPU)

Default: ``4``

Batch size per GPU device.

Memory impact: Linear (2× batch size ≈ 2× memory)

Guidance by GPU::

    - 11GB GPU (RTX 2080 Ti): 1
    - 24GB GPU (RTX A4000): 2-4
    - 48GB GPU (RTX A6000): 4-8
    - 80GB GPU (H100): 8-16

Training dynamics::

    - Larger batches: more stable gradients, slower per-step updates
    - Smaller batches: noisier gradients, faster per-step updates

Typical ranges::

    # Memory constrained
    python run_training.py training.micro_batch_size=1

    # Balanced
    python run_training.py training.micro_batch_size=4

    # High capacity
    python run_training.py training.micro_batch_size=8

**num_batches**

Type: ``integer``

Range: ``10`` to ``100000``

Default: ``3738``

Total number of training batches (steps).

Training duration::

    duration ≈ num_batches / (batches_per_hour)

Typical speeds::

    - 270M model: ~500 steps/hour
    - 1B model: ~300 steps/hour
    - 4B model: ~150 steps/hour

Examples::

    # Quick test (10 minutes)
    training.num_batches=50

    # Full training (varies)
    training.num_batches=3738

    # Long training (1-2 days)
    training.num_batches=10000

Relationship with num_epochs::

    total_steps = (dataset_size / batch_size) × num_epochs
    # But num_batches directly specifies total steps

**num_epochs**

Type: ``integer``

Range: ``1`` to ``10``

Default: ``1``

Number of complete passes through dataset.

Typical::

    - Most tasks: 1 epoch
    - Small datasets: 3-5 epochs
    - Large datasets: 1 epoch

Usually keep at 1 and adjust ``num_batches`` instead::

    # Good: directly specify steps
    training.num_batches=3738

    # Less common: use epochs
    training.num_epochs=2

**checkpoint_dir**

Type: ``string`` (path)

Default: ``./checkpoints/ckpts/``

Directory where model checkpoints saved.

Must be writable directory. Created if doesn't exist.

Absolute vs relative::

    # Relative to project root
    training.checkpoint_dir=./checkpoints/ckpts/

    # Absolute path
    training.checkpoint_dir=/full/path/to/checkpoints/ckpts/

Example::

    python run_training.py checkpoint_dir=~/my_models/checkpoints/

**save_interval_steps**

Type: ``integer``

Range: ``1`` to ``1000``

Default: ``100``

Save checkpoint every N steps.

Trade-offs::

    - Frequent saves (50): more disk, better coverage, can resume from recent step
    - Infrequent saves (1000): less disk, fewer checkpoints, coarser resume points

Disk usage estimate::

    disk = (checkpoint_size) × (num_batches / save_interval_steps)

For 1B model (≈4GB checkpoint)::

    - save_interval=50: 4GB × (3738/50) ≈ 300GB
    - save_interval=100: 4GB × (3738/100) ≈ 150GB
    - save_interval=500: 4GB × (3738/500) ≈ 30GB

Example::

    python run_training.py training.save_interval_steps=200

**eval_interval_steps**

Type: ``integer``

Range: ``100`` to ``5000``

Default: ``500``

Evaluate on validation set every N steps.

Lower = more frequent evaluation::

    - Every 100 steps: frequent feedback, slower training
    - Every 500 steps: good balance (default)
    - Every 1000 steps: less frequent, faster training

Example::

    python run_training.py training.eval_interval_steps=1000

**log_interval_steps**

Type: ``integer``

Range: ``1`` to ``100``

Default: ``10``

Log metrics every N steps.

Determines frequency of logged stats::

    - Every 1 step: very verbose, can slow training
    - Every 10 steps: good visibility (default)
    - Every 100 steps: less detailed

Example::

    python run_training.py training.log_interval_steps=5

**seed**

Type: ``integer``

Default: ``42``

Random seed for reproducibility.

Same seed = reproducible results across runs.

For different runs::

    python run_training.py seed=42
    python run_training.py seed=43
    python run_training.py seed=44

Reproducibility::

    python run_training.py seed=42  # Run 1
    python run_training.py seed=42  # Run 2 (identical to Run 1)

**device**

Type: ``string``

Options: ``cuda``, ``cpu``

Default: ``cuda``

Training device.

GPU (recommended)::

    python run_training.py device=cuda

CPU (very slow, for testing)::

    python run_training.py device=cpu

Memory Optimization Parameters
------------------------------

**gradient_accumulation_steps**

Type: ``integer``

Default: ``1``

Number of gradient accumulation steps before update.

Effectively increases batch size without more GPU memory::

    effective_batch_size = micro_batch_size × gradient_accumulation_steps

Example::

    # Effective batch size of 8 with 2GB GPU
    training.micro_batch_size=2
    training.gradient_accumulation_steps=4

**max_grad_norm** (in optimizer config)

Type: ``float``

Default: ``0.1``

Gradient clipping threshold.

Larger value = less clipping, more aggressive updates::

    - 0.01: strong clipping, stable but slow
    - 0.1: moderate clipping (default)
    - 1.0: weak clipping, aggressive

Complete Training Configuration Example
---------------------------------------

::

    # conf/training/default.yaml
    micro_batch_size: 4
    num_batches: 3738
    num_epochs: 1
    checkpoint_dir: ./checkpoints/ckpts/
    save_interval_steps: 100
    eval_interval_steps: 500
    log_interval_steps: 10
    seed: 42
    device: cuda

Custom training config::

    # conf/training/aggressive.yaml
    micro_batch_size: 8
    num_batches: 1000
    num_epochs: 1
    checkpoint_dir: ./checkpoints/ckpts/
    save_interval_steps: 50
    eval_interval_steps: 100
    log_interval_steps: 5
    seed: 42
    device: cuda

Use::

    python run_training.py training=aggressive

Common Configuration Patterns
-----------------------------

**Quick Test (10 steps)**

::

    python run_training.py \
        training.micro_batch_size=2 \
        training.num_batches=10 \
        training.save_interval_steps=10 \
        training.eval_interval_steps=10

Or use experiment::

    python run_training.py +experiment=quick_test

**Memory Constrained (11GB GPU)**

::

    python run_training.py \
        model=gemma3_270m \
        training.micro_batch_size=1 \
        training.num_batches=3738

**High Performance (H100 GPU)**

::

    python run_training.py \
        model=gemma3_4b \
        training.micro_batch_size=8 \
        training.num_batches=10000 \
        training.eval_interval_steps=1000

**Production Training**

::

    python run_training.py \
        model=gemma3_1b \
        training.micro_batch_size=4 \
        training.num_batches=10000 \
        training.save_interval_steps=100

Resuming Training
-----------------

Automatically resumes from latest checkpoint::

    python run_training.py checkpoint_dir=./checkpoints/ckpts/

Or from specific directory::

    python run_training.py checkpoint_dir=/path/to/checkpoints/ckpts/

The framework:

1. Finds latest checkpoint in directory
2. Loads model weights
3. Continues training from that step

Training Dynamics
-----------------

How configuration affects training:

1. **Learning Rate + Batch Size**::

       Larger batch → can use higher learning rate
       Smaller batch → need lower learning rate

2. **Warmup + Learning Rate**::

       Longer warmup (higher warmup_ratio) → more stable
       Short warmup → faster convergence but less stable

3. **Number of Batches + Evaluation Interval**::

       More batches → longer training, more progress
       Less frequent eval → faster training but less monitoring

4. **LoRA Rank + Learning Rate**::

       Higher rank → more parameters, may need lower LR
       Lower rank → fewer parameters, can use higher LR

Checkpoint Management
---------------------

**Disk Space Required**

::

    total_disk ≈ checkpoint_size × (num_batches / save_interval_steps)

For 1B model (≈4GB)::

    3738 steps, save every 100 steps
    total_disk ≈ 4GB × (3738/100) ≈ 150GB

**Keeping Only Important Checkpoints**

Save less frequently::

    python run_training.py training.save_interval_steps=500

Or manually delete old checkpoints::

    # Keep only last 5 checkpoints
    ls -dt checkpoints/ckpts/actor/*/ | tail -n +6 | xargs rm -rf

**Finding Checkpoint Sizes**

::

    du -sh checkpoints/ckpts/actor/*/

Monitoring Training Progress
-----------------------------

**Check Training Loss**

::

    tail -f outputs/tunix-grpo/YYYY-MM-DD/HH-MM-SS/train.log

**Use Weights & Biases**

Enabled by default. View at https://wandb.ai

**Use TensorBoard**

::

    make tensorboard
    # Open http://localhost:6006

Integration with Other Configs
------------------------------

Training settings interact with:

1. **Model**: Larger models need lower batch sizes
2. **GRPO**: num_generations multiplies memory usage
3. **Optimizer**: Learning rate should match batch size
4. **Scheduler**: warmup_ratio affects convergence

Coordinated tuning example::

    # For 4GB GPU
    python run_training.py \
        model=gemma3_270m \
        model.lora_rank=8 \
        training.micro_batch_size=1 \
        grpo.num_generations=2 \
        optimizer.learning_rate=1e-5

    # For 80GB GPU
    python run_training.py \
        model=gemma3_4b \
        model.lora_rank=64 \
        training.micro_batch_size=8 \
        grpo.num_generations=4 \
        optimizer.learning_rate=3e-6

Next Steps
----------

- :doc:`overview` - Configuration overview
- :doc:`../guide/training` - Training guide
- :doc:`../guide/hyperparameter_tuning` - Tuning strategies
- :doc:`../getting_started/configuration` - Configuration guide
