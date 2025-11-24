Hyperparameter Tuning
=====================

Overview
--------

Hyperparameter tuning is critical for model performance. This guide covers common tuning strategies.

Key Hyperparameters
--------------------

**Learning Rate**

Controls update step size. Too high causes divergence, too low causes slow convergence.

Range: ``1e-7`` to ``1e-4``

Default: ``3e-6``

Sweep::

    python run_training.py --multirun optimizer.learning_rate=1e-7,1e-6,1e-5,1e-4

**Batch Size**

Larger batches provide better gradient estimates but require more memory.

Range: ``1`` to ``8`` (depending on GPU)

Default: ``4``

Sweep::

    python run_training.py --multirun training.micro_batch_size=1,2,4

**LoRA Rank**

Higher rank provides more capacity but requires more memory and computation.

Range: ``4`` to ``64``

Default: ``32``

Sweep::

    python run_training.py --multirun model.lora_rank=8,16,32,64

**Number of Generations**

More generations provide better reward signal but increase computation.

Range: ``1`` to ``8``

Default: ``4``

Sweep::

    python run_training.py --multirun grpo.num_generations=2,4,8

**KL Beta**

Strength of KL divergence penalty. Higher values keep closer to reference model.

Range: ``0.01`` to ``1.0``

Default: ``0.08``

Sweep::

    python run_training.py --multirun grpo.beta=0.01,0.05,0.1,0.5

Tuning Strategies
-----------------

**1. Learning Rate Search**

Find optimal learning rate first::

    python run_training.py --multirun optimizer.learning_rate=1e-7,1e-6,1e-5,1e-4

Monitor loss curves and pick best one.

**2. Batch Size vs Learning Rate**

Larger batches often allow higher learning rates::

    python run_training.py --multirun training.micro_batch_size=1,2,4 optimizer.learning_rate=1e-6,3e-6,1e-5

**3. Model Capacity**

Test different model sizes::

    python run_training.py --multirun model=gemma3_270m,gemma3_1b

**4. Algorithm Parameters**

Tune GRPO-specific parameters::

    python run_training.py --multirun grpo.num_generations=2,4,8 grpo.beta=0.01,0.1,1.0

Monitoring During Tuning
------------------------

**1. Watch logs**::

    tail -f outputs/tunix-grpo/YYYY-MM-DD/HH-MM-SS/train.log

**2. Use Weights & Biases**

Compare runs at: https://wandb.ai

**3. Check tensorboard**::

    make tensorboard

Common Tuning Issues
--------------------

**Loss Not Decreasing**

- Learning rate too low: increase to 1e-5
- Batch size too small: increase to 2 or 4
- Model too small: try 1b model

**Loss Diverging (NaN)**

- Learning rate too high: reduce to 1e-7
- Gradient clipping insufficient: reduce ``max_grad_norm``
- Batch size too large: reduce

**Slow Convergence**

- Learning rate too low
- Batch size too small
- Not enough generations

**Mode Collapse**

Model stops improving. Try:

- Increase diversity: higher temperature
- Modify reward function
- Change LoRA rank

Efficient Search
----------------

**Grid Search**

Systematic search over parameter combinations::

    python run_training.py --multirun \
        optimizer.learning_rate=1e-6,3e-6,1e-5 \
        training.micro_batch_size=1,2,4

**Random Search**

Random sampling of parameter space::

    python run_training.py --multirun \
        optimizer.learning_rate='log_uniform(1e-7,1e-4)' \
        training.micro_batch_size='choice(1,2,4)'

**Early Stopping**

Stop unpromising runs early::

    # Quick test first
    python run_training.py +experiment=quick_test model=gemma3_1b

    # Then full training only for promising configs
    python run_training.py model=gemma3_1b

Memory-Aware Tuning
-------------------

**For 11GB GPU (RTX 2080 Ti)**::

    model.lora_rank: 8-16
    training.micro_batch_size: 1
    model: gemma3_270m
    grpo.num_generations: 2

**For 48GB GPU (RTX A6000)**::

    model.lora_rank: 32-64
    training.micro_batch_size: 4
    model: gemma3_1b
    grpo.num_generations: 4

**For 80GB GPU (H100)**::

    model.lora_rank: 64
    training.micro_batch_size: 8
    model: gemma3_4b
    grpo.num_generations: 8

Example Tuning Workflow
-----------------------

**Phase 1: Learning Rate Search** (1 hour)::

    python run_training.py +experiment=quick_test --multirun \
        optimizer.learning_rate=1e-7,1e-6,1e-5,1e-4

**Phase 2: Batch Size Tuning** (2 hours)::

    python run_training.py --multirun \
        training.micro_batch_size=1,2,4 \
        optimizer.learning_rate=3e-6

**Phase 3: Model Size** (varies)::

    python run_training.py --multirun \
        model=gemma3_270m,gemma3_1b \
        training.micro_batch_size=1

**Phase 4: Final Training**

Use best parameters from previous phases::

    python run_training.py \
        model=gemma3_1b \
        optimizer.learning_rate=1e-5 \
        training.micro_batch_size=2

Best Practices
--------------

1. **Start simple**: Tune one parameter at a time
2. **Use short runs**: Test with ``+experiment=quick_test``
3. **Log everything**: Enable W&B logging
4. **Save results**: Document best configurations
5. **Reproduce winners**: Verify best configs on full runs
6. **Monitor hardware**: Check GPU memory and utilization

Next Steps
----------

- :doc:`Training Guide </guide/training>`
- :doc:`Evaluation Guide </guide/evaluation>`
- :doc:`Configuration Guide </getting_started/configuration>`
