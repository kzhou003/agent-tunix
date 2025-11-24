Optimizer Configuration Reference
==================================

This section details optimizer configuration options and strategies.

Available Optimizers
--------------------

AdamW (Adam with Decoupled Weight Decay)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default optimizer for Agent-Tunix.

Configuration file: ``conf/optimizer/adamw.yaml``

::

    optimizer_name: adamw
    learning_rate: 3e-6
    weight_decay: 0.01
    betas: [0.9, 0.999]
    eps: 1e-8
    warmup_ratio: 0.1
    max_grad_norm: 0.1

Use::

    python run_training.py optimizer=adamw

**Why AdamW?**

- Adaptive learning rates per parameter
- Decoupled weight decay (correct L2 regularization)
- Good convergence properties
- Standard in modern deep learning

Optimizer Configuration Parameters
-----------------------------------

**optimizer_name**

Type: ``string``

Default: ``adamw``

Optimizer algorithm name.

Example::

    optimizer_name: adamw

**learning_rate**

Type: ``float``

Range: ``1e-7`` to ``1e-4`` (typical)

Default: ``3e-6``

Controls step size in gradient descent. Critical hyperparameter.

Guidance by model size::

    - 270M model: 1e-5 to 3e-5
    - 1B model: 1e-6 to 1e-5
    - 4B model: 1e-6 to 3e-6

Too high (divergence)::

    loss → NaN or ∞
    Solution: reduce by 10× (e.g., 1e-5 → 1e-6)

Too low (slow training)::

    loss decreases very slowly
    Solution: increase by 10× (e.g., 1e-7 → 1e-6)

Example::

    python run_training.py optimizer.learning_rate=1e-5

**weight_decay**

Type: ``float``

Range: ``0.0`` to ``0.1``

Default: ``0.01``

L2 regularization strength. Penalizes large weights.

Effects:

- Higher: stronger regularization, less overfitting
- Lower: more flexibility, potential overfitting

Typical values::

    - No regularization needed: 0.0
    - Standard: 0.01
    - Strong regularization: 0.05-0.1

Example::

    python run_training.py optimizer.weight_decay=0.05

**betas**

Type: ``list[float, float]``

Default: ``[0.9, 0.999]``

Exponential moving average coefficients for gradient moments.

Format: ``[beta1, beta2]``

- **beta1** (momentum): controls first moment exponential moving average
- **beta2** (second moment): controls second moment exponential moving average

Typical values::

    - Standard: [0.9, 0.999]
    - Aggressive (faster adaptation): [0.95, 0.99]
    - Conservative (smoother): [0.8, 0.999]

Default usually works well. Change only if:

- Training unstable → try [0.95, 0.99]
- Converging too slowly → try [0.8, 0.999]

Example::

    python run_training.py 'optimizer.betas=[0.95,0.99]'

**eps**

Type: ``float``

Default: ``1e-8``

Small value to prevent division by zero in adaptive learning rates.

Rarely needs adjustment. Only increase if numerical instability::

    python run_training.py optimizer.eps=1e-6

**warmup_ratio**

Type: ``float``

Range: ``0.0`` to ``1.0``

Default: ``0.1``

Fraction of training steps devoted to warmup.

Effect: Learning rate gradually increases from 0 to peak during warmup.

Benefits::

    - Stabilizes early training
    - Prevents gradient explosion
    - Improves final model quality

Common values::

    - No warmup: 0.0
    - Light warmup (5%): 0.05
    - Standard (10%): 0.1
    - Strong warmup (20%): 0.2

For short training (quick_test)::

    python run_training.py +experiment=quick_test optimizer.warmup_ratio=0.0

Example::

    python run_training.py optimizer.warmup_ratio=0.2

**max_grad_norm**

Type: ``float``

Range: ``0.01`` to ``1.0``

Default: ``0.1``

Gradient clipping threshold. Limits gradient magnitude to prevent exploding gradients.

Effect: If ||gradient|| > max_grad_norm, scale down to threshold.

When needed::

    - NaN/Inf loss → reduce to 0.01
    - Unstable training → reduce to 0.05
    - Smooth training → 0.1 (default)

Example::

    python run_training.py optimizer.max_grad_norm=0.05

Learning Rate Scheduling
------------------------

Combined with ``scheduler`` configuration::

    # conf/scheduler/warmup_cosine.yaml
    scheduler_name: warmup_cosine
    warmup_steps: null  # Computed from warmup_ratio
    total_steps: null   # Computed from num_batches
    lr_min: 1e-7

Default schedule: Warmup → Cosine decay

Warmup phase::

    lr(t) = learning_rate × (t / warmup_steps)

Cosine decay phase::

    lr(t) = lr_min + 0.5 × (lr_peak - lr_min) × (1 + cos(π × progress))

This provides:

- Stability in early training (warmup)
- Gradual cooling for convergence (cosine)

Optimizer Tuning Workflow
--------------------------

**Step 1: Find Baseline Learning Rate**

Quick search with small dataset::

    python run_training.py +experiment=quick_test \
        --multirun optimizer.learning_rate=1e-7,1e-6,1e-5,1e-4

Monitor training loss curves. Pick best.

**Step 2: Fine-tune Around Best Learning Rate**

Narrow range around best from step 1::

    python run_training.py --multirun \
        optimizer.learning_rate=1e-6,3e-6,1e-5,3e-5

**Step 3: Tune Warmup Ratio**

Try different warmup values::

    python run_training.py \
        optimizer.learning_rate=3e-6 \
        --multirun optimizer.warmup_ratio=0.05,0.1,0.2

**Step 4: Tune Weight Decay**

Reduce if overfitting, increase if underfitting::

    python run_training.py \
        optimizer.learning_rate=3e-6 \
        --multirun optimizer.weight_decay=0.0,0.01,0.05

**Step 5: Full Training**

Use best parameters::

    python run_training.py \
        optimizer.learning_rate=3e-6 \
        optimizer.warmup_ratio=0.1 \
        optimizer.weight_decay=0.01

Common Configuration Examples
-----------------------------

**Conservative (stable, slow)**

::

    optimizer:
      learning_rate: 1e-6
      warmup_ratio: 0.2
      weight_decay: 0.05
      max_grad_norm: 0.05

Use::

    python run_training.py \
        optimizer.learning_rate=1e-6 \
        optimizer.warmup_ratio=0.2 \
        optimizer.weight_decay=0.05 \
        optimizer.max_grad_norm=0.05

**Balanced (recommended)**

::

    optimizer:
      learning_rate: 3e-6
      warmup_ratio: 0.1
      weight_decay: 0.01
      max_grad_norm: 0.1

Use::

    python run_training.py optimizer=adamw  # Uses defaults

**Aggressive (fast, risky)**

::

    optimizer:
      learning_rate: 1e-5
      warmup_ratio: 0.05
      weight_decay: 0.0
      max_grad_norm: 0.1

Use::

    python run_training.py \
        optimizer.learning_rate=1e-5 \
        optimizer.warmup_ratio=0.05 \
        optimizer.weight_decay=0.0

Diagnosing Optimizer Issues
----------------------------

**Loss Diverging (NaN/Inf)**

Cause: Learning rate too high

Solution::

    python run_training.py optimizer.learning_rate=1e-7
    python run_training.py optimizer.max_grad_norm=0.01

**Loss Not Decreasing**

Cause: Learning rate too low or model not training

Solution::

    python run_training.py optimizer.learning_rate=1e-4

**Oscillating Loss (high variance)**

Cause: Learning rate borderline, warmup insufficient

Solution::

    python run_training.py \
        optimizer.learning_rate=1e-6 \
        optimizer.warmup_ratio=0.2

**Slow Convergence**

Cause: Learning rate too low or weight decay too high

Solution::

    python run_training.py \
        optimizer.learning_rate=1e-5 \
        optimizer.weight_decay=0.0

Complete Optimizer Configuration Example
----------------------------------------

::

    # conf/optimizer/custom.yaml
    optimizer_name: adamw
    learning_rate: 5e-6
    weight_decay: 0.02
    betas: [0.9, 0.999]
    eps: 1e-8
    warmup_ratio: 0.15
    max_grad_norm: 0.1

Use::

    python run_training.py optimizer=custom

Or override directly::

    python run_training.py \
        optimizer.learning_rate=5e-6 \
        optimizer.warmup_ratio=0.15

Interaction with Other Settings
--------------------------------

Optimizer settings interact with:

1. **Model size**: Larger models need lower learning rates
2. **Batch size**: Larger batches can use higher learning rates
3. **LoRA rank**: Doesn't directly affect learning rate
4. **Data size**: Smaller datasets need lower learning rates

Example adjustment::

    # Large model, small batch → lower LR
    model=gemma3_4b \
    training.micro_batch_size=1 \
    optimizer.learning_rate=1e-6

    # Small model, large batch → higher LR
    model=gemma3_270m \
    training.micro_batch_size=8 \
    optimizer.learning_rate=1e-4

Next Steps
----------

- :doc:`overview` - Configuration overview
- :doc:`../guide/hyperparameter_tuning` - Tuning strategies
- :doc:`../getting_started/configuration` - Configuration guide
- :doc:`../api/train` - Training API reference
