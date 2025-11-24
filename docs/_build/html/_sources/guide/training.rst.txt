Training Guide
==============

Overview
--------

Agent-Tunix implements GRPO (Group Relative Policy Optimization) training for language models with parameter-efficient fine-tuning via LoRA.

Basic Training
--------------

Start training with default configuration::

    python run_training.py

This will:

1. Load the reference model (frozen)
2. Create policy model with LoRA
3. Run GRPO training loop
4. Save checkpoints periodically

Training Process
----------------

The training process involves:

**1. Data Loading**

- Loads GSM8K dataset (grade school math)
- Splits into train/validation/test sets
- Tokenizes with model's tokenizer

**2. Model Setup**

- Loads base model from Kaggle
- Applies LoRA to specified layers
- Initializes reference and policy models

**3. GRPO Training**

For each training step:

- Generate multiple responses per prompt
- Compute rewards (correctness + format)
- Calculate policy gradient with KL penalty
- Update policy model with gradient

**4. Evaluation**

- Periodically evaluate on validation set
- Compute metrics (accuracy, format accuracy)
- Log to Weights & Biases

**5. Checkpointing**

- Save model checkpoints
- Save training logs
- Save configuration for reproducibility

Training Configuration
----------------------

Key hyperparameters::

    # Model
    model.lora_rank: 32              # LoRA rank
    model.model_size: 270m           # Model size

    # Training
    training.micro_batch_size: 4     # Batch per device
    training.num_batches: 3738       # Training batches
    training.num_epochs: 1           # Epochs

    # Optimization
    optimizer.learning_rate: 3e-6    # Peak learning rate
    optimizer.warmup_ratio: 0.1      # Warmup as % of total
    optimizer.max_grad_norm: 0.1     # Gradient clipping

    # GRPO
    grpo.num_generations: 4          # Responses per prompt
    grpo.beta: 0.08                  # KL divergence weight
    grpo.epsilon: 0.2                # PPO clipping

    # Generation
    generation.max_generation_steps: 512
    generation.temperature: 0.9
    generation.top_k: 50
    generation.top_p: 1.0

Monitoring Training
-------------------

**View logs**::

    tail -f outputs/tunix-grpo/YYYY-MM-DD/HH-MM-SS/train.log

**Weights & Biases**

Training logs are sent to W&B by default. View at: https://wandb.ai

To disable::

    python run_training.py wandb_disabled=true

**TensorBoard**

View training metrics::

    make tensorboard

Then open http://localhost:6006

Memory Optimization
-------------------

If training runs out of memory, try:

**Reduce batch size**::

    python run_training.py training.micro_batch_size=1

**Use smaller model**::

    python run_training.py model=gemma3_270m

**Reduce LoRA rank**::

    python run_training.py model.lora_rank=8

**Shorter sequences**::

    python run_training.py generation.max_prompt_length=128 generation.max_generation_steps=256

**Reduce number of generations**::

    python run_training.py grpo.num_generations=2

Distributed Training
--------------------

For multi-GPU training, modify mesh shape in model config::

    python run_training.py model.mesh_shape=[[2,2],["fsdp","tp"]]

See :doc:`Distributed Training Guide </advanced/distributed_training>` for details.

Resuming Training
-----------------

Resume from latest checkpoint::

    python run_training.py checkpoint_dir=./checkpoints/ckpts/

The framework automatically detects and continues from the latest checkpoint.

Custom Reward Functions
-----------------------

Modify reward functions in ``src/agent_tunix/rewards.py``:

- ``match_format_exactly``: Rewards format compliance
- ``check_answer``: Rewards correct answers
- ``check_numbers``: Extracts and validates numbers

See :doc:`Custom Rewards </advanced/custom_rewards>` for custom implementations.

Troubleshooting
---------------

**Out of Memory**

Reduce batch size or model size as shown in Memory Optimization section.

**Slow Training**

Check GPU utilization with::

    nvidia-smi -l 1

If utilization is low, data loading may be the bottleneck.

**NaN Loss**

May indicate:

- Learning rate too high: reduce with ``optimizer.learning_rate=1e-6``
- Gradient overflow: reduce ``optimizer.max_grad_norm``
- Data issue: inspect training data

**Model Not Improving**

Check:

- Learning rate too low
- Insufficient training data
- Reward function not properly calibrated

Tips and Best Practices
-----------------------

1. **Start small**: Use ``+experiment=quick_test`` first
2. **Monitor metrics**: Check logs and W&B frequently
3. **Save often**: Use ``training.save_interval_steps=50``
4. **Document runs**: Add tags to experiments in config
5. **Validate early**: Evaluate on validation set frequently
6. **Checkpoint management**: Keep important checkpoints

Example Training Runs
---------------------

**Quick Test** (10 steps, reduced)::

    python run_training.py +experiment=quick_test

**Single GPU Training** (270M model)::

    python run_training.py model=gemma3_270m training.micro_batch_size=1

**Ablation Study** (sweep learning rates)::

    python run_training.py --multirun optimizer.learning_rate=1e-6,3e-6,1e-5

**Production Run** (1B model, full training)::

    python run_training.py model=gemma3_1b +experiment=full_training

Next Steps
----------

- :doc:`Hyperparameter Tuning </guide/hyperparameter_tuning>`
- :doc:`Model Evaluation </guide/evaluation>`
- :doc:`Troubleshooting </advanced/troubleshooting>`
