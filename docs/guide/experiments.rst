Experiments Guide
=================

What are Experiments?
---------------------

Experiments are configuration presets that combine multiple settings for specific scenarios. They're stored in ``conf/experiment/`` as YAML files.

Using Built-in Experiments
---------------------------

**Quick Test**

Fast test with 10 steps, reduced configuration::

    python run_training.py +experiment=quick_test

Good for:

- Testing setup
- Validating data pipeline
- Debugging configuration

**Full Training**

Production settings with default parameters::

    python run_training.py +experiment=full_training

Good for:

- Real training runs
- Benchmarking
- Production models

Creating Custom Experiments
---------------------------

Create ``conf/experiment/my_exp.yaml``::

    # @package _global_
    # This is an experiment configuration

    defaults:
      - override /model: gemma3_1b
      - override /optimizer: adamw

    # Override specific values
    training:
      num_batches: 100
      micro_batch_size: 2

    optimizer:
      learning_rate: 1e-5

    # Metadata
    experiment_name: my_experiment
    tags: [custom, testing]

Use it::

    python run_training.py +experiment=my_exp

Experiment Structure
--------------------

**Package Declaration**

``# @package _global_`` is required and tells the system to merge experiment into global config.

**Defaults List**

Override default configurations::

    defaults:
      - override /model: gemma3_1b
      - override /optimizer: adamw

**Configuration Overrides**

Specific value overrides::

    training:
      num_batches: 100
      micro_batch_size: 4

    grpo:
      num_generations: 8

**Metadata** (optional)

Add tags for organization::

    experiment_name: learning_rate_ablation
    tags: [ablation, hyperparameter]
    description: Ablating learning rates from 1e-7 to 1e-4

Example Experiments
-------------------

**Memory-Efficient Training**

``conf/experiment/low_memory.yaml``::

    # @package _global_

    defaults:
      - override /model: gemma3_270m

    model:
      lora_rank: 8

    training:
      micro_batch_size: 1

    grpo:
      num_generations: 2

Use::

    python run_training.py +experiment=low_memory

**Large Batch Training**

``conf/experiment/large_batch.yaml``::

    # @package _global_

    defaults:
      - override /model: gemma3_1b

    model:
      lora_rank: 64

    training:
      micro_batch_size: 8

    optimizer:
      learning_rate: 1e-4

Use::

    python run_training.py +experiment=large_batch

**Ablation Study**

``conf/experiment/lr_ablation.yaml``::

    # @package _global_

    training:
      num_batches: 50

    experiment_name: learning_rate_ablation
    tags: [ablation, lr]

Then sweep::

    python run_training.py +experiment=lr_ablation --multirun \
        optimizer.learning_rate=1e-7,1e-6,1e-5,1e-4

Experiment Workflow
-------------------

**1. Design Experiment**

Create YAML file with settings.

**2. Test Configuration**

Verify config before running::

    python run_training.py +experiment=my_exp --cfg job

**3. Run Experiment**

Execute the experiment::

    python run_training.py +experiment=my_exp

**4. Evaluate Results**

Check metrics and logs::

    python evaluate.py checkpoint_dir=./checkpoints/ckpts/

**5. Document Results**

Record findings and best settings for future reference.

Combining Experiments with Sweeps
----------------------------------

Run an experiment with parameter sweep::

    python run_training.py +experiment=my_exp --multirun \
        optimizer.learning_rate=1e-6,3e-6,1e-5

This runs the experiment configuration with three different learning rates.

Advanced: Experiment Hierarchies
--------------------------------

Create base experiments and extend them::

**Base experiment** ``conf/experiment/base.yaml``::

    # @package _global_

    defaults:
      - override /model: gemma3_1b

    training:
      num_batches: 100

**Extended experiment** ``conf/experiment/extended.yaml``::

    # @package _global_

    defaults:
      - base

    training:
      num_batches: 500
      micro_batch_size: 4

Naming Conventions
------------------

Recommend consistent naming:

- **Descriptive**: ``lr_ablation``, ``batch_size_study``
- **Dated**: ``2024_01_experiment``
- **Tagged**: ``ablation_lr``, ``benchmark_v2``

Examples::

    conf/experiment/
    ├── quick_test.yaml
    ├── full_training.yaml
    ├── ablation_lr.yaml
    ├── ablation_batch_size.yaml
    ├── benchmark_v1.yaml
    ├── benchmark_v2.yaml
    └── memory_constrained.yaml

Best Practices
--------------

1. **Document each experiment**: Add description and tags
2. **Keep experiments reproducible**: Don't change configs after creating experiment
3. **Version experiments**: Use dates or version numbers
4. **Clean naming**: Use descriptive, self-explanatory names
5. **Track results**: Document findings with each experiment
6. **Store important configs**: Keep successful experiments in version control

Managing Experiments
--------------------

**List all available experiments**::

    python run_training.py --info config-groups | grep experiment

**Show experiment config**::

    python run_training.py +experiment=my_exp --cfg job

**Compare experiments**

Run multiple and compare results in W&B or TensorBoard.

Next Steps
----------

- :doc:`Hyperparameter Tuning </guide/hyperparameter_tuning>`
- :doc:`Training Guide </guide/training>`
- :doc:`Configuration Guide </getting_started/configuration>`
