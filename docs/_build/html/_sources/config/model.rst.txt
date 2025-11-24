Model Configuration Reference
=============================

This section details all model configuration options and available models.

Available Models
----------------

Gemma3 270M
~~~~~~~~~~~

Lightweight model for constrained environments.

Configuration file: ``conf/model/gemma3_270m.yaml``

::

    model_family: gemma3
    model_size: 270m
    lora_rank: 32
    lora_alpha: 32.0
    lora_module_path: ".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|.*attn_vec_einsum"
    mesh_shape: [[1, 1], ["fsdp", "tp"]]

Use::

    python run_training.py model=gemma3_270m

**Specifications**:

- Parameters: 270 million
- Memory: ~11GB with batch size 1
- Recommended GPU: RTX 2080 Ti, RTX A4000
- LoRA rank: 8-32
- Training speed: ~500 steps/hour

**Good for**:

- Testing setups
- Running on limited GPUs
- Quick prototyping
- Small datasets

Gemma3 1B
~~~~~~~~~

Standard model balancing performance and efficiency.

Configuration file: ``conf/model/gemma3_1b.yaml``

::

    model_family: gemma3
    model_size: 1b
    lora_rank: 32
    lora_alpha: 32.0
    lora_module_path: ".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|.*attn_vec_einsum"
    mesh_shape: [[1, 1], ["fsdp", "tp"]]

Use::

    python run_training.py model=gemma3_1b

**Specifications**:

- Parameters: 1 billion
- Memory: ~24GB with batch size 2, ~48GB with batch size 4
- Recommended GPU: RTX A6000, L40S, A100-40GB
- LoRA rank: 16-64
- Training speed: ~300 steps/hour

**Good for**:

- Production training
- Balanced quality/speed
- Most use cases
- Benchmarking

Gemma3 4B
~~~~~~~~~

Larger model for higher capacity tasks.

Configuration file: ``conf/model/gemma3_4b.yaml``

::

    model_family: gemma3
    model_size: 4b
    lora_rank: 64
    lora_alpha: 64.0
    lora_module_path: ".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|.*attn_vec_einsum"
    mesh_shape: [[1, 1], ["fsdp", "tp"]]

Use::

    python run_training.py model=gemma3_4b

**Specifications**:

- Parameters: 4 billion
- Memory: ~80GB with batch size 8 (single GPU)
- Recommended: H100 or multiple A100s
- LoRA rank: 32-128
- Training speed: ~150 steps/hour

**Good for**:

- Complex reasoning tasks
- Large datasets
- Multi-GPU setups
- High-quality models

Model Configuration Parameters
------------------------------

**model_family**

Type: ``string``

Default: ``gemma3``

Model architecture family. Currently only ``gemma3`` supported.

Example::

    model_family: gemma3

**model_size**

Type: ``string``

Options: ``270m``, ``1b``, ``4b``

Default: ``270m``

Model size variant.

Example::

    model_size: 1b

**lora_rank**

Type: ``integer``

Range: ``4`` to ``128``

Default: ``32``

Rank of LoRA matrices. Higher rank = more capacity but more memory.

Recommended values::

    - 270M model: 8-32
    - 1B model: 16-64
    - 4B model: 32-128

Memory impact::

    Memory ≈ baseline_memory × (1 + 2 × lora_rank / hidden_dim)

For 1B model with hidden_dim=2048::

    - rank 16: +1.6% memory
    - rank 32: +3.1% memory
    - rank 64: +6.3% memory

Example::

    lora_rank: 64

**lora_alpha**

Type: ``float``

Default: equals lora_rank

Scaling factor for LoRA. Usually equals ``lora_rank``.

Affects training dynamics:

- Higher alpha: stronger LoRA updates
- Lower alpha: weaker LoRA updates

Typically::

    lora_alpha: ${model.lora_rank}

Or set manually::

    lora_alpha: 16.0

**lora_module_path**

Type: ``string`` (regex pattern)

Regular expression matching layer names to apply LoRA.

Default for Gemma3::

    lora_module_path: ".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|.*attn_vec_einsum"

This applies LoRA to:

- Attention query/key/value projections
- MLP gate and projection layers

To apply LoRA to all layers (not recommended, memory intensive)::

    lora_module_path: ".*"

To apply LoRA only to attention::

    lora_module_path: ".*einsum"

To apply LoRA only to MLP::

    lora_module_path: ".*proj"

**mesh_shape**

Type: ``list[list]`` with dimension names

Default: ``[[1, 1], ["fsdp", "tp"]]``

Parallelism configuration for distributed training.

Format: ``[[num_devices_fsdp, num_devices_tp], ["fsdp", "tp"]]``

Where:

- **num_devices_fsdp**: GPUs for fully sharded data parallelism
- **num_devices_tp**: GPUs for tensor parallelism

Single GPU::

    mesh_shape: [[1, 1], ["fsdp", "tp"]]

Data parallel (4 GPUs)::

    mesh_shape: [[4, 1], ["fsdp", "tp"]]

Tensor parallel (4 GPUs)::

    mesh_shape: [[1, 4], ["fsdp", "tp"]]

Hybrid (8 GPUs, 2 data × 4 tensor)::

    mesh_shape: [[2, 4], ["fsdp", "tp"]]

See :doc:`../advanced/distributed_training` for details.

Complete Model Configuration Example
-------------------------------------

::

    # conf/model/custom.yaml
    model_family: gemma3
    model_size: 1b
    lora_rank: 64
    lora_alpha: 64.0
    lora_module_path: ".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|.*attn_vec_einsum"
    mesh_shape: [[1, 1], ["fsdp", "tp"]]

Use::

    python run_training.py model=custom

Memory Requirements by Configuration
-------------------------------------

RTX 2080 Ti (11GB VRAM)
~~~~~~~~~~~~~~~~~~~~~~~~

::

    model=gemma3_270m
    model.lora_rank: 8
    training.micro_batch_size: 1

RTX A6000 (48GB VRAM)
~~~~~~~~~~~~~~~~~~~~~

::

    model=gemma3_1b
    model.lora_rank: 32
    training.micro_batch_size: 4

H100 (80GB VRAM)
~~~~~~~~~~~~~~~~

::

    model=gemma3_4b
    model.lora_rank: 64
    training.micro_batch_size: 8

Multi-GPU (4× A100 80GB)
~~~~~~~~~~~~~~~~~~~~~~~~

::

    model=gemma3_4b
    model.lora_rank: 128
    model.mesh_shape: [[2, 2], ["fsdp", "tp"]]
    training.micro_batch_size: 8

Tuning LoRA Rank
----------------

**Finding Right Rank**

Start with default (32 for 1B) and adjust based on:

1. **Memory constraints**::

       # If OOM
       model.lora_rank=16

2. **Training quality**::

       # If poor performance
       model.lora_rank=64

3. **Speed/memory trade-off**::

       # Balance training speed and capacity
       model.lora_rank=32  # Default good balance

**Testing Different Ranks**

Create experiment to sweep ranks::

    # conf/experiment/rank_sweep.yaml
    # @package _global_
    training:
      num_batches: 100

Run::

    python run_training.py +experiment=rank_sweep \
        --multirun model.lora_rank=8,16,32,64

Compare metrics to find optimal rank.

Creating Custom Models
---------------------

To add support for a new model:

1. Create config file ``conf/model/newmodel.yaml``::

       model_family: newmodel
       model_size: 1b
       lora_rank: 32
       lora_alpha: 32.0
       lora_module_path: ".*pattern_matching_layers"
       mesh_shape: [[1, 1], ["fsdp", "tp"]]

2. Update code to load model (if needed)
3. Update tokenizer path if different

Then use::

    python run_training.py model=newmodel

Integration with Training
-------------------------

Model config integrates with training via:

1. **LoRA**: Only ``lora_rank``, ``lora_alpha``, ``lora_module_path`` matter for fine-tuning
2. **Distributed training**: ``mesh_shape`` controls parallelism
3. **Memory**: ``model_size`` + ``lora_rank`` determine memory usage

Optimal configuration depends on:

- Available GPU memory
- Training data size
- Time constraints
- Target model quality

Next Steps
----------

- :doc:`overview` - Configuration overview
- :doc:`../getting_started/configuration` - Configuration guide
- :doc:`../api/models` - Model API reference
- :doc:`../advanced/distributed_training` - Distributed training setup
