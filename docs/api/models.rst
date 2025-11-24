Models API
==========

.. py:module:: agent_tunix.models
   :noindex:

This module contains model architectures and utilities for loading and configuring models.

Available Models
----------------

**Gemma3 270M**

Lightweight model suitable for resource-constrained environments::

    # Configuration: conf/model/gemma3_270m.yaml
    model_family: gemma3
    model_size: 270m
    lora_rank: 32
    lora_alpha: 32.0

Use::

    python run_training.py model=gemma3_270m

Memory requirements:

- VRAM: ~11GB (with batch size 1)
- Single GPU: RTX 2080 Ti or RTX A4000
- LoRA rank: 8-32 (lower for memory constraints)

**Gemma3 1B**

Standard model for balanced performance and efficiency::

    # Configuration: conf/model/gemma3_1b.yaml
    model_family: gemma3
    model_size: 1b
    lora_rank: 32
    lora_alpha: 32.0

Use::

    python run_training.py model=gemma3_1b

Memory requirements:

- VRAM: ~48GB (with batch size 4)
- Single GPU: RTX A6000
- LoRA rank: 16-64

**Gemma3 4B**

Larger model for higher capacity tasks::

    model_family: gemma3
    model_size: 4b
    lora_rank: 64
    lora_alpha: 64.0

Use::

    python run_training.py model=gemma3_4b

Memory requirements:

- VRAM: ~80GB (with batch size 8)
- Multiple GPUs: H100 or A100
- LoRA rank: 32-128

Model Configuration
-------------------

Key configuration parameters::

    model:
      model_family: gemma3           # Model family name
      model_size: 1b                 # Size variant (270m, 1b, 4b)
      lora_rank: 32                  # LoRA rank for low-rank adaptation
      lora_alpha: 32.0               # LoRA alpha scaling factor
      lora_module_path: ".*pattern"  # Regex for which layers to apply LoRA
      mesh_shape: [[1,4],            # Parallelism shape
                    ["fsdp","tp"]]   # fsdp: fully sharded, tp: tensor parallel

LoRA (Low-Rank Adaptation)
----------------------------

LoRA is a parameter-efficient fine-tuning technique that:

1. Freezes the base model weights
2. Adds low-rank trainable adapters to specified layers
3. Reduces trainable parameters from 100% to ~1%

Configuration::

    lora_rank: 32           # Rank of adaptation matrices (4-128)
    lora_alpha: 32.0        # Scaling factor (usually = rank)

Higher rank = more capacity but more parameters and memory.

Typical settings:

- Memory constrained (11GB): rank 8-16
- Standard (48GB): rank 32-64
- High capacity (80GB+): rank 64-128

Module Selection
~~~~~~~~~~~~~~~~

Control which layers get LoRA::

    lora_module_path: ".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|.*attn_vec_einsum"

This regex pattern applies LoRA to:

- Attention projections (q, k, v einsum operations)
- MLP layers (gate, up, down projections)

Custom Model Configuration
---------------------------

Create a custom model in ``conf/model/custom.yaml``::

    model_family: gemma3
    model_size: 4b
    lora_rank: 16
    lora_alpha: 16.0
    lora_module_path: ".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|.*attn_vec_einsum"
    mesh_shape: [[1, 4], ["fsdp", "tp"]]

Use it::

    python run_training.py model=custom

Distributed Training
--------------------

For multi-GPU training, configure mesh shape::

    # Single GPU
    mesh_shape: [[1, 1], ["fsdp", "tp"]]

    # 2 GPUs in a row (data parallelism)
    mesh_shape: [[2, 1], ["fsdp", "tp"]]

    # 4 GPUs in a 2x2 grid
    mesh_shape: [[2, 2], ["fsdp", "tp"]]

    # 4 GPUs in a line (tensor parallelism)
    mesh_shape: [[1, 4], ["fsdp", "tp"]]

Where:

- **fsdp**: Fully Sharded Data Parallel - shards model across GPUs
- **tp**: Tensor Parallel - splits tensors across GPUs

Model Loading
-------------

Models are loaded from:

1. **Hugging Face Hub**: For public models
2. **Kaggle Models**: For Kaggle-hosted weights
3. **Local checkpoint**: For previous training runs

The framework automatically handles model downloading and caching.

Memory Requirements by Model
-----------------------------

Based on batch size and LoRA rank::

    RTX 2080 Ti (11GB):
    - Model: gemma3_270m
    - Batch size: 1
    - LoRA rank: 8-16

    RTX A6000 (48GB):
    - Model: gemma3_1b
    - Batch size: 4
    - LoRA rank: 32-64

    H100/A100 (80GB):
    - Model: gemma3_4b
    - Batch size: 8
    - LoRA rank: 64-128

Advanced Topics
---------------

See :doc:`../advanced/distributed_training` for:

- Tensor parallel training
- Fully sharded data parallel setup
- Multi-node distributed training

Next Steps
----------

- :doc:`../guide/training` - Training guide
- :doc:`../advanced/distributed_training` - Distributed training setup
- :doc:`train` - Training API reference
