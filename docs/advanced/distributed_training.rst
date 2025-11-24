Distributed Training
====================

Running training across multiple GPUs or nodes significantly speeds up training and enables handling larger models.

Overview
--------

Three parallelism strategies available:

- **Data Parallelism (FSDP)**: Replicate model on each GPU, shard data
- **Tensor Parallelism (TP)**: Shard model tensors across GPUs
- **Hybrid**: Combine both for very large models on many GPUs

Configuration
-------------

Configure parallelism via mesh shape in model config::

    model:
      mesh_shape: [[num_fsdp, num_tp], ["fsdp", "tp"]]

Where:

- ``num_fsdp``: Number of GPUs for data parallelism
- ``num_tp``: Number of GPUs for tensor parallelism

Single GPU Setup
----------------

::

    mesh_shape: [[1, 1], ["fsdp", "tp"]]

Usage::

    python run_training.py model.mesh_shape=[[1,1],["fsdp","tp"]]

Data Parallel (Multiple GPUs)
-----------------------------

Shard data across 4 GPUs, each with full model::

    mesh_shape: [[4, 1], ["fsdp", "tp"]]

Each GPU:

- Receives different batch of data
- Has full copy of model
- Communicates gradients with other GPUs

Usage::

    python run_training.py model.mesh_shape=[[4,1],["fsdp","tp"]]

Advantages:

- Simple to implement
- Good scaling up to 8-16 GPUs
- Minimal communication overhead

Tensor Parallel (Model Sharding)
--------------------------------

Shard large models across 4 GPUs::

    mesh_shape: [[1, 4], ["fsdp", "tp"]]

Each GPU:

- Has part of each tensor
- Needs to synchronize between forward/backward passes
- Suitable for models that don't fit on single GPU

Usage::

    python run_training.py model.mesh_shape=[[1,4],["fsdp","tp"]]

Best for:

- Very large models (10B+)
- Memory per GPU is limited
- Model parallelism is necessary

Hybrid Parallelism
------------------

Combine both strategies for large scale::

    mesh_shape: [[2, 2], ["fsdp", "tp"]]

This creates:

- 2 groups of 4 GPUs each
- Within each group: 2 data parallel, 2 tensor parallel
- Between groups: communication for data parallelism
- Within groups: communication for tensor parallelism

Suitable for:

- 4-16 GPU training
- Models >1B parameters
- Balancing communication overhead

Multi-Node Setup
----------------

For training across multiple machines/nodes::

    mesh_shape: [[8, 2], ["fsdp", "tp"]]

Configuration steps:

1. Ensure network connectivity between nodes
2. Set distributed environment variables::

       export MASTER_ADDR=<master-node-ip>
       export MASTER_PORT=29500
       export RANK=<node-rank>
       export WORLD_SIZE=<total-num-gpus>

3. Launch training::

       torchrun --nproc_per_node=4 run_training.py model.mesh_shape=[[8,2],["fsdp","tp"]]

Or with JAX::

       python -m jax.distributed.launch --nprocs=4 run_training.py

Performance Tuning
------------------

Memory Optimization
~~~~~~~~~~~~~~~~~~~

With FSDP, reduce per-GPU memory by sharding::

    # Without FSDP: each GPU needs full model
    model=gemma3_1b training.micro_batch_size=2

    # With FSDP across 4 GPUs: split model
    model=gemma3_1b training.micro_batch_size=4 model.mesh_shape=[[4,1],["fsdp","tp"]]

With FSDP, each GPU stores:

- 1/N-th of model weights
- Full activations for one batch
- Gradients (periodically)

Communication Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Minimize communication overhead::

    # Fewer gradient sync steps
    training:
      gradient_accumulation_steps: 4
      sync_gradients_every: 2

    # Use lower precision for communication
    training:
      mixed_precision: bf16

Load Balancing
~~~~~~~~~~~~~~

Ensure even work distribution::

    # Check GPU utilization
    nvidia-smi dmon

    # Adjust batch sizes if imbalanced
    python run_training.py training.micro_batch_size=8

Debugging Distributed Training
-------------------------------

**Check GPU Connectivity**

::

    python -c "
    import jax
    print(f'Number of devices: {jax.device_count()}')
    for i, device in enumerate(jax.devices()):
        print(f'Device {i}: {device}')
    "

**Monitor Communication**

::

    # Enable JAX profiling
    python run_training.py training.profile_communication=true

**Check Rank and Size**

::

    python -c "
    import os
    print(f'Rank: {os.environ.get(\"RANK\", \"0\")}')
    print(f'World size: {os.environ.get(\"WORLD_SIZE\", \"1\")}')
    "

Common Issues
-------------

**Device Mismatch**

Issue: Different GPUs have different compute capabilities::

    RuntimeError: Devices are not homogeneous

Solution: Use same GPU types across all nodes

**Communication Timeout**

Issue: Slow network or unresponsive nodes::

    TimeoutError: Communication timed out

Solution::

    # Increase timeout
    export NCCL_P2P_CONNECT_TIMEOUT=300

    # Check network connectivity
    ping <other-node-ip>

**Unbalanced Training**

Issue: Some GPUs finish earlier than others::

    Solution: Adjust batch sizes or model loading

**OOM During Communication**

Issue: Not enough GPU memory for communication buffers::

    Solution::

        # Use gradient accumulation
        training:
          gradient_accumulation_steps: 2

        # Reduce batch size
        training:
          micro_batch_size: 2

Example Distributed Setups
---------------------------

**4 GPUs on Single Machine**

::

    python run_training.py \
        model=gemma3_1b \
        model.mesh_shape=[[4,1],["fsdp","tp"]] \
        training.micro_batch_size=4

**8 GPUs Across 2 Machines (4 per machine)**

Machine 1::

    export MASTER_ADDR=192.168.1.10
    export MASTER_PORT=29500
    export RANK=0
    export WORLD_SIZE=8
    python -m jax.distributed.launch --nprocs=4 run_training.py \
        model=gemma3_1b \
        model.mesh_shape=[[8,1],["fsdp","tp"]]

Machine 2::

    export MASTER_ADDR=192.168.1.10
    export MASTER_PORT=29500
    export RANK=4
    export WORLD_SIZE=8
    python -m jax.distributed.launch --nprocs=4 run_training.py \
        model=gemma3_1b \
        model.mesh_shape=[[8,1],["fsdp","tp"]]

**16 GPUs with Hybrid Parallelism**

::

    python run_training.py \
        model=gemma3_4b \
        model.mesh_shape=[[8,2],["fsdp","tp"]] \
        training.micro_batch_size=8

Monitoring
----------

Watch distributed training::

    # Monitor all GPU processes
    watch -n 1 nvidia-smi

    # Check inter-GPU communication (requires profiling)
    tensorboard --logdir outputs/

    # Monitor training logs
    tail -f outputs/tunix-grpo/YYYY-MM-DD/HH-MM-SS/train.log

Best Practices
--------------

1. **Start small**: Test with 2 GPUs before scaling
2. **Use same GPU types**: Avoid heterogeneous clusters initially
3. **Profile communication**: Identify bottlenecks
4. **Use high-performance interconnect**: NVLink, InfiniBand for multi-node
5. **Monitor balance**: Ensure all GPUs have similar utilization
6. **Scale gradually**: Test each parallelism strategy separately
7. **Document configuration**: Keep notes on working mesh shapes

Next Steps
----------

- :doc:`../guide/training` - Training guide
- :doc:`../api/models` - Model configuration reference
- :doc:`../getting_started/configuration` - Configuration guide
