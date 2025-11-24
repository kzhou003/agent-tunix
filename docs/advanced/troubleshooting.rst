Troubleshooting
===============

Common issues and solutions for training and evaluation.

Installation Issues
-------------------

**CUDA Not Found**

Error::

    RuntimeError: CUDA is not available

Solution::

    # Verify CUDA installation
    python -m agent_tunix.utils check-gpu

    # Check NVIDIA drivers
    nvidia-smi

    # Install CUDA if missing (see installation guide)

**JAX Backend Issues**

Error::

    ModuleNotFoundError: No module named 'jax'

Solution::

    # Reinstall JAX with CUDA support
    pip install jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

    # Or with latest CUDA
    pip install jax[cuda12_cudnn83]

**GPU Out of Memory on Import**

Error::

    RuntimeError: CUDA out of memory

Solution::

    # Set XLA to only allocate needed memory
    export XLA_PYTHON_CLIENT_PREALLOCATE=false

    # Reduce TensorFlow/JAX memory allocation
    export JAX_PLATFORM_NAME=cpu  # Use CPU for testing

Training Issues
---------------

**Training Starts But Stops Immediately**

Error::

    ValueError: Data loading failed

Solution::

    1. Check data availability::

        python -c "from datasets import load_dataset; load_dataset('gsm8k')"

    2. Verify tokenizer exists::

        python run_training.py --cfg job | grep tokenizer

    3. Check configuration::

        python run_training.py --cfg job

**Loss is NaN**

Symptoms::

    loss: nan
    Training crashes after few steps

Causes and solutions:

1. **Learning rate too high**::

        python run_training.py optimizer.learning_rate=1e-7

2. **Gradient overflow**::

        python run_training.py optimizer.max_grad_norm=0.01

3. **Batch size too large**::

        python run_training.py training.micro_batch_size=1

4. **Bad data example**::

        # Check and clean data
        python -c "from agent_tunix.data import load_dataset; ds, _ = load_dataset(); print(ds[0])"

**Loss Not Decreasing**

Symptoms::

    Loss stays constant or increases

Causes and solutions:

1. **Learning rate too low**::

        python run_training.py optimizer.learning_rate=1e-4

2. **Model not training (frozen weights)**::

        # Check if parameters are trainable
        python -c "from agent_tunix.train import create_model; model = create_model(cfg); print(model.trainable_params())"

3. **Data too small**::

        # Use more training data or reduce num_batches
        python run_training.py training.num_batches=10000

4. **LoRA not properly configured**::

        python run_training.py model.lora_rank=64

**Memory Error During Training**

Error::

    RuntimeError: CUDA out of memory

Solutions (in order of impact):

1. Reduce batch size::

       python run_training.py training.micro_batch_size=1

2. Reduce model size::

       python run_training.py model=gemma3_270m

3. Reduce LoRA rank::

       python run_training.py model.lora_rank=8

4. Reduce sequence length::

       python run_training.py \
           generation.max_prompt_length=128 \
           generation.max_generation_steps=256

5. Reduce generations per prompt::

       python run_training.py grpo.num_generations=2

6. Use gradient accumulation::

       python run_training.py training.gradient_accumulation_steps=4

**Slow Training**

Check GPU utilization::

    # Monitor in another terminal
    watch -n 1 nvidia-smi

Solutions if utilization is low:

1. **Data loading bottleneck**::

        # Increase number of data loading workers
        python run_training.py training.num_workers=8

2. **Model too small**::

        python run_training.py model=gemma3_1b

3. **Check for CPU bottleneck**::

        # Monitor CPU usage
        top

4. **I/O bottleneck**::

        # Move data to faster storage (SSD)
        # Or use memory-mapped datasets

**Checkpoints Not Being Saved**

Error::

    No checkpoint directory created

Solutions:

1. Check checkpoint directory permissions::

        ls -la checkpoints/ckpts/

2. Verify checkpoint configuration::

        python run_training.py --cfg job | grep checkpoint

3. Create directory if missing::

        mkdir -p checkpoints/ckpts/

4. Use absolute path::

        python run_training.py checkpoint_dir=/path/to/checkpoints/ckpts/

**Model Not Improving on Validation**

Symptoms::

    Validation accuracy flat
    Training loss decreases but validation stagnates

Causes and solutions:

1. **Overfitting**::

        # Add validation set diversity
        # Reduce model capacity
        python run_training.py model.lora_rank=8

2. **Wrong reward signal**::

        # Check reward function
        python -c "
        from agent_tunix.rewards import check_answer
        print(check_answer('The answer is 4.', '4'))
        "

3. **Validation set too small**::

        # Use larger validation set
        # Or fewer validation steps
        python run_training.py training.eval_interval_steps=1000

4. **Distribution mismatch**::

        # Ensure test set matches training data
        # Or fine-tune on test distribution

Evaluation Issues
-----------------

**No Checkpoints Found**

Error::

    No checkpoint found in: checkpoints/ckpts/actor/

Solutions:

1. Verify training completed::

        ls -la outputs/tunix-grpo/*/

2. Check for checkpoints::

        find . -name "*actor*" -type d

3. Use absolute path::

        python evaluate.py checkpoint_dir=/absolute/path/to/checkpoints/ckpts/

4. Train if not done::

        python run_training.py +experiment=quick_test

**CUDA Memory During Evaluation**

Error::

    RuntimeError: CUDA out of memory during evaluation

Solutions:

1. Reduce batch size::

        python evaluate.py training.micro_batch_size=1

2. Reduce sequence length::

        python evaluate.py generation.max_generation_steps=256

3. Use CPU::

        python evaluate.py device=cpu

**Evaluation Takes Too Long**

Solutions:

1. Use greedy decoding (faster)::

        python evaluate.py inference_config=greedy

2. Reduce evaluation samples::

        python evaluate.py evaluation.num_samples=100

3. Reduce number of passes::

        python evaluate.py evaluation.num_passes=1

4. Use smaller model checkpoint::

        python evaluate.py step=100

**Metric Results Don't Match Training**

Causes:

1. **Different inference config**::

        # Use same as training
        python evaluate.py inference_config=greedy

2. **Different checkpoint**::

        # Use latest
        python evaluate.py

3. **Different data**::

        # Ensure same dataset
        python evaluate.py --cfg job | grep dataset

4. **Randomness**::

        # Set seed
        python evaluate.py seed=42

Configuration Issues
--------------------

**Invalid Configuration**

Error::

    ConfigError: Could not find 'model/custom.yaml'

Solutions:

1. List available configs::

        python run_training.py --info config-groups | grep model

2. Check file exists::

        ls conf/model/

3. Use default if custom missing::

        python run_training.py model=gemma3_1b

**Conflicting Overrides**

Error::

    ConfigCompositionException: Could not override

Solutions:

1. Check config hierarchy::

        python run_training.py --info defaults-tree

2. Use correct path::

        # Correct
        python run_training.py optimizer.learning_rate=1e-5

        # Wrong
        python run_training.py learning_rate=1e-5

3. Use force override if needed::

        python run_training.py ++optimizer.new_param=value

**Missing Configuration Group**

Error::

    ConfigCompositionException: Could not load group

Solutions:

1. List defaults::

        python run_training.py --info config-groups

2. Create missing config::

        touch conf/scheduler/custom.yaml

3. Update config.yaml defaults::

        # conf/config.yaml
        defaults:
          - scheduler: custom

Distributed Training Issues
----------------------------

**NCCL Errors**

Error::

    RuntimeError: NCCL operation failed

Solutions:

1. Check GPU connectivity::

        nvidia-smi -L

2. Enable NCCL debugging::

        export NCCL_DEBUG=INFO

3. Increase timeout::

        export NCCL_P2P_CONNECT_TIMEOUT=300

4. Use single GPU for testing::

        python run_training.py model.mesh_shape=[[1,1],["fsdp","tp"]]

**Device Mismatch**

Error::

    RuntimeError: Devices are not homogeneous

Causes:

- Different GPU types in cluster
- Different compute capabilities

Solution:

- Use same GPU type across all nodes
- Or use compatible GPUs

**Communication Timeout**

Error::

    TimeoutError: Communication timed out

Solutions:

1. Check network::

        ping <other-node>

2. Increase timeout::

        export NCCL_P2P_CONNECT_TIMEOUT=600

3. Use slower network::

        export NCCL_SOCKET_IFNAME=eth0  # Specific network interface

**Uneven GPU Utilization**

Issue: Some GPUs finish faster than others

Solutions:

1. Check loads::

        nvidia-smi dmon -s pm

2. Adjust batch size::

        python run_training.py training.micro_batch_size=8

3. Balance data distribution::

        python run_training.py training.data_seed=42

Debugging Tips
--------------

**Enable Verbose Logging**

::

    python run_training.py training.log_level=DEBUG

**Profile Training**

::

    python run_training.py training.profile=true

Check profile output in logs.

**Inspect Configuration**

::

    python run_training.py --cfg job --resolve

Shows all interpolations resolved.

**Dry Run Test**

::

    python run_training.py +experiment=quick_test --dry-run

Validates configuration without training.

**Check Versions**

::

    python -c "
    import jax
    import flax
    import transformers
    import hydra
    print(f'JAX: {jax.__version__}')
    print(f'Flax: {flax.__version__}')
    print(f'Transformers: {transformers.__version__}')
    print(f'Hydra: {hydra.__version__}')
    "

Getting Help
------------

When reporting issues, include:

1. Complete error message and traceback
2. Configuration used (``python run_training.py --cfg job``)
3. GPU information (``nvidia-smi``)
4. Environment info (Python version, package versions)
5. Minimal reproducible example

Common Patterns
---------------

**Testing Fix Before Full Run**

::

    # Test configuration and data loading
    python run_training.py +experiment=quick_test --dry-run

    # Run 10 steps to verify
    python run_training.py +experiment=quick_test

    # If successful, run full training
    python run_training.py +experiment=full_training

**Incremental Memory Reduction**

::

    # Start here
    python run_training.py

    # If OOM, reduce batch size
    python run_training.py training.micro_batch_size=2

    # If still OOM, use smaller model
    python run_training.py model=gemma3_270m training.micro_batch_size=1

    # If still OOM, reduce LoRA rank
    python run_training.py model=gemma3_270m model.lora_rank=8 training.micro_batch_size=1

Next Steps
----------

- :doc:`../guide/training` - Training guide
- :doc:`../api/train` - Training API
- :doc:`../getting_started/configuration` - Configuration reference
