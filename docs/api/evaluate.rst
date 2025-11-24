Evaluation API
==============

.. py:module:: agent_tunix.evaluate
   :noindex:

This module provides evaluation utilities for assessing model performance.

Main Entry Point
----------------

.. autofunction:: evaluate
   :members:

Evaluation Functions
--------------------

.. autofunction:: create_sampler
   :members:

.. autofunction:: evaluate_with_config
   :members:

Configuration Classes
---------------------

Evaluation configuration structure::

    evaluation:
      checkpoint_dir: ./checkpoints/ckpts/
      step: null                           # null for latest
      inference_config: greedy
      num_passes: 1
      verbose: true

Inference Configurations
------------------------

Three predefined inference strategies:

**Greedy**

Deterministic, always select highest probability token::

    inference_config: greedy
    # temperature: 1e-4
    # top_k: 1
    # top_p: 1.0

Best for: Reproducible results, benchmarking, production inference

**Standard**

Balanced sampling with moderate diversity::

    inference_config: standard
    # temperature: 0.7
    # top_k: 50
    # top_p: 0.95

Best for: Reasonable diversity while maintaining coherence

**Liberal**

High diversity, creative responses::

    inference_config: liberal
    # temperature: 0.85
    # top_k: 2000
    # top_p: 1.0

Best for: Exploring model capabilities, creative tasks

Evaluation Metrics
------------------

The framework computes three main metrics:

- **Accuracy**: Percentage of exactly correct answers (number matching expected output)
- **Partial Accuracy**: Percentage within 10% of correct value
- **Format Accuracy**: Percentage of responses matching expected format structure

Example output::

    Evaluation Results
    ==================
    Correct: 125/500
    Accuracy: 25.00%
    Partial Accuracy: 45.20%
    Format Accuracy: 78.50%

Checkpoint Selection
--------------------

Evaluate latest checkpoint::

    python evaluate.py

Evaluate specific step::

    python evaluate.py step=1000

Use custom checkpoint directory::

    python evaluate.py checkpoint_dir=/path/to/checkpoints/

List available checkpoints::

    ls -la checkpoints/ckpts/actor/

Advanced Configuration
----------------------

Create custom evaluation config in ``conf/evaluation/custom.yaml``::

    checkpoint_dir: ./checkpoints/ckpts/
    step: 500
    inference_config: greedy
    num_passes: 5
    verbose: true

Use it::

    python evaluate.py --config custom

Multiple Passes
---------------

Run multiple generation passes per question for uncertainty estimation::

    python evaluate.py num_passes=3

Useful for:

- Understanding model consistency
- Finding best response from multiple attempts
- Estimating confidence/uncertainty

Batch Evaluation
----------------

Evaluate multiple checkpoints::

    for step in 100 200 300 400 500; do
        python evaluate.py step=$step >> results.txt
    done

Or with Hydra sweeps::

    python evaluate.py --multirun step=100,200,300,400,500

Interpreting Results
--------------------

**High Accuracy, Low Format Accuracy**

Model produces correct answers but in wrong format. May indicate:

- Reward function not properly calibrated
- Format specification unclear to model
- Need for stricter format enforcement during training

**Low Accuracy, High Format Accuracy**

Model follows format but answers are incorrect. May need:

- More training data
- Better reward signal
- Longer training duration
- Different hyperparameters

**Low Accuracy, Low Format Accuracy**

Fundamental training issue. Check:

- Data quality and completeness
- Model size adequacy for task complexity
- Training configuration correctness
- Learning rate appropriateness
- Sufficient training steps

Troubleshooting
---------------

**No checkpoint found**

Verify training completed and checkpoints exist::

    ls -la checkpoints/ckpts/actor/

**CUDA out of memory**

Reduce batch size::

    python evaluate.py training.micro_batch_size=1

**Evaluation takes too long**

- Use greedy inference (faster)
- Reduce evaluation set size
- Evaluate fewer test batches

**Metric discrepancies**

Ensure using same inference configuration as training::

    python evaluate.py inference_config=greedy

Programmatic Usage
------------------

Use evaluation functions in custom scripts::

    from agent_tunix.evaluate import create_sampler, evaluate_with_config

    # Create sampler for generation
    sampler = create_sampler(model, tokenizer, model_config, 256, 512)

    # Evaluate with predefined config
    results = evaluate_with_config(test_dataset, sampler, "greedy")

    print(f"Accuracy: {results['accuracy']:.2f}%")
    print(f"Partial Accuracy: {results['partial_accuracy']:.2f}%")
    print(f"Format Accuracy: {results['format_accuracy']:.2f}%")

Example Evaluation Workflow
----------------------------

1. Evaluate latest checkpoint with greedy inference::

    python evaluate.py

2. Compare different inference strategies::

    for config in greedy standard liberal; do
        echo "=== $config ==="
        python evaluate.py inference_config=$config
    done

3. Get uncertainty estimates with multiple passes::

    python evaluate.py num_passes=5

4. Compare multiple checkpoints::

    python evaluate.py --multirun step=100,200,300,400,500

Next Steps
----------

- :doc:`../guide/evaluation` - Detailed evaluation guide
- :doc:`train` - Training API reference
- :doc:`models` - Model architecture reference
