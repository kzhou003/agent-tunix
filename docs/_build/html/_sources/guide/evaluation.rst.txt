Evaluation Guide
================

Model Evaluation
----------------

Evaluate a trained model on the test set::

    python evaluate.py

This will:

1. Load the trained model with LoRA weights
2. Create a sampler for text generation
3. Generate responses for test set questions
4. Compute evaluation metrics

Evaluation Metrics
------------------

The framework computes:

- **Accuracy**: Percentage of exactly correct answers
- **Partial Accuracy**: Answers within 10% of correct value
- **Format Accuracy**: Responses matching expected format

Example output::

    Evaluation Results
    ==================
    Correct: 125/500
    Accuracy: 25.00%
    Partial Accuracy: 45.20%
    Format Accuracy: 78.50%

Configuration
-------------

Evaluation settings are in ``conf/evaluation/``.

**Key parameters**::

    checkpoint_dir: ./checkpoints/ckpts/    # Model checkpoint directory
    step: null                              # Checkpoint step (null for latest)
    inference_config: greedy                # Inference strategy
    num_passes: 1                           # Generations per question
    verbose: true                           # Show progress

Override from command line::

    # Use specific checkpoint step
    python evaluate.py step=500

    # Different checkpoint directory
    python evaluate.py checkpoint_dir=/path/to/checkpoints/

Inference Strategies
--------------------

Three predefined inference configurations:

**Greedy**

Deterministic generation, always choose highest probability token::

    python evaluate.py inference_config=greedy

Temperature: 1e-4, top_k: 1, top_p: 1.0

**Standard**

Balanced sampling with reasonable diversity::

    python evaluate.py inference_config=standard

Temperature: 0.7, top_k: 50, top_p: 0.95

**Liberal**

More diverse, creative responses::

    python evaluate.py inference_config=liberal

Temperature: 0.85, top_k: 2000, top_p: 1.0

Multiple Passes
---------------

Run multiple generation passes per question::

    python evaluate.py num_passes=3

Useful for:

- Understanding model consistency
- Finding best response from multiple attempts
- Estimating uncertainty

Checkpoint Selection
--------------------

**Latest checkpoint** (default)::

    python evaluate.py

**Specific step**::

    python evaluate.py step=1000

**Custom directory**::

    python evaluate.py checkpoint_dir=./custom/checkpoints/

Finding Checkpoint Steps
^^^^^^^^^^^^^^^^^^^^^^^^

List available checkpoints::

    ls -la checkpoints/ckpts/actor/

Output shows directories like::

    0/
    50/
    100/
    150/
    ...

These correspond to training steps.

Advanced Configuration
----------------------

Custom evaluation configuration in ``conf/evaluation/custom.yaml``::

    checkpoint_dir: ./checkpoints/ckpts/
    step: 500
    inference_config: greedy
    num_passes: 5
    verbose: true

Use it::

    python evaluate.py --config custom

Batch Evaluation
----------------

Evaluate multiple checkpoints::

    for step in 100 200 300 400 500; do
        python evaluate.py step=$step >> results.txt
    done

Or with hyperparameter sweep::

    python evaluate.py --multirun step=100,200,300,400,500

Interpreting Results
--------------------

**High Accuracy, Low Format Accuracy**

Model produces correct answers but in wrong format. Check reward function calibration.

**Low Accuracy, High Format Accuracy**

Model follows format but answers are incorrect. May need:

- More training data
- Better reward signal
- Longer training

**Low Accuracy, Low Format Accuracy**

Fundamental training issue. Check:

- Data quality
- Model size adequacy
- Training configuration
- Learning rate

Troubleshooting Evaluation
---------------------------

**No checkpoint found**

Ensure training has completed and checkpoints exist::

    ls -la checkpoints/ckpts/actor/

**CUDA out of memory during eval**

Reduce batch size or model size::

    python evaluate.py training.micro_batch_size=1

**Evaluation takes too long**

- Use greedy inference (faster)
- Reduce evaluation set size
- Use fewer test batches

**Metric discrepancies**

Ensure using same inference configuration as training::

    python evaluate.py inference_config=greedy

Example Evaluation Workflow
---------------------------

**1. Evaluate specific checkpoint**::

    python evaluate.py step=500

**2. Try different inference strategies**::

    for config in greedy standard liberal; do
        echo "=== $config ==="
        python evaluate.py inference_config=$config
    done

**3. Multiple passes for uncertainty**::

    python evaluate.py num_passes=5

**4. Compare checkpoints**::

    python evaluate.py --multirun step=100,200,300,400,500

Programmatic Evaluation
-----------------------

Use evaluation functions directly::

    from agent_tunix.evaluate import evaluate, create_sampler, evaluate_with_config

    # Create sampler
    sampler = create_sampler(model, tokenizer, model_config, 256, 512)

    # Evaluate with predefined config
    results = evaluate_with_config(test_dataset, sampler, "greedy")

    print(f"Accuracy: {results['accuracy']:.2f}%")

See :doc:`API Reference </api/evaluate>` for details.

Next Steps
----------

- :doc:`Hyperparameter Tuning </guide/hyperparameter_tuning>`
- :doc:`Training Guide </guide/training>`
- :doc:`API Reference </api/evaluate>`
