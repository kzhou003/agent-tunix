Data API
========

.. py:module:: agent_tunix.data
   :noindex:

This module provides data loading and preprocessing utilities.

Dataset Support
---------------

**GSM8K (Grade School Math)**

Default dataset for math reasoning tasks::

    # Auto-loaded with default configuration

Dataset specifications:

- Task: Grade school math word problems
- Format: Question → Answer with step-by-step reasoning
- Size: ~8,000 training examples
- Data source: Hugging Face datasets

Example::

    Question: Natalia sold clips to 48 of her friends in April, and then she sold
    clips to 42 of her friends in May. If she got $2.50 for each clip, how much
    money did she earn in total?

    Answer: Natalia sold clips to 48 + 42 = 90 friends in total.
    If she got $2.50 for each clip, she earned 90 * $2.50 = $225.

Data Loading
------------

The framework automatically handles:

1. **Downloading**: Fetches dataset from source if not cached
2. **Splitting**: Creates train/validation/test splits
3. **Tokenization**: Converts text to token IDs
4. **Batching**: Creates mini-batches for training
5. **Padding**: Handles variable-length sequences

Custom Datasets
---------------

To use a custom dataset, create a data loading function in ``src/agent_tunix/data.py``::

    def load_custom_dataset(dataset_path, tokenizer, max_length=512):
        """Load custom dataset and tokenize."""
        # 1. Load data from your source
        examples = load_data(dataset_path)

        # 2. Format as (prompt, answer) pairs
        formatted = [
            {"prompt": ex["question"], "answer": ex["answer"]}
            for ex in examples
        ]

        # 3. Tokenize
        tokenized = tokenizer(
            [f"{ex['prompt']} {ex['answer']}" for ex in formatted],
            truncation=True,
            max_length=max_length,
            return_tensors="np"
        )

        # 4. Return dataset object
        return formatted, tokenized

Data Format Requirements
------------------------

Minimum required fields::

    {
        "prompt": "What is 2 + 2?",
        "answer": "The answer is 4."
    }

For math problems, we recommend step-by-step format::

    {
        "prompt": "What is 2 + 2?",
        "answer": "2 + 2 = 4. The answer is 4."
    }

Tokenization
-------------

The framework uses the model's tokenizer::

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-1b")

    # Tokenize input
    tokens = tokenizer("What is 2 + 2?", return_tensors="pt")

    # Token IDs
    print(tokens['input_ids'])

    # Attention masks (1 for real tokens, 0 for padding)
    print(tokens['attention_mask'])

Preprocessing Pipeline
----------------------

Standard preprocessing steps::

    1. Load raw data
    2. Split into train/val/test
    3. Tokenize with model tokenizer
    4. Pad sequences to max_length
    5. Create attention masks
    6. Create PyTorch/JAX datasets
    7. Define data loaders for batching

Data Splits
-----------

Default splits::

    Train:      80% of data
    Validation: 10% of data
    Test:       10% of data

Customize in configuration::

    training:
      train_split: 0.8
      val_split: 0.1
      test_split: 0.1

Batch Processing
----------------

Mini-batch configuration::

    training:
      micro_batch_size: 4          # Batch per device

    grpo:
      num_generations: 4           # Responses per prompt

Processing flow::

    1. Load batch of prompts
    2. For each prompt, generate K responses (num_generations)
    3. Compute rewards for each response
    4. Update model based on rewards

Sequence Lengths
----------------

Configure sequence lengths::

    generation:
      max_prompt_length: 256       # Maximum prompt tokens
      max_generation_steps: 512    # Maximum response tokens

Practical limits::

    - Longer prompts: more context but less generation space
    - Shorter sequences: faster training but less capacity
    - Balance based on your task requirements

Advanced: Custom Data Loading
------------------------------

For advanced use cases, create custom data loaders::

    class CustomDataLoader:
        def __init__(self, data_path, tokenizer, batch_size):
            self.data = load_json(data_path)
            self.tokenizer = tokenizer
            self.batch_size = batch_size

        def __iter__(self):
            for i in range(0, len(self.data), self.batch_size):
                batch = self.data[i:i+self.batch_size]
                yield self._process_batch(batch)

        def _process_batch(self, batch):
            prompts = [ex["prompt"] for ex in batch]
            answers = [ex["answer"] for ex in batch]

            # Tokenize
            inputs = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                return_tensors="np"
            )

            outputs = self.tokenizer(
                answers,
                padding=True,
                truncation=True,
                return_tensors="np"
            )

            return {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "target_ids": outputs["input_ids"],
                "target_mask": outputs["attention_mask"]
            }

Data Validation
---------------

Verify data quality::

    # Check dataset statistics
    python -c "
    from agent_tunix.data import load_dataset
    dataset, tokenizer = load_dataset()
    print(f'Dataset size: {len(dataset)}')
    print(f'Example: {dataset[0]}')
    "

Troubleshooting
---------------

**Data loading too slow**

- Use smaller max_length
- Reduce batch size temporarily
- Pre-tokenize and cache data

**Out of memory during batching**

- Reduce micro_batch_size
- Reduce max sequence lengths
- Use gradient accumulation instead of larger batches

**Tokenization mismatches**

- Use same tokenizer as model
- Check special token handling
- Verify padding configuration

Next Steps
----------

- :doc:`../guide/training` - Training guide
- :doc:`train` - Training API reference
- :doc:`../advanced/custom_rewards` - Custom reward functions for your data
