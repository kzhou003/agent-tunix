Rewards API
===========

.. py:module:: agent_tunix.rewards
   :noindex:

This module provides reward functions for evaluating model outputs during training.

Reward System Overview
----------------------

The reward system evaluates model-generated responses and provides numerical feedback to guide training. Rewards are computed for each generated response and used to:

1. Update policy gradients
2. Guide GRPO optimization
3. Track training progress

Reward computation::

    prompt → model → response → [reward functions] → reward score

Built-in Reward Functions
--------------------------

**Format Reward**

Checks if response matches expected format::

    match_format_exactly(response, expected_format)

Rewards:

- 1.0: Perfect format match
- 0.5: Partial format match
- 0.0: No format match

Example::

    response = "The answer is 4."
    expected_format = "The answer is [NUM]."
    reward = match_format_exactly(response, expected_format)

**Correctness Reward**

Checks if answer is mathematically correct::

    check_answer(response, ground_truth)

Rewards:

- 1.0: Exact match
- 0.5: Partial credit (within 10% of correct value)
- 0.0: Incorrect

Example::

    response = "2 + 2 = 4. The answer is 4."
    ground_truth = "4"
    reward = check_answer(response, ground_truth)

**Number Extraction**

Extracts and validates numbers from responses::

    check_numbers(response, expected_numbers)

Returns:

- Extracted numbers from response
- Validation results
- Matching scores

Example::

    response = "2 + 2 = 4. The answer is 4."
    numbers = check_numbers(response, [4])
    # Returns: {"extracted": [4], "matches": [True], "score": 1.0}

Combined Reward
---------------

Total reward is typically a combination::

    total_reward = alpha * format_reward + beta * correctness_reward

Default weighting::

    alpha = 0.3    # Format importance
    beta = 0.7     # Correctness importance

Custom Reward Functions
-----------------------

Create custom reward function in ``src/agent_tunix/rewards.py``::

    def custom_reward_function(response: str, prompt: str, metadata: dict) -> float:
        """
        Compute reward for a response.

        Args:
            response: Model-generated response
            prompt: Original input prompt
            metadata: Additional context (answer, format, etc.)

        Returns:
            Reward score (typically 0.0 to 1.0)
        """
        # Extract expected answer from metadata
        expected = metadata.get("answer", "")

        # Compute components
        format_score = evaluate_format(response, metadata.get("format"))
        correctness_score = evaluate_correctness(response, expected)
        completeness_score = evaluate_completeness(response)

        # Combine with weights
        reward = (
            0.3 * format_score +
            0.5 * correctness_score +
            0.2 * completeness_score
        )

        return reward

Using Custom Rewards
~~~~~~~~~~~~~~~~~~~~

Update reward function in ``conf/training/default.yaml``::

    training:
      reward_function: custom_reward_function

Or pass during training::

    python run_training.py +training.reward_function=custom_reward_function

Reward Design Patterns
----------------------

**Binary Reward** (Correct/Incorrect)

Simplest approach, 1.0 for correct, 0.0 for incorrect::

    def binary_reward(response, ground_truth):
        return 1.0 if is_correct(response, ground_truth) else 0.0

Best for: Clear right/wrong answers

**Partial Credit Reward** (Graduated)

Award points for partial correctness::

    def graduated_reward(response, ground_truth):
        if is_correct(response, ground_truth):
            return 1.0
        elif is_partially_correct(response, ground_truth):
            return 0.5
        else:
            return 0.0

Best for: Tasks with multiple acceptable answers

**Continuous Reward** (Magnitude-based)

Reward proportional to answer quality::

    def continuous_reward(response, ground_truth):
        error = abs(extract_number(response) - ground_truth)
        max_error = 100
        return max(0.0, 1.0 - (error / max_error))

Best for: Numerical tasks where closer is better

**Multi-aspect Reward** (Composite)

Combine multiple evaluation aspects::

    def composite_reward(response, prompt, ground_truth):
        # Evaluate different aspects
        relevance = evaluate_relevance(response, prompt)
        correctness = evaluate_correctness(response, ground_truth)
        clarity = evaluate_clarity(response)
        conciseness = evaluate_conciseness(response)

        # Weighted combination
        reward = (
            0.4 * correctness +
            0.3 * relevance +
            0.2 * clarity +
            0.1 * conciseness
        )
        return reward

Best for: Complex tasks requiring multiple quality dimensions

Reward Shaping
---------------

Reward shaping guides learning by providing intermediate signals::

    def shaped_reward(response, ground_truth):
        """Add shaping to guide model behavior."""
        base_reward = check_correctness(response, ground_truth)

        # Shape 1: Penalize very long responses
        length_penalty = -0.1 * len(response.split()) / 100

        # Shape 2: Reward attempting reasoning steps
        reasoning_bonus = 0.1 if has_reasoning_steps(response) else 0.0

        # Shape 3: Penalize hallucination
        hallucination_penalty = -0.2 if has_hallucination(response) else 0.0

        return base_reward + length_penalty + reasoning_bonus + hallucination_penalty

Guidelines:

- Keep shaping rewards relatively small compared to primary reward
- Ensure shaping aligns with task objectives
- Monitor reward distribution during training

Reward Debugging
----------------

Inspect rewards during training::

    # Enable verbose reward logging
    python run_training.py training.log_rewards=true

This logs:

- Reward distribution for each batch
- Min/max/mean rewards
- Reward statistics over training

Analyze reward patterns::

    # Save reward analysis
    python run_training.py training.save_reward_analysis=true

Outputs analysis of:

- Which response types get high/low rewards
- Reward distribution skewness
- Reward variance
- Common failure patterns

Common Reward Issues
--------------------

**Reward Always Near 0 or 1**

Issue: Sparse or binary rewards don't guide learning well

Solution: Use graduated rewards with intermediate values::

    def improved_reward(response, ground_truth):
        if exact_match(response, ground_truth):
            return 1.0
        elif close_match(response, ground_truth, tolerance=0.1):
            return 0.5
        else:
            return 0.0

**Rewards Too Noisy**

Issue: High variance in rewards prevents consistent learning

Solution: Smooth and normalize rewards::

    def stable_reward(response, ground_truth):
        base = check_correctness(response, ground_truth)
        # Add minimum reward to avoid exact zeros
        return max(base, 0.1)

**Reward Hacking**

Issue: Model learns to game the reward instead of solving the task

Solution: Include format/style constraints::

    def robust_reward(response, ground_truth):
        correctness = check_correctness(response, ground_truth)
        format_match = check_format(response, expected_format)

        # Both must be good
        if format_match < 0.8:
            return 0.0

        return correctness

Testing Rewards
---------------

Test reward functions on sample outputs::

    from agent_tunix.rewards import check_answer, match_format_exactly

    # Test data
    test_cases = [
        {
            "response": "2 + 2 = 4. The answer is 4.",
            "ground_truth": "4",
            "expected_format": "The answer is [NUM].",
            "expected_reward": 1.0
        },
        {
            "response": "2 plus 2 equals 4.",
            "ground_truth": "4",
            "expected_format": "The answer is [NUM].",
            "expected_reward": 0.5  # Correct but wrong format
        }
    ]

    # Evaluate
    for case in test_cases:
        format_reward = match_format_exactly(
            case["response"],
            case["expected_format"]
        )
        correctness_reward = check_answer(
            case["response"],
            case["ground_truth"]
        )
        total = 0.3 * format_reward + 0.7 * correctness_reward
        print(f"Expected: {case['expected_reward']}, Got: {total}")

Advanced Topics
---------------

See :doc:`../advanced/custom_rewards` for:

- Reward normalization and scaling
- Multi-task rewards
- Curriculum learning with rewards
- Reward model training

Next Steps
----------

- :doc:`../advanced/custom_rewards` - Detailed custom reward guide
- :doc:`train` - Training API reference
- :doc:`../guide/training` - Training guide
