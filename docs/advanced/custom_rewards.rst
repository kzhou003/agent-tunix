Custom Reward Functions
=======================

Creating custom reward functions allows you to tailor the training signal to your specific task requirements.

Basic Reward Function Structure
-------------------------------

A reward function takes a response and computes a numerical score::

    def my_reward_function(response: str, **kwargs) -> float:
        """
        Compute reward for a model response.

        Args:
            response: The model-generated response text
            **kwargs: Additional context (prompt, answer, etc.)

        Returns:
            Reward score, typically in range [0.0, 1.0]
        """
        # Your evaluation logic here
        reward = evaluate_response(response, kwargs)
        return reward

Argument Patterns
~~~~~~~~~~~~~~~~~

Different calling patterns depending on your needs::

    # Minimal
    def simple_reward(response: str) -> float:
        return 1.0 if is_correct(response) else 0.0

    # With context
    def contextual_reward(response: str, prompt: str = "", answer: str = "") -> float:
        return evaluate(response, answer)

    # Using kwargs for flexibility
    def flexible_reward(response: str, **metadata) -> float:
        answer = metadata.get("answer", "")
        prompt = metadata.get("prompt", "")
        return evaluate(response, answer, prompt)

Implementing Reward Functions
-----------------------------

Example 1: Math Problem Correctness
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For math problems, extract and check numerical answers::

    import re

    def math_reward(response: str, answer: str = "") -> float:
        """Reward correct mathematical answers."""
        if not answer:
            return 0.0

        try:
            # Extract number from response (last number mentioned)
            numbers = re.findall(r'-?\d+\.?\d*', response)
            if not numbers:
                return 0.0

            predicted = float(numbers[-1])
            expected = float(answer)

            # Exact match
            if predicted == expected:
                return 1.0

            # Partial credit for close answers (within 10%)
            error_pct = abs(predicted - expected) / expected
            if error_pct <= 0.1:
                return 0.5

            return 0.0

        except (ValueError, IndexError):
            return 0.0

Usage::

    from agent_tunix.rewards import register_reward_function
    register_reward_function("math", math_reward)

    python run_training.py training.reward_function=math


Example 2: Format and Content Combined
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reward both format compliance and correctness::

    def format_and_content_reward(
        response: str,
        answer: str = "",
        format_template: str = "The answer is [NUM]."
    ) -> float:
        """Reward responses that match format and are correct."""

        # Check format (0.0 to 1.0)
        format_score = evaluate_format(response, format_template)

        # Check correctness (0.0 to 1.0)
        content_score = evaluate_correctness(response, answer)

        # Require both format and correctness
        if format_score < 0.8:
            return 0.0  # Strict format requirement

        # Combine scores
        return content_score * format_score

    def evaluate_format(response, template):
        """Check if response matches format template."""
        # Example: check if contains "The answer is [NUMBER]."
        if "The answer is" in response and "." in response:
            return 1.0
        elif "answer" in response.lower():
            return 0.5
        return 0.0

    def evaluate_correctness(response, answer):
        """Check if answer is correct."""
        import re
        numbers = re.findall(r'-?\d+\.?\d*', response)
        if numbers and float(numbers[-1]) == float(answer):
            return 1.0
        return 0.0


Example 3: Length-Penalized Reward
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reward shorter, more concise correct answers::

    def conciseness_reward(response: str, answer: str = "") -> float:
        """Reward correct but concise answers."""

        # Base correctness
        correctness = evaluate_correctness(response, answer)

        # Penalize verbosity
        words = len(response.split())
        length_penalty = max(0.0, 1.0 - (words / 100))  # Penalty after 100 words

        # Combine
        return correctness * (0.7 + 0.3 * length_penalty)


Example 4: Multi-Aspect Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Evaluate multiple dimensions of quality::

    def multi_aspect_reward(response: str, **kwargs) -> float:
        """Evaluate response on multiple dimensions."""

        answer = kwargs.get("answer", "")
        prompt = kwargs.get("prompt", "")

        # Aspect 1: Correctness (0-1)
        correctness = evaluate_correctness(response, answer)

        # Aspect 2: Clarity (0-1)
        clarity = evaluate_clarity(response)

        # Aspect 3: Completeness (0-1)
        completeness = evaluate_completeness(response, prompt)

        # Aspect 4: Efficiency (0-1)
        efficiency = evaluate_efficiency(response)

        # Weighted combination
        reward = (
            0.5 * correctness +
            0.2 * clarity +
            0.15 * completeness +
            0.15 * efficiency
        )

        return reward

    def evaluate_clarity(response):
        """Check if response is clear and well-structured."""
        # Heuristics for clarity
        lines = len(response.split('\n'))
        sentences = len(response.split('.'))

        if sentences > 1:  # Multiple sentences
            clarity = 0.8
        elif lines > 1:  # Multiple lines
            clarity = 0.6
        else:
            clarity = 0.4

        return min(1.0, clarity)

    def evaluate_completeness(response, prompt):
        """Check if response fully addresses prompt."""
        # Simple heuristic: longer responses usually more complete
        if len(response) > 50:
            return 1.0
        elif len(response) > 20:
            return 0.5
        return 0.0

    def evaluate_efficiency(response):
        """Score response efficiency (short but complete)."""
        word_count = len(response.split())

        if word_count < 30:
            return 1.0
        elif word_count < 100:
            return 0.8
        elif word_count < 200:
            return 0.5
        else:
            return 0.2


Registering Custom Rewards
---------------------------

Create custom reward file::

    # src/agent_tunix/custom_rewards.py

    def my_reward_v1(response: str, answer: str = "") -> float:
        """Custom reward function v1."""
        # Implementation
        return score

    def my_reward_v2(response: str, answer: str = "") -> float:
        """Custom reward function v2."""
        # Implementation
        return score

Register in training config::

    # conf/training/default.yaml
    training:
      reward_function: custom_rewards.my_reward_v1

Or use inline::

    python run_training.py training.reward_function=custom_rewards.my_reward_v1

Iterating on Rewards
--------------------

Testing Process
~~~~~~~~~~~~~~~

Create test suite for reward functions::

    # test_rewards.py
    from agent_tunix.custom_rewards import my_reward

    test_cases = [
        {
            "response": "The answer is 4.",
            "answer": "4",
            "expected": 1.0
        },
        {
            "response": "Four",
            "answer": "4",
            "expected": 0.5
        },
        {
            "response": "The answer is 5.",
            "answer": "4",
            "expected": 0.0
        }
    ]

    for case in test_cases:
        reward = my_reward(case["response"], answer=case["answer"])
        print(f"Expected: {case['expected']}, Got: {reward}")
        assert abs(reward - case["expected"]) < 0.01

Run tests::

    python test_rewards.py

Analyzing Reward Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Log rewards during training to analyze distribution::

    python run_training.py training.log_reward_stats=true

Monitor:

- Mean reward per batch
- Reward variance
- Min/max rewards
- Reward histogram

If rewards are skewed::

    # Too sparse (mostly 0 or 1)
    → Add graduated levels (0.25, 0.5, 0.75)

    # Too clustered
    → Add more dimensions (format, content, efficiency)

    # High variance
    → Smooth with normalization

Reward Normalization
--------------------

Normalize rewards for stable training::

    def normalize_reward(reward: float, mean: float = 0.5, std: float = 0.2) -> float:
        """Normalize reward to stable distribution."""
        # Clip outliers
        reward = max(-3 * std + mean, min(3 * std + mean, reward))

        # Normalize to standard range
        normalized = (reward - mean) / (std + 1e-8)
        return normalized

Apply in your reward function::

    def normalized_reward(response: str, answer: str = "") -> float:
        raw_reward = evaluate(response, answer)
        return normalize_reward(raw_reward)

Reward Shaping for Better Training
-----------------------------------

Guidance Rewards
~~~~~~~~~~~~~~~~

Add small bonuses to guide model behavior::

    def shaped_reward(response: str, **kwargs) -> float:
        """Reward with training guidance."""

        base_reward = evaluate_correctness(response, kwargs.get("answer", ""))

        # Shape 1: Reward explicit reasoning
        if has_reasoning_steps(response):
            base_reward += 0.05

        # Shape 2: Penalize repetition
        if has_repetition(response):
            base_reward -= 0.05

        # Shape 3: Encourage specific format
        if follows_template(response):
            base_reward += 0.1

        return max(0.0, min(1.0, base_reward))

    def has_reasoning_steps(response):
        keywords = ["because", "therefore", "first", "then", "step"]
        return any(kw in response.lower() for kw in keywords)

    def has_repetition(response):
        words = response.split()
        return len(words) != len(set(words))

    def follows_template(response):
        return "The answer is" in response and response.endswith(".")

Curriculum Learning
~~~~~~~~~~~~~~~~~~~~

Start with easy rewards, progress to harder::

    def curriculum_reward(response: str, step: int, **kwargs) -> float:
        """Progressive reward based on training progress."""

        if step < 1000:
            # Early: Just check format
            return 1.0 if follows_format(response) else 0.0

        elif step < 5000:
            # Middle: Format + basic correctness
            format_score = evaluate_format(response, kwargs.get("format", ""))
            content_score = evaluate_correctness(response, kwargs.get("answer", ""))
            return 0.3 * format_score + 0.7 * content_score

        else:
            # Late: Full evaluation
            return multi_aspect_reward(response, **kwargs)

Common Issues and Solutions
---------------------------

**Reward Always 1.0**

Issue: Reward function too lenient

Solution::

    # Before
    def loose_reward(response):
        return 1.0 if "answer" in response else 0.0

    # After
    def strict_reward(response, answer=""):
        return 1.0 if extract_answer(response) == answer else 0.0

**Reward Always 0.0**

Issue: Reward function too strict

Solution::

    # Before
    def strict_reward(response, answer=""):
        return 1.0 if response == answer else 0.0

    # After
    def flexible_reward(response, answer=""):
        extracted = extract_answer(response)
        return 1.0 if extracted == answer else 0.5

**Model Collapses to Single Output**

Issue: Reward doesn't differentiate outputs well

Solution: Diversify reward signal::

    def diverse_reward(response: str, **kwargs) -> float:
        correctness = evaluate_correctness(response, kwargs.get("answer", ""))
        style = evaluate_style_diversity(response)
        return 0.8 * correctness + 0.2 * style

Debugging Reward Functions
---------------------------

Trace reward computation::

    def debug_reward(response: str, **kwargs) -> float:
        """Reward with debug output."""

        print(f"Response: {response[:50]}...")

        correctness = evaluate_correctness(response, kwargs.get("answer", ""))
        print(f"Correctness: {correctness}")

        format_match = evaluate_format(response, kwargs.get("format", ""))
        print(f"Format: {format_match}")

        reward = 0.7 * correctness + 0.3 * format_match
        print(f"Final reward: {reward}\n")

        return reward

Run with debug enabled::

    python run_training.py training.reward_function=debug_reward training.log_level=debug

Best Practices
--------------

1. **Start simple**: Begin with basic correctness reward
2. **Test first**: Create test cases before using in training
3. **Analyze distribution**: Plot reward histograms during training
4. **Iterate gradually**: Add one aspect at a time
5. **Balance components**: Ensure no single component dominates
6. **Document design**: Record why you chose specific weights
7. **Version experiments**: Keep records of reward versions used

Next Steps
----------

- :doc:`../api/rewards` - Reward API reference
- :doc:`../guide/training` - Training guide
- :doc:`../guide/hyperparameter_tuning` - Tuning strategies
