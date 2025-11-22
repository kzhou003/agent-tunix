"""Reward functions for GRPO training."""

import re
from typing import Any

from .data import REASONING_START, REASONING_END, SOLUTION_START, SOLUTION_END


# Regex for exact format matching
MATCH_FORMAT = re.compile(
    rf"^[\s]{{0,}}"
    rf"{REASONING_START}.+?{REASONING_END}.*?"
    rf"{SOLUTION_START}(.+?){SOLUTION_END}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)

# Regex for extracting numbers from answers
MATCH_NUMBERS = re.compile(
    rf"{SOLUTION_START}.*?([\d\.]{{1,}})",
    flags=re.MULTILINE | re.DOTALL,
)


def match_format_exactly(
    prompts: list[str],
    completions: list[str],
    **kwargs: Any,
) -> list[float]:
    """Reward if output format matches exactly (3 points)."""
    return [
        3.0 if MATCH_FORMAT.search(response) is not None else 0.0
        for response in completions
    ]


def match_format_approximately(
    prompts: list[str],
    completions: list[str],
    **kwargs: Any,
) -> list[float]:
    """Reward if output format matches partially."""
    scores = []

    for completion in completions:
        score = 0.0
        # Count how many keywords are seen - penalize if too many
        score += 0.5 if completion.count(REASONING_START) == 1 else -0.5
        score += 0.5 if completion.count(REASONING_END) == 1 else -0.5
        score += 0.5 if completion.count(SOLUTION_START) == 1 else -0.5
        score += 0.5 if completion.count(SOLUTION_END) == 1 else -0.5
        scores.append(score)
    return scores


def check_answer(
    prompts: list[str],
    completions: list[str],
    answer: list[str],
    **kwargs: Any,
) -> list[float]:
    """Reward if the answer is correct or partially correct."""
    extracted_responses = [
        guess.group(1) if (guess := MATCH_FORMAT.search(r)) is not None else None
        for r in completions
    ]

    scores = []
    assert len(extracted_responses) == len(answer), (
        f"{extracted_responses} and {answer} have mismatching length"
    )

    for guess, true_answer in zip(extracted_responses, answer):
        score = 0.0
        if guess is None:
            scores.append(0.0)
            continue

        # Correct answer gets 3 points
        if guess == true_answer:
            score += 3.0
        # Match if spaces are seen
        elif guess.strip() == true_answer.strip():
            score += 1.5
        else:
            # Reward if answer is close via ratios
            try:
                ratio = float(guess) / float(true_answer)
                if 0.9 <= ratio <= 1.1:
                    score += 0.5
                elif 0.8 <= ratio <= 1.2:
                    score += 0.25
                else:
                    score -= 1.0  # Penalize wrong answers
            except (ValueError, ZeroDivisionError):
                score -= 0.5  # Penalize parse errors
        scores.append(score)
    return scores


def check_numbers(
    prompts: list[str],
    completions: list[str],
    answer: list[str],
    **kwargs: Any,
) -> list[float]:
    """Extract numbers from answer and check correctness."""
    question = kwargs.get("question", [""] * len(completions))

    extracted_responses = [
        guess.group(1) if (guess := MATCH_NUMBERS.search(r)) is not None else None
        for r in completions
    ]

    scores = []

    # Debug logging for first sample
    if len(question) > 0 and len(completions) > 0:
        print("START ============================")
        print(f"Question: {question[0]}")
        print(f"Answer: {answer[0]}")
        print(f"Response: {completions[0]}")
        print(f"Extracted: {extracted_responses[0]}")
        print("END ==============================")

    for guess, true_answer in zip(extracted_responses, answer):
        if guess is None:
            scores.append(0.0)
            continue
        # Convert to numbers
        try:
            true_answer_float = float(true_answer.strip())
            guess_float = float(guess.strip())
            scores.append(1.5 if guess_float == true_answer_float else 0.0)
        except (ValueError, AttributeError):
            scores.append(0.0)
    return scores


def get_reward_functions() -> list:
    """Get all reward functions for GRPO training."""
    return [
        match_format_exactly,
        match_format_approximately,
        check_answer,
        check_numbers,
    ]
