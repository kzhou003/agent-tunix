"""Evaluation utilities for GRPO-trained models."""

from tqdm.auto import tqdm

from tunix.generate import sampler as sampler_lib

from .config import GRPOTrainingConfig, INFERENCE_CONFIGS
from .data import SYSTEM_PROMPT, TEMPLATE
from .rewards import MATCH_FORMAT, MATCH_NUMBERS


def create_sampler(
    model,
    tokenizer,
    model_config,
    max_prompt_length: int = 256,
    max_generation_steps: int = 512,
):
    """Create a sampler for text generation.

    Args:
        model: The model to use for generation.
        tokenizer: The tokenizer.
        model_config: Model configuration.
        max_prompt_length: Maximum prompt length.
        max_generation_steps: Maximum generation steps.

    Returns:
        Sampler instance.
    """
    cache_size = max_prompt_length + max_generation_steps + 256

    return sampler_lib.Sampler(
        transformer=model,
        tokenizer=tokenizer,
        cache_config=sampler_lib.CacheConfig(
            cache_size=cache_size,
            num_layers=model_config.num_layers,
            num_kv_heads=model_config.num_kv_heads,
            head_dim=model_config.head_dim,
        ),
    )


def generate(
    question: str | list[str],
    sampler,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.95,
    max_generation_steps: int = 256,
    seed: int | None = None,
) -> str | list[str]:
    """Generate response for given question(s).

    Args:
        question: Single question or list of questions.
        sampler: Sampler instance.
        temperature: Sampling temperature.
        top_k: Top-k sampling parameter.
        top_p: Top-p (nucleus) sampling parameter.
        max_generation_steps: Maximum tokens to generate.
        seed: Random seed for reproducibility.

    Returns:
        Generated response(s).
    """
    if isinstance(question, str):
        input_batch = [
            TEMPLATE.format(
                system_prompt=SYSTEM_PROMPT,
                question=question,
            ),
        ]
    else:
        input_batch = [
            TEMPLATE.format(
                system_prompt=SYSTEM_PROMPT,
                question=q,
            )
            for q in question
        ]

    out_data = sampler(
        input_strings=input_batch,
        max_generation_steps=max_generation_steps,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        echo=False,
        seed=seed,
        eos_tokens=[1, 106],
    )

    output = out_data.text
    if isinstance(question, str):
        return output[0]
    return output


def evaluate(
    dataset,
    sampler,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.95,
    num_passes: int = 1,
    verbose: bool = True,
) -> dict:
    """Evaluate model on a dataset.

    Args:
        dataset: Dataset to evaluate on.
        sampler: Sampler instance.
        temperature: Sampling temperature.
        top_k: Top-k sampling parameter.
        top_p: Top-p sampling parameter.
        num_passes: Number of generation passes per question.
        verbose: Whether to print progress.

    Returns:
        Dictionary with evaluation metrics.
    """
    corr = 0
    partially_corr = 0
    corr_format = 0
    total = 0

    iterator = tqdm(dataset) if verbose else dataset

    for batch in iterator:
        answers = batch["answer"]
        questions = batch["question"]

        multiple_call_responses = [[] for _ in range(len(questions))]

        for p in range(num_passes):
            responses = generate(
                questions, sampler, temperature, top_k, top_p, seed=p
            )
            for idx, response in enumerate(responses):
                multiple_call_responses[idx].append(response)

        for question, multiple_call_response, answer in zip(
            questions, multiple_call_responses, answers
        ):
            corr_ctr_per_question = 0
            partially_corr_per_question = 0
            corr_format_per_question = 0

            for response in multiple_call_response:
                extracted_response = (
                    guess.group(1)
                    if (guess := MATCH_NUMBERS.search(response)) is not None
                    else "-1000000"
                )

                try:
                    if float(extracted_response.strip()) == float(answer.strip()):
                        corr_ctr_per_question += 1

                    ratio = float(extracted_response.strip()) / float(answer.strip())
                    if 0.9 <= ratio <= 1.1:
                        partially_corr_per_question += 1
                except (ValueError, ZeroDivisionError):
                    if verbose:
                        print("SKIPPED")

                # Check format
                if MATCH_FORMAT.search(response) is not None:
                    corr_format_per_question += 1

                if (
                    corr_ctr_per_question > 0
                    and partially_corr_per_question > 0
                    and corr_format_per_question > 0
                ):
                    break

            if corr_ctr_per_question > 0:
                corr += 1
            if partially_corr_per_question > 0:
                partially_corr += 1
            if corr_format_per_question > 0:
                corr_format += 1

            total += 1

            if verbose and total % 10 == 0:
                print(
                    f"===> {corr=}, {total=}, "
                    f"accuracy={corr / total * 100:.2f}%, "
                    f"partial_accuracy={partially_corr / total * 100:.2f}%, "
                    f"format_accuracy={corr_format / total * 100:.2f}%"
                )

    return {
        "correct": corr,
        "total": total,
        "accuracy": corr / total * 100 if total > 0 else 0,
        "partial_accuracy": partially_corr / total * 100 if total > 0 else 0,
        "format_accuracy": corr_format / total * 100 if total > 0 else 0,
    }


def evaluate_with_config(
    dataset,
    sampler,
    config_name: str = "greedy",
    num_passes: int = 1,
    verbose: bool = True,
) -> dict:
    """Evaluate using a predefined inference configuration.

    Args:
        dataset: Dataset to evaluate on.
        sampler: Sampler instance.
        config_name: Name of inference config ("greedy", "standard", "liberal").
        num_passes: Number of generation passes per question.
        verbose: Whether to print progress.

    Returns:
        Dictionary with evaluation metrics.
    """
    if config_name not in INFERENCE_CONFIGS:
        raise ValueError(
            f"Unknown config: {config_name}. "
            f"Available: {list(INFERENCE_CONFIGS.keys())}"
        )

    gen_config = INFERENCE_CONFIGS[config_name]
    return evaluate(
        dataset,
        sampler,
        temperature=gen_config["temperature"],
        top_k=gen_config["top_k"],
        top_p=gen_config["top_p"],
        num_passes=num_passes,
        verbose=verbose,
    )
