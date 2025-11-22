"""Data loading and preprocessing for GRPO training."""

import csv
import os
import shutil
from pathlib import Path
from typing import Literal

import grain
import kagglehub
from datasets import load_dataset
import tensorflow_datasets as tfds


# Special tokens for structured output
REASONING_START = "<reasoning>"
REASONING_END = "</reasoning>"
SOLUTION_START = "<answer>"
SOLUTION_END = "</answer>"

SYSTEM_PROMPT = f"""You are given a problem. Think about the problem and \
provide your reasoning. Place it between {REASONING_START} and \
{REASONING_END}. Then, provide the final answer (i.e., just one numerical \
value) between {SOLUTION_START} and {SOLUTION_END}."""

TEMPLATE = """<start_of_turn>user
{system_prompt}

{question}<end_of_turn>
<start_of_turn>model"""


def extract_hash_answer(text: str) -> str | None:
    """Extract the answer after #### from GSM8K format."""
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def download_kaggle_dataset(target_dir: str = "./data/gsm8k") -> str:
    """Download GSM8K dataset from Kaggle."""
    os.makedirs(target_dir, exist_ok=True)
    src = kagglehub.dataset_download("thedevastator/grade-school-math-8k-q-a")
    src = Path(src)
    dst = Path(target_dir)

    for csv_file in src.glob("*.csv"):
        shutil.copy2(csv_file, dst / csv_file.name)
        print(f"Copied {csv_file.name} -> {dst/csv_file.name}")
    return target_dir


def get_dataset(
    data_dir: str,
    split: str = "train",
    source: Literal["tfds", "kaggle", "huggingface"] = "huggingface",
) -> grain.MapDataset:
    """Load and preprocess GSM8K dataset.

    Args:
        data_dir: Directory for data storage/caching.
        split: Dataset split ("train" or "test").
        source: Data source ("tfds", "kaggle", or "huggingface").

    Returns:
        Preprocessed grain MapDataset.
    """
    os.makedirs(data_dir, exist_ok=True)

    if source == "tfds":
        import tensorflow_datasets.text.gsm8k
        data = tfds.data_source(
            "gsm8k",
            split=split,
            data_dir=data_dir,
            builder_kwargs={"file_format": tfds.core.FileFormat.ARRAY_RECORD},
            download=True,
        )
    elif source == "kaggle":
        kaggle_dir = download_kaggle_dataset(data_dir)
        file_name = f"main_{split}.csv"
        csv_path = os.path.join(kaggle_dir, file_name)

        data = []
        with open(csv_path, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append({
                    "question": row["question"],
                    "answer": row["answer"],
                })
    elif source == "huggingface":
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        data = load_dataset("gsm8k", "main", split=split)
    else:
        raise ValueError(f"Unknown source: {source}")

    def _as_text(v):
        return v if isinstance(v, str) else v.decode("utf-8")

    dataset = (
        grain.MapDataset.source(data)
        .shuffle(seed=42)
        .map(
            lambda x: {
                "prompts": TEMPLATE.format(
                    system_prompt=SYSTEM_PROMPT,
                    question=_as_text(x["question"]),
                ),
                "question": _as_text(x["question"]),
                "answer": extract_hash_answer(_as_text(x["answer"])),
            }
        )
    )
    return dataset


def prepare_datasets(
    train_data_dir: str,
    test_data_dir: str,
    source: str = "huggingface",
    micro_batch_size: int = 4,
    num_batches: int = 3738,
    num_test_batches: int = 100,
    train_fraction: float = 1.0,
    num_epochs: int = 1,
) -> tuple:
    """Prepare train, validation, and test datasets.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    dataset = get_dataset(train_data_dir, "train", source).batch(micro_batch_size)[
        :num_batches
    ]

    if train_fraction == 1.0:
        train_dataset = dataset.repeat(num_epochs)
        val_dataset = None
    else:
        train_dataset = dataset[: int(len(dataset) * train_fraction)]
        train_dataset = train_dataset.repeat(num_epochs)
        val_dataset = dataset[int(len(dataset) * train_fraction):].repeat(num_epochs)

    test_dataset = get_dataset(test_data_dir, "test", source).batch(micro_batch_size)[
        :num_test_batches
    ]

    return train_dataset, val_dataset, test_dataset
