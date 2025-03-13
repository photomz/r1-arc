#!/usr/bin/env python3
"""
Script to restructure task dataset into train.jsonl and eval.jsonl files.
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

import typer
from tqdm import tqdm


def process_files(source_dir: Path, output_file: Path, desc: str) -> None:
    """Process all JSON files in source_dir and write to output_file."""
    files = list(source_dir.glob("*.json"))

    # Group examples by task_id
    tasks = {}

    # Process all files
    for file_path in tqdm(files, desc=f"Reading {desc}"):
        try:
            with open(file_path, "r") as f_in:
                data = json.load(f_in)

                # Extract task ID from filename
                task_id = file_path.stem

                # Create task structure
                task_data = {"task_id": task_id, "train": [], "test": []}

                # Add examples to appropriate lists
                for split in ["train", "test"]:
                    if split in data:
                        for example in data[split]:
                            task_data[split].append(
                                {"input": example["input"], "output": example["output"]}
                            )

                tasks[task_id] = task_data
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Write tasks to output file
    with open(output_file, "w") as f_out:
        for task_id in sorted(tasks.keys()):
            f_out.write(json.dumps(tasks[task_id]) + "\n")


def main(
    data_dir: Path = Path("src/dsl/data"),
    output_dir: Path = Path("src/data"),
    overwrite: bool = False,
) -> None:
    """
    Restructure task dataset into train.jsonl and eval.jsonl files.

    Args:
        data_dir: Directory containing training and evaluation data
        output_dir: Directory to write output files
        overwrite: Whether to overwrite existing output files
    """
    train_dir = data_dir / "training"
    eval_dir = data_dir / "evaluation"

    train_output = output_dir / "train.jsonl"
    eval_output = output_dir / "eval.jsonl"

    # Check if output files already exist
    if not overwrite and (train_output.exists() or eval_output.exists()):
        raise FileExistsError(
            f"Output files already exist. Use --overwrite to overwrite."
        )

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process training and evaluation files
    process_files(train_dir, train_output, "training files")
    process_files(eval_dir, eval_output, "evaluation files")

    print(f"Done! Output written to {train_output} and {eval_output}")


if __name__ == "__main__":
    typer.run(main)
