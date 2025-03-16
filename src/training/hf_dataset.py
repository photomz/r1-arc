from datasets import Dataset, DatasetDict, Features, Sequence, Value, load_dataset
from src.utils.tasks import TASKS, TaskDef
from src.utils.devtools import TOKENIZERS
from src.utils import ROOT, FORMATTERS, FormatterNames, get_shape
from src.prompts import render_prompt
from tqdm import tqdm
from collections import defaultdict
import os
import numpy as np
from dotenv import load_dotenv
import typer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

load_dotenv()

HfGrid = Sequence(Sequence(Value("int32")))
FEATURES = Features(
    {
        "id": Value("string"),
        # 4d array for lists of examples, each example has two 2D grids (in/out).
        "train": Sequence(Sequence(HfGrid)),
        "test": Sequence(Sequence(HfGrid)),
        "solver": Value("string"),
        "augment": Value("null"),
        "difficulty": Value("int32"),
    }
)


def create_hf_dataset() -> DatasetDict:
    dataset_dict = {}

    for split in ["train", "eval"]:
        # Filter tasks by split
        tasks_by_split = [
            (id, task) for id, task in TASKS.items() if task.split == split
        ]

        data = defaultdict(list)
        for id, task in tqdm(tasks_by_split):
            task_data = task.dumps()
            for key in task_data:
                data[key].append(task_data[key])

        # Create HF Dataset for this split
        dataset_dict[split] = Dataset.from_dict(data, features=FEATURES)

    return DatasetDict(dataset_dict)


app = typer.Typer()

DATASET_DIR = ROOT / "datasets"


def plot_prompt_length_frequency(dataset: Dataset):
    token_counts = []
    task_ids = []

    for split in dataset:
        for i, example in enumerate(
            tqdm(dataset[split], desc=f"Counting tokens in {split}")
        ):
            prompt = render_prompt(TaskDef.from_hf(example))
            token_count = len(TOKENIZERS["deepseek"]["encode"](prompt))
            token_counts.append(token_count)
            task_ids.append(example["id"])

    token_counts = np.array(token_counts)

    # Create frequency histogram with hover data showing task IDs
    fig = px.histogram(
        x=token_counts,
        labels={"x": "Token Count", "y": "Frequency"},
        title=f"Token Count Distribution for {split} split",
        hover_data=[task_ids],
    )

    # Add scatter plot for individual points with task IDs
    fig.add_trace(
        go.Scatter(
            x=token_counts,
            y=[0] * len(token_counts),  # Place at bottom of histogram
            mode="markers",
            marker=dict(size=5, color="rgba(0,0,0,0.5)"),
            text=task_ids,
            hoverinfo="text",
            showlegend=False,
        )
    )

    fig.update_layout(
        xaxis_title="Token Count",
        yaxis_title="Number of Tasks",
        hovermode="closest",
    )

    fig.show()


@app.command()
def create(name="arc_plain", username="photonmz"):
    dataset = create_hf_dataset()
    # plot_prompt_length_frequency(dataset)

    if input("Upload? (y/n): ") != "y":
        return

    DATASET_DIR.mkdir(exist_ok=True)
    dataset.save_to_disk(DATASET_DIR / name)
    dataset.push_to_hub(f"{username}/{name}", token=os.getenv("HF_TOKEN"))

    print(
        f"View uploaded dataset at: https://huggingface.co/datasets/{username}/{name}"
    )


fmt = FORMATTERS[FormatterNames.SPREADSHEET].format


@app.command()
def load(name="photonmz/arc_plain"):
    data0 = load_dataset(name).remove_columns(["augment"])
    data0 = data0.map(
        lambda x: {
            "prompt": [
                {"role": "user", "content": render_prompt(TaskDef.from_hf(x))},
            ],
            **x,
        }
    )
    print(f"Loaded dataset.")
    return data0


if __name__ == "__main__":
    app()
