from datasets import Dataset, DatasetDict, Features, Sequence, Value, load_dataset
from src.utils.tasks import TASKS, TaskDef
from src.utils import ROOT, FORMATTERS, FormatterNames, get_shape
from src.prompts import render_prompt
from tqdm import tqdm
from collections import defaultdict
import os
from dotenv import load_dotenv
import typer

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


@app.command()
def create(name="arc_plain", username="photonmz"):
    dataset = create_hf_dataset()
    DATASET_DIR.mkdir(exist_ok=True)
    dataset.save_to_disk(DATASET_DIR / name)
    dataset.push_to_hub(f"{username}/{name}", token=os.getenv("HF_TOKEN"))


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
