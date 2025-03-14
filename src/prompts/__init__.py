from src.utils import ROOT
from src.utils.tasks import TASKS
import jinja2
from pathlib import Path
from src.utils.models import TaskDef, Example

# Setup Jinja2 environment for templates
PROMPT_DIR = ROOT / "src/prompts"
jinja_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(PROMPT_DIR),
    trim_blocks=True,
    lstrip_blocks=True,
)


PROMPT = "basic.md"
COT = "cot_1f642eb9.md"
cot_repr = (PROMPT_DIR / COT).open().read()


def render_prompt(task: TaskDef) -> str:
    template = jinja_env.get_template(PROMPT)
    return template.render(cot=cot_repr, task=(task.format_prompt()))


if __name__ == "__main__":
    print(render_prompt(TASKS["1f642eb9"]))
