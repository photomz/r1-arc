# @pytest.mark.stress
# @pytest.mark.parametrize("func", ["solver_repr", "verifier_repr"])
# @pytest.mark.parametrize("task_id", train_ids)
# def test_solve_any_id_in_interpreter(task_id, func, snapshot):
#     task = TASKS[task_id]
#     test_Is = [ex.I for ex in task.train + task.test]

#     codestring = getattr(task, func)
#     res = Interpreter.run(codestring, inputs=test_Is, id=task_id, cleanup=False)
#     metrics = StaticMetrics.from_literal(codestring)

#     assert metrics.syntax_ok and res.run_ok

#     shell_Is = res.traces.I
#     assert shell_Is == test_Is

#     shell_Os = res.traces.O
#     test_Os = [deep_tuple(ex.O) for ex in task.train + task.test]
#     assert get_shape(shell_Os) == get_shape(test_Os)
#     for want, expect in zip(shell_Os, test_Os):
#         assert want == expect, fmt.diff(
#             want, expect, get_shape(want), get_shape(expect)
#         )

#     assert (res, metrics) == snapshot


from src.utils import TASKS, ROOT
from src.training.env import debug_in_terminal, REWARD_FNS
import pytest
import json

train_ids = [t.id for t in TASKS.values() if t.split == "train"]


def frontmatter(codestr: str) -> str:
    # Mock LLM gen of Markdown-formatted frontmatter.
    return f"""
# Solution
We observe that boo deee boo dee dooo ...
and so we implement the Python code as follows:
```py
{codestr}
```
Observe that blah blah, and we're done.
"""


@pytest.mark.parametrize("task_id", train_ids)
def test_solver_passes_env_reward(task_id, snapshot):
    codestring = frontmatter(TASKS[task_id].solver_repr)
    r = debug_in_terminal(codestring, task_id)
    assert r is not None
    assert r == snapshot


traceback = ROOT / "src/training/traceback.json"


def test_reward_args_regression(snapshot):
    args = json.load(traceback.open())

    kwargs = {k: [v] for k, v in args[2].items()}
    rs = REWARD_FNS[0](prompts=[args[0]], completions=[args[1]], **kwargs)
    assert rs == snapshot
