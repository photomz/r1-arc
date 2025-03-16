import re
import typer
import asyncio
from dotenv import load_dotenv
from src.code_interpreter.execution import Interpreter, InterpreterOutput, pprint
from src.code_interpreter.static_analysis import StaticMetrics
from src.prompts import render_prompt, TASKS, Example
from src.utils.tasks import deep_tuple, TaskDef
from src.utils import get_shape, FORMATTERS, FormatterNames, ROOT, debug
from dataclasses import dataclass
import time
import numpy as np
import orjson as json
import math
import traceback

load_dotenv()

from src.utils.devtools import debug
from src.providers.models import ModelName, ProviderName, providers

app = typer.Typer()

fmt = FORMATTERS[FormatterNames.CELL_DIFF]


@dataclass
class FirstSoftReward:
    p0 = 0.1
    p1 = 0.1
    p2 = 0.2
    r0 = 0.1
    r1 = 0.7
    r2 = 0.5
    e1 = 5
    c1 = 1
    r3 = 2
    r4 = 1

    @classmethod
    def score_style(cls, metrics: StaticMetrics) -> float:
        m = metrics
        # Only nest penalty after correct.
        nest_penalty = cls.p0 * (m.n_branch + m.n_loop)
        # Bloat after 3 funcs. Likely copied.
        funcdef_penalty = cls.p1 * max((m.n_funcs - 3), 0)
        # Steep penalty after 100 lines.
        llength_penalty = cls.p2 * max((m.n_lines - 100), 0)

        format_reward = cls.r0 * (m.syntax_ok and m.format_ok)
        dsl_reward = cls.r4 * math.sqrt(len(m.n_dsl))
        # Stagger reward payout; refuse to give run reward unless format is ok.

        r = (format_reward + dsl_reward) - (funcdef_penalty + llength_penalty)
        return r

    @classmethod
    def score_exec(cls, out: InterpreterOutput, ex: Example) -> float:
        """Interpreter Output -> R reward."""
        if not out.run_ok:
            return -0.5

        # format, compile, % per example, n correct. Weighted method `score()`
        run_reward = cls.r1 * out.run_ok
        r = run_reward

        O_hats = out.traces.O
        Os = [deep_tuple(e.O) for e in ex]

        # TODO: Should avg on output examples. In reality, most 1 test output so not a problem.
        for ohat, o in zip(O_hats, Os):
            A, B = np.array(ohat), np.array(o)
            oshape_ok = A.shape == B.shape
            shape_reward = cls.r2 * oshape_ok

            pcorrect = 0
            if oshape_ok:
                pcorrect = np.sum(A == B) / np.prod(A.shape)
                # Steep at last few pixels
                softcorrect_reward = cls.c1 * (pcorrect**cls.e1)
            debug(oshape_ok, pcorrect, softcorrect_reward)

            r += shape_reward + softcorrect_reward

        # if all correct, bonus reward
        if all(np.array_equal(A, B) for A, B in zip(O_hats, Os)):
            r += cls.r3

        return r


def extract_python(o) -> str:
    # TODO: n-tool call doesn't pipe through.
    o = re.search(r"```(?:py|python)\n(.*?)\n```", o, re.DOTALL)
    if not o:
        debug("No code block found.")
        return ""
    return o.group(1)


def iterate_kwargs(kw: dict):
    # Get the length of any list in kwargs (assuming all lists have same length)
    n = len(list(kw.values())[0])

    # For each index, create a dict with the i-th element of each list
    return [{k: v[i] for k, v in kw.items()} for i in range(n)]


formatnow = lambda: time.strftime("%Y%m%d_%H%M%S")


def log_badcode(title, attempt, error):
    filename = f"{title}-{formatnow()}.md"
    tmp_dir = ROOT / "src/tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fp = tmp_dir / filename

    with fp.open("w") as f:
        f.write(f"# {title}\n")
        f.write(f"\nTime: _{formatnow()}_\n")
        f.write(f"## Error\n{error}")
        f.write(f"## Bad Generation\n{attempt}\n")


def guardrail(reward_fn):
    # args are typedef'd in hf_dataset.py
    # and HF's GRPOTrainer adds completions, reads prompts.
    def wrap_r(completions, id, **kwargs) -> list[float]:
        rewards = []
        items = (
            zip(completions, iterate_kwargs({"id": id, **kwargs}))
            if kwargs
            else [(c, {}) for c in completions]
        )
        for c, kw in items:
            content = c[0].get("content")

            try:
                codestring = extract_python(
                    c[0]["content"] if isinstance(c, list) else c
                )
                print(f"<{id[0]} start={formatnow()}>")

                if not codestring:
                    print(">> No Python. End Completion is")
                    print(c[-1]["content"][-10000:])
                    rewards.append(0)
                    continue

                pprint(codestring)

                rewards.append(reward_fn(codestring, **kw))
            except Exception as e:
                debug(e)
                log_badcode(id[0], content, f"{str(e)}\n{traceback.format_exc()}")
                rewards.append(0)
            print(f"<{id[0]}/>")
        return rewards

    return wrap_r


# Simplified core reward functions that work with the guardrail
def reward_exec(codestring: str, **kwargs) -> float:
    task = TaskDef.from_hf(kwargs)
    Is = [t.I for t in task.data]
    res = Interpreter.run(
        codestring=codestring, inputs=Is, id=task.id, raise_error=False
    )
    return FirstSoftReward.score_exec(res, task.data)


def reward_style(codestring: str, **kwargs) -> float:
    metrics = StaticMetrics.from_literal(codestring)
    return FirstSoftReward.score_style(metrics)


# HF gets my reward func name for wandb logging
# https://github.com/huggingface/trl/blob/fc4dae256d924dfbb906af9c2e817bc6fb7b590b/trl/trainer/grpo_trainer.py#L833
exec_reward = guardrail(reward_exec)
exec_reward.__name__ = "exec"
style_reward = guardrail(reward_style)
style_reward.__name__ = "style"

REWARD_FNS = [exec_reward, style_reward]


def debug_in_terminal(o: str, tid="5521c0d9") -> None:
    # TODO: n-tool call doesn't pipe through.
    o = re.search(r"```(?:py|python)\n(.*?)\n```", o, re.DOTALL)
    if not o:
        debug("No code block found.")
        return None

    codestring = o.group(1)
    ex = TASKS[tid]
    testinputs = [t.I for t in ex.data]

    # Run interpreter and get results
    res = Interpreter.run(
        codestring=codestring, inputs=testinputs, id=tid, raise_error=False
    )
    metrics = StaticMetrics.from_literal(codestring)

    r1 = debug(FirstSoftReward.score_exec(res, ex.data))
    r2 = debug(FirstSoftReward.score_style(metrics))
    return r1, r2
