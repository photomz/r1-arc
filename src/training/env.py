import re
import typer
import asyncio
from dotenv import load_dotenv
from src.code_interpreter.execution import Interpreter, InterpreterOutput
from src.code_interpreter.static_analysis import StaticMetrics
from src.prompts import render_prompt, TASKS, Example
from src.utils.tasks import deep_tuple, TaskDef
from src.utils import get_shape, FORMATTERS, FormatterNames
from dataclasses import dataclass
import numpy as np
import math

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
    r1 = 0.5
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
        # format, compile, % per example, n correct. Weighted method `score()`
        run_reward = cls.r1 * out.run_ok

        r = run_reward

        O_hats = out.traces.O
        Os = [deep_tuple(e.O) for e in ex]

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


def reward_exec(prompts, completions, **kwargs) -> list[float]:
    rewards = []
    for p, c, k in zip(prompts, completions, kwargs.values()):
        try:
            task = TaskDef(**k)
            codestring = extract_python(c)

            if not codestring:
                debug("No Python", c)
                rewards.append(0)
                continue

            Is = [t.I for t in task.data]
            res = Interpreter.run(
                codestring=codestring, inputs=Is, id=task.id, raise_error=False
            )
            print(completions)
            rewards.append(FirstSoftReward.score_exec(res, task.data))

        except Exception as e:
            debug(e)  # WARN: Last resort zeros reward, want to catch earlier.
            rewards.append(0)
    return rewards


def reward_style(completions, **kwargs) -> list[float]:
    rewards = []
    for c, k in zip(completions, kwargs.values()):
        try:
            codestring = extract_python(c)
            if not codestring:
                debug("No Python", c)
                rewards.append(0)
                continue

            metrics = StaticMetrics.from_literal(codestring)
            rewards.append(FirstSoftReward.score_style(metrics))
        except Exception as e:
            debug(e)
            rewards.append(0)
    return rewards


REWARD_FNS = [reward_exec, reward_style]


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
