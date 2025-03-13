# Iteration 2. GRPO
 
03-12-2025

# Goal

Get a working plot of GRPO by tmw 1pm. 22h left.

## Motivation

Iteration 1 too slow. No time to rejection sample dataset at scale (DeepSeek providers too slow, healing is work). Skip SFT to direct 2.5 as baseline.
1. 2.5 baseline on self-hosted vLLM, plugged into Provider subclass.
2. Give multi-tool calling, can it heal?
3. Start GRPO regardless of "health". Will collect SFT later (detour).
4. Start 224n (need 10h) or poster design.

## Development

## TODO (Ordered)

> Commit by TODO done. Budget 8h total.

A. Code Interpreter.
> 2h/14 = 9m each.
- [ ] Move `dsl` in src. Has file import issues.
- [ ] Refactor devtools debug -> tracer.
- [ ] Script tasks as JSONL no test/sol split.
- [ ] Aggregate load of task/DSL funcs/type signatures/etc into core, tested util.
- [ ] MVP1: func(io) trace 1 string literal func
- [ ] MVP2: func(io) trace of any solver_* in Typer
- [ ] vardiff is purefunc of DSLFunctionCall in new `differ.py`
- [ ] Fuse types in dsl,src.
- [ ] -> vardiff w any grid view
- [ ] Good dataclass format of tracer stats
- [ ] MVP3: vardiff trace of any solver_*
- [ ] MVP4: trace any DSL fp w dataclass stats in JSON file
- [ ] MVP5: choice of arbitrary formatter for prompt iter
- [ ] Bonus: Live shell interpreter w `load(task)`, `dump(stats)`, `clear`, `help`, `man(func)`

B. sLM Baseline
> 2h/16 = 8m each.
- [ ] Github find vLLM script for 2.5B, 1xA100.
- [ ] Batch gen on same input
- [ ] ngrok API w OpenAI-mocked interface
- [ ] Override Provider `complete` for raw arequests
- [ ] MVP1: Prompt 'Hello world' from local
- [ ] Sanitize `evolutionary` prompts to call DSL
- [ ] Good grid format for id.
- [ ] MVP2: Prompt sLM to o DSL literal
- [ ] ```py format reward 
- [ ] Pipe to tempfile, use A.MVP4
- [ ] MVP3: 1 <tool_response> from gen vLLM
- [ ] Reward dataclass: format, compile, % per example, n correct. Weighted method `score()` 
- [ ] Prompt sLM to rethink from tool call
- [ ] MVP4: Better 2nd tool call
- [ ] Bonus: Stats of non-DSL libs (list, count, group)
- [ ] MVP5: Solve easy task eventually w tool iter

C. GRPO Training Run
>

> Broad idea: Refactor local caller exec, remote vLLM gen -> local vLLM trajectories on cuda:0, Dataset Generator on cpu, halt on ```py to cpu code interpreter; cuda:1-3 is backprop. Rest follows open-r1.