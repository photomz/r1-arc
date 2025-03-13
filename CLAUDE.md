# R1-ARC Code Guidelines

## Build Commands
- Package manager: `uv` not `pip`.
- Python in uv venv: `upython -m src.{module} --args`
- Run tests: `upython -m pytest`
- Run specific test: `upython -m pytest -k "glob" ` 

## Mandatory Code Style
- Always use type hints with strict mypy configured
- Always use `dataclass`es or `TypedDicts`s over untyped dicts.
- Prefer `@classmethod`s and static constructors in dataclasses `DataObject.from_some_format` over funcs that return dataclasses.
- Any class with internal state must be a dataclass. Prefer pure functions with minimal side effects, over complex object state manipulation.
- Use enums for named constants and type safety
- In every file, attach a Typer app at code logic with most uncertainty, so user can test quickly on cmd line.
- Write succinct comments only when needed for complex logic
- Use async patterns consistently with proper error handling
- Top-level `import`s only, top-level hardcoded data only (lists, dicts). Overriding this needs explicit user approval.
- Varnames should be very symbolic and succinct: rename `time_to_first_token` -> `t1`.

## Project Context
- Research codebase for ARC competition
- Focus on DSL (Domain Specific Language) for grid transformations
- Evolutionary sampling approach following Jeremy's method
- Grid formatters and diffing utilities for visualization
- Multiple model providers supported with async interfaces
- Read latest experiment Iteration in `docs/Iteration_{largest n}*.md` for more context.