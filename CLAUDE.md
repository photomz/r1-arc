# R1-ARC Code Guidelines

## Build Commands
- Run main project: `upython src/main.py`
- Run tests: `upython -m pytest`
- Run specific test: `upython -m pytest tests.py::test_function_name` 
- Generate stubs: `make stubs` (in arc-dsl directory)
- Lint: `ruff check .`
- Typecheck: `mypy src/`

## Code Style
- Use snake_case for variables/functions, PascalCase for classes, UPPER_CASE for constants
- Always use type hints with strict mypy configured
- Prefer dataclasses and class methods for formatting components
- Use enums for named constants and type safety
- Add Typer functions for rapid testing/debugging
- Follow clean symmetrical structure with symbolic variable names
- Write succinct comments only when needed for complex logic
- Use async patterns consistently with proper error handling
- Group imports: std lib, third-party, local modules
- Prefer frozen sets and tuples for immutable data structures

## Project Context
- Research codebase for ARC competition
- Focus on DSL (Domain Specific Language) for grid transformations
- Evolutionary sampling approach following Jeremy's method
- Grid formatters and diffing utilities for visualization
- Multiple model providers supported with async interfaces
- Read latest experiment Iteration in `docs/Iteration_{largest n}*.md` for more context.