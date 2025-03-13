"""
CLI for ARC execution server

This module provides a command-line interface for executing ARC-DSL solvers on tasks.
"""

import os
import sys
import json
import typer
import importlib
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

# Add root directory to path
root_dir = Path(__file__).resolve().parents[3]
sys.path.append(str(root_dir))
arc_dsl_dir = root_dir / "arc-dsl"
sys.path.append(str(arc_dsl_dir))

# Import execution functions
from src.utils.exec_server.execution import (
    run_python_transform_sync,
    PythonResult,
    TransformInput,
    ResourceLimits,
)
from src.utils.devtools import verbose_debug

app = typer.Typer()

def load_task(task_path: Path) -> Dict[str, Any]:
    """Load a task from a JSON file"""
    with open(task_path, "r") as f:
        return json.load(f)

def find_task_file(task_id: str, tasks_dir: Path) -> Optional[Path]:
    """Find the task file containing the given task ID"""
    potential_files = [
        tasks_dir / "train.json",
        tasks_dir / "evalpub.json", 
        tasks_dir / "train_sol.json",
        tasks_dir / "evalpub_sol.json",
    ]
    
    for file_path in potential_files:
        if file_path.exists():
            try:
                task_data = load_task(file_path)
                if task_id in task_data:
                    return file_path
            except json.JSONDecodeError:
                continue
    
    return None

def extract_solver_function(task_id: str) -> str:
    """Extract the solver function from arc-dsl/solvers.py for the given task ID"""
    # Import the solvers module from arc-dsl
    try:
        sys.path.insert(0, str(arc_dsl_dir))
        import solvers as arc_solvers
        
        solver_name = f"solve_{task_id}"
        if not hasattr(arc_solvers, solver_name):
            return f"# No solver found for task ID: {task_id}\ndef solve(I):\n    return I"
        
        # Get the source code of the solver function
        import inspect
        solver_func = getattr(arc_solvers, solver_name)
        source_code = inspect.getsource(solver_func)
        
        # Replace 'return 0' with 'return verbose_debug(0)'
        source_code = source_code.replace("return 0", "return verbose_debug(0)")
        
        # Add imports for verbose_debug
        source_code = "from src.utils.devtools import verbose_debug\n\n" + source_code
        
        return source_code
    except Exception as e:
        return f"# Error extracting solver: {str(e)}\ndef solve(I):\n    return I"

def prepare_grid_inputs(task_data: Dict[str, Any], task_id: str) -> List[List[List[int]]]:
    """Prepare grid inputs from task data for the given task ID"""
    task_instance = task_data[task_id]
    
    # Handle different task formats
    if isinstance(task_instance, dict) and "train" in task_instance:
        # Challenge format
        return [example["input"] for example in task_instance["train"]]
    elif isinstance(task_instance, list):
        if isinstance(task_instance[0], dict) and "attempt_1" in task_instance[0]:
            # Submission format
            return [attempt["attempt_1"] for attempt in task_instance]
        else:
            # Solution format
            return task_instance
    
    # Default case - empty grid
    return [[[0, 0], [0, 0]]]

@app.command()
def run(
    task_id: str = typer.Option(..., "--id", help="Task ID to run"),
    task_file: Optional[str] = typer.Option(None, "--task", help="Path to task file (defaults to src/tasks/*.json)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show verbose output"),
):
    """Run a solver on a task using the execution server"""
    # Find the task file
    if task_file:
        task_path = Path(task_file)
        if not task_path.exists():
            typer.echo(f"Task file not found: {task_file}")
            raise typer.Exit(1)
    else:
        # Look in src/tasks
        tasks_dir = root_dir / "src" / "tasks"
        task_path = find_task_file(task_id, tasks_dir)
        if not task_path:
            typer.echo(f"Task ID {task_id} not found in any of the task files")
            raise typer.Exit(1)
    
    # Load the task
    task_data = load_task(task_path)
    if task_id not in task_data:
        typer.echo(f"Task ID {task_id} not found in {task_path}")
        raise typer.Exit(1)
    
    # Extract solver function
    solver_code = extract_solver_function(task_id)
    
    # Prepare grid inputs
    grid_inputs = prepare_grid_inputs(task_data, task_id)
    
    # Run the solver using execution server
    typer.echo(f"Running solver for task {task_id} from {task_path}")
    
    result = run_python_transform_sync(
        code=solver_code,
        grid_lists=grid_inputs,
        resource_limits=ResourceLimits(max_time_seconds=30),
        raise_exception=False,
    )
    
    # Print results
    if result.return_code != 0:
        typer.echo(f"Error executing solver: {result.stderr}")
        raise typer.Exit(1)
    
    # Display debug outputs - these come from verbose_debug(0)
    if hasattr(result, 'debug_outputs') and result.debug_outputs:
        typer.echo("\nDebug Output (local variables when returning 0):")
        for debug_out in result.debug_outputs:
            typer.echo(debug_out)
    
    if result.transform_results:
        typer.echo("\nResults:")
        for i, res in enumerate(result.transform_results):
            typer.echo(f"\nExample {i+1}:")
            for row in res:
                typer.echo(row)
    else:
        typer.echo("No results returned")
    
    # Show stdout if verbose
    if verbose and result.stdout:
        typer.echo("\nFull Stdout:")
        typer.echo(result.stdout)

if __name__ == "__main__":
    app()