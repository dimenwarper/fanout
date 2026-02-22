#!/usr/bin/env python3
"""KernelBench eval script for fanout.

Usage: ./eval.py <solution_file> [task_file]

The solution file should contain a ModelNew class (a CUDA-optimized replacement
for the reference Model). This script:
  1. Loads the reference task (Model) and the solution (ModelNew)
  2. Checks correctness via torch.allclose
  3. Measures speedup over the reference
  4. Prints a score (0.0-1.0) on the last line

Requires: torch with CUDA support, the task file in tasks/
"""

from __future__ import annotations

import importlib.util
import re
import sys
import tempfile
from pathlib import Path


def strip_code_fences(text: str) -> str:
    """Strip markdown code fences (```python ... ```) from LLM output."""
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text.strip())
    return text


def load_module(path: str, name: str):
    source = Path(path).read_text()
    cleaned = strip_code_fences(source)

    if cleaned != source:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
        tmp.write(cleaned)
        tmp.close()
        path = tmp.name

    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <solution_file> <task_file>", file=sys.stderr)
        print("0.0")
        sys.exit(0)

    solution_path = sys.argv[1]
    task_path = sys.argv[2]

    try:
        import torch
    except ImportError:
        print("torch not installed", file=sys.stderr)
        print("0.0")
        sys.exit(0)

    try:
        task = load_module(task_path, "task")
        sol = load_module(solution_path, "solution")
    except Exception as e:
        print(f"Load error: {e}", file=sys.stderr)
        print("0.0")
        sys.exit(0)

    if not hasattr(sol, "ModelNew"):
        print("No ModelNew class found in solution", file=sys.stderr)
        print("0.0")
        sys.exit(0)

    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        print("0.0")
        sys.exit(0)

    device = torch.device("cuda")
    init_inputs = task.get_init_inputs()

    try:
        ref_model = task.Model(*init_inputs).to(device).eval()
        new_model = sol.ModelNew(*init_inputs).to(device).eval()
    except Exception as e:
        print(f"Model init error: {e}", file=sys.stderr)
        print("0.0")
        sys.exit(0)

    # Correctness check (3 trials)
    NUM_CORRECTNESS_TRIALS = 3
    for trial in range(NUM_CORRECTNESS_TRIALS):
        inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in task.get_inputs()]
        try:
            with torch.no_grad():
                ref_out = ref_model(*inputs)
                new_out = new_model(*inputs)
            if not torch.allclose(ref_out, new_out, atol=1e-4, rtol=1e-4):
                print(f"Correctness check failed on trial {trial}", file=sys.stderr)
                print("0.0")
                sys.exit(0)
        except Exception as e:
            print(f"Runtime error on trial {trial}: {e}", file=sys.stderr)
            print("0.0")
            sys.exit(0)

    # Speedup measurement
    NUM_WARMUP = 5
    NUM_TIMING = 20

    def time_model(model, get_inputs_fn):
        for _ in range(NUM_WARMUP):
            inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in get_inputs_fn()]
            with torch.no_grad():
                model(*inputs)
        torch.cuda.synchronize()

        times = []
        for _ in range(NUM_TIMING):
            inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in get_inputs_fn()]
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.no_grad():
                model(*inputs)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        return sum(times) / len(times)

    ref_time = time_model(ref_model, task.get_inputs)
    new_time = time_model(new_model, task.get_inputs)

    if new_time <= 0:
        print("0.0")
        sys.exit(0)

    speedup = ref_time / new_time

    # Score: 0.0 if slower, linear 0-1 mapped from 1x-3x speedup, capped at 1.0
    score = max(0.0, min(1.0, (speedup - 1.0) / 2.0))

    print(f"ref={ref_time:.2f}ms new={new_time:.2f}ms speedup={speedup:.2f}x", file=sys.stderr)
    print(f"{score:.4f}")


if __name__ == "__main__":
    main()
