#!/usr/bin/env python3
"""MNIST-Weights eval script for fanout.

Usage: ./eval.py <solution_file> [task_name]

task_name: classify_all | binary_0v1 | binary_3v8 | top5_accuracy

The solution file must define the task's entry function returning a weight dict.
Prints the accuracy score on the last line.

Anti-cheat: solutions must produce raw weight numbers directly — no training
allowed. Banned imports (sklearn, torch, tensorflow, etc.) and .fit() calls
are detected via static analysis, and a 2-second time limit is enforced.
"""

from __future__ import annotations

import importlib.util
import re
import signal
import sys
import time

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# ── Anti-cheat ──────────────────────────────────────────

BANNED_IMPORTS = [
    r'^\s*(?:from|import)\s+sklearn\b',
    r'^\s*(?:from|import)\s+torch\b',
    r'^\s*(?:from|import)\s+tensorflow\b',
    r'^\s*(?:from|import)\s+keras\b',
    r'^\s*(?:from|import)\s+jax\b',
    r'^\s*(?:from|import)\s+flax\b',
    r'^\s*(?:from|import)\s+xgboost\b',
    r'^\s*(?:from|import)\s+lightgbm\b',
    r'^\s*(?:from|import)\s+catboost\b',
]

BANNED_PATTERNS = [
    r'\.fit\s*\(',
    r'\.train\s*\(',
    r'\.backward\s*\(',
    r'optimizer\.step\s*\(',
    r'\.compile\s*\(',
]

FUNC_TIME_LIMIT = 2.0  # seconds


def _strip_comments_and_strings(source: str) -> str:
    """Remove comments, docstrings, and string literals to check only real code."""
    # Remove triple-quoted strings (docstrings)
    source = re.sub(r'"""[\s\S]*?"""', '', source)
    source = re.sub(r"'''[\s\S]*?'''", '', source)
    # Remove single-quoted and double-quoted strings
    source = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', '', source)
    source = re.sub(r"'[^'\\]*(?:\\.[^'\\]*)*'", '', source)
    # Remove line comments
    source = re.sub(r'#.*$', '', source, flags=re.MULTILINE)
    return source


def _check_source(path: str) -> str | None:
    """Scan source for banned imports/patterns. Returns error message or None."""
    try:
        with open(path) as f:
            source = f.read()
    except Exception:
        return None  # can't read, let load_module handle it

    code_only = _strip_comments_and_strings(source)

    for pattern in BANNED_IMPORTS:
        if re.search(pattern, code_only, re.MULTILINE):
            return (
                f"CHEATING DETECTED: source contains banned import matching '{pattern}'. "
                f"This benchmark requires you to produce raw weight numbers directly "
                f"(numpy arrays of floats). You may NOT import or use any ML framework "
                f"(sklearn, torch, tensorflow, etc.) to train a model. "
                f"Think about what patterns distinguish the digits and encode that "
                f"knowledge directly into the weight values."
            )

    for pattern in BANNED_PATTERNS:
        if re.search(pattern, code_only):
            return (
                f"CHEATING DETECTED: source contains banned pattern matching '{pattern}'. "
                f"This benchmark requires you to produce raw weight numbers directly — "
                f"no training, fitting, or optimization loops allowed. "
                f"You must craft the weight values yourself using mathematical reasoning "
                f"about digit patterns."
            )

    return None


class _TimeoutError(Exception):
    pass


def _call_with_timeout(func, timeout: float):
    """Call func() with a wall-clock time limit."""
    def _handler(signum, frame):
        raise _TimeoutError()

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        result = func()
    except _TimeoutError:
        raise RuntimeError(
            f"CHEATING DETECTED: function took longer than {timeout}s. "
            f"This benchmark requires you to produce raw weight numbers directly — "
            f"no training or optimization loops allowed. Your function should just "
            f"return numpy arrays of pre-computed float values, which takes milliseconds."
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
    return result


def load_module(path: str):
    spec = importlib.util.spec_from_file_location("solution", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Data loading ──────────────────────────────────────────

def _get_data_10class():
    """Load 8x8 digits, split with fixed seed."""
    digits = load_digits()
    X = digits.data / 16.0  # normalize to [0, 1]
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    return X_test, y_test


def _get_data_binary(digit_a: int, digit_b: int):
    """Load binary subset of 8x8 digits."""
    digits = load_digits()
    mask = (digits.target == digit_a) | (digits.target == digit_b)
    X = digits.data[mask] / 16.0
    y = (digits.target[mask] == digit_b).astype(np.float32)  # digit_b=1, digit_a=0
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    return X_test, y_test


# ── Forward passes ──────────────────────────────────────────

def _forward_10class(weights: dict, X: np.ndarray) -> np.ndarray:
    """Forward pass for 64->16->10 MLP, returns logits."""
    W1 = np.array(weights["W1"], dtype=np.float32)
    b1 = np.array(weights["b1"], dtype=np.float32)
    W2 = np.array(weights["W2"], dtype=np.float32)
    b2 = np.array(weights["b2"], dtype=np.float32)

    if W1.shape != (64, 16) or b1.shape != (16,):
        raise ValueError(f"Layer 1 shape mismatch: W1={W1.shape}, b1={b1.shape}")
    if W2.shape != (16, 10) or b2.shape != (10,):
        raise ValueError(f"Layer 2 shape mismatch: W2={W2.shape}, b2={b2.shape}")

    h = X @ W1 + b1
    h = np.maximum(h, 0)  # ReLU
    logits = h @ W2 + b2
    return logits


def _forward_binary(weights: dict, X: np.ndarray) -> np.ndarray:
    """Forward pass for 64->8->1 binary MLP, returns probabilities."""
    W1 = np.array(weights["W1"], dtype=np.float32)
    b1 = np.array(weights["b1"], dtype=np.float32)
    W2 = np.array(weights["W2"], dtype=np.float32)
    b2 = np.array(weights["b2"], dtype=np.float32)

    if W1.shape != (64, 8) or b1.shape != (8,):
        raise ValueError(f"Layer 1 shape mismatch: W1={W1.shape}, b1={b1.shape}")
    if W2.shape != (8, 1) or b2.shape != (1,):
        raise ValueError(f"Layer 2 shape mismatch: W2={W2.shape}, b2={b2.shape}")

    h = X @ W1 + b1
    h = np.maximum(h, 0)  # ReLU
    logit = (h @ W2 + b2).flatten()
    prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -500, 500)))  # sigmoid
    return prob


# ── Evaluators ──────────────────────────────────────────

def _validate_weights(weights, expected_keys):
    if not isinstance(weights, dict):
        print(f"Expected dict, got {type(weights)}", file=sys.stderr)
        return False
    for key in expected_keys:
        if key not in weights:
            print(f"Missing key: {key}", file=sys.stderr)
            return False
    return True


def eval_classify_all(sol) -> float:
    BENCHMARK = 0.50

    if not hasattr(sol, "classify_all"):
        print("Missing classify_all()", file=sys.stderr)
        return 0.0

    try:
        weights = _call_with_timeout(sol.classify_all, FUNC_TIME_LIMIT)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 0.0
    if not _validate_weights(weights, ["W1", "b1", "W2", "b2"]):
        return 0.0

    X_test, y_test = _get_data_10class()

    try:
        logits = _forward_10class(weights, X_test)
    except ValueError as e:
        print(f"Forward error: {e}", file=sys.stderr)
        return 0.0

    preds = np.argmax(logits, axis=1)
    acc = np.mean(preds == y_test)
    print(f"accuracy={acc:.4f} benchmark={BENCHMARK}", file=sys.stderr)
    return float(acc)


def eval_binary_0v1(sol) -> float:
    BENCHMARK = 0.90

    if not hasattr(sol, "binary_0v1"):
        print("Missing binary_0v1()", file=sys.stderr)
        return 0.0

    try:
        weights = _call_with_timeout(sol.binary_0v1, FUNC_TIME_LIMIT)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 0.0
    if not _validate_weights(weights, ["W1", "b1", "W2", "b2"]):
        return 0.0

    X_test, y_test = _get_data_binary(0, 1)

    try:
        probs = _forward_binary(weights, X_test)
    except ValueError as e:
        print(f"Forward error: {e}", file=sys.stderr)
        return 0.0

    preds = (probs >= 0.5).astype(np.float32)
    acc = np.mean(preds == y_test)
    print(f"accuracy={acc:.4f} benchmark={BENCHMARK}", file=sys.stderr)
    return float(acc)


def eval_binary_3v8(sol) -> float:
    BENCHMARK = 0.80

    if not hasattr(sol, "binary_3v8"):
        print("Missing binary_3v8()", file=sys.stderr)
        return 0.0

    try:
        weights = _call_with_timeout(sol.binary_3v8, FUNC_TIME_LIMIT)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 0.0
    if not _validate_weights(weights, ["W1", "b1", "W2", "b2"]):
        return 0.0

    X_test, y_test = _get_data_binary(3, 8)

    try:
        probs = _forward_binary(weights, X_test)
    except ValueError as e:
        print(f"Forward error: {e}", file=sys.stderr)
        return 0.0

    preds = (probs >= 0.5).astype(np.float32)
    acc = np.mean(preds == y_test)
    print(f"accuracy={acc:.4f} benchmark={BENCHMARK}", file=sys.stderr)
    return float(acc)


def eval_top5_accuracy(sol) -> float:
    BENCHMARK = 0.85

    if not hasattr(sol, "top5_accuracy"):
        print("Missing top5_accuracy()", file=sys.stderr)
        return 0.0

    try:
        weights = _call_with_timeout(sol.top5_accuracy, FUNC_TIME_LIMIT)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 0.0
    if not _validate_weights(weights, ["W1", "b1", "W2", "b2"]):
        return 0.0

    X_test, y_test = _get_data_10class()

    try:
        logits = _forward_10class(weights, X_test)
    except ValueError as e:
        print(f"Forward error: {e}", file=sys.stderr)
        return 0.0

    # Top-5: check if true label is in top 5 predictions
    top5 = np.argsort(logits, axis=1)[:, -5:]
    correct = np.array([y_test[i] in top5[i] for i in range(len(y_test))])
    acc = np.mean(correct)
    print(f"top5_accuracy={acc:.4f} benchmark={BENCHMARK}", file=sys.stderr)
    return float(acc)


EVALUATORS = {
    "classify_all": eval_classify_all,
    "binary_0v1": eval_binary_0v1,
    "binary_3v8": eval_binary_3v8,
    "top5_accuracy": eval_top5_accuracy,
}


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <solution_file> [task_name]", file=sys.stderr)
        print("0.0")
        sys.exit(0)

    solution_path = sys.argv[1]
    task_name = sys.argv[2] if len(sys.argv) > 2 else "classify_all"

    # Static analysis: check for banned imports/patterns
    cheat_msg = _check_source(solution_path)
    if cheat_msg:
        print(cheat_msg, file=sys.stderr)
        print("0.0")
        sys.exit(0)

    try:
        sol = load_module(solution_path)
    except Exception as e:
        print(f"Load error: {e}", file=sys.stderr)
        print("0.0")
        sys.exit(0)

    if task_name not in EVALUATORS:
        print(f"Unknown task: {task_name}. Available: {list(EVALUATORS)}", file=sys.stderr)
        print("0.0")
        sys.exit(0)

    try:
        score = EVALUATORS[task_name](sol)
    except Exception as e:
        print(f"Eval error: {e}", file=sys.stderr)
        score = 0.0

    print(f"{score:.4f}")


if __name__ == "__main__":
    main()
