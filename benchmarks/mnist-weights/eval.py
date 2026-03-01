#!/usr/bin/env python3
"""MNIST-Weights eval script for fanout.

Usage: ./eval.py <solution_file> [task_name]

task_name: classify_all | binary_0v1 | binary_3v8 | top5_accuracy

The solution file must define the task's entry function returning a weight dict.
Prints the accuracy score on the last line.
"""

from __future__ import annotations

import importlib.util
import sys

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


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

    weights = sol.classify_all()
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

    weights = sol.binary_0v1()
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

    weights = sol.binary_3v8()
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

    weights = sol.top5_accuracy()
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
