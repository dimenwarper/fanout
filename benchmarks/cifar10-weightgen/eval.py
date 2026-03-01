#!/usr/bin/env python3
"""CIFAR-10 WeightGen eval script for fanout.

Usage: ./eval.py <solution_file> [task_name]

task_name: generate_cnn_small | generate_cnn_medium | generate_linear_only

The solution file must define the task's entry function returning a weight dict.
Prints the accuracy score on the last line.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np


def load_module(path: str):
    spec = importlib.util.spec_from_file_location("solution", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Data loading ──────────────────────────────────────────

def _get_cifar10_subset(n_test: int = 2000):
    """Load CIFAR-10 test subset. Auto-downloads to .data/ cache."""
    import torch
    import torchvision
    import torchvision.transforms as transforms

    cache_dir = os.path.join(os.path.dirname(__file__), ".data")
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    testset = torchvision.datasets.CIFAR10(
        root=cache_dir, train=False, download=True, transform=transform,
    )
    # Use fixed subset for reproducibility
    rng = np.random.RandomState(42)
    indices = rng.choice(len(testset), size=min(n_test, len(testset)), replace=False)

    images = []
    labels = []
    for idx in indices:
        img, label = testset[idx]
        images.append(img)
        labels.append(label)

    import torch
    images_tensor = torch.stack(images)
    labels_tensor = torch.tensor(labels)
    return images_tensor, labels_tensor


# ── Model builders ──────────────────────────────────────────

def _build_cnn_small(weights: dict):
    """Build small CNN and load weights."""
    import torch
    import torch.nn as nn

    class SmallCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
            self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
            self.pool = nn.MaxPool2d(4)
            self.fc = nn.Linear(64, 10)

        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))   # -> 8x8x8
            x = self.pool(torch.relu(self.conv2(x)))   # -> 16x2x2
            x = x.view(x.size(0), -1)                  # -> 64
            x = self.fc(x)
            return x

    model = SmallCNN()
    expected_keys = ["conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias", "fc.weight", "fc.bias"]
    state = {}
    for key in expected_keys:
        if key not in weights:
            raise ValueError(f"Missing key: {key}")
        state[key] = torch.tensor(np.array(weights[key]), dtype=torch.float32)
    model.load_state_dict(state)
    model.eval()
    return model


def _build_cnn_medium(weights: dict):
    """Build medium 3-conv CNN and load weights."""
    import torch
    import torch.nn as nn

    class MediumCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.conv3 = nn.Conv2d(32, 16, 3, padding=1)
            self.pool = nn.MaxPool2d(2)
            self.fc = nn.Linear(256, 10)

        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))   # -> 16x16x16
            x = self.pool(torch.relu(self.conv2(x)))   # -> 32x8x8
            x = self.pool(torch.relu(self.conv3(x)))   # -> 16x4x4
            x = x.view(x.size(0), -1)                  # -> 256
            x = self.fc(x)
            return x

    model = MediumCNN()
    expected_keys = [
        "conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias",
        "conv3.weight", "conv3.bias", "fc.weight", "fc.bias",
    ]
    state = {}
    for key in expected_keys:
        if key not in weights:
            raise ValueError(f"Missing key: {key}")
        state[key] = torch.tensor(np.array(weights[key]), dtype=torch.float32)
    model.load_state_dict(state)
    model.eval()
    return model


def _build_linear_only(weights: dict):
    """Build linear-only model and load weights."""
    import torch
    import torch.nn as nn

    class LinearModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3072, 10)

        def forward(self, x):
            x = x.view(x.size(0), -1)  # -> 3072
            x = self.fc(x)
            return x

    model = LinearModel()
    expected_keys = ["fc.weight", "fc.bias"]
    state = {}
    for key in expected_keys:
        if key not in weights:
            raise ValueError(f"Missing key: {key}")
        state[key] = torch.tensor(np.array(weights[key]), dtype=torch.float32)
    model.load_state_dict(state)
    model.eval()
    return model


# ── Evaluators ──────────────────────────────────────────

def _eval_model(model, images, labels) -> float:
    """Run inference and return accuracy."""
    import torch
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
    return correct / total


def eval_generate_cnn_small(sol) -> float:
    BENCHMARK = 0.20

    if not hasattr(sol, "generate_cnn_small"):
        print("Missing generate_cnn_small()", file=sys.stderr)
        return 0.0

    weights = sol.generate_cnn_small()
    if not isinstance(weights, dict):
        print(f"Expected dict, got {type(weights)}", file=sys.stderr)
        return 0.0

    try:
        model = _build_cnn_small(weights)
    except (ValueError, RuntimeError) as e:
        print(f"Model build error: {e}", file=sys.stderr)
        return 0.0

    images, labels = _get_cifar10_subset()
    acc = _eval_model(model, images, labels)
    print(f"accuracy={acc:.4f} benchmark={BENCHMARK}", file=sys.stderr)
    return acc


def eval_generate_cnn_medium(sol) -> float:
    BENCHMARK = 0.20

    if not hasattr(sol, "generate_cnn_medium"):
        print("Missing generate_cnn_medium()", file=sys.stderr)
        return 0.0

    weights = sol.generate_cnn_medium()
    if not isinstance(weights, dict):
        print(f"Expected dict, got {type(weights)}", file=sys.stderr)
        return 0.0

    try:
        model = _build_cnn_medium(weights)
    except (ValueError, RuntimeError) as e:
        print(f"Model build error: {e}", file=sys.stderr)
        return 0.0

    images, labels = _get_cifar10_subset()
    acc = _eval_model(model, images, labels)
    print(f"accuracy={acc:.4f} benchmark={BENCHMARK}", file=sys.stderr)
    return acc


def eval_generate_linear_only(sol) -> float:
    BENCHMARK = 0.15

    if not hasattr(sol, "generate_linear_only"):
        print("Missing generate_linear_only()", file=sys.stderr)
        return 0.0

    weights = sol.generate_linear_only()
    if not isinstance(weights, dict):
        print(f"Expected dict, got {type(weights)}", file=sys.stderr)
        return 0.0

    try:
        model = _build_linear_only(weights)
    except (ValueError, RuntimeError) as e:
        print(f"Model build error: {e}", file=sys.stderr)
        return 0.0

    images, labels = _get_cifar10_subset()
    acc = _eval_model(model, images, labels)
    print(f"accuracy={acc:.4f} benchmark={BENCHMARK}", file=sys.stderr)
    return acc


EVALUATORS = {
    "generate_cnn_small": eval_generate_cnn_small,
    "generate_cnn_medium": eval_generate_cnn_medium,
    "generate_linear_only": eval_generate_linear_only,
}


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <solution_file> [task_name]", file=sys.stderr)
        print("0.0")
        sys.exit(0)

    solution_path = sys.argv[1]
    task_name = sys.argv[2] if len(sys.argv) > 2 else "generate_cnn_small"

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
