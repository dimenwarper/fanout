# KernelBench

CUDA kernel optimization benchmark adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench) (Stanford Scaling Intelligence Lab).

## Task

Given a PyTorch `Model` class, write a CUDA-optimized `ModelNew` class that produces identical outputs but runs faster.

## Tasks included

| Task | Operation | Description |
|------|-----------|-------------|
| `matmul.py` | Matrix multiplication | Square matrix multiply (4096x4096) |
| `relu.py` | ReLU activation | Elementwise ReLU on large tensor |
| `layernorm.py` | Layer normalization | LayerNorm over 3D feature shape |
| `conv2d.py` | 2D convolution | AlexNet-style conv layer |
| `sum_reduce.py` | Sum reduction | Reduction over one dimension |

## Eval

Requires a CUDA-capable GPU with PyTorch.

```bash
chmod +x eval.sh

# Score a single solution against a task
./eval.sh solution.py tasks/matmul.py

# With fanout RSA
fanout run "$(cat tasks/matmul.py)

Write a CUDA-optimized ModelNew class that replaces Model. Use torch.autograd.Function with custom CUDA kernels or triton. Output only the Python file with ModelNew." \
  -m openai/gpt-4o-mini -n 3 \
  -s rsa --k-agg 2 -r 3 \
  --eval-script "./eval.sh" --materializer file --file-ext .py \
  -e script
```

## Scoring

- Correctness: output must match reference within `atol=rtol=1e-4`
- Speed: score is linear from 0.0 (no speedup) to 1.0 (3x+ speedup)
- If correctness fails, score is 0.0
