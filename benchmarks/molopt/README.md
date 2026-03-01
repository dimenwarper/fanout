# MolOpt

Evolve SMILES strings to optimize molecular properties via RDKit. The LLM generates lists of valid SMILES that are scored on drug-likeness metrics.

## Tasks

| Task | Function | Output | Benchmark | Description |
|------|----------|--------|-----------|-------------|
| `maximize_qed.py` | `maximize_qed()` | 10 SMILES | avg QED >= 0.9 | Maximize Quantitative Estimate of Drug-likeness |
| `qed_logp_balance.py` | `qed_logp_balance()` | 10 SMILES | balanced >= 0.85 | Balance QED and LogP (lipophilicity in [1,3]) |
| `aspirin_rediscovery.py` | `aspirin_rediscovery()` | 5 SMILES | Tanimoto >= 0.95 | Rediscover Aspirin by Tanimoto similarity |
| `drug_candidate.py` | `drug_candidate()` | 10 SMILES | composite >= 0.80 | Multi-objective: QED, MW, LogP, HBD, HBA |

## Dependencies

`rdkit>=2023.9` (installed via `--extra benchmarks`)

## Eval

```bash
# Score a solution for a specific task
python eval.py solution.py maximize_qed
python eval.py solution.py aspirin_rediscovery
```

## Run

```bash
# Single task, agent mode
uv run --extra benchmarks python benchmarks/molopt/run_benchmark.py \
  --tasks maximize_qed -m openai/gpt-4o-mini --mode agent --max-steps 3 -n 1

# All tasks, sample mode
uv run --extra benchmarks python benchmarks/molopt/run_benchmark.py \
  -m openai/gpt-4o-mini --rounds 3

# Record results
uv run --extra benchmarks python benchmarks/molopt/run_benchmark.py \
  --tasks maximize_qed -m openai/gpt-4o-mini --mode agent --max-steps 5 -n 1 \
  --record my-run
```

## Scoring

Each eval returns a float in [0, 1]. Invalid SMILES score 0. The average across all molecules in the list is the final score.
