# MolOpt

Evolve SMILES strings to optimize molecular properties via RDKit. All tasks require **100 diverse, valid SMILES**. Scoring is deliberately harsh:

- **Score = minimum** per-molecule score (weakest molecule determines the score)
- **Diversity enforced**: all pairwise Tanimoto similarities (Morgan FP, r=2) must be < 0.6 — any violation scores 0
- **Duplicates rejected**: canonical SMILES must be unique
- **Invalid SMILES rejected**: any unparseable molecule scores 0

## Tasks

| Task | Function | Per-Molecule Score | Benchmark | Description |
|------|----------|--------------------|-----------|-------------|
| `maximize_qed.py` | `maximize_qed()` | QED | min >= 0.9 | Maximize drug-likeness across 100 diverse molecules |
| `qed_logp_balance.py` | `qed_logp_balance()` | (QED + LogP_score) / 2 | min >= 0.85 | Balance QED and LogP (target [1.0, 3.0]) |
| `constrained_generation.py` | `constrained_generation()` | Fraction of 6 constraints met | min >= 0.85 | Hit 6 conflicting property windows simultaneously |
| `drug_candidate.py` | `drug_candidate()` | Fraction of 7 criteria met | min >= 0.85 | Lipinski + QED + rotatable bonds + TPSA |

### Constrained generation windows

QED >= 0.75, 250 <= MW <= 400, 1.5 <= LogP <= 3.5, 2 <= rings <= 4, HBD <= 3, 40 <= TPSA <= 90

### Drug candidate criteria

QED >= 0.7, 200 <= MW <= 500, 0 <= LogP <= 5, HBD <= 5, HBA <= 10, rotatable bonds <= 10, TPSA <= 140

## Dependencies

`rdkit>=2023.9` (installed via `--extra benchmarks`)

## Eval

```bash
python eval.py solution.py maximize_qed
python eval.py solution.py constrained_generation
```

## Run

```bash
# Single task, agent mode
uv run --extra benchmarks python benchmarks/molopt/run_benchmark.py \
  --tasks maximize_qed -m openai/gpt-4o-mini --mode agent --max-steps 3 -n 1

# All tasks
uv run --extra benchmarks python benchmarks/molopt/run_benchmark.py \
  -m openai/gpt-4o-mini --rounds 3

# Record results
uv run --extra benchmarks python benchmarks/molopt/run_benchmark.py \
  --tasks maximize_qed -m openai/gpt-4o-mini --mode agent --max-steps 5 -n 1 \
  --record my-run
```

## Scoring

Score = min(per_molecule_scores) across all 100 molecules. Invalid SMILES, duplicates, or diversity violations immediately score 0. The min-scoring makes this very hard — one bad molecule tanks the entire score.
