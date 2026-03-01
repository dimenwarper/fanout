"""Balance QED and LogP across 100 Diverse Molecules.

Objective: Generate 100 valid, diverse SMILES where every molecule balances
drug-likeness and lipophilicity. Per-molecule score:
  - QED component (0-1)
  - LogP component: 1.0 if LogP in [1.0, 3.0], linear decay outside (hits 0 at
    distance 3.0 from the window)

Score = median per-molecule balanced score across all 100, multiplied by a
diversity penalty (1 - fraction of pairs with Tanimoto >= 0.6).

Diversity penalty: pairs with Tanimoto similarity (Morgan FP, r=2) >= 0.6
reduce the score proportionally.

Benchmark: median balanced * diversity >= 0.85 across 100 molecules

Output: list of 100 SMILES strings.
"""

BENCHMARK_VALUE = 0.85


def qed_logp_balance() -> list[str]:
    """Return a list of 100 diverse SMILES balancing QED and LogP."""
    # Baseline: simple molecules (only 10, need 100)
    return [
        "c1ccccc1",
        "CCO",
        "CC(=O)O",
        "c1ccncc1",
        "CC(N)C(=O)O",
        "c1ccc(O)cc1",
        "CCCC",
        "CC(=O)NC",
        "c1ccc2ccccc2c1",
        "CC(=O)Oc1ccccc1",
    ]
