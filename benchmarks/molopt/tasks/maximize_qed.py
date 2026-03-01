"""Maximize QED (Quantitative Estimate of Drug-likeness).

Objective: Generate 100 valid, diverse SMILES strings where every molecule has
high QED. Score = median QED across all 100 molecules, multiplied by a diversity
penalty (1 - fraction of pairs with Tanimoto >= 0.6).

Diversity penalty: pairs with Tanimoto similarity (Morgan FP, r=2) >= 0.6
reduce the score proportionally.

Benchmark: median QED * diversity >= 0.9 across 100 molecules

Output: list of 100 SMILES strings.
"""

BENCHMARK_VALUE = 0.9


def maximize_qed() -> list[str]:
    """Return a list of 100 diverse SMILES strings with high QED scores."""
    # Baseline: simple drug-like molecules (will score low — only 10, need 100)
    return [
        "c1ccccc1",
        "CC(=O)O",
        "CCO",
        "CC(=O)Oc1ccccc1",
        "c1ccncc1",
        "CC(N)C(=O)O",
        "c1ccc2ccccc2c1",
        "CC(=O)NC",
        "c1ccc(O)cc1",
        "CC(=O)NCC",
    ]
