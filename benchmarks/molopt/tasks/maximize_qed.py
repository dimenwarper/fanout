"""Maximize QED (Quantitative Estimate of Drug-likeness).

Objective: Generate 100 valid, diverse SMILES strings where every molecule has
high QED. Score = minimum QED across all 100 molecules, so the weakest molecule
determines the score.

Diversity constraint: all pairwise Tanimoto similarities (Morgan FP, r=2) must
be < 0.6. If any pair exceeds this threshold, the score is 0.

Benchmark: min QED >= 0.9 across 100 diverse molecules

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
