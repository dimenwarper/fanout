"""Aspirin Rediscovery.

Objective: Generate 5 SMILES strings that are as structurally similar to
Aspirin (acetylsalicylic acid) as possible, measured by Tanimoto similarity
using Morgan fingerprints.

Aspirin SMILES: CC(=O)Oc1ccccc1C(=O)O

Benchmark: average Tanimoto similarity >= 0.95

Output: list of 5 SMILES strings.
"""

BENCHMARK_VALUE = 0.95


def aspirin_rediscovery() -> list[str]:
    """Return a list of 5 SMILES strings similar to Aspirin."""
    # Baseline: aspirin-related molecules
    return [
        "CC(=O)Oc1ccccc1C(=O)O",  # aspirin itself
        "OC(=O)c1ccccc1O",        # salicylic acid
        "CC(=O)Oc1ccccc1",        # phenyl acetate
        "OC(=O)c1ccccc1",         # benzoic acid
        "c1ccccc1O",              # phenol
    ]
