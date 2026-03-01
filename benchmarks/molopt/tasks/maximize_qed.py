"""Maximize QED (Quantitative Estimate of Drug-likeness).

Objective: Generate 10 valid SMILES strings that maximize the average QED score.
QED ranges from 0 to 1, with 1 being the most drug-like.

Benchmark: average QED >= 0.9

Output: list of 10 SMILES strings.
"""

BENCHMARK_VALUE = 0.9


def maximize_qed() -> list[str]:
    """Return a list of 10 SMILES strings with high QED scores."""
    # Baseline: simple drug-like molecules
    return [
        "c1ccccc1",        # benzene
        "CC(=O)O",         # acetic acid
        "CCO",             # ethanol
        "CC(=O)Oc1ccccc1", # phenyl acetate
        "c1ccncc1",        # pyridine
        "CC(N)C(=O)O",     # alanine
        "c1ccc2ccccc2c1",  # naphthalene
        "CC(=O)NC",        # N-methylacetamide
        "c1ccc(O)cc1",     # phenol
        "CC(=O)NCC",       # N-ethylacetamide
    ]
