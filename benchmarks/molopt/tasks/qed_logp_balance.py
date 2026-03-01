"""Balance QED and LogP.

Objective: Generate 10 valid SMILES strings that balance drug-likeness (QED)
and lipophilicity (LogP). The score is the average of:
  - QED (higher is better, max 1.0)
  - LogP penalty: 1.0 if LogP in [1.0, 3.0], decaying outside that range

Benchmark: average balanced score >= 0.85

Output: list of 10 SMILES strings.
"""

BENCHMARK_VALUE = 0.85


def qed_logp_balance() -> list[str]:
    """Return a list of 10 SMILES strings balancing QED and LogP."""
    # Baseline: simple molecules
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
