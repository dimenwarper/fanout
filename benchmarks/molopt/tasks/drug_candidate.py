"""Drug Candidate Multi-Objective Optimization.

Objective: Generate 10 SMILES strings that satisfy multiple drug-likeness criteria
simultaneously (Lipinski's Rule of Five plus QED):
  - QED >= 0.6 (drug-likeness)
  - 150 <= MW <= 500 (molecular weight)
  - LogP in [0, 5] (lipophilicity)
  - HBD <= 5 (hydrogen bond donors)
  - HBA <= 10 (hydrogen bond acceptors)

Score is the fraction of criteria met, averaged across molecules.

Benchmark: average composite score >= 0.80

Output: list of 10 SMILES strings.
"""

BENCHMARK_VALUE = 0.80


def drug_candidate() -> list[str]:
    """Return a list of 10 SMILES strings that are good drug candidates."""
    # Baseline: simple organic molecules
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
