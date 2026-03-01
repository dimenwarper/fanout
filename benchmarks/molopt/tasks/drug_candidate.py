"""Drug Candidate Multi-Objective Optimization.

Objective: Generate 100 valid, diverse SMILES where every molecule satisfies
Lipinski's Rule of Five plus additional drug-likeness criteria:

  1. QED >= 0.7             (drug-likeness, stricter threshold)
  2. 200 <= MW <= 500       (molecular weight)
  3. 0 <= LogP <= 5         (lipophilicity)
  4. HBD <= 5               (hydrogen bond donors)
  5. HBA <= 10              (hydrogen bond acceptors)
  6. Rotatable bonds <= 10  (flexibility)
  7. TPSA <= 140            (oral bioavailability proxy)

Per-molecule score = fraction of 7 criteria met.
Score = median per-molecule score across all 100, multiplied by a diversity
penalty (1 - fraction of pairs with Tanimoto >= 0.6).

Diversity penalty: pairs with Tanimoto similarity (Morgan FP, r=2) >= 0.6
reduce the score proportionally.

Benchmark: median composite * diversity >= 0.85 across 100 molecules

Output: list of 100 SMILES strings.
"""

BENCHMARK_VALUE = 0.85


def drug_candidate() -> list[str]:
    """Return a list of 100 diverse SMILES that are good drug candidates."""
    # Baseline: simple organic molecules (only 10, need 100)
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
