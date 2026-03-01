"""Constrained Multi-Property Generation.

Objective: Generate 100 valid, diverse SMILES where every molecule simultaneously
satisfies 6 conflicting property constraints:

  1. QED >= 0.75            (drug-likeness)
  2. 250 <= MW <= 400       (tight molecular weight window)
  3. 1.5 <= LogP <= 3.5     (moderate lipophilicity)
  4. Num rings in [2, 4]    (structural complexity)
  5. HBD <= 3               (hydrogen bond donors)
  6. 40 <= TPSA <= 90       (topological polar surface area)

Per-molecule score = fraction of constraints satisfied (0 to 1).
Score = median per-molecule score across all 100, multiplied by a diversity
penalty (1 - fraction of pairs with Tanimoto >= 0.6).

These constraints deliberately conflict: high QED favors simple molecules but
the ring count pushes toward complexity; the tight MW and LogP windows further
constrain the design space.

Diversity penalty: pairs with Tanimoto similarity (Morgan FP, r=2) >= 0.6
reduce the score proportionally.

Benchmark: median constraint fraction * diversity >= 0.85 across 100 molecules

Output: list of 100 SMILES strings.
"""

BENCHMARK_VALUE = 0.85


def constrained_generation() -> list[str]:
    """Return a list of 100 diverse SMILES satisfying multiple constraints."""
    # Baseline: simple molecules (only 10, need 100, constraints not targeted)
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
