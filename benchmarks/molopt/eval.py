#!/usr/bin/env python3
"""MolOpt eval script for fanout.

Usage: ./eval.py <solution_file> [task_name]

task_name: maximize_qed | qed_logp_balance | constrained_generation | drug_candidate

All tasks require 100 diverse SMILES. Score = min per-molecule score.
Diversity enforced: all pairwise Tanimoto (Morgan FP, r=2) must be < 0.6.
"""

from __future__ import annotations

import importlib.util
import sys


def load_module(path: str):
    spec = importlib.util.spec_from_file_location("solution", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Helpers ──────────────────────────────────────────

def _safe_mol(smiles: str):
    from rdkit import Chem
    if not isinstance(smiles, str):
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def _qed(mol):
    from rdkit.Chem import QED as QEDModule
    return QEDModule.qed(mol)


def _logp(mol):
    from rdkit.Chem import Descriptors
    return Descriptors.MolLogP(mol)


def _mw(mol):
    from rdkit.Chem import Descriptors
    return Descriptors.MolWt(mol)


def _hbd(mol):
    from rdkit.Chem import Descriptors
    return Descriptors.NumHDonors(mol)


def _hba(mol):
    from rdkit.Chem import Descriptors
    return Descriptors.NumHAcceptors(mol)


def _tpsa(mol):
    from rdkit.Chem import Descriptors
    return Descriptors.TPSA(mol)


def _num_rings(mol):
    return mol.GetRingInfo().NumRings()


def _num_rotatable_bonds(mol):
    from rdkit.Chem import Descriptors
    return Descriptors.NumRotatableBonds(mol)


def _morgan_fp(mol):
    from rdkit.Chem import AllChem
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def _check_diversity(mols: list, max_sim: float = 0.6) -> tuple[bool, float]:
    """Check all pairwise Tanimoto < max_sim. Returns (passed, max_found)."""
    from rdkit.Chem import DataStructs
    fps = [_morgan_fp(m) for m in mols]
    worst = 0.0
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            worst = max(worst, sim)
            if sim >= max_sim:
                print(f"  Diversity violation: mol {i} vs {j} Tanimoto={sim:.4f} >= {max_sim}", file=sys.stderr)
                return False, worst
    return True, worst


def _parse_and_validate(smiles_list, expected_count: int, func_name: str):
    """Parse SMILES, validate count, check diversity. Returns (mols, scores_or_none).
    If validation fails, returns (None, 0.0)."""
    if not isinstance(smiles_list, list):
        print(f"Return value is not a list", file=sys.stderr)
        return None, 0.0

    if len(smiles_list) < expected_count:
        print(f"Only {len(smiles_list)} SMILES, need {expected_count}", file=sys.stderr)
        return None, 0.0

    # Deduplicate by canonical SMILES
    from rdkit import Chem
    seen = set()
    mols = []
    for i, smi in enumerate(smiles_list[:expected_count]):
        mol = _safe_mol(smi)
        if mol is None:
            print(f"  [{i}] Invalid SMILES: {smi!r}", file=sys.stderr)
            return None, 0.0
        canon = Chem.MolToSmiles(mol)
        if canon in seen:
            print(f"  [{i}] Duplicate molecule: {smi!r} (canonical: {canon})", file=sys.stderr)
            return None, 0.0
        seen.add(canon)
        mols.append(mol)

    # Check diversity
    print(f"  Checking pairwise diversity ({len(mols)} molecules)...", file=sys.stderr)
    diverse, worst_sim = _check_diversity(mols)
    print(f"  Max pairwise Tanimoto: {worst_sim:.4f}", file=sys.stderr)
    if not diverse:
        return None, 0.0

    return mols, None  # None means "continue scoring"


# ── Evaluators ──────────────────────────────────────────

EXPECTED_COUNT = 100
MAX_SIMILARITY = 0.6


def eval_maximize_qed(sol) -> float:
    BENCHMARK = 0.9

    if not hasattr(sol, "maximize_qed"):
        print("Missing maximize_qed()", file=sys.stderr)
        return 0.0

    mols, early_score = _parse_and_validate(sol.maximize_qed(), EXPECTED_COUNT, "maximize_qed")
    if mols is None:
        return early_score

    min_qed = float("inf")
    for i, mol in enumerate(mols):
        q = _qed(mol)
        if q < min_qed:
            min_qed = q
        if i < 10 or q < 0.5:  # print first 10 and any low scorers
            print(f"  [{i}] QED={q:.4f}", file=sys.stderr)

    print(f"min_qed={min_qed:.4f} benchmark={BENCHMARK}", file=sys.stderr)
    return min_qed


def eval_qed_logp_balance(sol) -> float:
    BENCHMARK = 0.85

    if not hasattr(sol, "qed_logp_balance"):
        print("Missing qed_logp_balance()", file=sys.stderr)
        return 0.0

    mols, early_score = _parse_and_validate(sol.qed_logp_balance(), EXPECTED_COUNT, "qed_logp_balance")
    if mols is None:
        return early_score

    min_balanced = float("inf")
    for i, mol in enumerate(mols):
        q = _qed(mol)
        lp = _logp(mol)

        if 1.0 <= lp <= 3.0:
            lp_score = 1.0
        else:
            dist = min(abs(lp - 1.0), abs(lp - 3.0))
            lp_score = max(0.0, 1.0 - dist / 3.0)

        balanced = (q + lp_score) / 2.0
        if balanced < min_balanced:
            min_balanced = balanced
        if i < 10 or balanced < 0.5:
            print(f"  [{i}] QED={q:.4f} LogP={lp:.2f} lp_score={lp_score:.4f} balanced={balanced:.4f}", file=sys.stderr)

    print(f"min_balanced={min_balanced:.4f} benchmark={BENCHMARK}", file=sys.stderr)
    return min_balanced


def eval_constrained_generation(sol) -> float:
    BENCHMARK = 0.85

    if not hasattr(sol, "constrained_generation"):
        print("Missing constrained_generation()", file=sys.stderr)
        return 0.0

    mols, early_score = _parse_and_validate(sol.constrained_generation(), EXPECTED_COUNT, "constrained_generation")
    if mols is None:
        return early_score

    min_frac = float("inf")
    for i, mol in enumerate(mols):
        q = _qed(mol)
        mw = _mw(mol)
        lp = _logp(mol)
        rings = _num_rings(mol)
        hbd = _hbd(mol)
        tpsa = _tpsa(mol)

        total = 6
        met = 0
        if q >= 0.75:
            met += 1
        if 250 <= mw <= 400:
            met += 1
        if 1.5 <= lp <= 3.5:
            met += 1
        if 2 <= rings <= 4:
            met += 1
        if hbd <= 3:
            met += 1
        if 40 <= tpsa <= 90:
            met += 1

        frac = met / total
        if frac < min_frac:
            min_frac = frac
        if i < 10 or frac < 0.7:
            print(f"  [{i}] QED={q:.3f} MW={mw:.1f} LogP={lp:.2f} rings={rings} HBD={hbd} TPSA={tpsa:.1f} -> {met}/{total}={frac:.2f}", file=sys.stderr)

    print(f"min_constraint_frac={min_frac:.4f} benchmark={BENCHMARK}", file=sys.stderr)
    return min_frac


def eval_drug_candidate(sol) -> float:
    BENCHMARK = 0.85

    if not hasattr(sol, "drug_candidate"):
        print("Missing drug_candidate()", file=sys.stderr)
        return 0.0

    mols, early_score = _parse_and_validate(sol.drug_candidate(), EXPECTED_COUNT, "drug_candidate")
    if mols is None:
        return early_score

    min_frac = float("inf")
    for i, mol in enumerate(mols):
        q = _qed(mol)
        mw = _mw(mol)
        lp = _logp(mol)
        hbd = _hbd(mol)
        hba = _hba(mol)
        rot = _num_rotatable_bonds(mol)
        tpsa = _tpsa(mol)

        total = 7
        met = 0
        if q >= 0.7:
            met += 1
        if 200 <= mw <= 500:
            met += 1
        if 0 <= lp <= 5:
            met += 1
        if hbd <= 5:
            met += 1
        if hba <= 10:
            met += 1
        if rot <= 10:
            met += 1
        if tpsa <= 140:
            met += 1

        frac = met / total
        if frac < min_frac:
            min_frac = frac
        if i < 10 or frac < 0.7:
            print(f"  [{i}] QED={q:.3f} MW={mw:.1f} LogP={lp:.2f} HBD={hbd} HBA={hba} rot={rot} TPSA={tpsa:.1f} -> {met}/{total}={frac:.2f}", file=sys.stderr)

    print(f"min_score={min_frac:.4f} benchmark={BENCHMARK}", file=sys.stderr)
    return min_frac


EVALUATORS = {
    "maximize_qed": eval_maximize_qed,
    "qed_logp_balance": eval_qed_logp_balance,
    "constrained_generation": eval_constrained_generation,
    "drug_candidate": eval_drug_candidate,
}


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <solution_file> [task_name]", file=sys.stderr)
        print("0.0")
        sys.exit(0)

    solution_path = sys.argv[1]
    task_name = sys.argv[2] if len(sys.argv) > 2 else "maximize_qed"

    try:
        sol = load_module(solution_path)
    except Exception as e:
        print(f"Load error: {e}", file=sys.stderr)
        print("0.0")
        sys.exit(0)

    if task_name not in EVALUATORS:
        print(f"Unknown task: {task_name}. Available: {list(EVALUATORS)}", file=sys.stderr)
        print("0.0")
        sys.exit(0)

    try:
        score = EVALUATORS[task_name](sol)
    except Exception as e:
        print(f"Eval error: {e}", file=sys.stderr)
        score = 0.0

    print(f"{score:.4f}")


if __name__ == "__main__":
    main()
