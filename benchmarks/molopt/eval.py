#!/usr/bin/env python3
"""MolOpt eval script for fanout.

Usage: ./eval.py <solution_file> [task_name]

task_name: maximize_qed | qed_logp_balance | aspirin_rediscovery | drug_candidate

The solution file must define the task's entry function.
Prints the score on the last line.
"""

from __future__ import annotations

import importlib.util
import math
import sys


def load_module(path: str):
    spec = importlib.util.spec_from_file_location("solution", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Helpers ──────────────────────────────────────────

def _safe_mol(smiles: str):
    """Parse SMILES, return Mol or None."""
    from rdkit import Chem
    if not isinstance(smiles, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
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


def _tanimoto(mol, ref_mol):
    from rdkit.Chem import AllChem, DataStructs
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2, nBits=2048)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


# ── Evaluators ──────────────────────────────────────────


def eval_maximize_qed(sol) -> float:
    BENCHMARK = 0.9
    EXPECTED_COUNT = 10

    if not hasattr(sol, "maximize_qed"):
        print("Missing maximize_qed()", file=sys.stderr)
        return 0.0

    smiles_list = sol.maximize_qed()
    if not isinstance(smiles_list, list):
        print("Return value is not a list", file=sys.stderr)
        return 0.0

    scores = []
    for i, smi in enumerate(smiles_list[:EXPECTED_COUNT]):
        mol = _safe_mol(smi)
        if mol is None:
            print(f"  [{i}] Invalid SMILES: {smi!r}", file=sys.stderr)
            scores.append(0.0)
        else:
            q = _qed(mol)
            scores.append(q)
            print(f"  [{i}] {smi} -> QED={q:.4f}", file=sys.stderr)

    # Penalize if fewer molecules than expected
    while len(scores) < EXPECTED_COUNT:
        scores.append(0.0)

    avg = sum(scores) / EXPECTED_COUNT
    print(f"avg_qed={avg:.4f} benchmark={BENCHMARK}", file=sys.stderr)
    return avg


def eval_qed_logp_balance(sol) -> float:
    BENCHMARK = 0.85
    EXPECTED_COUNT = 10

    if not hasattr(sol, "qed_logp_balance"):
        print("Missing qed_logp_balance()", file=sys.stderr)
        return 0.0

    smiles_list = sol.qed_logp_balance()
    if not isinstance(smiles_list, list):
        print("Return value is not a list", file=sys.stderr)
        return 0.0

    scores = []
    for i, smi in enumerate(smiles_list[:EXPECTED_COUNT]):
        mol = _safe_mol(smi)
        if mol is None:
            print(f"  [{i}] Invalid SMILES: {smi!r}", file=sys.stderr)
            scores.append(0.0)
            continue

        q = _qed(mol)
        lp = _logp(mol)

        # LogP penalty: 1.0 if in [1,3], decay outside
        if 1.0 <= lp <= 3.0:
            lp_score = 1.0
        else:
            dist = min(abs(lp - 1.0), abs(lp - 3.0))
            lp_score = max(0.0, 1.0 - dist / 3.0)

        balanced = (q + lp_score) / 2.0
        scores.append(balanced)
        print(f"  [{i}] {smi} -> QED={q:.4f} LogP={lp:.2f} lp_score={lp_score:.4f} balanced={balanced:.4f}", file=sys.stderr)

    while len(scores) < EXPECTED_COUNT:
        scores.append(0.0)

    avg = sum(scores) / EXPECTED_COUNT
    print(f"avg_balanced={avg:.4f} benchmark={BENCHMARK}", file=sys.stderr)
    return avg


def eval_aspirin_rediscovery(sol) -> float:
    from rdkit import Chem

    BENCHMARK = 0.95
    EXPECTED_COUNT = 5
    ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"

    if not hasattr(sol, "aspirin_rediscovery"):
        print("Missing aspirin_rediscovery()", file=sys.stderr)
        return 0.0

    smiles_list = sol.aspirin_rediscovery()
    if not isinstance(smiles_list, list):
        print("Return value is not a list", file=sys.stderr)
        return 0.0

    ref_mol = Chem.MolFromSmiles(ASPIRIN_SMILES)

    scores = []
    for i, smi in enumerate(smiles_list[:EXPECTED_COUNT]):
        mol = _safe_mol(smi)
        if mol is None:
            print(f"  [{i}] Invalid SMILES: {smi!r}", file=sys.stderr)
            scores.append(0.0)
            continue

        sim = _tanimoto(mol, ref_mol)
        scores.append(sim)
        print(f"  [{i}] {smi} -> Tanimoto={sim:.4f}", file=sys.stderr)

    while len(scores) < EXPECTED_COUNT:
        scores.append(0.0)

    avg = sum(scores) / EXPECTED_COUNT
    print(f"avg_tanimoto={avg:.4f} benchmark={BENCHMARK}", file=sys.stderr)
    return avg


def eval_drug_candidate(sol) -> float:
    BENCHMARK = 0.80
    EXPECTED_COUNT = 10

    if not hasattr(sol, "drug_candidate"):
        print("Missing drug_candidate()", file=sys.stderr)
        return 0.0

    smiles_list = sol.drug_candidate()
    if not isinstance(smiles_list, list):
        print("Return value is not a list", file=sys.stderr)
        return 0.0

    scores = []
    for i, smi in enumerate(smiles_list[:EXPECTED_COUNT]):
        mol = _safe_mol(smi)
        if mol is None:
            print(f"  [{i}] Invalid SMILES: {smi!r}", file=sys.stderr)
            scores.append(0.0)
            continue

        q = _qed(mol)
        lp = _logp(mol)
        mw = _mw(mol)
        hbd = _hbd(mol)
        hba = _hba(mol)

        criteria_met = 0
        total_criteria = 5
        if q >= 0.6:
            criteria_met += 1
        if 150 <= mw <= 500:
            criteria_met += 1
        if 0 <= lp <= 5:
            criteria_met += 1
        if hbd <= 5:
            criteria_met += 1
        if hba <= 10:
            criteria_met += 1

        frac = criteria_met / total_criteria
        scores.append(frac)
        print(f"  [{i}] {smi} -> QED={q:.3f} MW={mw:.1f} LogP={lp:.2f} HBD={hbd} HBA={hba} score={frac:.2f}", file=sys.stderr)

    while len(scores) < EXPECTED_COUNT:
        scores.append(0.0)

    avg = sum(scores) / EXPECTED_COUNT
    print(f"avg_score={avg:.4f} benchmark={BENCHMARK}", file=sys.stderr)
    return avg


EVALUATORS = {
    "maximize_qed": eval_maximize_qed,
    "qed_logp_balance": eval_qed_logp_balance,
    "aspirin_rediscovery": eval_aspirin_rediscovery,
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
