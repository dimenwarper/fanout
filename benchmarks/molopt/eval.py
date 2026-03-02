#!/usr/bin/env python3
"""MolOpt eval script for fanout.

Usage: ./eval.py <solution_file> [task_name]

task_name: maximize_qed | qed_logp_balance | constrained_generation | drug_candidate

All tasks require 100 diverse SMILES. Score = median per-molecule score,
penalized by diversity: final = median * (1 - violation_fraction), where
violation_fraction = pairs with Tanimoto >= 0.6 / total pairs.
"""

from __future__ import annotations

import importlib.util
import re
import signal
import statistics
import sys


# ── Anti-cheat ──────────────────────────────────────────

BANNED_IMPORTS = [
    r'^\s*(?:from|import)\s+rdkit\b',
    r'^\s*(?:from|import)\s+openbabel\b',
    r'^\s*(?:from|import)\s+pybel\b',
    r'^\s*(?:from|import)\s+chembl',
    r'^\s*(?:from|import)\s+pubchempy\b',
]

BANNED_PATTERNS = [
    r'__import__\s*\(',
    r'\bimportlib\b',
    r'\bMolFromSmiles\b',
    r'\bMolToSmiles\b',
    r'\bChem\.\b',
    r'\bDescriptors\.\b',
    r'\bAllChem\.\b',
]

FUNC_TIME_LIMIT = 1.0  # seconds — returning a list of strings is instant


def _strip_comments_and_strings(source: str) -> str:
    """Remove comments, docstrings, and string literals to check only real code."""
    source = re.sub(r'"""[\s\S]*?"""', '', source)
    source = re.sub(r"'''[\s\S]*?'''", '', source)
    source = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', '', source)
    source = re.sub(r"'[^'\\]*(?:\\.[^'\\]*)*'", '', source)
    source = re.sub(r'#.*$', '', source, flags=re.MULTILINE)
    return source


def _check_source(path: str) -> str | None:
    """Scan source for banned imports/patterns. Returns error message or None."""
    try:
        with open(path) as f:
            source = f.read()
    except Exception:
        return None

    code_only = _strip_comments_and_strings(source)

    for pattern in BANNED_IMPORTS:
        if re.search(pattern, code_only, re.MULTILINE):
            return (
                f"CHEATING DETECTED: source contains banned import matching '{pattern}'. "
                f"This benchmark requires you to return a list of SMILES strings directly. "
                f"You may NOT import or use any chemistry toolkit (rdkit, openbabel, etc.) "
                f"to generate or validate molecules at runtime. "
                f"Craft the SMILES strings using your knowledge of molecular structure."
            )

    for pattern in BANNED_PATTERNS:
        if re.search(pattern, code_only):
            return (
                f"CHEATING DETECTED: source contains banned pattern matching '{pattern}'. "
                f"This benchmark requires you to return a list of SMILES strings directly — "
                f"no runtime molecule generation, filtering, or property checking allowed. "
                f"You must craft valid, diverse SMILES using your knowledge of chemistry."
            )

    return None


class _TimeoutError(Exception):
    pass


def _call_with_timeout(func, timeout: float):
    """Call func() with a wall-clock time limit."""
    def _handler(signum, frame):
        raise _TimeoutError()

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        result = func()
    except _TimeoutError:
        raise RuntimeError(
            f"CHEATING DETECTED: function took longer than {timeout}s. "
            f"This benchmark requires you to return a list of SMILES strings directly — "
            f"no runtime molecule generation or search loops allowed. Your function should "
            f"just return a hardcoded list, which takes milliseconds."
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
    return result


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


def _diversity_penalty(mols: list, max_sim: float = 0.6) -> float:
    """Compute diversity multiplier: 1.0 if all pairs below threshold, decreasing
    toward 0.0 as more pairs violate. Returns (1 - violation_fraction)."""
    from rdkit.Chem import DataStructs
    fps = [_morgan_fp(m) for m in mols]
    total_pairs = len(fps) * (len(fps) - 1) // 2
    violations = 0
    worst = 0.0
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            worst = max(worst, sim)
            if sim >= max_sim:
                violations += 1
    frac = violations / total_pairs if total_pairs > 0 else 0.0
    multiplier = 1.0 - frac
    print(f"  Diversity: {violations}/{total_pairs} pairs violate threshold {max_sim}, "
          f"max_sim={worst:.4f}, multiplier={multiplier:.4f}", file=sys.stderr)
    return multiplier


def _parse_and_validate(smiles_list, expected_count: int, func_name: str):
    """Parse SMILES, validate count, compute diversity penalty.
    Returns (mols, diversity_multiplier) on success, or (None, 0.0) on hard failure."""
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

    # Compute diversity penalty
    print(f"  Checking pairwise diversity ({len(mols)} molecules)...", file=sys.stderr)
    multiplier = _diversity_penalty(mols)

    return mols, multiplier


# ── Evaluators ──────────────────────────────────────────

EXPECTED_COUNT = 100
MAX_SIMILARITY = 0.6


def eval_maximize_qed(sol) -> float:
    BENCHMARK = 0.9

    if not hasattr(sol, "maximize_qed"):
        print("Missing maximize_qed()", file=sys.stderr)
        return 0.0

    try:
        smiles_list = _call_with_timeout(sol.maximize_qed, FUNC_TIME_LIMIT)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 0.0
    mols, div_mult = _parse_and_validate(smiles_list, EXPECTED_COUNT, "maximize_qed")
    if mols is None:
        return div_mult  # 0.0

    scores = []
    for i, mol in enumerate(mols):
        q = _qed(mol)
        scores.append(q)
        if i < 10 or q < 0.5:
            print(f"  [{i}] QED={q:.4f}", file=sys.stderr)

    med = statistics.median(scores)
    final = med * div_mult
    print(f"median_qed={med:.4f} diversity_mult={div_mult:.4f} final={final:.4f} benchmark={BENCHMARK}", file=sys.stderr)
    return final


def eval_qed_logp_balance(sol) -> float:
    BENCHMARK = 0.85

    if not hasattr(sol, "qed_logp_balance"):
        print("Missing qed_logp_balance()", file=sys.stderr)
        return 0.0

    try:
        smiles_list = _call_with_timeout(sol.qed_logp_balance, FUNC_TIME_LIMIT)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 0.0
    mols, div_mult = _parse_and_validate(smiles_list, EXPECTED_COUNT, "qed_logp_balance")
    if mols is None:
        return div_mult

    scores = []
    for i, mol in enumerate(mols):
        q = _qed(mol)
        lp = _logp(mol)

        if 1.0 <= lp <= 3.0:
            lp_score = 1.0
        else:
            dist = min(abs(lp - 1.0), abs(lp - 3.0))
            lp_score = max(0.0, 1.0 - dist / 3.0)

        balanced = (q + lp_score) / 2.0
        scores.append(balanced)
        if i < 10 or balanced < 0.5:
            print(f"  [{i}] QED={q:.4f} LogP={lp:.2f} lp_score={lp_score:.4f} balanced={balanced:.4f}", file=sys.stderr)

    med = statistics.median(scores)
    final = med * div_mult
    print(f"median_balanced={med:.4f} diversity_mult={div_mult:.4f} final={final:.4f} benchmark={BENCHMARK}", file=sys.stderr)
    return final


def eval_constrained_generation(sol) -> float:
    BENCHMARK = 0.85

    if not hasattr(sol, "constrained_generation"):
        print("Missing constrained_generation()", file=sys.stderr)
        return 0.0

    try:
        smiles_list = _call_with_timeout(sol.constrained_generation, FUNC_TIME_LIMIT)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 0.0
    mols, div_mult = _parse_and_validate(smiles_list, EXPECTED_COUNT, "constrained_generation")
    if mols is None:
        return div_mult

    scores = []
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
        scores.append(frac)
        if i < 10 or frac < 0.7:
            print(f"  [{i}] QED={q:.3f} MW={mw:.1f} LogP={lp:.2f} rings={rings} HBD={hbd} TPSA={tpsa:.1f} -> {met}/{total}={frac:.2f}", file=sys.stderr)

    med = statistics.median(scores)
    final = med * div_mult
    print(f"median_constraint_frac={med:.4f} diversity_mult={div_mult:.4f} final={final:.4f} benchmark={BENCHMARK}", file=sys.stderr)
    return final


def eval_drug_candidate(sol) -> float:
    BENCHMARK = 0.85

    if not hasattr(sol, "drug_candidate"):
        print("Missing drug_candidate()", file=sys.stderr)
        return 0.0

    try:
        smiles_list = _call_with_timeout(sol.drug_candidate, FUNC_TIME_LIMIT)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 0.0
    mols, div_mult = _parse_and_validate(smiles_list, EXPECTED_COUNT, "drug_candidate")
    if mols is None:
        return div_mult

    scores = []
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
        scores.append(frac)
        if i < 10 or frac < 0.7:
            print(f"  [{i}] QED={q:.3f} MW={mw:.1f} LogP={lp:.2f} HBD={hbd} HBA={hba} rot={rot} TPSA={tpsa:.1f} -> {met}/{total}={frac:.2f}", file=sys.stderr)

    med = statistics.median(scores)
    final = med * div_mult
    print(f"median_score={med:.4f} diversity_mult={div_mult:.4f} final={final:.4f} benchmark={BENCHMARK}", file=sys.stderr)
    return final


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

    # Static analysis: check for banned imports/patterns
    cheat_msg = _check_source(solution_path)
    if cheat_msg:
        print(cheat_msg, file=sys.stderr)
        print("0.0")
        sys.exit(0)

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
