"""SQLite CRUD operations."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from fanout.db.models import Evaluation, Run, Solution, SolutionWithScores
from fanout.db.schema import init_db


class Store:
    """Persistent storage backed by SQLite."""

    def __init__(self, db_path: Path | None = None):
        self.conn = init_db(db_path)
        self.conn.row_factory = sqlite3.Row

    # ── Runs ──────────────────────────────────────────────

    def save_run(self, run: Run) -> Run:
        self.conn.execute(
            "INSERT INTO runs (id, prompt, config, current_round, total_rounds, parent_run_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (run.id, run.prompt, json.dumps(run.config), run.current_round,
             run.total_rounds, run.parent_run_id, run.created_at.isoformat()),
        )
        self.conn.commit()
        return run

    def get_run(self, run_id: str) -> Run | None:
        row = self.conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        return Run(
            id=row["id"], prompt=row["prompt"], config=json.loads(row["config"]),
            current_round=row["current_round"], total_rounds=row["total_rounds"],
            parent_run_id=row["parent_run_id"], created_at=row["created_at"],
        )

    def update_run_round(self, run_id: str, current_round: int) -> None:
        self.conn.execute("UPDATE runs SET current_round = ? WHERE id = ?", (current_round, run_id))
        self.conn.commit()

    def list_runs(self) -> list[Run]:
        rows = self.conn.execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()
        return [
            Run(id=r["id"], prompt=r["prompt"], config=json.loads(r["config"]),
                current_round=r["current_round"], total_rounds=r["total_rounds"],
                parent_run_id=r["parent_run_id"], created_at=r["created_at"])
            for r in rows
        ]

    # ── Solutions ─────────────────────────────────────────

    def save_solution(self, sol: Solution) -> Solution:
        self.conn.execute(
            "INSERT INTO solutions (id, run_id, round_num, model, output, prompt_tokens, "
            "completion_tokens, latency_ms, cost_usd, parent_solution_id, metadata, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (sol.id, sol.run_id, sol.round_num, sol.model, sol.output,
             sol.prompt_tokens, sol.completion_tokens, sol.latency_ms, sol.cost_usd,
             sol.parent_solution_id, json.dumps(sol.metadata), sol.created_at.isoformat()),
        )
        self.conn.commit()
        return sol

    def get_solutions_for_run(self, run_id: str, round_num: int | None = None) -> list[Solution]:
        if round_num is not None:
            rows = self.conn.execute(
                "SELECT * FROM solutions WHERE run_id = ? AND round_num = ? ORDER BY created_at",
                (run_id, round_num),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM solutions WHERE run_id = ? ORDER BY created_at",
                (run_id,),
            ).fetchall()
        return [self._row_to_solution(r) for r in rows]

    def _row_to_solution(self, row: sqlite3.Row) -> Solution:
        return Solution(
            id=row["id"], run_id=row["run_id"], round_num=row["round_num"],
            model=row["model"], output=row["output"],
            prompt_tokens=row["prompt_tokens"], completion_tokens=row["completion_tokens"],
            latency_ms=row["latency_ms"], cost_usd=row["cost_usd"],
            parent_solution_id=row["parent_solution_id"],
            metadata=json.loads(row["metadata"]), created_at=row["created_at"],
        )

    # ── Evaluations ───────────────────────────────────────

    def save_evaluation(self, ev: Evaluation) -> Evaluation:
        self.conn.execute(
            "INSERT INTO evaluations (id, solution_id, evaluator, score, raw_score, details, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (ev.id, ev.solution_id, ev.evaluator, ev.score, ev.raw_score,
             json.dumps(ev.details), ev.created_at.isoformat()),
        )
        self.conn.commit()
        return ev

    def get_evaluations_for_solution(self, solution_id: str) -> list[Evaluation]:
        rows = self.conn.execute(
            "SELECT * FROM evaluations WHERE solution_id = ? ORDER BY evaluator",
            (solution_id,),
        ).fetchall()
        return [
            Evaluation(
                id=r["id"], solution_id=r["solution_id"], evaluator=r["evaluator"],
                score=r["score"], raw_score=r["raw_score"],
                details=json.loads(r["details"]), created_at=r["created_at"],
            )
            for r in rows
        ]

    # ── Composites ────────────────────────────────────────

    def get_solutions_with_scores(self, run_id: str, round_num: int | None = None) -> list[SolutionWithScores]:
        solutions = self.get_solutions_for_run(run_id, round_num)
        result = []
        for sol in solutions:
            evals = self.get_evaluations_for_solution(sol.id)
            agg = sum(e.score for e in evals) / len(evals) if evals else 0.0
            result.append(SolutionWithScores(solution=sol, evaluations=evals, aggregate_score=agg))
        result.sort(key=lambda s: s.aggregate_score, reverse=True)
        return result
