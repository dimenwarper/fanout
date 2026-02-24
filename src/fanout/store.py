"""Typed domain facade over a Channel backend."""

from __future__ import annotations

from fanout.channel import Channel
from fanout.channels.memory import MemoryChannel
from fanout.db.models import Evaluation, Run, Solution, SolutionWithScores


class Store:
    """Persistent storage backed by a Channel."""

    def __init__(self, channel: Channel | None = None):
        if channel is None:
            channel = MemoryChannel()
        self.ch = channel

    # ── Runs ──────────────────────────────────────────────

    def save_run(self, run: Run) -> Run:
        self.ch.put(
            "runs", run.id, run.model_dump(mode="json"),
            parent_run_id=run.parent_run_id or "",
        )
        return run

    def get_run(self, run_id: str) -> Run | None:
        data = self.ch.get("runs", run_id)
        if data is None:
            return None
        return Run(**data)

    def update_run_round(self, run_id: str, current_round: int) -> None:
        data = self.ch.get("runs", run_id)
        if data is None:
            return
        data["current_round"] = current_round
        self.ch.put(
            "runs", run_id, data,
            parent_run_id=data.get("parent_run_id") or "",
        )

    def list_runs(self) -> list[Run]:
        return [Run(**d) for d in self.ch.list("runs")]

    # ── Solutions ─────────────────────────────────────────

    def save_solution(self, sol: Solution) -> Solution:
        self.ch.put(
            "solutions", sol.id, sol.model_dump(mode="json"),
            run_id=sol.run_id,
            round_num=str(sol.round_num),
        )
        return sol

    def get_solutions_for_run(self, run_id: str, round_num: int | None = None) -> list[Solution]:
        filters: dict[str, str] = {"run_id": run_id}
        if round_num is not None:
            filters["round_num"] = str(round_num)
        return [Solution(**d) for d in self.ch.list("solutions", **filters)]

    # ── Evaluations ───────────────────────────────────────

    def save_evaluation(self, ev: Evaluation) -> Evaluation:
        self.ch.put(
            "evaluations", ev.id, ev.model_dump(mode="json"),
            solution_id=ev.solution_id,
        )
        return ev

    def get_evaluations_for_solution(self, solution_id: str) -> list[Evaluation]:
        return [
            Evaluation(**d)
            for d in self.ch.list("evaluations", solution_id=solution_id)
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
