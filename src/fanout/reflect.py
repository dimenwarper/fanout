"""Reflective mutation — LLM-powered failure diagnosis between rounds.

Inspired by GEPA's ReflectiveMutationProposer: after evaluating a round of
solutions an LLM reads the full execution traces (scores, stderr, stdout) and
produces a concise *improvement brief* that is prepended to the next round's
prompt.  This is more targeted than showing raw error output because the LLM
can identify *why* an approach failed and propose concrete fixes.

The reflection is intentionally kept short (≤ 400 tokens by default) so it
adds context without bloating the prompt.  A cheap fast model (e.g. Gemini
Flash) is used by default to keep latency and cost low.
"""

from __future__ import annotations

import os

import httpx

from fanout.db.models import SolutionWithScores
from fanout.solution_format import extract_solution

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

_REFLECTION_SYSTEM_PROMPT = """\
You are an expert code reviewer and optimizer helping an evolutionary search \
process improve solutions across rounds.

You will be given a set of candidate solutions from the current round, each \
with a score (0–1, higher is better) and any evaluation output (stdout/stderr).

Your job:
1. **Diagnose** — identify the specific failure modes (wrong algorithm, edge \
cases missed, performance bottleneck, incorrect API usage, logic errors, etc.)
2. **Hypothesize** — for each failure mode, state a concrete technical fix
3. **Synthesize** — write a brief "improvement brief" the next generation \
should act on

Rules:
- Be specific and technical, not generic ("try a different approach" is useless)
- Focus on the *why* behind failures, not just the what
- If all solutions scored 1.0, say so and suggest exploring diversity instead
- Keep the brief under 300 words
- Do NOT reproduce any solution code; only describe what to change"""

_REFLECTION_USER_TEMPLATE = """\
Round {round_num} results ({n} solutions, best score: {best_score:.3f}):

{entries}

Write the improvement brief for the next round:"""


def _build_entries(selected: list[SolutionWithScores], max_output_chars: int = 800) -> str:
    lines: list[str] = []
    for i, s in enumerate(selected, 1):
        output_preview = extract_solution(s.solution.output)[:max_output_chars]
        lines.append(f"--- Solution {i} | model={s.solution.model} | score={s.aggregate_score:.3f} ---")
        lines.append(output_preview)
        for ev in s.evaluations:
            stderr = ev.details.get("stderr", "").strip()
            stdout = ev.details.get("stdout", "").strip()
            exit_code = ev.details.get("exit_code")
            if exit_code is not None or stderr or stdout:
                lines.append(f"[{ev.evaluator}] exit={exit_code}")
                if stderr:
                    lines.append(f"  stderr: {stderr[:400]}")
                if stdout and not stderr:
                    lines.append(f"  stdout: {stdout[:200]}")
        lines.append("")
    return "\n".join(lines)


def reflect(
    selected: list[SolutionWithScores],
    round_num: int,
    *,
    model: str = "google/gemini-2.0-flash-001",
    api_key: str | None = None,
    max_tokens: int = 400,
) -> str | None:
    """Call an LLM to produce a targeted improvement brief from this round's results.

    Returns the brief as a string, or ``None`` if the call fails (so the caller
    can fall back gracefully to the unannotated prompt).
    """
    if not selected:
        return None

    api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        return None

    best_score = max(s.aggregate_score for s in selected)
    entries = _build_entries(selected)
    user_msg = _REFLECTION_USER_TEMPLATE.format(
        round_num=round_num,
        n=len(selected),
        best_score=best_score,
        entries=entries,
    )

    try:
        resp = httpx.post(
            OPENROUTER_API_URL,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": _REFLECTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                "temperature": 0.3,
                "max_tokens": max_tokens,
            },
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            return None
        content = choices[0]["message"]["content"]
        return content.strip() if content else None
    except Exception:
        return None
