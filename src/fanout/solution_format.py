"""Solution format presets — pairs a system prompt with an extraction function."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable


def extract_solution(text: str) -> str:
    """Extract the final solution from LLM output.

    Looks for <solution>...</solution> tags first. Falls back to stripping
    markdown code fences. Returns the raw text if neither is found.
    """
    # Try <solution> tags (last match wins, in case CoT contains earlier tags)
    matches = re.findall(r"<solution>(.*?)</solution>", text, re.DOTALL)
    if matches:
        return strip_code_fences(matches[-1].strip())

    # Fallback: strip code fences from the whole output
    return strip_code_fences(text)


def extract_diff(text: str) -> str:
    """Extract a diff from LLM output.

    Like extract_solution but keeps raw diff content (only strips code fences,
    no other transformations).
    """
    matches = re.findall(r"<solution>(.*?)</solution>", text, re.DOTALL)
    if matches:
        return strip_code_fences(matches[-1].strip())

    # Fallback: return raw text
    return text


def extract_raw(text: str) -> str:
    """Identity extractor — returns text unchanged."""
    return text


def strip_code_fences(text: str) -> str:
    """Strip markdown code fences (```python ... ```) from LLM output."""
    stripped = text.strip()
    stripped = re.sub(r"^```[a-zA-Z]*\n?", "", stripped)
    stripped = re.sub(r"\n?```$", "", stripped.strip())
    return stripped


@dataclass
class SolutionFormat:
    name: str
    system_prompt: str | None
    prompt_suffix: str | None
    extract: Callable[[str], str]


_CODE_SYSTEM_PROMPT = (
    "You are a helpful assistant that solves tasks carefully. "
    "You may think step-by-step."
)

_CODE_PROMPT_SUFFIX = (
    "You MUST place your final solution inside <solution> and </solution> tags "
    "at the end of your response.\n"
    "The content inside <solution> tags should be ONLY the deliverable "
    "(code, proof, text, etc.) with no commentary or explanation."
)

_DIFF_SYSTEM_PROMPT = (
    "You are a helpful assistant that solves tasks carefully. "
    "You may think step-by-step."
)

_DIFF_PROMPT_SUFFIX = (
    "You MUST place your final solution inside <solution> and </solution> tags "
    "at the end of your response.\n"
    "The content inside <solution> tags should be ONLY a unified diff "
    "(as produced by `diff -u` or `git diff`) with no commentary or explanation."
)

_REGISTRY: dict[str, SolutionFormat] = {
    "code": SolutionFormat(
        name="code",
        system_prompt=_CODE_SYSTEM_PROMPT,
        prompt_suffix=_CODE_PROMPT_SUFFIX,
        extract=extract_solution,
    ),
    "diff": SolutionFormat(
        name="diff",
        system_prompt=_DIFF_SYSTEM_PROMPT,
        prompt_suffix=_DIFF_PROMPT_SUFFIX,
        extract=extract_diff,
    ),
    "raw": SolutionFormat(
        name="raw",
        system_prompt=None,
        prompt_suffix=None,
        extract=extract_raw,
    ),
}


def get_format(name: str) -> SolutionFormat:
    """Look up a solution format by name."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown solution format: {name!r}. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]
