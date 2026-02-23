"""Unit tests for fanout.solution_format."""

from __future__ import annotations

import pytest

from fanout.solution_format import (
    SolutionFormat,
    extract_diff,
    extract_raw,
    extract_solution,
    get_format,
    strip_code_fences,
)


# ── code format extraction ───────────────────────────────


class TestExtractSolution:
    def test_cot_with_solution_tags(self):
        text = (
            "Let me think step by step...\n"
            "First we need to consider X.\n"
            "<solution>\ndef hello():\n    return 'world'\n</solution>"
        )
        assert extract_solution(text) == "def hello():\n    return 'world'"

    def test_solution_tags_with_code_fences(self):
        text = "<solution>\n```python\ndef hello():\n    return 'world'\n```\n</solution>"
        assert extract_solution(text) == "def hello():\n    return 'world'"

    def test_multiple_solution_blocks_uses_last(self):
        text = (
            "<solution>first attempt</solution>\n"
            "Wait, let me reconsider.\n"
            "<solution>final answer</solution>"
        )
        assert extract_solution(text) == "final answer"

    def test_no_tags_just_code_fences(self):
        text = "```python\nprint('hi')\n```"
        assert extract_solution(text) == "print('hi')"

    def test_no_tags_code_fences_with_preamble(self):
        """When fences aren't at start of text, strip_code_fences can't remove them."""
        text = "Here is the solution:\n```python\nprint('hi')\n```"
        # strip_code_fences only strips leading/trailing fences
        assert extract_solution(text) == "Here is the solution:\n```python\nprint('hi')"

    def test_no_tags_no_fences(self):
        text = "def hello():\n    return 'world'"
        assert extract_solution(text) == "def hello():\n    return 'world'"

    def test_empty_solution_tags(self):
        text = "Some reasoning\n<solution></solution>"
        assert extract_solution(text) == ""

    def test_solution_tags_whitespace(self):
        text = "<solution>  \n  code  \n  </solution>"
        assert extract_solution(text) == "code"


# ── diff format extraction ───────────────────────────────


class TestExtractDiff:
    def test_cot_with_solution_tags(self):
        diff = "--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new"
        text = f"Let me think...\n<solution>\n{diff}\n</solution>"
        assert extract_diff(text) == diff

    def test_diff_with_code_fences_inside_tags(self):
        diff = "--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new"
        text = f"<solution>\n```diff\n{diff}\n```\n</solution>"
        assert extract_diff(text) == diff

    def test_no_tags_returns_raw(self):
        text = "--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new"
        assert extract_diff(text) == text


# ── raw format extraction ────────────────────────────────


class TestExtractRaw:
    def test_any_input_unchanged(self):
        text = "<solution>some code</solution>"
        assert extract_raw(text) == text

    def test_empty_string(self):
        assert extract_raw("") == ""

    def test_code_fences_preserved(self):
        text = "```python\nprint('hi')\n```"
        assert extract_raw(text) == text


# ── strip_code_fences ────────────────────────────────────


class TestStripCodeFences:
    def test_python_fence(self):
        assert strip_code_fences("```python\ncode\n```") == "code"

    def test_bare_fence(self):
        assert strip_code_fences("```\ncode\n```") == "code"

    def test_no_fence(self):
        assert strip_code_fences("just code") == "just code"

    def test_language_variants(self):
        assert strip_code_fences("```lean\ntheorem\n```") == "theorem"
        assert strip_code_fences("```diff\n-old\n+new\n```") == "-old\n+new"


# ── format registry ──────────────────────────────────────


class TestFormatRegistry:
    def test_get_code_format(self):
        fmt = get_format("code")
        assert isinstance(fmt, SolutionFormat)
        assert fmt.name == "code"
        assert fmt.system_prompt is not None
        assert fmt.prompt_suffix is not None
        assert "<solution>" in fmt.prompt_suffix
        assert "<thinking>" in fmt.prompt_suffix
        assert fmt.extract is extract_solution

    def test_get_diff_format(self):
        fmt = get_format("diff")
        assert fmt.name == "diff"
        assert fmt.system_prompt is not None
        assert fmt.prompt_suffix is not None
        assert "diff" in fmt.prompt_suffix.lower()
        assert fmt.extract is extract_diff

    def test_get_raw_format(self):
        fmt = get_format("raw")
        assert fmt.name == "raw"
        assert fmt.system_prompt is None
        assert fmt.prompt_suffix is None
        assert fmt.extract is extract_raw

    def test_unknown_format_raises(self):
        with pytest.raises(KeyError, match="Unknown solution format"):
            get_format("nonexistent")

    def test_code_format_extract_works(self):
        fmt = get_format("code")
        text = "thinking...\n<solution>answer</solution>"
        assert fmt.extract(text) == "answer"

    def test_diff_format_extract_works(self):
        fmt = get_format("diff")
        diff = "--- a/f\n+++ b/f\n@@ -1 +1 @@\n-a\n+b"
        text = f"<solution>\n{diff}\n</solution>"
        assert fmt.extract(text) == diff

    def test_raw_format_extract_works(self):
        fmt = get_format("raw")
        text = "<solution>stuff</solution>"
        assert fmt.extract(text) == text
