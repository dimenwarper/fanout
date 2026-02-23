"""Fanout exception types."""

from __future__ import annotations


class FanoutError(Exception):
    """Base exception for fanout."""


class TruncatedOutputError(FanoutError):
    """Raised when a model response was truncated due to max_tokens limit."""

    def __init__(self, model: str, max_tokens: int, output: str):
        self.model = model
        self.max_tokens = max_tokens
        self.output = output
        super().__init__(
            f"Response from {model} was truncated at {max_tokens} tokens. "
            f"Increase --max-tokens to avoid this."
        )
