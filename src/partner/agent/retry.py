"""Retry a call on specific, expected exception types only."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def with_retries(
    fn: Callable[[], T],
    *,
    retry_on: tuple[type[Exception], ...],
    max_retries: int = 2,
) -> T:
    """Call fn(); on an exception listed in retry_on, call it again, up to
    max_retries extra times. The last failure is re-raised, never swallowed.
    Exceptions not in retry_on propagate on the first occurrence.
    """
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except retry_on:
            if attempt == max_retries:
                raise
    raise AssertionError("unreachable")
