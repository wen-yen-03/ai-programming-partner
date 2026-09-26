"""with_retries: retry only listed exception types, then re-raise the last one."""

import pytest

from partner.agent.errors import MalformedOutputError
from partner.agent.retry import with_retries


class FlakyCall:
    def __init__(self, failures: int, exc: Exception) -> None:
        self.failures = failures
        self.exc = exc
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        if self.calls <= self.failures:
            raise self.exc
        return "ok"


def malformed() -> MalformedOutputError:
    return MalformedOutputError("bad json", raw_content="{")


def test_succeeds_after_two_failures() -> None:
    call = FlakyCall(failures=2, exc=malformed())
    assert with_retries(call, retry_on=(MalformedOutputError,)) == "ok"
    assert call.calls == 3


def test_reraises_after_three_failures() -> None:
    call = FlakyCall(failures=3, exc=malformed())
    with pytest.raises(MalformedOutputError):
        with_retries(call, retry_on=(MalformedOutputError,))
    assert call.calls == 3


def test_unlisted_exception_is_not_retried() -> None:
    call = FlakyCall(failures=1, exc=KeyError("bug"))
    with pytest.raises(KeyError):
        with_retries(call, retry_on=(MalformedOutputError,))
    assert call.calls == 1


def test_negative_max_retries_is_rejected() -> None:
    with pytest.raises(ValueError):
        with_retries(lambda: "ok", retry_on=(MalformedOutputError,), max_retries=-1)
