"""Expected failures inside agent steps. Nodes turn these into outcome=error.

Anything that is NOT one of these (or another type listed in
nodes.RECOVERABLE_ERRORS) is treated as a bug and allowed to propagate.
"""


class AgentStepError(Exception):
    """Base class for an agent step failing in an expected, reportable way."""


class MalformedOutputError(AgentStepError):
    """A model response was not valid JSON or failed Pydantic validation."""

    def __init__(self, message: str, *, raw_content: str) -> None:
        super().__init__(message)
        self.raw_content = raw_content


class ModelTimeoutError(AgentStepError):
    """A model call did not finish within its timeout."""


class SessionReaderUnavailableError(AgentStepError):
    """The MCP session reader is not configured or cannot be reached."""
