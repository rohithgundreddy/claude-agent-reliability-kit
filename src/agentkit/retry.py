"""Retry with exponential backoff and full jitter for Claude API calls.

The SDK already retries 429/5xx twice. Under a sustained spike that is not
enough, and when its retries run out the exception propagates and, in a
webhook handler, the submission is lost. This wrapper adds a longer, bounded
retry schedule, honours ``retry-after``, and treats 529 ``overloaded_error``
as retryable. Anything non-retryable (400, 401, 403, 404) is raised at once.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Callable, TypeVar

import anthropic

from .logs import get_logger, log

T = TypeVar("T")

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


class RetryExhausted(RuntimeError):
    """Raised when every attempt failed with a retryable error."""

    def __init__(self, attempts: int, last: BaseException):
        super().__init__(f"gave up after {attempts} attempts: {last!r}")
        self.attempts = attempts
        self.last = last


def classify(exc: BaseException) -> tuple[bool, float | None]:
    """Return (retryable, server_suggested_delay_seconds)."""
    if isinstance(exc, anthropic.RateLimitError):
        return True, _retry_after(exc)
    if isinstance(exc, anthropic.APIStatusError):
        if exc.status_code in RETRYABLE_STATUS or getattr(exc, "type", None) == "overloaded_error":
            return True, _retry_after(exc)
        return False, None
    if isinstance(exc, (anthropic.APIConnectionError, anthropic.APITimeoutError)):
        return True, None
    return False, None


def _retry_after(exc: anthropic.APIStatusError) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    raw = headers.get("retry-after")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


def call_with_backoff(
    fn: Callable[[], T],
    *,
    max_attempts: int = 6,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
    logger: logging.Logger | None = None,
    **log_fields: Any,
) -> T:
    """Call ``fn`` until it succeeds or a non-retryable error occurs.

    Delay schedule: ``min(max_delay, base_delay * 2**attempt)`` with full
    jitter, overridden by ``retry-after`` when the server sends one.
    """
    logger = logger or get_logger()
    rng = rng or random.Random()
    last: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - classified below
            retryable, suggested = classify(exc)
            if not retryable:
                log(logger, "api_error_fatal", logging.ERROR, error=repr(exc), **log_fields)
                raise
            last = exc
            if attempt == max_attempts - 1:
                break
            delay = suggested if suggested is not None else rng.uniform(
                0, min(max_delay, base_delay * (2**attempt))
            )
            log(
                logger,
                "api_error_retry",
                logging.WARNING,
                attempt=attempt + 1,
                max_attempts=max_attempts,
                delay_s=round(delay, 2),
                status=getattr(exc, "status_code", None),
                error_type=getattr(exc, "type", type(exc).__name__),
                **log_fields,
            )
            sleep(delay)
    assert last is not None
    log(logger, "api_error_exhausted", logging.ERROR, attempts=max_attempts, error=repr(last), **log_fields)
    raise RetryExhausted(max_attempts, last)
