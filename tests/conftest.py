"""Test doubles for the Claude Messages API. No network, no API key."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import pytest

try:  # anthropic 1.x is built on httpx2
    import httpx2 as httpx
except ImportError:  # pragma: no cover - older SDKs
    import httpx  # type: ignore

import anthropic


def text(t: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=t)


def tool_use(id_: str, name: str, inp: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=inp)


def resp(content: list[Any], stop_reason: str, *, cache_read: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        stop_details=None,
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=20,
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=0,
        ),
        _request_id="req_test",
    )


def api_error(status: int, error_type: str = "api_error", retry_after: str | None = None) -> anthropic.APIStatusError:
    headers = {"retry-after": retry_after} if retry_after else {}
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status, headers=headers, request=request)
    body = {"type": "error", "error": {"type": error_type, "message": error_type}}
    cls: type[anthropic.APIStatusError]
    if status == 429:
        cls = anthropic.RateLimitError
    elif status == 400:
        cls = anthropic.BadRequestError
    else:
        cls = anthropic.APIStatusError
    return cls(error_type, response=response, body=body)


class FakeMessages:
    def __init__(self, script: list[Any]):
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        # Snapshot: the loop mutates its messages list in place after the call.
        self.calls.append({**kwargs, "messages": copy.deepcopy(kwargs["messages"])})
        if not self.script:
            raise AssertionError("fake client ran out of scripted responses")
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class FakeClient:
    def __init__(self, script: list[Any]):
        self.messages = FakeMessages(script)
        self.beta = SimpleNamespace(messages=self.messages)


@pytest.fixture
def no_sleep() -> list[float]:
    delays: list[float] = []

    def _sleep(d: float) -> None:
        delays.append(d)

    _sleep.delays = delays  # type: ignore[attr-defined]
    return _sleep  # type: ignore[return-value]
