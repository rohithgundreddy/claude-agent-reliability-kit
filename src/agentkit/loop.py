"""A hardened manual tool-use loop for the Claude Messages API.

Guards built in:

* ``max_iterations`` hard cap on model turns per job.
* Repeated-call guard: an identical ``(tool, input)`` call is answered from
  a per-job cache with a nudge instead of hitting the backend again; a third
  identical call ends the run with ``LoopDetected`` rather than burning tokens.
* Tool exceptions become ``is_error`` tool results, never crashes.
* All parallel tool results go back in ONE user message (the API expects it).
* Final answer is constrained with ``output_config.format`` (server-side JSON
  schema) AND validated client-side with Pydantic; a validation failure or a
  ``max_tokens`` truncation triggers a bounded re-ask.
* ``stop_reason`` of ``refusal`` and ``pause_turn`` handled explicitly.
* Prompt caching: ``cache_control`` on the system prompt and on the last tool
  definition, with tools sorted by name so the prefix is byte-stable.
* Every API call logs usage including cache hits and the request id.

Works with any object exposing ``messages.create`` (the real SDK client or a
test double).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from .logs import get_logger, log

ToolHandler = Callable[[dict[str, Any]], Any]


class LoopDetected(RuntimeError):
    pass


class AgentFailed(RuntimeError):
    pass


@dataclass
class AgentResult:
    output: BaseModel | None
    text: str
    iterations: int
    tool_calls: int
    usage: dict[str, int] = field(default_factory=dict)
    stop_reason: str | None = None


def _canonical(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Pydantic schema with ``additionalProperties: false`` on every object."""
    schema = model.model_json_schema()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" and "additionalProperties" not in node:
                node["additionalProperties"] = False
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(schema)
    return schema


def _text_of(content: list[Any]) -> str:
    return "".join(getattr(b, "text", "") for b in content if getattr(b, "type", None) == "text")


def _usage_dict(usage: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
        v = getattr(usage, k, None)
        if isinstance(v, int):
            out[k] = v
    return out


def run_agent(
    client: Any,
    *,
    model: str,
    system: str,
    tools: list[dict[str, Any]],
    handlers: dict[str, ToolHandler],
    user_content: Any,
    output_model: type[BaseModel] | None = None,
    max_iterations: int = 8,
    max_output_retries: int = 2,
    max_tokens: int = 16000,
    effort: str | None = None,
    use_beta_fallbacks: bool = False,
    job_id: str | None = None,
    logger: logging.Logger | None = None,
) -> AgentResult:
    """Run one tool-use task to completion and return a validated result.

    ``use_beta_fallbacks`` opts into Anthropic's server-side refusal fallback
    (``betas=["server-side-fallback-2026-07-01"], fallbacks="default"``) via
    ``client.beta.messages.create``; leave off for the plain endpoint.
    """
    logger = logger or get_logger()
    tools_sorted = sorted(tools, key=lambda t: t["name"])
    tools_param = [dict(t) for t in tools_sorted]
    if tools_param:
        tools_param[-1] = {**tools_param[-1], "cache_control": {"type": "ephemeral"}}
    system_param = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]

    request: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_param,
        "tools": tools_param,
    }
    if output_model is not None:
        request["output_config"] = {"format": {"type": "json_schema", "schema": _strict_schema(output_model)}}
    if effort is not None:
        request.setdefault("output_config", {})["effort"] = effort
    if use_beta_fallbacks:
        create = client.beta.messages.create
        request["betas"] = ["server-side-fallback-2026-07-01"]
        request["fallbacks"] = "default"
    else:
        create = client.messages.create

    messages: list[dict[str, Any]] = [{"role": "user", "content": user_content}]
    seen_calls: dict[str, tuple[int, Any]] = {}
    totals: dict[str, int] = {}
    tool_calls = 0
    output_retries = 0
    response = None

    for iteration in range(1, max_iterations + 1):
        started = time.perf_counter()
        response = create(messages=messages, **request)
        usage = _usage_dict(getattr(response, "usage", None))
        for k, v in usage.items():
            totals[k] = totals.get(k, 0) + v
        log(
            logger,
            "api_call",
            job_id=job_id,
            iteration=iteration,
            request_id=getattr(response, "_request_id", None),
            stop_reason=response.stop_reason,
            latency_ms=round((time.perf_counter() - started) * 1000),
            **usage,
        )

        stop = response.stop_reason
        if stop == "refusal":
            raise AgentFailed(f"model refused: {getattr(response, 'stop_details', None)}")

        if stop == "pause_turn":
            messages.append({"role": "assistant", "content": response.content})
            continue

        tool_uses = [b for b in response.content if getattr(b, "type", None) == "tool_use"]

        if stop == "tool_use" or tool_uses:
            messages.append({"role": "assistant", "content": response.content})
            results = []
            for block in tool_uses:
                tool_calls += 1
                key = f"{block.name}:{_canonical(block.input)}"
                count, cached = seen_calls.get(key, (0, None))
                if count >= 2:
                    log(logger, "tool_loop_abort", logging.ERROR, job_id=job_id, tool=block.name, repeats=count)
                    raise LoopDetected(f"{block.name} called {count + 1}x with identical input")
                if count == 1:
                    log(logger, "tool_repeat_served_from_cache", logging.WARNING, job_id=job_id, tool=block.name)
                    seen_calls[key] = (2, cached)
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": (
                                f"[Repeated call] You already called {block.name} with these exact arguments. "
                                f"The result has not changed:\n{cached}\nDo not call it again with the same "
                                "arguments; use this result to finish the task."
                            ),
                        }
                    )
                    continue
                handler = handlers.get(block.name)
                t0 = time.perf_counter()
                try:
                    if handler is None:
                        raise KeyError(f"no handler registered for tool {block.name!r}")
                    out = handler(block.input)
                    content = out if isinstance(out, str) else json.dumps(out, default=str)
                    results.append({"type": "tool_result", "tool_use_id": block.id, "content": content})
                    seen_calls[key] = (1, content)
                    log(logger, "tool_ok", job_id=job_id, tool=block.name, ms=round((time.perf_counter() - t0) * 1000))
                except Exception as exc:  # noqa: BLE001 - surfaced to the model
                    msg = f"Error: {type(exc).__name__}: {exc}"
                    results.append({"type": "tool_result", "tool_use_id": block.id, "content": msg, "is_error": True})
                    seen_calls[key] = (1, msg)
                    log(logger, "tool_error", logging.WARNING, job_id=job_id, tool=block.name, error=msg)
            messages.append({"role": "user", "content": results})
            continue

        if stop == "max_tokens":
            output_retries += 1
            if output_retries > max_output_retries:
                raise AgentFailed("output truncated by max_tokens after retries")
            log(logger, "output_truncated_retry", logging.WARNING, job_id=job_id, attempt=output_retries)
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": "Your answer was cut off. Reply again with only the complete JSON object, as short as possible."})
            continue

        # end_turn (or anything else): final answer.
        text = _text_of(response.content)
        if output_model is None:
            return AgentResult(None, text, iteration, tool_calls, totals, stop)
        try:
            parsed = output_model.model_validate_json(_extract_json(text))
        except (ValidationError, ValueError) as exc:
            output_retries += 1
            log(logger, "output_invalid", logging.WARNING, job_id=job_id, attempt=output_retries, error=str(exc)[:500])
            if output_retries > max_output_retries:
                raise AgentFailed(f"invalid structured output after {max_output_retries} retries: {exc}") from exc
            messages.append({"role": "assistant", "content": response.content})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous reply did not match the required JSON schema. "
                        f"Validation error: {str(exc)[:800]}\n"
                        "Reply again with ONLY a JSON object that matches the schema. No prose, no code fences."
                    ),
                }
            )
            continue
        log(logger, "job_done", job_id=job_id, iterations=iteration, tool_calls=tool_calls, **totals)
        return AgentResult(parsed, text, iteration, tool_calls, totals, stop)

    log(logger, "iteration_cap_hit", logging.ERROR, job_id=job_id, max_iterations=max_iterations)
    raise LoopDetected(f"no final answer after {max_iterations} iterations")


def _extract_json(text: str) -> str:
    """Tolerate code fences or stray prose around a JSON object."""
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found in model output")
    return s[start : end + 1]
