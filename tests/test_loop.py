from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from agentkit.loop import AgentFailed, LoopDetected, run_agent
from conftest import FakeClient, resp, text, tool_use


class Claim(BaseModel):
    policy_number: str
    incident_date: str
    damage_description: str
    policy_active: bool


TOOLS = [
    {
        "name": "lookup_policy",
        "description": "Look up a policy by number.",
        "input_schema": {
            "type": "object",
            "properties": {"policy_number": {"type": "string"}},
            "required": ["policy_number"],
            "additionalProperties": False,
        },
    },
    {
        "name": "a_first_tool",
        "description": "Sorted before lookup_policy on purpose.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
]

GOOD = json.dumps(
    {"policy_number": "P-1", "incident_date": "2026-10-01", "damage_description": "rear bumper", "policy_active": True}
)


def _kwargs(client, **over):
    base = dict(
        model="claude-opus-5-5",
        system="Extract claims.",
        tools=TOOLS,
        handlers={"lookup_policy": lambda i: {"active": True, "policy": i["policy_number"]}},
        user_content="Claim text here",
        output_model=Claim,
        max_iterations=6,
    )
    base.update(over)
    return base


def test_happy_path_tool_then_validated_output():
    client = FakeClient(
        [
            resp([tool_use("t1", "lookup_policy", {"policy_number": "P-1"})], "tool_use"),
            resp([text(GOOD)], "end_turn"),
        ]
    )
    result = run_agent(client, **_kwargs(client))
    assert result.output == Claim.model_validate_json(GOOD)
    assert result.iterations == 2 and result.tool_calls == 1

    first, second = client.messages.calls
    # caching: system block and last (sorted) tool carry cache_control; tools are name-sorted
    assert first["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert [t["name"] for t in first["tools"]] == ["a_first_tool", "lookup_policy"]
    assert first["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in first["tools"][0]
    # schema enforcement on the server side
    schema = first["output_config"]["format"]["schema"]
    assert schema["additionalProperties"] is False and "policy_active" in schema["required"]
    # tool result went back as one user message with the matching id
    tool_msg = second["messages"][-1]
    assert tool_msg["role"] == "user" and tool_msg["content"][0]["tool_use_id"] == "t1"
    assert json.loads(tool_msg["content"][0]["content"]) == {"active": True, "policy": "P-1"}


def test_repeated_identical_call_is_cached_then_aborted():
    calls = []
    same = lambda: resp([tool_use("t", "lookup_policy", {"policy_number": "P-9"})], "tool_use")  # noqa: E731
    client = FakeClient([same(), same(), same(), same()])
    kwargs = _kwargs(client, handlers={"lookup_policy": lambda i: calls.append(i) or "ok"})
    with pytest.raises(LoopDetected):
        run_agent(client, **kwargs)
    # backend hit exactly once; second call answered from cache with a nudge; third aborted
    assert calls == [{"policy_number": "P-9"}]
    nudge = client.messages.calls[2]["messages"][-1]["content"][0]["content"]
    assert "Repeated call" in nudge and "ok" in nudge
    assert len(client.messages.calls) == 3


def test_invalid_json_is_retried_with_validation_error_fed_back():
    client = FakeClient(
        [
            resp([text('{"policy_number": "P-1", "incident_date": "2026-10-01"}')], "end_turn"),  # missing fields
            resp([text("```json\n" + GOOD + "\n```")], "end_turn"),  # fenced, still accepted
        ]
    )
    result = run_agent(client, **_kwargs(client))
    assert result.output.policy_active is True
    correction = client.messages.calls[1]["messages"][-1]["content"]
    assert "did not match the required JSON schema" in correction
    assert "damage_description" in correction


def test_invalid_json_gives_up_after_retries():
    bad = resp([text("not json at all")], "end_turn")
    client = FakeClient([bad, bad, bad, bad])
    with pytest.raises(AgentFailed):
        run_agent(client, **_kwargs(client, max_output_retries=2))
    assert len(client.messages.calls) == 3


def test_max_tokens_truncation_is_retried():
    client = FakeClient([resp([text('{"policy_number": "P-1", "incid')], "max_tokens"), resp([text(GOOD)], "end_turn")])
    result = run_agent(client, **_kwargs(client))
    assert result.output.policy_number == "P-1"
    assert "cut off" in client.messages.calls[1]["messages"][-1]["content"]


def test_parallel_tools_one_message_and_errors_are_is_error():
    def boom(_):
        raise ConnectionError("policy service down")

    client = FakeClient(
        [
            resp(
                [tool_use("a", "lookup_policy", {"policy_number": "P-1"}), tool_use("b", "a_first_tool", {})],
                "tool_use",
            ),
            resp([text(GOOD)], "end_turn"),
        ]
    )
    result = run_agent(client, **_kwargs(client, handlers={"lookup_policy": lambda i: "fine", "a_first_tool": boom}))
    assert result.tool_calls == 2
    results = client.messages.calls[1]["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["a", "b"]
    assert results[1]["is_error"] is True and "policy service down" in results[1]["content"]


def test_iteration_cap():
    client = FakeClient([resp([tool_use(f"t{i}", "lookup_policy", {"policy_number": f"P-{i}"})], "tool_use") for i in range(10)])
    with pytest.raises(LoopDetected):
        run_agent(client, **_kwargs(client, max_iterations=3))
    assert len(client.messages.calls) == 3


def test_refusal_is_surfaced():
    client = FakeClient([resp([], "refusal")])
    with pytest.raises(AgentFailed):
        run_agent(client, **_kwargs(client))


def test_beta_fallbacks_flag_adds_params():
    client = FakeClient([resp([text(GOOD)], "end_turn")])
    run_agent(client, **_kwargs(client, use_beta_fallbacks=True))
    call = client.messages.calls[0]
    assert call["betas"] == ["server-side-fallback-2026-07-01"] and call["fallbacks"] == "default"
