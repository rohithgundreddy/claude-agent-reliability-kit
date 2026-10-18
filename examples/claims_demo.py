"""End-to-end demo of the hardened claims agent.

    uv run python examples/claims_demo.py          # offline: scripted fake API
    uv run python examples/claims_demo.py --live   # real API (needs ANTHROPIC_API_KEY or `ant auth login`)

The offline run replays the exact failure modes from production (a repeated
tool call, a malformed first answer, a 529) and shows the kit absorbing them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from pydantic import BaseModel, Field  # noqa: E402

from agentkit import JobQueue  # noqa: E402
from agentkit.worker import process_one  # noqa: E402


class ClaimRecord(BaseModel):
    policy_number: str = Field(description="Policy number exactly as written on the submission")
    incident_date: str = Field(description="ISO date YYYY-MM-DD")
    vehicle: str
    damage_description: str
    policy_active: bool
    confidence: float = Field(ge=0, le=1)


TOOLS = [
    {
        "name": "lookup_policy",
        "description": "Check whether a policy number is active. Call at most once per policy number.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"policy_number": {"type": "string"}},
            "required": ["policy_number"],
            "additionalProperties": False,
        },
    }
]

SYSTEM = (
    "You extract structured auto-damage claim records from free-text submissions. "
    "Look up each policy number exactly once with lookup_policy, then answer with the JSON record. "
    "If a field is missing from the submission, use an empty string and lower the confidence."
)


def policy_api(inp: dict) -> dict:
    # Stand-in for the client's internal policy API.
    return {"policy_number": inp["policy_number"], "active": inp["policy_number"].startswith("P-")}


def build_kwargs(payload: dict) -> dict:
    return dict(
        model="claude-opus-5-5",
        system=SYSTEM,
        tools=TOOLS,
        handlers={"lookup_policy": policy_api},
        user_content=payload["submission"],
        output_model=ClaimRecord,
        max_iterations=6,
        effort="low",
    )


def scripted_client():
    """Replays production symptoms: repeated tool call, bad JSON, then recovery."""
    from conftest import FakeClient, api_error, resp, text, tool_use

    good = ClaimRecord(
        policy_number="P-44810", incident_date="2026-10-12", vehicle="2019 Honda Civic",
        damage_description="rear bumper and tail light", policy_active=True, confidence=0.93,
    ).model_dump_json()
    return FakeClient(
        [
            api_error(529, "overloaded_error"),
            resp([tool_use("t1", "lookup_policy", {"policy_number": "P-44810"})], "tool_use"),
            resp([tool_use("t2", "lookup_policy", {"policy_number": "P-44810"})], "tool_use"),  # repeat
            resp([text('{"policy_number": "P-44810", "vehicle": "Civic"')], "end_turn"),  # malformed
            resp([text(good)], "end_turn", cache_read=1800),
        ]
    )


def main() -> None:
    live = "--live" in sys.argv
    if live:
        import anthropic

        client = anthropic.Anthropic()
    else:
        client = scripted_client()

    queue = JobQueue(":memory:")
    queue.enqueue(
        {
            "submission": (
                "Policy P-44810. On 2026-10-12 my 2019 Honda Civic was rear-ended at a light; "
                "rear bumper and tail light damaged. Other driver admitted fault."
            )
        }
    )
    processed = process_one(
        queue,
        client,
        build_kwargs=build_kwargs,
        on_result=lambda jid, r: print(json.dumps({"job": jid, "record": r.output.model_dump(), "usage": r.usage}, indent=2)),
        alert=lambda kind, data: print("ALERT", kind, data),
        sleep=lambda s: None if not live else __import__("time").sleep(s),
    )
    print("queue:", queue.counts(), "processed:", processed)


if __name__ == "__main__":
    main()
