from __future__ import annotations

import random

import pytest
from pydantic import BaseModel

from agentkit.loop import LoopDetected
from agentkit.queue import JobQueue
from agentkit.retry import RetryExhausted, call_with_backoff
from agentkit.worker import process_one
from conftest import FakeClient, api_error, resp, text, tool_use


# ---------------------------------------------------------------- retry ----
def test_backoff_retries_429_and_529_then_succeeds(no_sleep):
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise api_error(429, "rate_limit_error", retry_after="2.5")
        if attempts["n"] == 2:
            raise api_error(529, "overloaded_error")
        return "ok"

    out = call_with_backoff(flaky, sleep=no_sleep, rng=random.Random(0), base_delay=1.0)
    assert out == "ok" and attempts["n"] == 3
    delays = no_sleep.delays
    assert delays[0] == 2.5  # retry-after honoured exactly
    assert 0 <= delays[1] <= 2.0  # jittered, capped at base*2**1


def test_backoff_gives_up_with_retry_exhausted(no_sleep):
    def always():
        raise api_error(529, "overloaded_error")

    with pytest.raises(RetryExhausted) as e:
        call_with_backoff(always, max_attempts=3, sleep=no_sleep)
    assert e.value.attempts == 3 and len(no_sleep.delays) == 2


def test_backoff_does_not_retry_400(no_sleep):
    import anthropic

    def bad():
        raise api_error(400, "invalid_request_error")

    with pytest.raises(anthropic.BadRequestError):
        call_with_backoff(bad, sleep=no_sleep)
    assert no_sleep.delays == []


# ---------------------------------------------------------------- queue ----
def test_queue_lifecycle(tmp_path):
    q = JobQueue(str(tmp_path / "q.sqlite3"), lease_seconds=10)
    jid = q.enqueue({"claim": "A"}, max_attempts=2)
    assert q.enqueue({"claim": "A"}, job_id=jid) == jid  # idempotent re-delivery

    job = q.claim(now=1000)
    assert job is not None and job.id == jid and job.attempts == 1
    assert q.claim(now=1001) is None  # leased, not visible

    # worker dies: lease expires and the job is claimable again
    job2 = q.claim(now=1011)
    assert job2 is not None and job2.attempts == 2

    assert q.fail(jid, "boom", now=1012) == "dead"  # attempts == max_attempts
    assert q.counts() == {"dead": 1}
    assert q.dead_letters()[0]["last_error"] == "boom"


def test_queue_retry_delay_and_ack(tmp_path):
    q = JobQueue(str(tmp_path / "q.sqlite3"))
    jid = q.enqueue({"claim": "B"}, max_attempts=3)
    q.claim(now=0)
    assert q.fail(jid, "529", retry_in=30, now=0) == "pending"
    assert q.claim(now=10) is None  # not before 30s
    job = q.claim(now=31)
    assert job is not None
    q.ack(jid, {"ok": True})
    assert q.counts() == {"done": 1}


# --------------------------------------------------------------- worker ----
class Out(BaseModel):
    policy_number: str


TOOLS = [
    {
        "name": "lookup_policy",
        "description": "x",
        "input_schema": {"type": "object", "properties": {"policy_number": {"type": "string"}}, "required": ["policy_number"], "additionalProperties": False},
    }
]


def _build(payload):
    return dict(
        model="claude-opus-5-5",
        system="s",
        tools=TOOLS,
        handlers={"lookup_policy": lambda i: "active"},
        user_content=payload["text"],
        output_model=Out,
        max_iterations=4,
    )


def test_worker_acks_success_and_dead_letters_loops(tmp_path, no_sleep):
    q = JobQueue(str(tmp_path / "q.sqlite3"))
    good = q.enqueue({"text": "claim 1"})
    looping = q.enqueue({"text": "claim 2"})

    same = lambda: resp([tool_use("t", "lookup_policy", {"policy_number": "P"})], "tool_use")  # noqa: E731
    client = FakeClient(
        [
            api_error(529, "overloaded_error"),  # first job: transient error then success
            resp([text('{"policy_number": "P-1"}')], "end_turn"),
            same(), same(), same(),  # second job: loops -> LoopDetected -> dead
        ]
    )
    alerts = []
    results = {}
    assert process_one(q, client, build_kwargs=_build, on_result=lambda j, r: results.__setitem__(j, r), alert=lambda k, d: alerts.append(k), sleep=no_sleep)
    assert process_one(q, client, build_kwargs=_build, alert=lambda k, d: alerts.append(k), sleep=no_sleep)
    assert not process_one(q, client, build_kwargs=_build, sleep=no_sleep)

    assert results[good].output.policy_number == "P-1"
    assert q.counts() == {"done": 1, "dead": 1}
    assert q.dead_letters()[0]["id"] == looping and alerts == ["agent_failure"]
