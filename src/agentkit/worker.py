"""Queue worker: ties the durable queue, backoff and the hardened loop together.

    webhook -> queue.enqueue(payload) -> 202
    worker  -> queue.claim() -> run_agent(...) under call_with_backoff -> ack / fail

A job reaches ``dead`` only after ``max_attempts`` failures, and every failure
is logged with the job id so nothing disappears silently.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from .logs import get_logger, log
from .loop import AgentFailed, AgentResult, LoopDetected, run_agent
from .queue import JobQueue
from .retry import RetryExhausted, call_with_backoff

AlertHook = Callable[[str, dict[str, Any]], None]


def process_one(
    queue: JobQueue,
    client: Any,
    *,
    build_kwargs: Callable[[dict[str, Any]], dict[str, Any]],
    on_result: Callable[[str, AgentResult], None] | None = None,
    alert: AlertHook | None = None,
    logger: logging.Logger | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Claim and process one job. Returns False when the queue is empty."""
    logger = logger or get_logger()
    job = queue.claim()
    if job is None:
        return False
    log(logger, "job_start", job_id=job.id, attempt=job.attempts, max_attempts=job.max_attempts)
    try:
        result = call_with_backoff(
            lambda: run_agent(client, job_id=job.id, logger=logger, **build_kwargs(job.payload)),
            sleep=sleep,
            logger=logger,
            job_id=job.id,
        )
    except (LoopDetected, AgentFailed) as exc:
        # Model-behaviour failures: retrying the same prompt rarely helps. Dead-letter now.
        queue.fail(job.id, f"agent: {exc}", retry_in=0)
        queue.conn.execute("UPDATE jobs SET state='dead' WHERE id=?", (job.id,))
        log(logger, "job_dead", logging.ERROR, job_id=job.id, reason=str(exc))
        if alert:
            alert("agent_failure", {"job_id": job.id, "reason": str(exc)})
        return True
    except RetryExhausted as exc:
        state = queue.fail(job.id, f"api: {exc}", retry_in=120)
        log(logger, "job_retry_later" if state == "pending" else "job_dead", logging.ERROR, job_id=job.id, state=state)
        if alert and state == "dead":
            alert("api_failure", {"job_id": job.id, "reason": str(exc)})
        return True
    except Exception as exc:  # noqa: BLE001 - unknown: retry with backoff via queue
        state = queue.fail(job.id, f"unexpected: {exc!r}", retry_in=60)
        log(logger, "job_error", logging.ERROR, job_id=job.id, state=state, error=repr(exc))
        if alert and state == "dead":
            alert("unexpected_failure", {"job_id": job.id, "reason": repr(exc)})
        return True

    payload = result.output.model_dump() if result.output is not None else {"text": result.text}
    queue.ack(job.id, payload)
    if on_result:
        on_result(job.id, result)
    return True


def run_forever(queue: JobQueue, client: Any, *, idle_sleep: float = 1.0, **kwargs: Any) -> None:
    while True:
        if not process_one(queue, client, **kwargs):
            time.sleep(idle_sleep)
