"""Reliability primitives for Claude tool-use agents.

Fixes the five production failure modes that show up once volume grows:

1. Tool-call loops           -> ``loop.run_agent`` (iteration cap + repeated-call guard)
2. Malformed JSON output     -> ``loop.run_agent`` (schema-enforced output + validate + retry)
3. 429/529 silently dropped  -> ``retry.call_with_backoff`` + ``queue.JobQueue`` (durable, acked)
4. Cost and latency creep    -> prompt caching on system + tools, usage logged per call
5. No visibility             -> ``logs`` structured JSON events with request ids
"""

from .loop import AgentResult, LoopDetected, run_agent
from .queue import Job, JobQueue
from .retry import RetryExhausted, call_with_backoff

__all__ = [
    "AgentResult",
    "Job",
    "JobQueue",
    "LoopDetected",
    "RetryExhausted",
    "call_with_backoff",
    "run_agent",
]
