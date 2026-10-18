# Claude agent reliability kit

Drop-in hardening for a Claude tool-use agent that started failing under load. Built around the five symptoms that show up when claim volume doubles:

| Symptom in production | Root cause | What the kit does |
| --- | --- | --- |
| Agent stuck calling the policy API in a loop | No iteration cap; identical calls re-executed; tool description does not say "once" | `run_agent` caps iterations, answers a repeated identical call from a per-job cache with a nudge, and aborts on the third repeat with `LoopDetected` |
| Intermittent malformed JSON breaks the n8n workflow | Free-text answer parsed downstream; `max_tokens` truncation | Server-side `output_config.format` JSON schema plus client-side Pydantic validation; invalid or truncated output triggers a bounded re-ask with the validation error fed back |
| 429 / 529 errors silently lose submissions | API called inline in the webhook; SDK retries run out; nothing persists the job | Webhook enqueues to a SQLite queue and returns 202; a worker claims with a lease, retries with jittered backoff that honours `retry-after`, and dead-letters after N attempts with an alert |
| API cost tripled, latency up | System prompt and tools resent uncached every turn; loops multiply calls | `cache_control` on the system block and the last tool, tools sorted by name so the prefix is byte-stable; cache hits logged per call; loop guard removes wasted turns; `effort` exposed |
| Failures hard to trace | Minimal logging | One JSON log line per API call with request id, iteration, stop reason, latency, token and cache counts; per-job events from `job_start` to `job_done`, `job_retry_later` or `job_dead` |

## Layout

```
src/agentkit/
  loop.py     hardened manual tool-use loop (run_agent)
  retry.py    call_with_backoff: 429/529/5xx/connection errors, retry-after, full jitter
  queue.py    JobQueue: durable SQLite queue with leases, retries, dead letters
  worker.py   process_one / run_forever: queue -> backoff -> loop -> ack/fail
  logs.py     structured JSON logging
examples/claims_demo.py   offline replay of the production failure modes, or --live
tests/                    15 tests, no network, no API key
```

## Run

```bash
uv sync --extra dev
uv run pytest -q
uv run python examples/claims_demo.py
```

## Wiring into n8n

1. Webhook node receives the claim submission and POSTs it to a tiny HTTP endpoint (FastAPI or Flask) that calls `JobQueue.enqueue(payload)` and returns `202 {"job_id": ...}`.
2. A worker process runs `run_forever(queue, client, build_kwargs=...)`. On success `on_result` POSTs the validated record to the n8n "claims system" webhook. On dead letter, `alert` posts to Slack or email.
3. n8n never waits on the model, so a 529 storm delays claims instead of losing them, and the dead-letter table is the audit trail.

## Notes

- Default model is `claude-opus-5-5`. Thinking is adaptive on that model; depth is controlled with `effort` (`low` is enough for extraction).
- `use_beta_fallbacks=True` opts into Anthropic's server-side refusal fallback. Off by default to keep the plain endpoint.
- Tool definitions should say "call at most once per X". The guard is a backstop, not a substitute for a clear tool description.
- The repeated-call cache is per job. If a tool is legitimately re-callable with identical input (a poll), give it a `nonce` or timestamp argument so calls differ.
