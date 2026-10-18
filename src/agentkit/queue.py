"""Durable SQLite job queue with leases, retries and a dead-letter state.

Why: a webhook handler that calls Claude inline loses the submission the
moment the process restarts or the API returns 529 for longer than the SDK
retries. Accept the webhook, ``enqueue`` the payload, return 202, and let a
worker ``claim`` jobs. A job is removed only by ``ack``; a crashed worker's
lease expires and the job is claimed again.

States: pending -> leased -> done | pending (retry) | dead
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL,
    lease_until REAL,
    not_before REAL NOT NULL DEFAULT 0,
    last_error TEXT,
    result TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_state_idx ON jobs(state, not_before);
"""


@dataclass
class Job:
    id: str
    payload: dict[str, Any]
    attempts: int
    max_attempts: int


class JobQueue:
    def __init__(self, path: str = "agent_jobs.sqlite3", *, lease_seconds: float = 300.0):
        self.conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self.lease_seconds = lease_seconds

    # -- producer -----------------------------------------------------------
    def enqueue(self, payload: dict[str, Any], *, max_attempts: int = 5, job_id: str | None = None) -> str:
        job_id = job_id or uuid.uuid4().hex
        now = time.time()
        self.conn.execute(
            "INSERT OR IGNORE INTO jobs(id,payload,max_attempts,created_at,updated_at) VALUES(?,?,?,?,?)",
            (job_id, json.dumps(payload, sort_keys=True), max_attempts, now, now),
        )
        return job_id

    # -- consumer -----------------------------------------------------------
    def claim(self, *, now: float | None = None) -> Job | None:
        """Atomically lease the oldest runnable job, recovering expired leases."""
        now = time.time() if now is None else now
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            self.conn.execute(
                "UPDATE jobs SET state='pending', updated_at=? WHERE state='leased' AND lease_until < ?",
                (now, now),
            )
            row = self.conn.execute(
                "SELECT id,payload,attempts,max_attempts FROM jobs "
                "WHERE state='pending' AND not_before <= ? ORDER BY created_at LIMIT 1",
                (now,),
            ).fetchone()
            if row is None:
                self.conn.execute("COMMIT")
                return None
            job_id, payload, attempts, max_attempts = row
            self.conn.execute(
                "UPDATE jobs SET state='leased', attempts=attempts+1, lease_until=?, updated_at=? WHERE id=?",
                (now + self.lease_seconds, now, job_id),
            )
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        return Job(id=job_id, payload=json.loads(payload), attempts=attempts + 1, max_attempts=max_attempts)

    def ack(self, job_id: str, result: dict[str, Any] | None = None) -> None:
        self.conn.execute(
            "UPDATE jobs SET state='done', result=?, lease_until=NULL, updated_at=? WHERE id=?",
            (json.dumps(result, sort_keys=True) if result is not None else None, time.time(), job_id),
        )

    def fail(self, job_id: str, error: str, *, retry_in: float = 30.0, now: float | None = None) -> str:
        """Record a failure. Returns the new state: 'pending' (will retry) or 'dead'."""
        now = time.time() if now is None else now
        row = self.conn.execute("SELECT attempts,max_attempts FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        attempts, max_attempts = row
        state = "dead" if attempts >= max_attempts else "pending"
        self.conn.execute(
            "UPDATE jobs SET state=?, last_error=?, lease_until=NULL, not_before=?, updated_at=? WHERE id=?",
            (state, error[:2000], now + retry_in if state == "pending" else 0, now, job_id),
        )
        return state

    # -- observability --------------------------------------------------------
    def counts(self) -> dict[str, int]:
        rows = self.conn.execute("SELECT state, COUNT(*) FROM jobs GROUP BY state").fetchall()
        return {state: n for state, n in rows}

    def dead_letters(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT id,payload,attempts,last_error FROM jobs WHERE state='dead' ORDER BY updated_at"
        ).fetchall()
        return [
            {"id": i, "payload": json.loads(p), "attempts": a, "last_error": e} for i, p, a, e in rows
        ]
