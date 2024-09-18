"""Database-backed background job queue with an in-process worker thread.

* durable: jobs live in the ``jobs`` table and survive restarts
* atomic claiming (``UPDATE ... WHERE status='queued'``) so several workers can run safely
  (``DL_WORKER_MODE=external`` + ``python -m app.cli worker`` runs a separate worker process)
* progress + heartbeat updates, cooperative cancellation, bounded automatic retries
* stale ``running`` jobs (crashed worker) are re-queued or failed on startup
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
from collections.abc import Callable
from datetime import timedelta

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.metrics import METRICS
from app.models import Job, User, utcnow

log = logging.getLogger("defectlens.jobs")


class JobCancelled(Exception):
    pass


class PermanentJobError(Exception):
    """Errors that must not be retried (invalid input etc.)."""


class JobContext:
    def __init__(self, job_id: str, params: dict, user_id: int | None):
        self.job_id = job_id
        self.params = params
        self.user_id = user_id
        self._last_write = 0.0
        self._last_cancel_check = 0.0
        self._cancelled = False

    def progress(self, frac: float, message: str = "", force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_write < 0.5:
            return
        self._last_write = now
        with SessionLocal() as db:
            db.execute(update(Job).where(Job.id == self.job_id).values(
                progress=max(0.0, min(1.0, float(frac))), message=message[:500], heartbeat_at=utcnow()))
            db.commit()

    def cancelled(self) -> bool:
        now = time.monotonic()
        if self._cancelled:
            return True
        if now - self._last_cancel_check < 0.5:
            return False
        self._last_cancel_check = now
        with SessionLocal() as db:
            self._cancelled = bool(db.execute(select(Job.cancel_requested).where(Job.id == self.job_id)).scalar())
        return self._cancelled

    def check_cancel(self) -> None:
        if self.cancelled():
            raise JobCancelled()


Handler = Callable[[JobContext], dict]
HANDLERS: dict[str, Handler] = {}


def handler(name: str):
    def deco(fn: Handler) -> Handler:
        HANDLERS[name] = fn
        return fn

    return deco


def enqueue(db: Session, job_type: str, params: dict, user: User | None, max_attempts: int = 2) -> Job:
    if job_type not in HANDLERS:
        raise ValueError(f"Unknown job type {job_type}")
    job = Job(type=job_type, params=params, created_by_id=user.id if user else None, max_attempts=max_attempts)
    db.add(job)
    db.flush()
    METRICS.inc("jobs_enqueued_total", type=job_type)
    return job


def request_cancel(db: Session, job: Job) -> Job:
    if job.status == "queued":
        job.status = "cancelled"
        job.finished_at = utcnow()
        job.message = "Cancelled before start"
    elif job.status == "running":
        job.cancel_requested = True
        job.message = "Cancellation requested"
    return job


def retry(db: Session, job: Job) -> Job:
    if job.status not in ("failed", "cancelled"):
        raise ValueError("Only failed or cancelled jobs can be retried")
    job.status = "queued"
    job.cancel_requested = False
    job.error = ""
    job.progress = 0.0
    job.attempts = 0
    job.message = "Re-queued"
    job.finished_at = None
    return job


def recover_stale(stale_seconds: int) -> int:
    cutoff = utcnow() - timedelta(seconds=stale_seconds)
    n = 0
    with SessionLocal() as db:
        for job in db.execute(select(Job).where(Job.status == "running")).scalars():
            hb = job.heartbeat_at or job.started_at
            if hb is not None and hb.tzinfo is None:
                from datetime import UTC

                hb = hb.replace(tzinfo=UTC)
            if hb is None or hb < cutoff or stale_seconds == 0:
                if job.cancel_requested:
                    job.status = "cancelled"
                elif job.attempts < job.max_attempts:
                    job.status = "queued"
                    job.message = "Recovered after worker interruption; re-queued"
                else:
                    job.status = "failed"
                    job.error = "Worker interrupted and retry budget exhausted"
                job.finished_at = utcnow() if job.status != "queued" else None
                n += 1
        db.commit()
    if n:
        log.warning("recovered %d stale job(s)", n)
    return n


def claim_next() -> Job | None:
    with SessionLocal() as db:
        cand = db.execute(select(Job.id).where(Job.status == "queued").order_by(Job.created_at).limit(5)).scalars().all()
        for jid in cand:
            res = db.execute(update(Job).where(Job.id == jid, Job.status == "queued").values(
                status="running", attempts=Job.attempts + 1, started_at=utcnow(), heartbeat_at=utcnow(),
                message="Started"))
            db.commit()
            if res.rowcount == 1:  # type: ignore[attr-defined]  # CursorResult for UPDATE
                return db.get(Job, jid)
    return None


def run_job(job: Job) -> None:
    ctx = JobContext(job.id, job.params or {}, job.created_by_id)
    fn = HANDLERS.get(job.type)
    t0 = time.time()
    status, result, error = "succeeded", None, ""
    try:
        if fn is None:
            raise PermanentJobError(f"No handler for job type {job.type}")
        result = fn(ctx)
    except JobCancelled:
        status, error = "cancelled", ""
    except PermanentJobError as exc:
        status, error = "failed", str(exc)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        status, error = "failed", f"{exc.__class__.__name__}: {exc}"
    except Exception as exc:
        if ctx.cancelled() or "cancelled" in str(exc).lower():
            status, error = "cancelled", ""
        else:
            status, error = "retry", f"{exc.__class__.__name__}: {exc}"
            log.error("job %s failed: %s", job.id, traceback.format_exc(), extra={"job_id": job.id})
    with SessionLocal() as db:
        j = db.get(Job, job.id)
        if j is None:
            return
        if status == "retry":
            if j.attempts < j.max_attempts and not j.cancel_requested:
                j.status, j.message = "queued", f"Attempt {j.attempts} failed; retrying"
            else:
                j.status, j.message = "failed", "Failed"
            j.error = error
        else:
            j.status = status
            j.error = error
            j.result = result
            j.message = {"succeeded": "Completed", "cancelled": "Cancelled", "failed": "Failed"}[status]
            if status == "succeeded":
                j.progress = 1.0
        if j.status != "queued":
            j.finished_at = utcnow()
        db.commit()
        METRICS.inc("jobs_finished_total", type=j.type, status=j.status)
        log.info("job %s %s -> %s in %.1fs", j.id, j.type, j.status, time.time() - t0, extra={"job_id": j.id})


class Worker(threading.Thread):
    def __init__(self, poll_seconds: float = 1.0):
        super().__init__(name="defectlens-worker", daemon=True)
        self.poll = poll_seconds
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        log.info("job worker started")
        while not self._stop.is_set():
            try:
                job = claim_next()
            except Exception:
                log.exception("job claim failed")
                job = None
            if job is None:
                self._stop.wait(self.poll)
                continue
            run_job(job)


def run_pending_sync(max_jobs: int = 100) -> int:
    """Run queued jobs in the current thread (tests / CLI)."""
    n = 0
    while n < max_jobs:
        job = claim_next()
        if job is None:
            break
        run_job(job)
        n += 1
    return n
