"""Strong references and error reporting for fire-and-forget asyncio tasks.

asyncio keeps only a WEAK reference to a running Task. A task nobody else holds
can therefore be garbage-collected mid-execution, and the CPython docs say
plainly: "Save a reference to the result of this function, to avoid a task
disappearing mid-execution."

libucks had six bare `asyncio.ensure_future(...)` calls. The exposed work
included `MitosisService.split()` (fired from Librarian when a bucket crosses
its threshold) and the stale-bucket reindex in QueryOrchestrator — one-shot
coroutines with many await points, where a drop means the bucket silently never
splits and the index silently never refreshes. `HealthMonitor.run()` and
`NovelBucketService.run()`, both named in CLAUDE.md as core mechanisms, are the
same shape; a loop parked in `asyncio.sleep` is usually kept alive by its timer
handle, so the practical risk there is lower — but the failure mode when it
does bite ("auto-mitosis just stopped, no error anywhere") is close to
undebuggable.

The bare form has a second defect: an exception in the coroutine surfaces only
at GC, as asyncio's bare "Task exception was never retrieved", with no bucket
id or operation name attached. `spawn` logs it immediately, with context.
"""
from __future__ import annotations

import asyncio
from typing import Any, Coroutine, Set

import structlog

log = structlog.get_logger(__name__)

# Strong references to in-flight tasks. Entries are removed by the done
# callback, so this is bounded by concurrency, not by total tasks ever created.
_TASKS: Set[asyncio.Task] = set()


def spawn(coro: Coroutine[Any, Any, Any], *, name: str) -> asyncio.Task:
    """Schedule `coro` as a background task that cannot be GC'd or fail silently.

    `name` identifies the work in logs — prefer something locatable such as
    ``f"mitosis.split:{bucket_id}"`` over a bare verb.

    Returns the Task, so a caller that does want to await or cancel it can.
    """
    task = asyncio.ensure_future(coro)
    _TASKS.add(task)
    task.add_done_callback(_on_done(name))
    return task


def _on_done(name: str):
    def _cb(task: asyncio.Task) -> None:
        _TASKS.discard(task)
        if task.cancelled():
            # Cancellation is how shutdown works; it is not a failure.
            return
        exc = task.exception()
        if exc is not None:
            log.error(
                "background_task.failed",
                task=name,
                error=repr(exc),
                consequence=f"{name} did not complete; its effect has not been applied",
            )

    return _cb


def pending_count() -> int:
    """Number of in-flight tasks. For tests and diagnostics."""
    return len(_TASKS)
