"""Fire-and-forget tasks must not vanish, and must not fail silently.

Sweep finding (2026-07-28, ruff RUF006): six `asyncio.ensure_future(...)` calls
store no reference to the returned Task. asyncio keeps only a WEAK reference to
running tasks, so a task with no strong referent can be garbage-collected
mid-execution — the CPython docs say explicitly to save a reference. The two
worst instances are one-shot coroutines doing real work:

  * librarian.py:245        -> mitosis_service.split()   (bucket never splits)
  * query_orchestrator.py:52 -> reindex of stale buckets  (staleness persists)

`health_monitor.run()` and `novel_bucket_service.run()` — both named in
CLAUDE.md as core mechanisms — are the same shape. A loop parked in
asyncio.sleep is usually kept alive by its timer handle, so this is a latent
footgun rather than a guaranteed failure; the failure mode when it does bite
("auto-mitosis just stops, no error") is near-undebuggable.

There is a second defect at the same sites: a bare ensure_future swallows
exceptions until GC, when asyncio prints "Task exception was never retrieved"
with no application context.
"""
from __future__ import annotations

import asyncio

import pytest

from libucks.background_tasks import pending_count, spawn


class TestStrongReference:
    async def test_task_is_retained_while_running(self):
        started = asyncio.Event()
        release = asyncio.Event()

        async def work():
            started.set()
            await release.wait()

        spawn(work(), name="held")
        await started.wait()
        assert pending_count() == 1, "a running task must be strongly referenced"
        release.set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    async def test_reference_is_released_on_completion(self):
        async def work():
            return 1

        t = spawn(work(), name="quick")
        await t
        await asyncio.sleep(0)
        assert pending_count() == 0, "completed tasks must not leak forever"

    async def test_returns_the_task_so_callers_can_await_it(self):
        async def work():
            return 42

        assert await spawn(work(), name="ret") == 42


class TestExceptionsAreSurfaced:
    async def test_failure_is_logged_not_swallowed(self, monkeypatch):
        seen: list[tuple] = []
        import libucks.background_tasks as bt

        monkeypatch.setattr(bt.log, "error", lambda ev, **kw: seen.append((ev, kw)))

        async def boom():
            raise RuntimeError("mitosis blew up")

        spawn(boom(), name="boom")
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert seen, "a fire-and-forget failure must be logged, not silently dropped"
        assert "mitosis blew up" in repr(seen[0][1])

    async def test_task_name_is_reported(self, monkeypatch):
        seen: list[tuple] = []
        import libucks.background_tasks as bt

        monkeypatch.setattr(bt.log, "error", lambda ev, **kw: seen.append((ev, kw)))

        async def boom():
            raise ValueError("x")

        spawn(boom(), name="split:abc123")
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert "split:abc123" in repr(seen[0][1])

    async def test_cancellation_is_not_reported_as_an_error(self, monkeypatch):
        seen: list[tuple] = []
        import libucks.background_tasks as bt

        monkeypatch.setattr(bt.log, "error", lambda ev, **kw: seen.append((ev, kw)))

        async def forever():
            await asyncio.sleep(3600)

        t = spawn(forever(), name="cancelme")
        await asyncio.sleep(0)
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t
        assert not seen, "shutdown cancellation is not an error"


class TestCallSitesUseIt:
    """The helper is worthless if the risky sites keep calling ensure_future."""

    @pytest.mark.parametrize("module", [
        "libucks/librarian.py",
        "libucks/query_orchestrator.py",
        "libucks/mcp_bridge.py",
    ])
    def test_no_bare_ensure_future(self, module):
        from pathlib import Path

        root = Path(__file__).resolve().parents[2]
        src = (root / module).read_text()
        assert "asyncio.ensure_future(" not in src, (
            f"{module} still creates un-referenced tasks; use background_tasks.spawn"
        )
