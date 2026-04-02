"""QueryOrchestrator — embed query, route to buckets, gather Representations."""
from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, List, Optional

import numpy as np
import structlog

from libucks.central_agent import CentralAgent
from libucks.librarian import Librarian
from libucks.models.events import QueryEvent
from libucks.thinking.base import Representation

log = structlog.get_logger(__name__)


class QueryOrchestrator:
    def __init__(
        self,
        central_agent: CentralAgent,
        librarians: dict[str, Librarian],
        embed_fn: Callable[[str], np.ndarray],
        top_k: int = 3,
        stale_checker: object = None,  # Optional[StaleChecker] — avoid circular import
        reindex_fn: Optional[Callable[[List[str]], Awaitable[None]]] = None,
    ) -> None:
        self._agent = central_agent
        self._librarians = librarians
        self._embed_fn = embed_fn
        self._top_k = top_k
        self._stale_checker = stale_checker
        self._reindex_fn = reindex_fn

    async def query(self, text: str) -> List[Representation]:
        embedding = self._embed_fn(text)
        bucket_ids = self._agent.route(embedding, self._top_k)
        if not bucket_ids:
            return []

        # ---- JIT stale check (Phase 6-C) -------------------------------------
        if self._stale_checker is not None:
            stale_result = await self._stale_checker.check(bucket_ids)
            if stale_result.is_stale:
                log.warning(
                    "query.stale_buckets_detected",
                    level=stale_result.level,
                    reason=stale_result.reason,
                    stale_bucket_ids=stale_result.stale_bucket_ids,
                )
                if self._reindex_fn is not None:
                    asyncio.ensure_future(
                        self._reindex_fn(stale_result.stale_bucket_ids)
                    )
        # ---- Answer with possibly-stale data (eventual consistency) ----------

        async def _query_one(bucket_id: str) -> Optional[Representation]:
            librarian = self._librarians.get(bucket_id)
            if librarian is None:
                return None
            event = QueryEvent(query=text, bucket_id=bucket_id)
            return await librarian.handle(event)

        results = await asyncio.gather(*(_query_one(bid) for bid in bucket_ids))
        return [r for r in results if r is not None]
