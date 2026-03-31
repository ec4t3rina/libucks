"""QueryOrchestrator — embed query, route to buckets, gather Representations."""
from __future__ import annotations

import asyncio
from typing import Callable, List, Optional

import numpy as np

from libucks.central_agent import CentralAgent
from libucks.librarian import Librarian
from libucks.models.events import QueryEvent
from libucks.thinking.base import Representation


class QueryOrchestrator:
    def __init__(
        self,
        central_agent: CentralAgent,
        librarians: dict[str, Librarian],
        embed_fn: Callable[[str], np.ndarray],
        top_k: int = 3,
    ) -> None:
        self._agent = central_agent
        self._librarians = librarians
        self._embed_fn = embed_fn
        self._top_k = top_k

    async def query(self, text: str) -> List[Representation]:
        embedding = self._embed_fn(text)
        bucket_ids = self._agent.route(embedding, self._top_k)
        if not bucket_ids:
            return []

        async def _query_one(bucket_id: str) -> Optional[Representation]:
            librarian = self._librarians.get(bucket_id)
            if librarian is None:
                return None
            event = QueryEvent(query=text, bucket_id=bucket_id)
            return await librarian.handle(event)

        results = await asyncio.gather(*(_query_one(bid) for bid in bucket_ids))
        return [r for r in results if r is not None]
