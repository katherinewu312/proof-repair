"""Similar-lemma retrieval: LeanExplore API, an offline name-overlap fallback, and a JSONL cache."""
from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
import re
import threading
import time
from typing import Any, Protocol, Sequence

try:
    from .mutation_utils import JsonlJournal
except ImportError:  # Support direct execution from contributions/scripts.
    from mutation_utils import JsonlJournal


DEFAULT_PACKAGES = ("Mathlib", "Batteries", "Init", "Std")
QUERY_STRATEGIES = ("name", "statement", "name_then_statement")


@dataclass(frozen=True)
class Candidate:
    name: str
    module: str | None = None
    source: str = ""
    source_link: str | None = None


class NeighborRetriever(Protocol):
    backend_id: str

    def neighbors(self, lemma_full_name: str, lemma_statement: str, k: int) -> list[Candidate]: ...


def dedupe(candidates: Sequence[Candidate]) -> list[Candidate]:
    seen: set[str] = set()
    unique: list[Candidate] = []
    for candidate in candidates:
        if candidate.name and candidate.name not in seen:
            seen.add(candidate.name)
            unique.append(candidate)
    return unique


def package_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def retry_delay(error: Exception, fallback: float) -> float:
    """Honour Retry-After on HTTP 429; otherwise the fallback."""
    response = getattr(error, "response", None)
    if response is not None and getattr(response, "status_code", None) == 429:
        try:
            return float(response.headers.get("Retry-After", fallback))
        except (TypeError, ValueError):
            return fallback
    return fallback


class _LeanExploreBase:
    backend_id = "leanexplore"

    def __init__(
        self,
        *,
        packages: Sequence[str] = DEFAULT_PACKAGES,
        query_strategy: str = "name",
        retries: int = 6,
    ) -> None:
        if query_strategy not in QUERY_STRATEGIES:
            raise ValueError(f"unsupported query strategy: {query_strategy}")
        self.packages = list(packages)
        self.query_strategy = query_strategy
        self.retries = max(1, retries)
        self.version = package_version("lean-explore")
        self._lock = threading.Lock()

    async def _search_async(self, query: str, k: int) -> Any:
        raise NotImplementedError

    def search(self, query: str, k: int) -> list[Candidate]:
        delay = 2.0
        for attempt in range(self.retries):
            try:
                with self._lock:
                    response = asyncio.run(self._search_async(query, k))
                break
            except Exception as error:
                if attempt + 1 == self.retries:
                    raise RuntimeError(
                        f"LeanExplore search failed after {self.retries} attempts ({type(error).__name__})"
                    ) from error
                time.sleep(retry_delay(error, delay))
                delay = min(delay * 2, 60.0)
        return [
            Candidate(
                name=str(result.name),
                module=getattr(result, "module", None),
                source=self.backend_id,
                source_link=getattr(result, "source_link", None),
            )
            for result in getattr(response, "results", None) or []
        ]

    def neighbors(self, lemma_full_name: str, lemma_statement: str, k: int) -> list[Candidate]:
        if self.query_strategy == "statement":
            return dedupe(self.search(lemma_statement or lemma_full_name, k))
        results = dedupe(self.search(lemma_full_name, k))
        others = [candidate for candidate in results if candidate.name != lemma_full_name]
        if self.query_strategy == "name_then_statement" and len(others) < 2 and lemma_statement:
            results = dedupe(results + self.search(lemma_statement, k))
        return results


class LeanExploreRetriever(_LeanExploreBase):
    """Remote API (no key needed)."""

    backend_id = "leanexplore-api"

    def __init__(self, *, timeout: float = 30.0, **options: Any) -> None:
        try:
            from lean_explore.api import ApiClient
        except ImportError as error:
            raise RuntimeError("install lean-explore (requirements.txt) to use LeanExplore retrieval") from error
        super().__init__(**options)
        self._client_class = ApiClient
        self.timeout = timeout

    async def _search_async(self, query: str, k: int) -> Any:
        client = self._client_class(timeout=self.timeout)
        return await client.search(query, limit=k, packages=self.packages or None)


class LeanExploreLocalRetriever(_LeanExploreBase):
    """Local index (`pip install lean-explore[local]`, `lean-explore data fetch`)."""

    backend_id = "leanexplore-local"

    def __init__(self, **options: Any) -> None:
        try:
            from lean_explore.search import Service
        except ImportError as error:
            raise RuntimeError("install lean-explore[local] and run `lean-explore data fetch`") from error
        super().__init__(**options)
        self._service_class = Service
        self._service: Any = None

    async def _search_async(self, query: str, k: int) -> Any:
        if self._service is None:
            self._service = self._service_class()
        return await self._service.search(query=query, limit=k, packages=self.packages or None)


NAME_SEGMENT = re.compile(r"[._]+")


def name_segments(name: str) -> frozenset[str]:
    return frozenset(part.lower() for part in NAME_SEGMENT.split(name) if part)


class NameOverlapRetriever:
    """Offline fallback: rank names from a file (one per line) by shared name segments."""

    backend_id = "name-overlap"

    def __init__(self, declarations_file: Path) -> None:
        if not declarations_file.is_file():
            raise ValueError(f"declarations file does not exist: {declarations_file}")
        names = sorted({
            line.strip() for line in declarations_file.read_text(encoding="utf-8").splitlines() if line.strip()
        })
        self._index = [(name, name_segments(name)) for name in names]
        self.version = None

    def neighbors(self, lemma_full_name: str, lemma_statement: str, k: int) -> list[Candidate]:
        target = name_segments(lemma_full_name)
        if not target:
            return []
        scored: list[tuple[float, str]] = []
        for name, segments in self._index:
            if name == lemma_full_name:
                continue
            overlap = len(target & segments)
            if overlap:
                scored.append((overlap / len(target | segments), name))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [Candidate(name=name, source=self.backend_id) for _, name in scored[:k]]


class CachedRetriever:
    def __init__(self, inner: Any, cache: JsonlJournal, *, query_strategy: str) -> None:
        self.inner = inner
        self.cache = cache
        self.query_strategy = query_strategy
        self.backend_id = inner.backend_id
        self.version = getattr(inner, "version", None)
        self._lock = threading.Lock()

    def key(self, lemma_full_name: str, k: int) -> str:
        return f"{self.backend_id}|{self.query_strategy}|{k}|{lemma_full_name}"

    def neighbors(self, lemma_full_name: str, lemma_statement: str, k: int) -> list[Candidate]:
        key = self.key(lemma_full_name, k)
        with self._lock:
            cached = self.cache.get(key)
        if cached is not None:
            return [Candidate(**value) for value in cached["candidates"]]
        candidates = self.inner.neighbors(lemma_full_name, lemma_statement, k)
        with self._lock:
            if self.cache.get(key) is None:
                self.cache.append({"key": key, "candidates": [asdict(candidate) for candidate in candidates]})
        return candidates
