"""Lean-Interact verification helpers for generated proof mutations."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import tempfile
import threading
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class Diagnostic:
    line: int
    col: int
    error: str
    line_at_error: str
    state_at_error: str


@dataclass(frozen=True)
class Verification:
    valid: bool
    diagnostic: Diagnostic | None = None


def _position_tuple(position: Any) -> tuple[int, int]:
    return int(position.line), int(position.column)


def _contains(start: tuple[int, int], end: tuple[int, int], point: tuple[int, int]) -> bool:
    return start <= point <= end


def state_from_response(response: Any, line: int, col: int) -> str | None:
    """Read an infoview goal from structured tactic/info-tree data only."""
    point = (line, col)
    containing: list[tuple[tuple[int, int], tuple[int, int], str]] = []
    preceding: list[tuple[tuple[int, int], str]] = []
    for tactic in getattr(response, "tactics", None) or []:
        start = _position_tuple(tactic.start_pos)
        end = _position_tuple(tactic.end_pos)
        goals = str(getattr(tactic, "goals", "") or "").strip()
        if goals and _contains(start, end, point):
            containing.append((start, end, goals))
        if goals and start <= point:
            preceding.append((end, goals))
    if containing:
        containing.sort(key=lambda item: (item[1][0] - item[0][0], item[1][1] - item[0][1]))
        return containing[0][2]

    nodes: list[Any] = []

    def visit(node: Any) -> None:
        if node is None:
            return
        if isinstance(node, (list, tuple)):
            for child in node:
                visit(child)
            return
        info_node = getattr(node, "node", None)
        if info_node is not None and getattr(info_node, "goals_before", None) is not None:
            nodes.append(info_node)
        for child in getattr(node, "children", None) or []:
            visit(child)

    visit(getattr(response, "infotree", None))
    for node in nodes:
        syntax = getattr(node, "stx", None)
        node_range = getattr(syntax, "range", None)
        start_pos = getattr(node_range, "start", None)
        end_pos = getattr(node_range, "finish", None)
        if start_pos is None or end_pos is None:
            continue
        start, end = _position_tuple(start_pos), _position_tuple(end_pos)
        before_goals = getattr(node, "goals_before", None) or []
        after_goals = getattr(node, "goals_after", None) or []
        before = "\n\n".join(str(goal).strip() for goal in before_goals if str(goal).strip())
        after = "\n\n".join(str(goal).strip() for goal in after_goals if str(goal).strip())
        if _contains(start, end, point) and before:
            containing.append((start, end, before))
        if end <= point and (after or before):
            preceding.append((end, after or before))
    if containing:
        containing.sort(key=lambda item: (item[1][0] - item[0][0], item[1][1] - item[0][1]))
        return containing[0][2]
    if preceding:
        preceding.sort(key=lambda item: item[0])
        return preceding[-1][1]
    return None


class LeanInteractVerifier:
    """One reusable Lean-Interact server tied to a Lean project."""

    def __init__(self, project_dir: Path, timeout: float) -> None:
        try:
            from lean_interact import LeanREPLConfig, LeanServer, LocalProject
        except ImportError as error:
            raise RuntimeError("install lean-interact to verify mutations") from error
        self.timeout = timeout
        project = LocalProject(directory=project_dir.resolve(), auto_build=False)
        config = LeanREPLConfig(
            project=project,
            enable_parallel_elaboration=False,
            enable_incremental_optimization=False,
        )
        self.server = LeanServer(config)

    def verify(self, source: str) -> Verification:
        from lean_interact import FileCommand

        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", suffix=".lean", prefix=".line-mutation-", delete=False
            ) as handle:
                handle.write(source)
                temporary = Path(handle.name)
            response = self.server.run(
                FileCommand(path=str(temporary), all_tactics=True, root_goals=True, infotree="full"),
                timeout=self.timeout,
            )
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        if not hasattr(response, "messages"):
            message = getattr(response, "message", response)
            raise RuntimeError(f"Lean-Interact failed before elaboration: {message}")
        errors = [message for message in (response.messages or []) if message.severity == "error"]
        if not errors:
            return Verification(valid=True)
        errors.sort(key=lambda message: _position_tuple(message.start_pos))
        first = errors[0]
        line, col = _position_tuple(first.start_pos)
        source_lines = source.splitlines()
        line_text = source_lines[line - 1].strip() if 1 <= line <= len(source_lines) else ""
        state = state_from_response(response, line, col)
        diagnostic = None
        if state:
            diagnostic = Diagnostic(line, col, str(first.data).strip(), line_text, state)
        return Verification(valid=False, diagnostic=diagnostic)


class VerifierPool:
    """Give each worker thread its own reusable verifier process."""

    def __init__(self, factory: Callable[[], Any]) -> None:
        self.factory = factory
        self.local = threading.local()

    def verify(self, source: str) -> Verification:
        verifier = getattr(self.local, "verifier", None)
        if verifier is None:
            verifier = self.factory()
            self.local.verifier = verifier
        return verifier.verify(source)


def verify_parallel(
    jobs: Iterable[tuple[str, str]],
    verify: Callable[[str], Verification],
    workers: int,
) -> Iterable[tuple[str, Verification]]:
    materialized = list(jobs)
    if workers == 1:
        for key, source in materialized:
            yield key, verify(source)
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(verify, source): key for key, source in materialized}
        for future in as_completed(futures):
            yield futures[future], future.result()
