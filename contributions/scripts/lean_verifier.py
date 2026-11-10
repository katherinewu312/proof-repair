"""Lean-Interact verification helpers for generated proof mutations."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import tempfile
import threading
from typing import Any, Callable, Iterable, Sequence


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


# Environment sessions (theorem mutation).


@dataclass(frozen=True)
class ErrorMessage:
    line: int
    col: int
    end_line: int | None
    end_col: int | None
    message: str


@dataclass(frozen=True)
class TacticSpan:
    start_line: int
    start_col: int
    end_line: int
    end_col: int
    used_constants: tuple[str, ...] = ()


@dataclass(frozen=True)
class CompileResult:
    """Compile outcome; positions are full-file."""

    errors: tuple[ErrorMessage, ...] = ()
    line_at_error: str | None = None
    state_at_error: str | None = None
    tactics: tuple[TacticSpan, ...] = ()
    uses_sorry: bool = False
    timed_out: bool = False
    crashed: bool = False
    prefix_failed: bool = False
    detail: str | None = None

    @property
    def ok(self) -> bool:
        return not (self.errors or self.timed_out or self.crashed or self.prefix_failed)

    @property
    def first_error(self) -> ErrorMessage | None:
        return self.errors[0] if self.errors else None

    def diagnostic(self) -> Diagnostic | None:
        first = self.first_error
        if first is None or not self.state_at_error:
            return None
        return Diagnostic(first.line, first.col, first.message, self.line_at_error or "", self.state_at_error)

    def to_dict(self, *, include_tactics: bool = False) -> dict[str, Any]:
        value = asdict(self)
        if not include_tactics:
            value.pop("tactics")
        return value


class PrefixError(RuntimeError):
    """The prefix does not elaborate."""


def unsolved_goals_state(message: str) -> str | None:
    if not message.startswith("unsolved goals"):
        return None
    _, _, goals = message.partition("\n")
    return goals.strip() or None


def compile_result_from_response(response: Any, line_offset: int, full_source: str) -> CompileResult:
    """Convert a response; lines are shifted by ``line_offset``."""
    messages = list(getattr(response, "messages", None) or [])
    errors = sorted(
        (message for message in messages if message.severity == "error"),
        key=lambda message: _position_tuple(message.start_pos),
    )
    converted = tuple(
        ErrorMessage(
            line=int(message.start_pos.line) + line_offset,
            col=int(message.start_pos.column),
            end_line=int(message.end_pos.line) + line_offset if getattr(message, "end_pos", None) else None,
            end_col=int(message.end_pos.column) if getattr(message, "end_pos", None) else None,
            message=str(message.data).strip(),
        )
        for message in errors
    )
    tactics = tuple(
        TacticSpan(
            int(tactic.start_pos.line) + line_offset,
            int(tactic.start_pos.column),
            int(tactic.end_pos.line) + line_offset,
            int(tactic.end_pos.column),
            tuple(str(name) for name in (getattr(tactic, "used_constants", None) or ())),
        )
        for tactic in (getattr(response, "tactics", None) or [])
    )
    uses_sorry = any("declaration uses 'sorry'" in str(message.data) for message in messages)
    line_at_error = state = None
    if errors:
        line, col = _position_tuple(errors[0].start_pos)
        state = state_from_response(response, line, col) or unsolved_goals_state(converted[0].message)
        source_lines = full_source.splitlines()
        full_line = line + line_offset
        line_at_error = source_lines[full_line - 1].strip() if 1 <= full_line <= len(source_lines) else ""
    return CompileResult(converted, line_at_error, state, tactics, uses_sorry)


def lean_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t")
    return f'"{escaped}"'


RESOLVER_MARKER = "THM_RESOLVE "
EXPORT_MARKER = "THM_EXPORT "

# API spellings differ across Lean versions; the first variant that elaborates is kept.
_MODULE_NAME_EXPRS = (
    "env.header.moduleNames[idx.toNat]?",
    "env.header.moduleNames[idx]?",
    "env.allImportedModuleNames[idx.toNat]?",
    "env.allImportedModuleNames[idx]?",
)
_DEPRECATED_EXPRS = (
    "Lean.Linter.getDeprecatedNewName env c",
    "(none : Option Lean.Name)",
)
_CONSTANT_LIST_EXPRS = ("env.constants.toList", "env.constants.map₁.toList")
RESOLVER_VARIANTS = tuple((module, deprecated) for deprecated in _DEPRECATED_EXPRS for module in _MODULE_NAME_EXPRS)
EXPORT_VARIANTS = tuple((constants, module) for constants in _CONSTANT_LIST_EXPRS for module in _MODULE_NAME_EXPRS)

# Fully qualified names only: `open ... in` would change how queries resolve.
_RESOLVER_TEMPLATE = """\
#eval show Lean.Elab.Command.CommandElabM Unit from do
  let env ← Lean.getEnv
  let mut results : Array Lean.Json := #[]
  for q in ([@QUERIES@] : List String) do
    let mut found : Array Lean.Json := #[]
    let candidates ← Lean.Elab.Command.liftTermElabM (Lean.resolveGlobalName q.toName)
    for (c, fields) in candidates do
      if !fields.isEmpty then continue
      let some info := env.find? c | continue
      let kindStr : String := match info with
        | .thmInfo _ => "theorem"
        | .defnInfo _ => "def"
        | .axiomInfo _ => "axiom"
        | _ => "other"
      let modName : Option Lean.Name := match env.getModuleIdxFor? c with
        | some idx => @MODULE_EXPR@
        | none => none
      let modJson : Lean.Json := match modName with
        | some m => Lean.Json.str (toString m)
        | none => Lean.Json.null
      let newName : Option Lean.Name := @DEPRECATED_EXPR@
      let newJson : Lean.Json := match newName with
        | some n => Lean.Json.str (toString n)
        | none => Lean.Json.null
      let mut typeStr : String := ""
      try
        let fmt ← Lean.Elab.Command.liftTermElabM (Lean.Meta.ppExpr info.type)
        typeStr := toString fmt
      catch _ =>
        typeStr := ""
      found := found.push (Lean.Json.mkObj [
        ("name", Lean.Json.str (toString c)),
        ("kind", Lean.Json.str kindStr),
        ("module", modJson),
        ("type", Lean.Json.str typeStr),
        ("deprecated_alias_of", newJson)])
    results := results.push (Lean.Json.mkObj [("query", Lean.Json.str q), ("matches", Lean.Json.arr found)])
  Lean.logInfo (Lean.MessageData.ofFormat (Std.Format.text (@MARKER@ ++ (Lean.Json.arr results).compress)))
"""

_EXPORT_TEMPLATE = """\
#eval show Lean.Elab.Command.CommandElabM Unit from do
  let env ← Lean.getEnv
  let packages : List String := [@PACKAGES@]
  let mut names : Array String := #[]
  for (c, info) in @CONSTANTS_EXPR@ do
    if c.isInternal then continue
    let isThm : Bool := match info with
      | .thmInfo _ => true
      | _ => false
    if !isThm then continue
    let some idx := env.getModuleIdxFor? c | continue
    let some m := @MODULE_EXPR@ | continue
    if packages.contains m.getRoot.toString then
      names := names.push (toString c)
  IO.FS.writeFile @PATH@ ("\\n".intercalate names.toList ++ "\\n")
  Lean.logInfo (Lean.MessageData.ofFormat (Std.Format.text (@MARKER@ ++ toString names.size)))
"""


def resolver_command(names: Sequence[str], variant: int = 0) -> str:
    module_expr, deprecated_expr = RESOLVER_VARIANTS[variant]
    return (
        _RESOLVER_TEMPLATE
        .replace("@QUERIES@", ", ".join(lean_string(name) for name in names))
        .replace("@MODULE_EXPR@", module_expr)
        .replace("@DEPRECATED_EXPR@", deprecated_expr)
        .replace("@MARKER@", lean_string(RESOLVER_MARKER))
    )


def export_command(path: Path, packages: Sequence[str], variant: int = 0) -> str:
    constants_expr, module_expr = EXPORT_VARIANTS[variant]
    return (
        _EXPORT_TEMPLATE
        .replace("@PACKAGES@", ", ".join(lean_string(package) for package in packages))
        .replace("@CONSTANTS_EXPR@", constants_expr)
        .replace("@MODULE_EXPR@", module_expr)
        .replace("@PATH@", lean_string(str(path)))
        .replace("@MARKER@", lean_string(EXPORT_MARKER))
    )


def marked_info(messages: Iterable[Any], marker: str) -> str | None:
    for message in messages or []:
        data = str(getattr(message, "data", ""))
        if getattr(message, "severity", None) == "info" and data.startswith(marker):
            return data[len(marker):].strip()
    return None


def parse_resolver_output(messages: Iterable[Any]) -> dict[str, list[dict[str, Any]]] | None:
    payload = marked_info(messages, RESOLVER_MARKER)
    if payload is None:
        return None
    return {str(item["query"]): list(item.get("matches") or []) for item in json.loads(payload)}


def _describe(error: BaseException) -> str:
    text = str(error).strip().splitlines()
    return f"{type(error).__name__}: {text[0][:300]}" if text else type(error).__name__


class LeanEnvSession:
    """Lean-Interact server with a cached prefix environment."""

    def __init__(
        self,
        project_dir: Path,
        timeout: float,
        *,
        prefix_timeout: float | None = None,
        restart_between_proofs: bool = True,
        server_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.project_dir = project_dir.resolve()
        self.timeout = timeout
        self.prefix_timeout = prefix_timeout if prefix_timeout is not None else 4 * timeout
        self.restart_between_proofs = restart_between_proofs
        self._server_factory = server_factory or self._default_server
        self._config: Any = None
        self._server: Any = None
        self._current_proof: str | None = None
        self._env_key: str | None = None
        self._env_id: int | None = None
        self._resolver_variant: int | None = None
        self._export_variant: int | None = None

    def _default_server(self) -> Any:
        try:
            from lean_interact import LeanREPLConfig, LeanServer, LocalProject
        except ImportError as error:
            raise RuntimeError("install lean-interact to verify mutations") from error
        if self._config is None:
            self._config = LeanREPLConfig(
                project=LocalProject(directory=self.project_dir, auto_build=False),
                enable_parallel_elaboration=False,
                enable_incremental_optimization=False,
            )
        return LeanServer(self._config)

    @staticmethod
    def _command(**fields: Any) -> Any:
        from lean_interact import Command

        return Command(**fields)

    @staticmethod
    def _file_command(**fields: Any) -> Any:
        from lean_interact import FileCommand

        return FileCommand(**fields)

    def restart(self) -> None:
        """Drop the server and cached environments."""
        server, self._server = self._server, None
        self._env_key = self._env_id = None
        if server is None:
            return
        for method in ("kill", "close", "stop"):
            stop = getattr(server, method, None)
            if callable(stop):
                try:
                    stop()
                except Exception:
                    pass
                break

    def begin_proof(self, key: str) -> None:
        """Restart between proofs so environments don't accumulate."""
        if self.restart_between_proofs and self._current_proof not in (None, key):
            self.restart()
        self._current_proof = key

    def _run(self, request: Any, timeout: float) -> Any:
        if self._server is None:
            self._server = self._server_factory()
        return self._server.run(request, timeout=timeout)

    def _checked_run(self, request: Any, timeout: float) -> Any:
        response = self._run(request, timeout)
        if not hasattr(response, "messages"):
            raise RuntimeError(f"Lean-Interact error: {getattr(response, 'message', response)}")
        return response

    def _prefix_env(self, key: str, prefix: str) -> int | None:
        if not prefix.strip():
            return None
        if self._env_key == key and self._env_id is not None:
            return self._env_id
        response = self._checked_run(self._command(cmd=prefix), self.prefix_timeout)
        errors = [message for message in response.messages if message.severity == "error"]
        if errors:
            first = errors[0]
            raise PrefixError(f"line {first.start_pos.line}: {str(first.data).strip()[:300]}")
        self._env_key, self._env_id = key, int(response.env)
        return self._env_id

    def compile_declaration(
        self, key: str, prefix: str, declaration: str, line_offset: int, full_source: str
    ) -> CompileResult:
        """Compile ``declaration`` on the ``prefix`` environment."""
        try:
            env = self._prefix_env(key, prefix)
        except PrefixError as error:
            return CompileResult(prefix_failed=True, detail=str(error))
        except TimeoutError:
            self.restart()
            return CompileResult(timed_out=True, detail="prefix timed out")
        except Exception as error:
            self.restart()
            return CompileResult(crashed=True, detail=_describe(error))
        fields: dict[str, Any] = {"cmd": declaration, "all_tactics": True, "infotree": "full"}
        if env is not None:
            fields["env"] = env
        try:
            response = self._run(self._command(**fields), self.timeout)
        except TimeoutError:
            self.restart()
            return CompileResult(timed_out=True)
        except Exception as error:
            self.restart()
            return CompileResult(crashed=True, detail=_describe(error))
        if not hasattr(response, "messages"):
            self.restart()
            return CompileResult(crashed=True, detail=str(getattr(response, "message", response))[:300])
        return compile_result_from_response(response, line_offset, full_source)

    def compile_file(self, source: str) -> CompileResult:
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", suffix=".lean", prefix=".theorem-mutation-", delete=False
            ) as handle:
                handle.write(source)
                temporary = Path(handle.name)
            request = self._file_command(path=str(temporary), all_tactics=True, root_goals=True, infotree="full")
            try:
                response = self._run(request, self.prefix_timeout)
            except TimeoutError:
                self.restart()
                return CompileResult(timed_out=True)
            except Exception as error:
                self.restart()
                return CompileResult(crashed=True, detail=_describe(error))
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        if not hasattr(response, "messages"):
            self.restart()
            return CompileResult(crashed=True, detail=str(getattr(response, "message", response))[:300])
        return compile_result_from_response(response, 0, source)

    def resolve(
        self, key: str, prefix: str, names: Sequence[str], *, inline: bool = False
    ) -> dict[str, list[dict[str, Any]]]:
        """Resolve names in the scope at the end of ``prefix``.

        ``inline=True`` appends the resolver to ``prefix`` and runs one command.
        Raises RuntimeError when no variant elaborates.
        """
        queries = sorted(set(names))
        if not queries:
            return {}
        variants = (
            [self._resolver_variant] if self._resolver_variant is not None else list(range(len(RESOLVER_VARIANTS)))
        )
        failures: list[str] = []
        for variant in variants:
            command = resolver_command(queries, variant)
            if inline:
                body = prefix.rstrip()
                prefix_lines = body.count("\n") + 1
                response = self._checked_run(self._command(cmd=body + "\n\n" + command), self.prefix_timeout)
                errors = [message for message in response.messages if message.severity == "error"]
                own = [message for message in errors if message.start_pos.line <= prefix_lines]
                if own:
                    raise PrefixError(f"line {own[0].start_pos.line}: {str(own[0].data).strip()[:300]}")
            else:
                env = self._prefix_env(key, prefix)
                fields: dict[str, Any] = {"cmd": command}
                if env is not None:
                    fields["env"] = env
                response = self._checked_run(self._command(**fields), self.timeout)
                errors = [message for message in response.messages if message.severity == "error"]
            if errors:
                failures.append(str(errors[0].data).strip()[:200])
                continue
            parsed = parse_resolver_output(response.messages)
            if parsed is None:
                failures.append("resolver printed no output")
                continue
            self._resolver_variant = variant
            return parsed
        raise RuntimeError("Lean name resolver failed for every API variant: " + " | ".join(failures[:3]))

    def export_theorem_names(self, key: str, prefix: str, path: Path, packages: Sequence[str]) -> int:
        """Write library theorem names to ``path``."""
        env = self._prefix_env(key, prefix)
        variants = [self._export_variant] if self._export_variant is not None else list(range(len(EXPORT_VARIANTS)))
        failures: list[str] = []
        path.parent.mkdir(parents=True, exist_ok=True)
        for variant in variants:
            fields: dict[str, Any] = {"cmd": export_command(path.resolve(), packages, variant)}
            if env is not None:
                fields["env"] = env
            response = self._checked_run(self._command(**fields), self.prefix_timeout)
            errors = [message for message in response.messages if message.severity == "error"]
            if errors:
                failures.append(str(errors[0].data).strip()[:200])
                continue
            count = marked_info(response.messages, EXPORT_MARKER)
            if count is None:
                failures.append("export printed no output")
                continue
            self._export_variant = variant
            return int(count)
        raise RuntimeError("Lean theorem export failed for every API variant: " + " | ".join(failures[:3]))
