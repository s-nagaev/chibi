"""Tripwire for the ACK sentinel gate lifecycle at every ``emit_delta`` call site.

The ACK gate lives in the shared ``emit_delta`` channel, but its lifecycle is
manual: each streaming attempt must call ``reset_ack_sentinel_gate`` before it
starts emitting deltas and ``flush_ack_sentinel_gate`` after the stream ends.
Forgetting ``reset`` splices stale buffered text across turns; forgetting
``flush`` silently drops a held-back partial-sentinel tail of a legitimate
answer. This test enumerates every ``emit_delta`` call site in ``chibi/`` and
fails when its enclosing function lacks either lifecycle call, so a future
streaming adapter that forgets the wiring fails CI loudly instead of silently.

Call-site matching resolves import aliases: ``from ... import emit_delta as _emit``
plus ``_emit(...)`` and module-attribute forms such as
``from chibi.services.providers import streaming`` plus ``streaming.emit_delta(...)``
are enumerated the same way as the canonical plain-name form. Any module that
imports the streaming layer but contributes no resolvable call site is itself
reported, so a binding style the matcher cannot resolve fails CI instead of
passing silently.
"""

import ast
from dataclasses import dataclass, field
from pathlib import Path

CHIBI_ROOT = Path(__file__).resolve().parents[3] / "chibi"

EMIT_DELTA = "emit_delta"
RESET_GATE = "reset_ack_sentinel_gate"
FLUSH_GATE = "flush_ack_sentinel_gate"
STREAMING_MODULE = "chibi.services.providers.streaming"
STREAMING_PACKAGE = "chibi.services.providers"
STREAMING_LOCAL_NAME = "streaming"


@dataclass
class _Bindings:
    """Local names bound to the streaming emitter, its lifecycle helpers, or the module itself."""

    functions: dict[str, set[str]] = field(default_factory=dict)
    module_aliases: set[str] = field(default_factory=set)
    imports_streaming_module: bool = False

    def bind_function(self, source_name: str, local_name: str) -> None:
        self.functions.setdefault(source_name, set()).add(local_name)

    def local_names(self, source_name: str) -> set[str]:
        return self.functions.get(source_name, set())


def _resolve_import_from(node: ast.ImportFrom, package: str) -> str | None:
    """Resolve an ``ImportFrom`` to an absolute module path, handling relative imports."""
    if node.level == 0:
        return node.module
    if not package:
        return None
    parts = package.split(".")
    base = parts[: len(parts) - (node.level - 1)] if node.level > 1 else parts
    if node.module:
        base = base + node.module.split(".")
    return ".".join(base)


def _bindings(tree: ast.Module, package: str) -> _Bindings:
    """Collect the module's bindings to the streaming layer from its import statements."""
    bindings = _Bindings()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            resolved = _resolve_import_from(node, package)
            if resolved is None:
                continue
            if resolved == STREAMING_MODULE:
                bindings.imports_streaming_module = True
                for alias in node.names:
                    bindings.bind_function(alias.name, alias.asname or alias.name)
            elif resolved == STREAMING_PACKAGE:
                for alias in node.names:
                    if alias.name == STREAMING_LOCAL_NAME:
                        bindings.module_aliases.add(alias.asname or alias.name)
                        bindings.imports_streaming_module = True
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == STREAMING_MODULE:
                    bindings.imports_streaming_module = True
                    if alias.asname:
                        bindings.module_aliases.add(alias.asname)
    return bindings


def _package_of(path: Path, root: Path) -> str:
    """Return the dotted package path of ``path`` relative to ``root``."""
    rel = path.relative_to(root).with_suffix("")
    if rel.name == "__init__":
        rel = rel.parent
    return rel.as_posix().replace("/", ".")


def _dotted_name(node: ast.AST) -> str | None:
    """Return the dotted path of a ``Name``/``Attribute`` chain, or None."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _is_streaming_module_ref(dotted: str, bindings: _Bindings) -> bool:
    """True when ``dotted`` resolves to the streaming module under a known binding."""
    return dotted == STREAMING_MODULE or dotted in bindings.module_aliases


def _is_emitter_call(func: ast.AST, bindings: _Bindings) -> bool:
    """True when ``func`` is a call of ``emit_delta`` through any supported binding form."""
    if isinstance(func, ast.Name):
        return func.id in bindings.local_names(EMIT_DELTA)
    if isinstance(func, ast.Attribute) and func.attr == EMIT_DELTA:
        dotted = _dotted_name(func.value)
        return dotted is not None and _is_streaming_module_ref(dotted, bindings)
    return False


def _enclosing_function(
    node: ast.AST, parents: dict[ast.AST, ast.AST]
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """Return the nearest function definition enclosing ``node``, if any."""
    current = parents.get(node)
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current
        current = parents.get(current)
    return None


def _called_function_names(function: ast.FunctionDef | ast.AsyncFunctionDef, bindings: _Bindings) -> set[str]:
    """Collect the source-level function names called inside ``function``, aliases resolved.

    A plain-name call is mapped through the import aliases; an attribute call is
    mapped through the module bindings (``streaming.flush_ack_sentinel_gate``).
    Unknown callees are reported under their literal spelling.
    """
    aliases = {local: source for source, locals_ in bindings.functions.items() for local in locals_}
    called: set[str] = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            called.add(aliases.get(node.func.id, node.func.id))
        elif isinstance(node.func, ast.Attribute):
            dotted = _dotted_name(node.func.value)
            if dotted is not None and _is_streaming_module_ref(dotted, bindings):
                called.add(node.func.attr)
    return called


@dataclass
class _ScanResult:
    """Outcome of scanning a source tree for emitter call sites."""

    violations: list[str] = field(default_factory=list)
    checked: int = 0


def _scan_for_lifecycle_violations(root: Path) -> _ScanResult:
    """Enumerate every ``emit_delta`` call site under ``root`` and check the gate lifecycle.

    Modules that import the streaming layer but yield no resolvable call site are
    reported as violations, so a binding style the matcher does not understand
    fails loudly instead of silently skipping the module.
    """
    result = _ScanResult()
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        bindings = _bindings(tree, _package_of(path, root))
        if not bindings.imports_streaming_module:
            continue
        relative = path.relative_to(root).as_posix()
        module_sites = 0

        parents: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent

        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and _is_emitter_call(node.func, bindings)):
                continue
            module_sites += 1
            result.checked += 1
            function = _enclosing_function(node, parents)
            if function is None:
                result.violations.append(
                    f"{relative}:{node.lineno}: {EMIT_DELTA} called outside of any function; "
                    f"the gate lifecycle cannot be verified"
                )
                continue
            called = _called_function_names(function, bindings)
            missing = [name for name in (RESET_GATE, FLUSH_GATE) if name not in called]
            if missing:
                result.violations.append(
                    f"{relative}:{node.lineno}: {EMIT_DELTA} call in {function.name!r} is missing "
                    f"the ACK gate lifecycle call(s): {', '.join(missing)}. Every streaming "
                    f"attempt must call {RESET_GATE}() before emitting deltas and {FLUSH_GATE}() "
                    f"after the stream loop (see {STREAMING_MODULE.replace('.', '/')}.py)."
                )

        if module_sites == 0:
            result.violations.append(
                f"{relative}: imports the ACK streaming layer but no {EMIT_DELTA} call site "
                f"could be resolved in it. Either the module has no call sites (remove the "
                f"unused import) or it calls the emitter through a binding form this tripwire "
                f"cannot resolve; extend the matcher instead of letting the module go unchecked."
            )
    return result


def test_every_emit_delta_call_site_wires_the_ack_gate_lifecycle() -> None:
    """Every enclosing function of an ``emit_delta`` call must reset and flush the gate."""
    result = _scan_for_lifecycle_violations(CHIBI_ROOT)
    assert result.checked > 0, (
        "No emit_delta call sites were found; the enumeration logic no longer "
        "matches the codebase and this tripwire is vacuous."
    )
    assert not result.violations, "ACK gate lifecycle is broken at some emit_delta call site(s):\n" + "\n".join(
        result.violations
    )


def test_emit_delta_call_sites_were_found_in_streaming_adapters() -> None:
    """The shared emitter must still be reachable from the known provider adapters."""
    adapter_files = {
        "services/providers/provider.py",
        "services/providers/openai.py",
        "services/providers/gemini_native.py",
        "services/providers/mistralai_native.py",
    }
    found = set()
    for path in sorted(CHIBI_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if _bindings(tree, _package_of(path, CHIBI_ROOT)).local_names(EMIT_DELTA):
            found.add(path.relative_to(CHIBI_ROOT).as_posix())
    missing = adapter_files - found
    assert not missing, (
        f"emit_delta is no longer imported by known streaming adapter(s): "
        f"{sorted(missing)}. Update this tripwire if the adapter layout changed."
    )


def _write_module(root: Path, relative: str, source: str) -> None:
    """Write a synthetic source module into ``root`` for matcher self-tests."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def test_tripwire_catches_aliased_emit_delta_call_site(tmp_path: Path) -> None:
    """``from ... import emit_delta as _emit`` with no lifecycle must fail."""
    _write_module(
        tmp_path,
        "services/providers/newadapter.py",
        "from chibi.services.providers.streaming import emit_delta as _emit\n\n\n"
        "async def stream_new_thing(interface):\n    await _emit(interface, 'x')\n",
    )
    result = _scan_for_lifecycle_violations(tmp_path)
    assert result.checked == 1
    assert any(
        "services/providers/newadapter.py" in violation
        and "'stream_new_thing'" in violation
        and RESET_GATE in violation
        and FLUSH_GATE in violation
        for violation in result.violations
    ), result.violations


def test_tripwire_catches_module_attribute_emit_delta_call_site(tmp_path: Path) -> None:
    """``streaming.emit_delta(...)`` with no lifecycle must fail."""
    _write_module(
        tmp_path,
        "services/providers/newadapter.py",
        "from chibi.services.providers import streaming\n\n\n"
        "async def stream_new_thing(interface):\n    await streaming.emit_delta(interface, 'x')\n",
    )
    result = _scan_for_lifecycle_violations(tmp_path)
    assert result.checked == 1
    assert any(
        "services/providers/newadapter.py" in violation
        and "'stream_new_thing'" in violation
        and RESET_GATE in violation
        and FLUSH_GATE in violation
        for violation in result.violations
    ), result.violations


def test_tripwire_accepts_aliased_call_site_with_full_lifecycle(tmp_path: Path) -> None:
    """An aliased call site whose function wires both lifecycle calls passes."""
    _write_module(
        tmp_path,
        "services/providers/newadapter.py",
        "from chibi.services.providers.streaming import (\n"
        "    emit_delta as _emit,\n"
        f"    {RESET_GATE} as _reset,\n"
        f"    {FLUSH_GATE} as _flush,\n"
        ")\n\n\n"
        "async def stream_new_thing(interface):\n    _reset(interface)\n"
        "    await _emit(interface, 'x')\n    _flush(interface)\n",
    )
    result = _scan_for_lifecycle_violations(tmp_path)
    assert result.checked == 1
    assert not result.violations, result.violations


def test_tripwire_accepts_module_attribute_call_site_with_full_lifecycle(tmp_path: Path) -> None:
    """``streaming.emit_delta`` wired through module-attribute lifecycle calls passes."""
    _write_module(
        tmp_path,
        "services/providers/newadapter.py",
        "from chibi.services.providers import streaming\n\n\n"
        "async def stream_new_thing(interface):\n"
        "    streaming.reset_ack_sentinel_gate(interface)\n"
        "    await streaming.emit_delta(interface, 'x')\n"
        "    streaming.flush_ack_sentinel_gate(interface)\n",
    )
    result = _scan_for_lifecycle_violations(tmp_path)
    assert result.checked == 1
    assert not result.violations, result.violations


def test_tripwire_flags_streaming_import_without_resolvable_call_site(tmp_path: Path) -> None:
    """A module importing the streaming layer with no resolvable call site fails loudly."""
    _write_module(
        tmp_path,
        "services/providers/newadapter.py",
        "import chibi.services.providers.streaming\n\n\nasync def stream_new_thing(interface):\n    pass\n",
    )
    result = _scan_for_lifecycle_violations(tmp_path)
    assert result.checked == 0
    assert any(
        "newadapter.py" in violation and f"no {EMIT_DELTA} call site" in violation for violation in result.violations
    ), result.violations
