"""Tripwire for the ACK sentinel gate lifecycle at every ``emit_delta`` call site.

The ACK gate lives in the shared ``emit_delta`` channel, but its lifecycle is
manual: each streaming attempt must call ``reset_ack_sentinel_gate`` before it
starts emitting deltas and ``flush_ack_sentinel_gate`` after the stream ends.
Forgetting ``reset`` splices stale buffered text across turns; forgetting
``flush`` silently drops a held-back partial-sentinel tail of a legitimate
answer. This test enumerates every ``emit_delta`` call site in ``chibi/`` and
fails when its enclosing function lacks either lifecycle call, so a future
streaming adapter that forgets the wiring fails CI loudly instead of silently.

Call-site matching resolves the emitter through its import bindings rather than
matching import spellings: every import is resolved to an absolute module path
(relative imports included, with Python's own level semantics), module aliases
are bound to the paths they denote, and dotted references are resolved through
those bindings by longest-prefix matching. That covers the canonical plain-name
form, import aliases, module-attribute chains rooted at the streaming module or
its parent packages, and fully dotted absolute chains.

Any module that imports the streaming module itself but contributes no
resolvable call site is reported outright, so a binding style the matcher cannot
resolve fails CI instead of passing silently. A package-level import
(``from chibi.services import providers``) alone does not trip that fence,
because the package is legitimately imported for non-streaming names elsewhere;
its ``<alias>.streaming.emit_delta(...)`` calls are still resolved and checked.
"""

import ast
from dataclasses import dataclass, field
from pathlib import Path

CHIBI_ROOT = Path(__file__).resolve().parents[3] / "chibi"

EMIT_DELTA = "emit_delta"
RESET_GATE = "reset_ack_sentinel_gate"
FLUSH_GATE = "flush_ack_sentinel_gate"

# Absolute name of the streaming module and its root-relative spellings. Import
# resolution happens in root-relative space so the same matcher works on the real
# ``chibi`` tree and on synthetic roots laid out as ``services/providers/*.py``.
ABS_STREAMING_MODULE = "chibi.services.providers.streaming"
REL_STREAMING_MODULE = "services.providers.streaming"
REL_STREAMING_PACKAGE = "services.providers"
_ABS_PREFIX = "chibi."


def _normalize(dotted: str) -> str:
    """Strip the ``chibi.`` package prefix so absolute and relative names compare equal."""
    return dotted[len(_ABS_PREFIX) :] if dotted.startswith(_ABS_PREFIX) else dotted


@dataclass
class _Bindings:
    """Local names bound to the streaming emitter, its lifecycle helpers, or its modules."""

    functions: dict[str, set[str]] = field(default_factory=dict)
    module_aliases: dict[str, str] = field(default_factory=dict)
    imports_streaming_module: bool = False

    def bind_function(self, source_name: str, local_name: str) -> None:
        self.functions.setdefault(source_name, set()).add(local_name)

    def local_names(self, source_name: str) -> set[str]:
        return self.functions.get(source_name, set())


def _containing_package(path: Path, root: Path) -> tuple[str, ...]:
    """Return the root-relative dotted package containing ``path``.

    For ``services/providers/foo.py`` this is ``services.providers``; for an
    ``__init__.py`` it is the package the file defines. This is exactly the base
    a level-1 relative import resolves against, for modules and packages alike.
    """
    return path.relative_to(root).with_suffix("").parts[:-1]


def _resolve_import_from(node: ast.ImportFrom, package_parts: tuple[str, ...]) -> str | None:
    """Resolve an ``ImportFrom`` node to a root-relative absolute module path.

    Level 0 resolves the written absolute module, normalized to drop the
    ``chibi.`` prefix. Level N resolves against the containing package climbed
    ``N - 1`` levels up, matching Python's relative-import semantics.
    """
    if node.level == 0:
        return _normalize(node.module) if node.module else None
    if len(package_parts) < node.level - 1:
        return None
    base = package_parts[: len(package_parts) - (node.level - 1)]
    if node.module:
        base = base + tuple(node.module.split("."))
    return ".".join(base)


def _collect_bindings(tree: ast.Module, package_parts: tuple[str, ...]) -> _Bindings:
    """Collect the module's bindings to the streaming layer from its import statements."""
    bindings = _Bindings()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            resolved = _resolve_import_from(node, package_parts)
            if resolved is None:
                continue
            if resolved == REL_STREAMING_MODULE:
                bindings.imports_streaming_module = True
                for alias in node.names:
                    bindings.bind_function(alias.name, alias.asname or alias.name)
                continue
            for alias in node.names:
                target = f"{resolved}.{alias.name}"
                if target == REL_STREAMING_MODULE:
                    bindings.imports_streaming_module = True
                    bindings.module_aliases[alias.asname or alias.name] = REL_STREAMING_MODULE
                elif target == REL_STREAMING_PACKAGE:
                    bindings.module_aliases[alias.asname or alias.name] = REL_STREAMING_PACKAGE
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if _normalize(alias.name) == REL_STREAMING_MODULE:
                    bindings.imports_streaming_module = True
                    if alias.asname:
                        bindings.module_aliases[alias.asname] = REL_STREAMING_MODULE
    return bindings


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


def _resolves_to_streaming_module(dotted: str, bindings: _Bindings) -> bool:
    """True when a dotted reference resolves to the streaming module.

    Handles the full absolute chain (``chibi.services.providers.streaming``) and
    any prefix bound by an import: a module alias (``streaming``, ``st``), a
    parent-package alias (``providers`` bound to ``chibi.services.providers``),
    and their combinations (``providers.streaming``, ``p.streaming``).
    """
    ref = _normalize(dotted)
    if ref == REL_STREAMING_MODULE:
        return True
    parts = ref.split(".")
    for i in range(1, len(parts) + 1):
        absolute = bindings.module_aliases.get(".".join(parts[:i]))
        if absolute is not None and ".".join([absolute, *parts[i:]]) == REL_STREAMING_MODULE:
            return True
    return False


def _is_emitter_call(func: ast.AST, bindings: _Bindings) -> bool:
    """True when ``func`` is a call of ``emit_delta`` through any resolvable binding."""
    if isinstance(func, ast.Name):
        return func.id in bindings.local_names(EMIT_DELTA)
    if isinstance(func, ast.Attribute) and func.attr == EMIT_DELTA:
        dotted = _dotted_name(func.value)
        return dotted is not None and _resolves_to_streaming_module(dotted, bindings)
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
    resolved through the module bindings (``streaming.flush_ack_sentinel_gate``,
    ``providers.streaming.flush_ack_sentinel_gate``). Unknown callees are
    reported under their literal spelling.
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
            if dotted is not None and _resolves_to_streaming_module(dotted, bindings):
                called.add(node.func.attr)
    return called


@dataclass
class _ScanResult:
    """Outcome of scanning a source tree for emitter call sites."""

    violations: list[str] = field(default_factory=list)
    checked: int = 0


def _scan_for_lifecycle_violations(root: Path) -> _ScanResult:
    """Enumerate every ``emit_delta`` call site under ``root`` and check the gate lifecycle.

    Modules that import the streaming module itself but yield no resolvable call
    site are reported as violations, so a binding style the matcher does not
    understand fails loudly instead of silently skipping the module.
    """
    result = _ScanResult()
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        bindings = _collect_bindings(tree, _containing_package(path, root))
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
                    f"after the stream loop (see {ABS_STREAMING_MODULE.replace('.', '/')}.py)."
                )

        if bindings.imports_streaming_module and module_sites == 0:
            result.violations.append(
                f"{relative}: imports the ACK streaming module but no {EMIT_DELTA} call site "
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
        if _collect_bindings(tree, _containing_package(path, CHIBI_ROOT)).local_names(EMIT_DELTA):
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


def test_tripwire_catches_relative_from_import_call_site(tmp_path: Path) -> None:
    """``from .streaming import emit_delta`` with no lifecycle must fail."""
    _write_module(
        tmp_path,
        "services/providers/newadapter.py",
        "from .streaming import emit_delta\n\n\n"
        "async def stream_new_thing(interface):\n    await emit_delta(interface, 'x')\n",
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


def test_tripwire_catches_relative_package_import_call_site(tmp_path: Path) -> None:
    """``from . import streaming`` plus ``streaming.emit_delta(...)`` with no lifecycle must fail."""
    _write_module(
        tmp_path,
        "services/providers/newadapter.py",
        "from . import streaming\n\n\n"
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


def test_tripwire_catches_parent_package_attribute_chain_call_site(tmp_path: Path) -> None:
    """``from chibi.services import providers`` plus ``providers.streaming.emit_delta`` must fail."""
    _write_module(
        tmp_path,
        "services/providers/newadapter.py",
        "from chibi.services import providers\n\n\n"
        "async def stream_new_thing(interface):\n"
        "    await providers.streaming.emit_delta(interface, 'x')\n",
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


def test_tripwire_catches_full_dotted_chain_without_streaming_import(tmp_path: Path) -> None:
    """A fully dotted emitter call is checked even when only an ancestor package is imported."""
    _write_module(
        tmp_path,
        "services/providers/newadapter.py",
        "import chibi.services\n\n\n"
        "async def stream_new_thing(interface):\n"
        "    await chibi.services.providers.streaming.emit_delta(interface, 'x')\n",
    )
    result = _scan_for_lifecycle_violations(tmp_path)
    assert result.checked == 1
    assert any(
        "services/providers/newadapter.py" in violation and "'stream_new_thing'" in violation
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


def test_tripwire_accepts_relative_from_import_with_full_lifecycle(tmp_path: Path) -> None:
    """A relative-import call site whose function wires both lifecycle calls passes."""
    _write_module(
        tmp_path,
        "services/providers/newadapter.py",
        "from .streaming import emit_delta, reset_ack_sentinel_gate as _reset, "
        "flush_ack_sentinel_gate as _flush\n\n\n"
        "async def stream_new_thing(interface):\n    _reset(interface)\n"
        "    await emit_delta(interface, 'x')\n    _flush(interface)\n",
    )
    result = _scan_for_lifecycle_violations(tmp_path)
    assert result.checked == 1
    assert not result.violations, result.violations


def test_tripwire_accepts_relative_package_import_with_full_lifecycle(tmp_path: Path) -> None:
    """``from . import streaming`` wired through module-attribute lifecycle calls passes."""
    _write_module(
        tmp_path,
        "services/providers/newadapter.py",
        "from . import streaming\n\n\n"
        "async def stream_new_thing(interface):\n"
        "    streaming.reset_ack_sentinel_gate(interface)\n"
        "    await streaming.emit_delta(interface, 'x')\n"
        "    streaming.flush_ack_sentinel_gate(interface)\n",
    )
    result = _scan_for_lifecycle_violations(tmp_path)
    assert result.checked == 1
    assert not result.violations, result.violations


def test_tripwire_accepts_parent_package_attribute_chain_with_full_lifecycle(tmp_path: Path) -> None:
    """``providers.streaming.*`` lifecycle calls credit the enclosing function."""
    _write_module(
        tmp_path,
        "services/providers/newadapter.py",
        "from chibi.services import providers\n\n\n"
        "async def stream_new_thing(interface):\n"
        "    providers.streaming.reset_ack_sentinel_gate(interface)\n"
        "    await providers.streaming.emit_delta(interface, 'x')\n"
        "    providers.streaming.flush_ack_sentinel_gate(interface)\n",
    )
    result = _scan_for_lifecycle_violations(tmp_path)
    assert result.checked == 1
    assert not result.violations, result.violations


def test_package_level_import_without_streaming_usage_is_not_flagged(tmp_path: Path) -> None:
    """Importing the providers package for non-streaming names must not trip the fence."""
    _write_module(
        tmp_path,
        "services/registry.py",
        "from chibi.services import providers\n\n\ndef known_providers():\n    return providers.RegisteredProviders\n",
    )
    result = _scan_for_lifecycle_violations(tmp_path)
    assert result.checked == 0
    assert not result.violations, result.violations


def test_tripwire_flags_streaming_import_without_resolvable_call_site(tmp_path: Path) -> None:
    """A module importing the streaming module with no resolvable call site fails loudly."""
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
