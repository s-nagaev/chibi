"""Tripwire for the ACK sentinel gate lifecycle at every ``emit_delta`` call site.

The ACK gate lives in the shared ``emit_delta`` channel, but its lifecycle is
manual: each streaming attempt must call ``reset_ack_sentinel_gate`` before it
starts emitting deltas and ``flush_ack_sentinel_gate`` after the stream ends.
Forgetting ``reset`` splices stale buffered text across turns; forgetting
``flush`` silently drops a held-back partial-sentinel tail of a legitimate
answer. This test enumerates every ``emit_delta`` call site in ``chibi/`` and
fails when its enclosing function lacks either lifecycle call, so a future
streaming adapter that forgets the wiring fails CI loudly instead of silently.
"""

import ast
from pathlib import Path

CHIBI_ROOT = Path(__file__).resolve().parents[3] / "chibi"

EMIT_DELTA = "emit_delta"
RESET_GATE = "reset_ack_sentinel_gate"
FLUSH_GATE = "flush_ack_sentinel_gate"
STREAMING_MODULE = "chibi.services.providers.streaming"


def _streaming_imports(tree: ast.Module) -> dict[str, str]:
    """Map local names imported from the streaming module to their source names."""
    imported: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == STREAMING_MODULE and node.level == 0:
            for alias in node.names:
                imported[alias.asname or alias.name] = alias.name
    return imported


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


def _called_names(function: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Collect the plain-name callees inside ``function``."""
    return {
        node.func.id for node in ast.walk(function) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


def _emit_delta_call_sites() -> list[tuple[Path, ast.Module]]:
    """Return (file, tree) pairs for every ``chibi`` module importing the emitter."""
    sites: list[tuple[Path, ast.Module]] = []
    for path in sorted(CHIBI_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if EMIT_DELTA in _streaming_imports(tree):
            sites.append((path, tree))
    return sites


def test_every_emit_delta_call_site_wires_the_ack_gate_lifecycle() -> None:
    """Every enclosing function of an ``emit_delta`` call must reset and flush the gate."""
    sites = _emit_delta_call_sites()
    violations: list[str] = []
    checked = 0

    for path, tree in sites:
        parents: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent

        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            if node.func.id != EMIT_DELTA:
                continue
            checked += 1
            function = _enclosing_function(node, parents)
            if function is None:
                violations.append(
                    f"{path.relative_to(CHIBI_ROOT)}:{node.lineno}: {EMIT_DELTA} called "
                    f"outside of any function; the gate lifecycle cannot be verified"
                )
                continue
            called = _called_names(function)
            missing = [name for name in (RESET_GATE, FLUSH_GATE) if name not in called]
            if missing:
                violations.append(
                    f"{path.relative_to(CHIBI_ROOT)}:{node.lineno}: {EMIT_DELTA} call in "
                    f"{function.name!r} is missing the ACK gate lifecycle call(s): "
                    f"{', '.join(missing)}. Every streaming attempt must call "
                    f"{RESET_GATE}() before emitting deltas and {FLUSH_GATE}() after the "
                    f"stream loop (see chibi/services/providers/streaming.py)."
                )

    assert checked > 0, (
        "No emit_delta call sites were found; the enumeration logic no longer "
        "matches the codebase and this tripwire is vacuous."
    )
    assert not violations, "ACK gate lifecycle is broken at some emit_delta call site(s):\n" + "\n".join(violations)


def test_emit_delta_call_sites_were_found_in_streaming_adapters() -> None:
    """The shared emitter must still be reachable from the known provider adapters."""
    adapter_files = {
        "services/providers/provider.py",
        "services/providers/openai.py",
        "services/providers/gemini_native.py",
        "services/providers/mistralai_native.py",
    }
    found = {path.relative_to(CHIBI_ROOT).as_posix() for path, _ in _emit_delta_call_sites()}
    missing = adapter_files - found
    assert not missing, (
        f"emit_delta is no longer imported by known streaming adapter(s): "
        f"{sorted(missing)}. Update this tripwire if the adapter layout changed."
    )
