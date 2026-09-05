"""Static validation of a generated skill module — the load-bearing defense,
independent of whatever isolation the sandbox manages to provide (PRD:
"when only the subprocess path is available, isolation is weaker").

Three checks, all via `ast` — the module is never imported or executed here:
1. A complete, literal `MANIFEST = SkillManifest(...)` assignment.
2. A top-level `def run(ctx, **params):`.
3. No import or call from the PRD's denylist unless the matching permission
   was declared: `socket`/`http`/`urllib`/`requests`/`httpx`/`ftplib`/
   `smtplib` need `"net"`; `subprocess`/`ctypes`/`os.system`/`os.popen`/
   `eval`/`exec`/`__import__` need `"shell"`; a builtin `open(...)` call needs
   `"fs_read"` or `"fs_write"`.
"""

from __future__ import annotations

import ast

from jarvis.skills.contract import SkillManifest

NET_MODULES = frozenset({"socket", "http", "urllib", "requests", "httpx", "ftplib", "smtplib"})
SHELL_MODULES = frozenset({"subprocess", "ctypes"})
SHELL_BUILTIN_CALLS = frozenset({"eval", "exec", "__import__"})


class ValidationError(RuntimeError):
    """The generated module failed a structural or permission check."""


def validate(
    module_source: str,
    known_names: frozenset[str] = frozenset(),
    *,
    allow_name: str | None = None,
) -> SkillManifest:
    try:
        tree = ast.parse(module_source)
    except SyntaxError as exc:
        raise ValidationError(f"generated module has a syntax error: {exc}") from exc

    manifest = _extract_manifest(tree)
    if manifest.name != allow_name and manifest.name in known_names:
        raise ValidationError(
            f"a skill named '{manifest.name}' already exists — pick a different name "
            "or use edit_skill instead"
        )

    if not _has_run_function(tree):
        raise ValidationError("no top-level `def run(ctx, **params):` found")

    violations = _denylist_violations(tree, manifest.permissions)
    if violations:
        raise ValidationError(
            f"'{manifest.name}' needs permission(s) it didn't declare: "
            + "; ".join(violations)
        )

    return manifest


def _extract_manifest(tree: ast.Module) -> SkillManifest:
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "MANIFEST" for t in node.targets):
            continue
        call = node.value
        if not (isinstance(call, ast.Call) and _dotted_name(call.func) == "SkillManifest"):
            raise ValidationError("MANIFEST must be assigned `SkillManifest(...)` directly")
        if call.args:
            raise ValidationError("MANIFEST's SkillManifest(...) call must use keyword arguments only")
        kwargs = {}
        for kw in call.keywords:
            if kw.arg is None:
                raise ValidationError("MANIFEST must not use `**kwargs` expansion")
            try:
                kwargs[kw.arg] = ast.literal_eval(kw.value)
            except ValueError as exc:
                raise ValidationError(
                    f"MANIFEST.{kw.arg} must be a literal, not an expression"
                ) from exc
        try:
            return SkillManifest(**kwargs)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"MANIFEST is invalid: {exc}") from exc
    raise ValidationError("no `MANIFEST = SkillManifest(...)` assignment found")


def _has_run_function(tree: ast.Module) -> bool:
    return any(
        isinstance(node, ast.FunctionDef) and node.name == "run" and node.args.args
        for node in tree.body
    )


def _dotted_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return None


def _root_name(node: ast.expr) -> str | None:
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _denylist_violations(tree: ast.Module, permissions: frozenset[str]) -> list[str]:
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            roots = (
                [alias.name.split(".")[0] for alias in node.names]
                if isinstance(node, ast.Import)
                else [(node.module or "").split(".")[0]]
            )
            for root in roots:
                if root in NET_MODULES and "net" not in permissions:
                    violations.append(f"`import {root}` requires 'net'")
                elif root in SHELL_MODULES and "shell" not in permissions:
                    violations.append(f"`import {root}` requires 'shell'")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                if func.id in SHELL_BUILTIN_CALLS and "shell" not in permissions:
                    violations.append(f"`{func.id}(...)` requires 'shell'")
                elif func.id == "open" and not ({"fs_read", "fs_write"} & permissions):
                    violations.append("`open(...)` requires 'fs_read' or 'fs_write'")
            elif isinstance(func, ast.Attribute):
                root = _root_name(func.value)
                if root == "os" and func.attr in {"system", "popen"} and "shell" not in permissions:
                    violations.append(f"`os.{func.attr}(...)` requires 'shell'")
                elif root in SHELL_MODULES and "shell" not in permissions:
                    violations.append(f"`{root}.{func.attr}(...)` requires 'shell'")
                elif root in NET_MODULES and "net" not in permissions:
                    violations.append(f"`{root}.{func.attr}(...)` requires 'net'")
    return violations
