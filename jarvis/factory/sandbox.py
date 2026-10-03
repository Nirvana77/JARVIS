"""`SubprocessSandbox` — runs a generated skill's tests and a dry-run `run()`
call in an isolated `python -I` subprocess.

This is defense-in-depth *behind* `validate.py`'s AST denylist, not the
primary defense — PRD: "When only the subprocess path is available, JARVIS
warns once that isolation is weaker." Network isolation (`unshare --user
--net`) is attempted but optional; the rlimits and `-I` (isolated mode: no
user site-packages, no implicit script-dir on `sys.path`) always apply.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

_UNSHARE_CMD = ["unshare", "--user", "--map-root-user", "--net"]
#: repo root, so the sandboxed `-I` subprocess (which ignores PYTHONPATH and
#: doesn't auto-add cwd) can still `import jarvis...` — generated skills import
#: `jarvis.skills.contract.SkillManifest`.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


#: what sandboxed code may see of the brain's environment: nothing secret
#: (``ANTHROPIC_API_KEY``, ``HF_TOKEN`` and the edge tokens are all in it)
_ENV_ALLOWED = ("PATH", "HOME", "LANG", "LC_ALL", "TZ", "TMPDIR")


def sandbox_env() -> dict[str, str]:
    """The environment a sandboxed run gets: a short allowlist, plus
    ``JARVIS_SANDBOX=1`` so ``jarvis.config`` does not read ``.env`` either."""
    env = {k: os.environ[k] for k in _ENV_ALLOWED if k in os.environ}
    env["JARVIS_SANDBOX"] = "1"
    return env


@dataclass(frozen=True)
class SandboxResult:
    ok: bool
    stdout: str
    stderr: str
    returncode: int


class SandboxError(RuntimeError):
    """The sandbox itself couldn't run (missing interpreter, timeout, ...)."""


def _rlimits(mem_mb: int, cpu_s: int):
    def _set() -> None:
        import resource

        mem_bytes = mem_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))

    return _set


class SubprocessSandbox:
    def __init__(
        self,
        timeout_s: float = 10.0,
        mem_mb: int = 256,
        cpu_s: int = 5,
        repo_timeout_s: float = 180.0,
        repo_mem_mb: int = 2048,
        repo_cpu_s: int = 180,
    ) -> None:
        self.timeout_s = timeout_s
        self.mem_mb = mem_mb
        self.cpu_s = cpu_s
        #: M8: the repo's own tests import the whole package and its fixtures
        self.repo_limits = (
            max(timeout_s, repo_timeout_s), max(mem_mb, repo_mem_mb), max(cpu_s, repo_cpu_s)
        )
        self.net_isolated = self._probe_unshare()

    @staticmethod
    def _probe_unshare() -> bool:
        if not shutil.which("unshare"):
            return False
        try:
            probe = subprocess.run(
                [*_UNSHARE_CMD, "true"], capture_output=True, timeout=5
            )
            return probe.returncode == 0
        except Exception:  # noqa: BLE001 — any failure means "not available"
            return False

    def _run(
        self, script: str, cwd: Path, permissions: frozenset[str], limits=None
    ) -> SandboxResult:
        timeout_s, mem_mb, cpu_s = limits or (self.timeout_s, self.mem_mb, self.cpu_s)
        runner = cwd / "_sandbox_runner.py"
        runner.write_text(script, encoding="utf-8")

        cmd = [sys.executable, "-I", str(runner)]
        if self.net_isolated and "net" not in permissions:
            cmd = [*_UNSHARE_CMD, *cmd]
        elif "net" not in permissions:
            log.warning(
                "unshare unavailable — sandboxed skill runs without kernel-level "
                "network isolation (validate.py's AST ban is still enforced)"
            )

        try:
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                    preexec_fn=_rlimits(mem_mb, cpu_s),
                    env=sandbox_env(),
                )
            except subprocess.TimeoutExpired as exc:
                return SandboxResult(
                    ok=False,
                    stdout=exc.stdout or "",
                    stderr=(exc.stderr or "") + f"\n[sandbox] timed out after {timeout_s}s",
                    returncode=-1,
                )
            return SandboxResult(
                ok=proc.returncode == 0, stdout=proc.stdout, stderr=proc.stderr, returncode=proc.returncode
            )
        finally:
            runner.unlink(missing_ok=True)
            pycache = cwd / "__pycache__"
            if pycache.is_dir():
                shutil.rmtree(pycache, ignore_errors=True)
            pytest_cache = cwd / ".pytest_cache"
            if pytest_cache.is_dir():
                shutil.rmtree(pytest_cache, ignore_errors=True)

    def run_tests(
        self,
        module_path: Path,
        test_path: Path,
        permissions: frozenset[str],
    ) -> SandboxResult:
        """Run the generated pytest module against the generated skill module,
        both importable from `module_path.parent`."""
        script = textwrap.dedent(
            f"""\
            import sys
            sys.path.insert(0, {str(_REPO_ROOT)!r})
            sys.path.insert(0, {str(module_path.parent)!r})
            import pytest
            raise SystemExit(pytest.main(["-q", {str(test_path)!r}]))
            """
        )
        return self._run(script, module_path.parent, permissions)

    def run_repo_tests(
        self,
        module_path: Path,
        name: str,
        test_files: list[str],
        permissions: frozenset[str],
    ) -> SandboxResult:
        """M8: the repo's own tests for builtin ``name``, with the rewrite at
        ``module_path`` standing in for ``jarvis.skills.builtin.<name>``. Only
        tests whose name mentions the skill are run (``-k``); none selected
        counts as a pass. Same isolation as every sandboxed run — the
        scrubbed environment means the repo's fixtures load no secrets."""
        script = textwrap.dedent(
            f"""\
            import importlib.util
            import sys
            sys.path.insert(0, {str(_REPO_ROOT)!r})
            import jarvis.skills.builtin as _pkg
            _full = "jarvis.skills.builtin." + {name!r}
            _spec = importlib.util.spec_from_file_location(_full, {str(module_path)!r})
            _mod = importlib.util.module_from_spec(_spec)
            sys.modules[_full] = _mod
            _spec.loader.exec_module(_mod)
            setattr(_pkg, {name!r}, _mod)
            import pytest
            code = pytest.main(["-q", "-p", "no:cacheprovider", "--rootdir", {str(_REPO_ROOT)!r},
                                "-k", {name!r}, *{list(test_files)!r}])
            raise SystemExit(0 if code in (0, 5) else int(code))
            """
        )
        return self._run(script, module_path.parent, permissions, self.repo_limits)

    def dry_run(
        self,
        module_path: Path,
        params: dict,
        permissions: frozenset[str],
    ) -> SandboxResult:
        """Import the module and call `run()` once with sample params against a
        minimal stand-in `Context`, asserting it returns a `str`."""
        script = textwrap.dedent(
            f"""\
            import importlib.util
            import sys
            sys.path.insert(0, {str(_REPO_ROOT)!r})

            class _FakeContext:
                def __init__(self):
                    self.llm = None
                    self.config = None
                    self.edges = None
                    self.memory = None
                    self.knowledge = None
                def now(self):
                    import datetime
                    return datetime.datetime.now().astimezone()
                def say(self, text):
                    print("[say]", text)
                @property
                def data_dir(self):
                    import pathlib, tempfile
                    d = pathlib.Path(tempfile.mkdtemp())
                    return d

            spec = importlib.util.spec_from_file_location({module_path.stem!r}, {str(module_path)!r})
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            result = mod.run(_FakeContext(), **{params!r})
            assert isinstance(result, str), f"run() returned {{type(result).__name__}}, expected str"
            print("[dry-run ok]", result)
            """
        )
        return self._run(script, module_path.parent, permissions)
