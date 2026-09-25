"""The requirements files have to match what the code actually imports.

"I added an import and forgot the requirements file" is invisible on the
machine that already has the package, and breaks the next fresh clone — which
is exactly the machine nobody is sitting at. So the import graph is checked
against the files, by parsing both.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "requirements.txt"
EDGE = ROOT / "requirements-edge.txt"

#: import name -> distribution name, where they differ
_DISTRIBUTION = {
    "sklearn": "scikit-learn",
    "dotenv": "python-dotenv",
    "faster_whisper": "faster-whisper",
    "piper": "piper-tts",
    "huggingface_hub": "huggingface-hub",
    "sqlite_vec": "sqlite-vec",
    "yaml": "pyyaml",
}

#: imported but deliberately not required: optional extras the code guards with
#: a try/except and degrades without.
_OPTIONAL = {"gpiozero"}


def _requirements(path: Path) -> set[str]:
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        names.add(re.split(r"[<>=!~\[]", line, maxsplit=1)[0].strip().lower())
    return names


def _imports(*paths: Path) -> set[str]:
    """Top-level third-party modules imported under `paths` — each a directory
    to walk or a single file."""
    found: set[str] = set()
    for base in paths:
        files = base.rglob("*.py") if base.is_dir() else [base]
        for file in files:
            # generated skills may import anything; they are not the repo's deps
            if "learned" in file.parts or "staging" in file.parts:
                continue
            tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    found.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    found.add(node.module.split(".")[0])
    return {
        name
        for name in found
        if name not in sys.stdlib_module_names and name not in ("jarvis", "tests")
    }


def _distribution(module: str) -> str:
    return _DISTRIBUTION.get(module, module).lower()


def test_the_requirements_files_exist_and_parse():
    for path in (MAIN, EDGE, ROOT / "services/whisper/requirements.txt",
                 ROOT / "services/voder/requirements.txt"):
        assert path.is_file(), f"{path} is missing"
        assert _requirements(path), f"{path} lists nothing"


def test_everything_the_package_imports_is_required():
    required = _requirements(MAIN)
    missing = sorted(
        module
        for module in _imports(ROOT / "jarvis")
        if module not in _OPTIONAL and _distribution(module) not in required
    )
    assert not missing, f"imported by jarvis/ but not in requirements.txt: {missing}"


def test_everything_the_tests_import_is_required():
    required = _requirements(MAIN)
    missing = sorted(
        module
        for module in _imports(ROOT / "tests")
        if module not in _OPTIONAL and _distribution(module) not in required
    )
    assert not missing, f"imported by tests/ but not in requirements.txt: {missing}"


def test_the_edge_requirements_stay_tiny():
    """The edge's whole promise is that it installs on a Raspberry Pi. Three
    packages, and none of them a model runtime."""
    required = _requirements(EDGE)
    assert required == {"numpy", "sounddevice", "websockets"}, required
    for heavy in ("faster-whisper", "piper-tts", "fastembed", "openwakeword",
                  "scikit-learn", "anthropic", "onnxruntime", "torch"):
        assert heavy not in required


#: the modules an edge box actually loads (`tests/test_edge_imports.py` proves
#: this is the whole graph, in a subprocess)
_EDGE_FILES = (
    "jarvis/remote/edge.py",
    "jarvis/remote/protocol.py",
    "jarvis/audio/segment.py",
    "jarvis/audio/player.py",
    "jarvis/audio/pipewire.py",
    "jarvis/config.py",
)


def test_the_edge_requirements_cover_what_the_edge_imports():
    """The other direction, and the one that would actually strand a Pi."""
    required = _requirements(EDGE)
    modules = _imports(*(ROOT / name for name in _EDGE_FILES))
    missing = sorted(
        module
        for module in modules
        if module not in _OPTIONAL and _distribution(module) not in required
    )
    # `dotenv` is imported by config.py behind a try/except — it reads secrets,
    # and an edge has only its own token in the environment — so it is allowed
    # to be absent. Nothing else may be.
    assert missing in ([], ["dotenv"]), missing


def test_the_services_require_only_their_own_model():
    """Each service runs in its own venv and imports nothing from `jarvis`."""
    whisper = _requirements(ROOT / "services/whisper/requirements.txt")
    voder = _requirements(ROOT / "services/voder/requirements.txt")
    assert "faster-whisper" in whisper and "piper-tts" not in whisper
    assert "piper-tts" in voder and "faster-whisper" not in voder
    for service in (ROOT / "services/whisper/serve.py", ROOT / "services/voder/serve.py"):
        assert "import jarvis" not in service.read_text(encoding="utf-8")


@pytest.mark.parametrize("module", sorted(_DISTRIBUTION))
def test_the_name_map_is_not_stale(module):
    """Every rename in the map should still be a module something imports —
    otherwise the map is quietly hiding a missing requirement."""
    if module in ("yaml", "sqlite_vec"):
        pytest.skip("listed for the phase-0 stack, not imported by jarvis/ yet")
    assert module in _imports(ROOT / "jarvis") | _imports(ROOT / "tests")
