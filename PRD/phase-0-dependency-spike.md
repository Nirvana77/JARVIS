# Phase 0 — dependency spike: outcome

**Date:** 2026-09-03
**Ran on:** Fedora (Linux 7.1.12, x86-64), existing repo `.venv`
**Result:** ✅ **Python 3.14 is viable for the entire 2026 rebuild stack. No 3.12 fallback is needed.**

This closes the Phase 0 item in `jarvis-2026-rebuild.md` ("Confirm CPU wheels on the
target Python … pick the interpreter. Extend `check_setup.py` with these rows.").

---

## Interpreter decision

**Use Python 3.14** (`.venv`, currently 3.14.7). Every dependency in the target
architecture resolves to a CPU binary wheel on cp314 and imports cleanly. The
PRD's contingency ("fall back to a 3.12 `.venv` if any are missing") is **not
exercised** — drop it from the plan unless a later milestone hits a wall.

The old TensorFlow blocker is gone by construction: the rebuild drops TF, and
nothing in the new stack needs it.

## Wheel / import matrix (Python 3.14.7, CPU)

| PRD dependency | Installed version | Wheel tag | Import | Notes |
|---|---|---|---|---|
| `faster-whisper` | 1.2.1 | `py3-none-any` | ✓ | pulls `ctranslate2`, `av`, `onnxruntime`, `tokenizers`, `huggingface-hub` |
| `ctranslate2` | 4.8.2 | `cp314` manylinux_2_28 | ✓ | faster-whisper's inference backend |
| `openwakeword` | **0.4.0** | `py3-none-any` | ✓ | **see caveat below** — 0.5+/0.6 need `tflite-runtime` (no PyPI wheel); 0.4.0 uses the ONNX path |
| `piper-tts` | 1.7.0 | `cp39-abi3` manylinux_2_28 | ✓ | **`piper-phonemize` is obsolete** — 1.7 bundles phonemization; it is not a separate dependency anymore |
| `piper-phonemize` | — | — | — | **not on PyPI at all**; no longer required (see above). Remove from the PRD package list. |
| `onnxruntime` | 1.29.0 | `cp314` manylinux_2_28 | ✓ | shared backend for fastembed, openwakeword 0.4, piper |
| `fastembed` | 0.8.0 | `py3-none-any` | ✓ | model files download on first `TextEmbedding(...)` use, not at import |
| `sqlite-vec` | 0.1.9 | `py3-none-manylinux2014` | ✓ | loadable extension verified: `vec_version()` → `v0.1.9` in an in-memory DB |
| `sounddevice` | 0.5.6 | `py3-none-any` | ✓ | PortAudio already present system-wide (`libportaudio.so.2`); `query_devices()` → 20 devices |
| `scikit-learn` | 1.9.0 | `cp314` manylinux_2_28 | ✓ | pulls `scipy` 1.18.1 (`cp314`), `joblib`, `threadpoolctl` |
| `pypdf` | 6.16.2 | `py3-none-any` | ✓ | pure Python |

Notable transitive wheels that also resolved on cp314: `scipy==1.18.1`,
`av==18.1.0` (`cp311-abi3`), `tokenizers==0.23.2`, `numpy==2.5.2` (already
present), `protobuf==7.36.1`, `mmh3==5.3.0`, `py-rust-stemmers==0.1.8`,
`cffi==2.1.1`, `pillow==12.3.0`.

## Caveats to carry into M1

1. **Pin `openwakeword==0.4.0`.** Versions ≥ 0.5 declare
   `tflite-runtime<3,>=2.8.0` on Linux, and `tflite-runtime` has **no
   distribution on PyPI** for any interpreter, so pip silently back-solves to
   0.4.0. 0.4.0 supports `inference_framework="onnx"`, which is the path we
   want (onnxruntime is already in the stack). If a newer openwakeword feature
   is needed later, the options are: vendor `tflite-runtime` from Google's
   wheel index, or switch wake-word engine. Verify 0.4.0's bundled "hey
   jarvis" / custom "Jarvis" ONNX model actually loads and fires during M1
   `audio/wake.py` work.
2. **`piper-phonemize` removed.** The PRD package layout and the setup docs
   still mention it; `piper-tts` 1.7 is self-contained. Update
   `jarvis-2026-rebuild.md` (done) and the eventual `pyproject.toml`.
3. **`fastembed` / `faster-whisper` first-run downloads.** Both fetch ONNX /
   CT2 model files from Hugging Face on first use. That is a one-time setup
   cost, not a hot-path network dependency — but M1 setup instructions should
   pre-warm them (or ship a `jarvis models pull` command) so the first live
   run isn't a surprise.
4. **`anthropic==1.3.0` in the venv is very old.** Unrelated to Phase 0, but
   the skill factory (M2) will want a current SDK. Bump it when M2 starts.

## What changed in the repo

- `check_setup.py` — new **"2026 rebuild stack (PRD Phase 0)"** section:
  ten import rows plus functional probes for PortAudio and the `sqlite-vec`
  loadable extension. These rows count toward the pass/fail exit code, since
  every one has a 3.14 wheel — a failure here is a real setup gap.
- `PRD/jarvis-2026-rebuild.md` — Phase 0 marked done; `piper-phonemize`
  dropped; `openwakeword` pinned to 0.4.0 in the degradation table.
- The ten packages above (and their transitive deps) are installed in `.venv`.

## Reproduce

```bash
.venv/bin/python -m pip install \
  faster-whisper 'openwakeword==0.4.0' piper-tts fastembed \
  sqlite-vec sounddevice scikit-learn pypdf
.venv/bin/python check_setup.py     # "2026 rebuild stack" section all ✓
```

The only failing row in `check_setup.py` after this is the pre-existing
`ANTHROPIC_WORKSPACE_ID` blocker, which is unrelated and the user's to resolve.
