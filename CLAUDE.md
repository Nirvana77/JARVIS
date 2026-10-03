# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository and authorship

Owner: **Kevin Lundell** <nocktok123@gmail.com> — GitHub `Nirvana77`, repo
`Nirvana77/JARVIS`. Earlier commits also appear as `Navanda` (same address),
which is the alias in the LICENSE.

**Commit as that identity.** It is set repo-locally:

```bash
git config user.name "Kevin Lundell"
git config user.email "nocktok123@gmail.com"
```

so a machine whose *global* git identity belongs to somebody else still records
the right author here. Before committing, check `git log -1 --format='%an <%ae>'`
— if it is anyone else, fix the repo-local config rather than the commit. This
is written down because it has already gone wrong once: fifteen commits on
`develop` are authored by the machine's global identity instead of the owner's.

Pushing needs the `Nirvana77` credential (`gh auth switch -u Nirvana77`, or pin
it with `git remote set-url origin https://Nirvana77@github.com/Nirvana77/JARVIS.git`);
another account gets a 403 even with the right author set, since authorship and
push permission are unrelated.

**Where things are.** `PRD/jarvis-2026-rebuild.md` is the canonical spec;
`PRD/milestone-N-*.md` are per-milestone plans and `PRD/milestone-N-*-outcome.md`
their outcomes once shipped. `PRD/known-issues.md` lists what is known-broken
and not yet fixed — **read it before debugging something that looks new**, and
add to it rather than fixing in passing when a bug is out of the current
milestone's scope. The code is the `jarvis/` package (`python -m jarvis ...`);
the 2024 scripts (`libs/`, `actions/`, the TensorFlow model) are gone, and
`main.py` is a shim for `python -m jarvis`. Tests: `python -m pytest`
(`tests/`).

## One worktree and branch per conversation

Every agent or Claude Code conversation that changes files does so in **its
own git worktree, on its own branch** — never directly in the main checkout.
Two conversations editing one tree overwrite each other's uncommitted work and
leave a `git status` nobody can attribute; a worktree each keeps them apart,
and keeps the main checkout runnable while the work is in progress. Do this
first, before the first edit:

```bash
git worktree add ../JARVIS-<branch> -b <branch> develop   # short kebab-case: timer-names, ma-est
cd ../JARVIS-<branch>
```

Claude Code's own worktree support (`claude --worktree <name>`, the
`EnterWorktree` tool, `isolation: "worktree"` on a subagent) is the same thing
and fine to use, provided the branch starts from `develop`. A conversation that
only reads — a question, a review — needs neither.

What does not come along, and will bite:

- **Uncommitted work in the main checkout.** The branch starts from `develop`'s
  last commit. If the task builds on changes that are not committed yet, say so
  and ask, rather than copying files across by hand.
- **The gitignored per-machine files**: `.env`, `config.toml`, `data/`, `.venv`.
  `jarvis.config` resolves them from the checkout it runs in, so a bare
  worktree has no secrets, no config, and re-downloads every model. Link them
  from the main checkout:

  ```bash
  main=$(git worktree list --porcelain | head -1 | cut -d' ' -f2-)
  ln -s "$main"/{.env,config.toml,data,.venv} .
  echo /data >> "$(git rev-parse --git-common-dir)/info/exclude"   # once per clone
  ```

  The second line is there because `.gitignore`'s `data/` matches a directory
  and a symlink is not one — without it the link shows up as untracked and a
  `git add -A` commits it.

  `data/` is then *shared*: a dry-run that teaches or learns writes to the real
  install's corpus and skills. Copy it instead of linking when the work touches
  what is stored there.

What does come along is the repo-local git identity — worktrees share
`.git/config` — so the authorship rule above holds unchanged; still check it
before the first commit.

When the work is done, commit it on the branch and report the branch name.
Merging into `develop` is the owner's call (history uses
`Merge branch '<branch>': <summary>`); once merged, `git worktree remove
../JARVIS-<branch>`. Never remove a worktree that still has uncommitted changes
without asking.

## Implementing PRD work

Follow this order for any PRD/milestone item — don't skip or reorder steps:

1. **Write the tests first**, derived from the PRD's "Verification" section
   and the milestone plan's acceptance criteria — before writing the
   implementation itself. Match the conventions already in `tests/`
   (dependency-injected fakes per component; `tests/conftest.py`'s
   `config`/`embedding_model`/`embedder` fixtures).
2. **Implement** the code the tests describe.
3. **Run the full suite**: `python -m pytest`. If anything fails, **fix the
   code — never the test** — the one exception is a test that turns out to
   be factually wrong about what the PRD/plan actually specified, and even
   then say so explicitly rather than quietly loosening it.
4. Once the suite is green, **dry-run the program** — actually converse with
   it — via `python -m jarvis text` (or `--script FILE` for a saved
   conversation; see `jarvis/audio/text_io.py`). This is a separate check
   from the unit tests passing: it confirms the *real* wiring (NLU, registry,
   persona, the skill factory) behaves the way the PRD describes end-to-end,
   not just that the pieces work in isolation behind fakes.
5. **Monitor thread spawning** during both the test run (step 3) and the
   dry-run (step 4). JARVIS starts threads in several places:
   `jarvis/factory/jobs.py` (background learning), `jarvis/core/interrupt.py`,
   `jarvis/core/orchestrator.py` (including short-lived threads that end with
   the call they make: `compound` for a sentence with "and"/"then" in it, and
   `mishear-guess` / `reason` per unclear command when Ollama is up — M4,
   M4.5), `jarvis/audio/stt.py`,
   plus sounddevice's audio callback threads — and, on the M3 remote path, every
   `asyncio.to_thread` the orchestrator makes through `RemoteLink`
   (`jarvis/remote/server.py`) plus the services' `ThreadingHTTPServer` — and,
   since M5, the knowledge scan (`jarvis/knowledge/__init__.py`:
   `Knowledge.watch` runs each interval scan in an `asyncio.to_thread` worker;
   cancelling the task waits for it) — and, since M7, the phrasing retrain
   (`Orchestrator._retrain_phrasings`: the same `RetrainWorker` process as a
   learning job, plus one `asyncio.to_thread` for the regression gate) and
   the autonomous build and repair jobs, which are ordinary M2.5 learning jobs. A leaked or runaway thread doesn't fail a test on
   its own, so check for one explicitly:
   - **Tests**: compare `threading.enumerate()` before and after the code
     under test. A test that starts a thread must leave none behind once it
     finishes (join or stop it). Any count that keeps growing across tests is
     a leak.
   - **Dry-run**: watch the live process with
     `watch -n1 "ps -T -p $(pgrep -f 'python -m jarvis') | tail -n +2 | wc -l"`
     (or `top -H -p <pid>`). The thread count should go back to its baseline
     after each turn, each learning job, and each cancel/interrupt. If it
     climbs turn after turn, treat that as a bug.
   Report the thread counts you observed along with the test results.

## Setup

```bash
pip install -r requirements.txt       # Python 3.12+; the edge and the two services have their own
sudo dnf install portaudio-devel      # or portaudio19-dev on Debian — sounddevice needs it
cp config.example.toml config.toml    # per-machine, git-ignored; the example documents every key
python -m jarvis models pull          # Whisper + the Piper voice
python check_setup.py                 # ✓/✗/– report; exits non-zero only on a required row
```

**Secrets are in `.env` only**, never `config.toml`: `ANTHROPIC_API_KEY` (or the
older `api_key`; used by the skill factory and nothing else), optional
`ANTHROPIC_MODEL` (default `claude-opus-5`) and `ANTHROPIC_WORKSPACE_ID` (sent as
the `anthropic-workspace-id` header; needed only for identity-linked keys),
`HF_TOKEN` (authenticated model downloads), `JARVIS_EDGE_TOKENS` (brain) /
`JARVIS_EDGE_TOKEN` (edge), `JARVIS_GITHUB_TOKEN` (M8: a fine-grained token,
contents:write on this repo only, with which JARVIS pushes its own builtin
rewrites to `jarvis/self`; `scripts/set-github-token` puts it in `.env` and
the pod's `jarvis-env` Secret and restarts the pod, without the token ever
appearing on a command line). Env overrides of config: `JARVIS_PERSONA`,
`language`.

## Running

```bash
python -m jarvis                 # all-in-one voice loop (wake word + mic + speaker)
python -m jarvis text            # the same brain, typed in and printed out (--script FILE)
python -m jarvis --selftest      # load NLU + persona, list skills, exit 0
python -m jarvis nlu rebuild     # retrain the classifier now
python -m jarvis serve           # M3: the brain, waiting for an audio edge
python -m jarvis edge            # M3: the audio satellite (a mic, a speaker, a socket)
python -m jarvis knowledge scan  # M5: index the docs folder now, print what changed
python -m jarvis knowledge status  # M5: list indexed sources, counts, index backend
python -m jarvis pair <code> | devices | power | notify "text"   # the watch / paired edges
python -m jarvis learning [status|log --since 2d|phrasings|undo ID|enable SKILL]  # M7
```

**Time.** Both of the owner's machines run on UTC. `[general] timezone` (an IANA
zone) is what "now" means for the `clock` skill, the reasoner's prompt and
`ctx.now()`, which skills, including Claude-written ones, use instead of
`datetime.now()`.

`./jarvis-run <args>` is the same with the repo's `.venv`, from any directory.

**All-in-one is still the default.** `serve` / `edge` are the opt-in split
from Milestone 3 (`PRD/milestone-3-remote-edge.md`): the brain runs where the
GPU is, the edge runs in the room. The edge in daily use is the ESP32-S3 watch
(firmware repo `Nirvana77/esp32-s3-touch-amoled-2.06`, `PRD/milestone-3.5-esp32-edge.md`).

### The knowledge base (M5)

`PRD/milestone-5-knowledge-base.md` is the plan; the code is `jarvis/knowledge/`.

- **Where documents go**: `[knowledge] docs_dir` (default `~/jarvis/knowledge`,
  not created until something is written there). `.txt`, `.md` and `.pdf`
  (via `pypdf`) anywhere under it are indexed; dot-folders and other suffixes
  are ignored. Spoken facts ("remember that …") are indexed too, with no file
  behind them, and a folder scan never drops them.
- **The index**: chunks embedded with the NLU's own MiniLM model into
  `data/knowledge/kb.sqlite`. The float32 BLOBs in that file are the source of
  truth; `sqlite-vec` indexes them when the extension loads, otherwise search is
  brute-force cosine in numpy with the same ranking — either backend reads a
  database the other wrote. `knowledge status` and `--selftest` say which one
  you got.
- **When it is scanned**: once at startup, *before* JARVIS says it is ready (so
  a first-run embedding of a large folder is visible there, not hidden behind
  the first turns), then every `scan_interval_s` by `Knowledge.watch` — only
  files whose mtime+size changed are re-embedded, removed files are dropped.
- **Skills**: two new — `recall` ("what do my notes say about …", "where did i
  park") and `remember` ("remember that …"). Two changed — `note` also appends
  each note to `<docs_dir>/dictated-notes.md` and indexes it at once, and
  `search` ("what is …") asks the notes first and goes to Wikipedia only when
  nothing clears the bar.
- **Answers**: composed by the local Ollama reasoner from the top-k chunks when
  it is up; otherwise the best snippet verbatim — "From your notes, sir: … —
  from <source>." **Claude is never called for a knowledge answer**, by design
  (PRD § "Knowledge base (RAG)") — don't add it as a fallback.
- **Two score bars**: `min_score` (0.48) is what `recall` accepts; `search_min_score`
  (0.6) is the stricter bar for answering "what is …" from the notes, because
  there a note that merely mentions the subject must not beat Wikipedia.

**Tests must never index the developer's own `~/jarvis/knowledge` or write to
the repo's `data/`.** Point a real config at tmp directories with
`with_knowledge_paths(config, tmp_path)` from `tests/knowledge_harness.py`
(it also has `make_store` / `make_knowledge` and fake embedders/reasoners). A
test that builds JARVIS from the real config without it scans whatever is in
the developer's folder — slow, and nondeterministic.

### The remote edge (M3)

`python -m jarvis serve` **starts the two services itself** — separate
processes, adopted if already running, restarted if they die, stopped with the
brain (unless adopted). `[whisper] autostart` / `[voder] autostart` turn that
off; `[whisper] python` / `[voder] python` point at a separate venv, which is
what you want on a GPU box so only that venv carries the CUDA wheels.

To run them by hand instead, each in its own venv (they deliberately do **not**
import `jarvis`):

```bash
python services/whisper/serve.py --model small.en --device cuda --port 3461
# on a CPU brain use --model base.en: measured 415 ms vs small.en's 1050 ms on
# the same clips, with identical transcripts (the cost is fixed overhead, not
# decoding, so the bigger model is pure latency there)
python services/voder/serve.py   --voice-dir data/models/piper --port 3462
```

Both ship a systemd unit beside them. `check_setup.py` prints a ✓/✗ row per
service from its `/healthz`. The brain starts without them and says so
("Cannot hear you: the transcription service is not running"); `text` mode
keeps working regardless.

On the Pi, the whole install is:

```bash
pip install -r requirements-edge.txt   # numpy, sounddevice, websockets
sudo apt install libportaudio2         # + gpiozero for a push-to-talk button
```

`tests/test_edge_imports.py` enforces that: the edge's import graph must not
reach fastembed, faster-whisper, Piper, sklearn, anthropic or openwakeword. If
you add an import to `jarvis/remote/edge.py`, `jarvis/audio/segment.py`,
`jarvis/audio/player.py`, `jarvis/remote/protocol.py` or `jarvis/config.py`,
that test is the one that will tell you it can no longer run on a Pi.

Secrets live in `.env`: the edge reads `JARVIS_EDGE_TOKEN`, the brain reads
`JARVIS_EDGE_TOKENS="livingroom:s3cret,kitchen:other"`. `[server]` refuses to
listen on a routable address without TLS unless `allow_insecure = true`.

#### Over the internet, via Cloudflare Tunnel

The recommended shape, because it needs no certificate, no open port and no
port-forwarding — `cloudflared` runs **on the brain machine** and dials out:

```toml
[server]
host = "127.0.0.1"                        # the tunnel is the only way in
port = 8765
trusted_proxy_header = "CF-Connecting-IP" # see below
```

```yaml
# ~/.cloudflared/config.yml
tunnel: <tunnel-id>
credentials-file: /home/<user>/.cloudflared/<tunnel-id>.json
ingress:
  - hostname: jarvis.example.com
    service: http://127.0.0.1:8765        # cloudflared proxies the WS upgrade
  - service: http_status:404
```

```bash
cloudflared tunnel login
cloudflared tunnel create jarvis
cloudflared tunnel route dns jarvis jarvis.example.com
cloudflared tunnel run jarvis            # or: cloudflared service install
```

The edge then uses `server_url = "wss://jarvis.example.com"` — port 443, no
`:8765`, and `tls_ca` stays empty because Cloudflare's certificate is publicly
trusted. Cloudflare's WebSocket support is on by default; `[server]
ping_interval_s = 20` is what keeps an idle conversation from being dropped as
idle by the edge network.

**`trusted_proxy_header` is not optional here.** Through a tunnel every
connection reaches the brain from 127.0.0.1, so the per-address auth backoff
cannot tell your edge from anyone hammering your public hostname — without it,
a stranger's failed guesses lock out your own Pi. The header is believed only
when the connection came from loopback, i.e. from `cloudflared` on the same
machine; a header from any other address is ignored, because there it is the
client's claim about itself.

Binding `0.0.0.0` as well is a separate decision and a worse one: the tunnel
does not need it, and it puts unencrypted audio on your LAN. If you want it
anyway (a second edge on the LAN that skips the tunnel), that is what
`allow_insecure = true` is for, and the warning it logs is accurate.

##### `cloudflared` in Docker, and tunnel replicas

From a container, `cloudflared` reaches the brain over the Docker bridge, not
loopback. Two consequences:

- the brain must bind something the container can reach — `network_mode: host`
  (then everything above applies unchanged, loopback and all), or `0.0.0.0`
  with `allow_insecure = true` and a firewall;
- the header has to be trusted by address instead:

```toml
[server]
host = "0.0.0.0"
allow_insecure = true                       # the LAN hop is now in the clear
trusted_proxy_header = "CF-Connecting-IP"
trusted_proxy_peers = ["172.17.0.0/16"]     # the CONTAINER's subnet, not the gateway
```

Which address to trust is easy to get backwards, and getting it wrong fails
*silently* — the header is ignored and the backoff quietly lumps every client
together again. The container **dials** the bridge gateway (`172.17.0.1`, or
`host.docker.internal` with `extra_hosts: ["host.docker.internal:host-gateway"]`),
but the address the brain **sees** is the container's own (`172.17.0.2`, and it
changes when the container is recreated). So `trusted_proxy_peers` names the
container's subnet, not the gateway. The brain logs `refused <device> from
<address>` — that address is the one to put in the list.

Anything inside `trusted_proxy_peers` can claim to be any client, so name the
proxy's own address rather than its subnet where you can. A mistyped entry stops
the brain at startup rather than quietly disabling the distinction.

**Running the same tunnel on two machines does not give JARVIS failover.**
Cloudflare load-balances across replicas of a tunnel rather than ordering them
primary/secondary, so a replica elsewhere (a NAS, say) will receive connections
and must be able to reach the brain — over the LAN, in the clear, with the
brain's port open to it. And it cannot help anyway: the brain runs on exactly
one machine, so a second replica adds a path to a service that is down whenever
that machine is. **Serve the JARVIS hostname from a tunnel replica on the brain
machine only**, and let other hostnames use the other replicas.

If replicas on more than one machine must carry it, the ingress rule is stored
centrally and every replica gets it, so it has to name an address they can
*all* reach — the brain's LAN address, not `localhost`. The brain then binds
the LAN, and the hop from any replica that is not on the brain machine crosses
it unencrypted:

```toml
[server]
host = "0.0.0.0"
allow_insecure = true                       # the LAN hop is in the clear
trusted_proxy_header = "CF-Connecting-IP"
trusted_proxy_peers = [
    "192.168.0.20",      # a NAS replica, by its LAN address
    "172.17.0.0/16",     # the brain box's own cloudflared container
]
```

with `service: http://<brain-lan-ip>:8765`, and the port firewalled to just
those sources (`ufw allow from … to any port 8765 proto tcp`, then `ufw deny
8765`). To encrypt that hop as well, give the brain a self-signed certificate
and use `https://` with `originRequest: {noTLSVerify: true}`, or carry it over
WireGuard/Tailscale. The edge is unaffected either way — it only ever talks to
Cloudflare.

Worth adding on top: a Cloudflare Access policy or WAF rate-limit on the
hostname. The device tokens are the real authentication, but there is no reason
to let the whole internet reach the handshake.

**Privacy, stated plainly.** The edge has no wake word — the addressing modes
are the wake word — so *every* segment of speech it hears is transcribed on
your own brain machine. ByName decides what reaches a skill, not what is
transcribed. The ways to actually stop that are **PushToTalk** (the mic is
genuinely off until the button is held) and "turn off the mic"; Ignore still
listens and discards. Nothing is sent to a third party, and audio and tokens
are never logged.

## Architecture (`jarvis/`)

One asyncio loop — the **orchestrator** — owns the NLU and the skill registry
and drives a turn: hear → classify → dispatch a skill → speak. Everything that
touches hardware is injected in four roles (`wake`, `mic`, `stt`, `tts`), so the
same orchestrator runs with a real mic (`audio/`), typed text
(`audio/text_io.py`) or a remote edge (`remote/server.py`'s `RemoteLink`).

| package | what is there |
|---|---|
| `core/` | `orchestrator.py` (turns, standby, follow-up window, merge gate, safe points, the unclear-turn path), `context.py` (what a skill gets: `ctx.say/data_dir/llm/http/edges/knowledge/memory`), `persona.py`, `reasoner.py` (optional Ollama), `mishear.py` + `reasoning.py` (M4/M4.5 prompts and parsing), `memory.py` + `forgetting.py` (per-device memory), `interrupt.py`, `speech.py` |
| `audio/` | `wake.py` (openwakeword 0.4.0), `capture.py`, `stt.py` (faster-whisper), `tts.py` + `player.py` (Piper), `segment.py` (the edge's VAD), the whisper/voder service clients, `text_io.py` |
| `nlu/` | `corpus.py` (`intents.json` seed + every skill's `MANIFEST.examples`), `train.py` (MiniLM embeddings + `LogisticRegression`, versions in `data/models/nlu/v<N>/`, last 3 kept, corpus digest in `meta.json`), `classifier.py` (threshold + similarity floor → `unknown`), `slots.py`, `compound.py`, `retrain_worker.py` |
| `skills/` | `contract.py` (`SkillManifest`, `run(ctx, **params) -> str`), `registry.py` (builtin + learned + edge tools; `origin` is set from where a skill was found), `builtin/`, `learned/` (the package only: factory output lives in `data/skills/learned/`), `edge.py` (an edge's declared tools as skills) |
| `factory/` | the skill factory: `flows.py` (teach/edit/revert/remove dialogs), `jobs.py` (background build → validate → permission → sandbox), `claude_client.py` (the only Anthropic caller), `validate.py` (AST allowlist), `sandbox.py` |
| `remote/` | M3 brain/edge split: `protocol.py`, `server.py`, `edge.py`, `addressing.py`, `intake.py`, `pairing.py`, `firmware.py` (OTA), `powerlog.py`, `supervisor.py` (starts/adopts the two services) |
| `knowledge/` | M5 RAG: `store.py`, `ingest.py`, `answer.py`, `Knowledge.watch` |
| `learning/` | M7: `interactions.py` (the per-turn log), `phrasings.py` (words learned from confirmed turns, the corpus's third source), `state.py` (daily caps, disabled skills, events, confusions), `cli.py`. Locked atomic writes: `data/` is shared with the pod |
| `app.py`, `__main__.py`, `config.py` | assembly per mode, the CLI, `config.toml` + `.env` |

Rules worth knowing before changing things:

- **Claude teaches, never answers.** It is called only by the factory. Answers
  come from skills, the knowledge base and the optional local reasoner.
- **The reasoner never picks a skill.** It suggests phrases; each goes through
  the real classifier, and nothing it suggests runs unconfirmed.
- **Skills import nothing from the core stack**; they only use `ctx`.
- **Retraining never blocks a turn**: it runs in a worker, and the new model is
  swapped in only at an idle safe point (`_merge_gate`). `ensure_nlu` at
  startup retrains when a skill is unknown to the model or the corpus digest
  changed — learned phrasings are part of that digest.
- **Every corpus goes through `Orchestrator._corpus`** (or `app.training_corpus`
  at startup), which adds the learned phrasings. A retrain built from
  `build_corpus` alone would silently unlearn them.
- **M7 learns by itself, within bounds** (`[learning]`): phrasings only for
  skills, never for a guarded/meta action; builds capped per day; repairs
  capped per skill per day, then the skill is switched off; a permission
  beyond pure/notify still waits for a spoken yes unless `auto_permissions`.
  Since M8 a builtin that fails is rewritten too, as an **override** in
  `data/skills/overrides/<name>.py` that both brains load in place of the
  packaged one. Its gate adds the repo's own tests for that builtin, run in the
  sandbox with the rewrite standing in, and its recent good calls replayed. It
  is on probation for `probation_calls` uses (a failure or a correction puts
  the previous version back), "undo that" reverts it, and it is pushed to the
  `jarvis/self` branch, never to `develop`. The failure is still written to
  `data/learning/builtin-failures.jsonl`.
- **The sandbox gets no secrets.** `factory.sandbox.sandbox_env` is a short
  allowlist plus `JARVIS_SANDBOX=1`, under which `jarvis.config` does not read
  `.env`. Never pass the brain's environment to sandboxed code: on this machine
  `unshare` is unavailable, so there is no network isolation either.
- **The edge's import graph stays light** (`tests/test_edge_imports.py`).

### Adding a builtin skill

A module in `jarvis/skills/builtin/` with a `MANIFEST = SkillManifest(...)`
(name = module name, description, `examples` — these train the classifier —
`params`, `permissions`, `voice="jarvis"`) and `def run(ctx, **params) -> str`
returning the line to speak. **Write that line in the persona's voice** ("Noted,
sir.") and declare it with `voice`: a skill in the active persona's voice is
spoken exactly as written. Anything else is rewritten by the local LLM at run
time (0.4–0.9 s), and that rewrite is thrown away if it lost a number or a
content word (`jarvis/core/voice.py`). The factory writes learned skills in
voice the same way (`claude_client.VoiceGuide`). A canned reply to an
`intents.json` intent is voiced by a `reply_<tag>` line in the persona's
`responses.toml`. If its params need extracting from the utterance, add its rule to
`nlu/slots.py`. Add it to `tests/test_skills.py`'s roster. The model retrains on
the next start.

### Adding an intent handled by the orchestrator

Seed patterns go in `intents.json` (`tag`, `patterns`, `responses`, `action`) —
for session and meta actions (greeting, goodbye, teach, forget …) whose
handling lives in the orchestrator, not in a skill. The trainer never writes to
`intents.json`.

### Adding a watch tool

Tools are declared by the edge in `hello`, not written here: see
`docs/edge-tools.md` in the watch repo. The brain stores them in
`data/remote/tools/<device_id>.json` and learns them in the background.

### `data/` on this machine

`data/` is per-machine and git-ignored, and **on the owner's machine it is also
the production k8s pod's `/app/data`** (same directory). Anything that retrains
the NLU or writes under `data/` changes what the pod loads on its next restart.
In a worktree, copy `data/` rather than linking it when the work trains,
teaches or writes there.

Since M7 the pod writes there continuously: `data/interactions/` (every turn)
and `data/learning/` (learned phrasings, the learning state). **Both brains
learn**: HAProxy sends the watch to the dev brain whenever it runs, so that is
the brain being talked to. Each retrains only on the phrasings it learned
itself (`Orchestrator._own_pending`) and picks up the other's at its next start
through the corpus digest, so the two never retrain the same batch. Learned *skills* are in `data/skills/learned/` too
(known issue #19, fixed in M7): discovery points the `jarvis.skills.learned`
package there, and a skill still in the old place — the checkout's
`jarvis/skills/learned/`, or the pod's PVC mounted over it — is copied over at
startup, never overwriting the shared copy.
