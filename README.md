# J.A.R.V.I.S.

Just A Rather Very Intelligent System, after the one in Iron Man: a voice
assistant that runs on your own machines. Hearing, understanding, answering and
speaking are all local. A small embedding-based classifier routes what you say
to a skill, and Piper speaks the answer in the persona's voice.

Two things reach outside, and only when you ask for them. Claude **teaches**
JARVIS new skills: "learn how to flip a coin" has Claude write a Python skill,
which JARVIS validates, sandboxes, keeps after you confirm, and runs itself from
then on. Some skills fetch from the web (Wikipedia, YouTube). Claude is never
used to answer questions. Answers come from the skills, your own notes and,
optionally, a local Ollama model.

`PRD/jarvis-2026-rebuild.md` is the specification; `PRD/` also holds the plan and
outcome of each milestone and `PRD/known-issues.md`.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

sudo dnf install portaudio-devel      # Fedora — sounddevice needs PortAudio
# sudo apt install portaudio19-dev    # Debian/Ubuntu

cp config.example.toml config.toml    # then edit — config.toml is per-machine
python -m jarvis models pull          # fetch Whisper + the Piper voice
python check_setup.py                 # ✓/✗ report
```

There are four requirement sets, one per virtualenv, because the pieces are
meant to fail independently:

| file | for |
|---|---|
| `requirements.txt` | the all-in-one box, and the brain |
| `requirements-edge.txt` | the audio satellite — three packages, no models |
| `services/whisper/requirements.txt` | the warm transcription service |
| `services/voder/requirements.txt` | the warm speech service |

## Configure

Secrets live only in `.env`, in the project root:

```
ANTHROPIC_API_KEY="sk-ant-..."          # the skill factory; nothing else uses it
# ANTHROPIC_WORKSPACE_ID="wrkspc_..."   # only for identity-linked keys
# ANTHROPIC_MODEL="claude-opus-5"       # optional override
# HF_TOKEN="hf_..."                     # authenticated model downloads
# JARVIS_EDGE_TOKENS="livingroom:s3cret"  # on the brain: one per edge device
# JARVIS_EDGE_TOKEN="s3cret"              # on an edge: its own
```

Everything else is in `config.toml`. `config.example.toml` documents every
setting. `JARVIS_PERSONA` (env) picks the persona (`jarvis` or `plain`, a
folder under `personas/`).

## Run

```bash
python -m jarvis                 # the voice loop — say "hey jarvis"
python -m jarvis text            # the same brain, typed in and printed out
python -m jarvis --selftest      # load NLU + persona, list skills, exit 0
python -m jarvis mic             # live microphone meter, for tuning the VAD
python -m jarvis nlu rebuild     # retrain the classifier now
python -m jarvis serve           # the brain, waiting for an audio edge
python -m jarvis edge            # the audio satellite
python -m jarvis knowledge scan  # index the knowledge docs folder now
python -m jarvis knowledge status  # what is indexed, and which search backend
python -m jarvis pair <code>     # approve a device that is pairing
python -m jarvis devices         # who may connect
python -m jarvis power           # the watch's power log
python -m jarvis notify "text"   # send a notification to the edge
```

`./jarvis-run <args>` does the same using the repo's own virtualenv, from any
directory. `python main.py` still works; it is the same as `python -m jarvis`.

## Teaching it a skill

*"Learn how to flip a coin."* JARVIS asks a few questions (a name, what it
should do, an example) and then works on it **in the background**. You can
keep using it meanwhile. Claude writes the module and its tests. JARVIS checks
the code statically (no network, shell or file access unless you grant it by
voice), runs the tests in a sandbox, and retrains its classifier. At the next
pause it asks *"Shall I keep it?"*, and after a yes the skill is live. *"Edit
the coin flip skill"*, *"revert the coin flip skill"* and *"forget how to flip
a coin"* manage what it has learned. The last three versions of each skill and
of the classifier are kept.

The classifier retrains on its own when `intents.json` or a skill's examples
change.

## When it isn't sure what you said

A command is classified by a small local model and run by a skill; nothing
else is needed for that. What happens when the classifier gives up depends on
whether a local LLM is running (Ollama, `[reasoner]` in `config.toml` —
optional, and never Claude):

| you say | without Ollama | with Ollama |
|---|---|---|
| *"set a timer for five minutes and find my watch"* | both run, in order | the same — this needs no model |
| *"flip a corn"* (a mishearing) | "I didn't catch that" | *"Did you mean 'flip a coin', sir?"* — runs on a yes |
| *"wake me in five and beep the watch"* | "I didn't catch that" | *"Did you mean 'set a timer for 5 minutes, then find my watch'?"* — runs on a yes |
| *"how many days are in a leap year?"* | "I didn't catch that" | it answers, in the persona's voice |

Two rules hold throughout: the LLM never chooses a skill — it suggests
*phrases*, and each one goes through the ordinary classifier — and nothing it
suggests runs before you have said yes. Each step can be turned off on its own
(`[nlu] compound`, `[reasoner] correct_misheard`, `plan_commands`,
`answer_questions`).

### Memory, per device

Each device — an edge's id (`watch`, `livingroom`), or `local` without one —
has its own memory:

- **The conversation**: the last few turns, in RAM, so *"and a normal year?"*
  has the question before it. Gone after 15 quiet minutes or a restart.
- **What you asked it to keep**: *"remember that I parked on level two"*,
  *"note that the wifi code is 1234"*. On disk, in
  `data/memory/<device>.json`, as plain text — and in the knowledge base
  below, which is shared by every device.

Both are given to the local LLM when it answers, so *"where did I park?"*
works. *"What do you remember?"* reads the list back and *"forget everything
I told you"* clears it after a yes — neither needs the LLM. Forgetting clears
that memory and the facts this device gave the knowledge base; it does not
touch the notes files (`data/skills/note/notes.txt`, and the knowledge base's
`dictated-notes.md`), which keep every note ever given. One thing at a time:
*"forget about the park"*, *"remove the wifi note"* — JARVIS names what it
found and asks first, then takes it out of everything, notes files included.
*"What should I remember today?"* reads back what you asked it to remember
today. What the watch
was told, the living room does not know. `[memory]` in `config.toml` sets the
sizes.

## Knowledge base: your own notes

JARVIS answers from your own documents, locally. Put `.txt`, `.md` or `.pdf`
files in `[knowledge] docs_dir` (default `~/jarvis/knowledge`); they are
indexed at startup and re-scanned every `scan_interval_s`. Facts you say are
kept too:

- *"Remember that I parked on level three"* — stored as a remembered fact.
- *"Where did I park?"*, *"What do my notes say about the boiler?"* — answered
  from the notes and facts.
- *"Note that the wifi code is 1234"* — also lands in
  `<docs_dir>/dictated-notes.md`, so it is answerable on the next turn.
- *"What is …"* checks your notes first (with a stricter match,
  `search_min_score`), then Wikipedia.

With a local Ollama reasoner the answer is composed from the best excerpts;
without one, JARVIS reads the best excerpt and names its source (*"From your
notes, sir: … — from boiler manual."*). Claude is never asked. The index is
`data/knowledge/kb.sqlite`, searched with `sqlite-vec` when the extension loads
and by brute force otherwise. `[knowledge] enabled = false` turns it all off.

## Remote edge: brain here, ears there

The best speech models want a GPU, but the place an assistant needs to *hear
and speak* is a room — and a room is where a small, silent, cheap device
belongs. So JARVIS can be split in two:

- **the brain** (`python -m jarvis serve`) runs where the GPU is. It does
  speech-to-text, NLU, skills, persona and speech synthesis.
- **the edge** (`python -m jarvis edge`) runs on a Raspberry Pi-class box in
  the room. It owns the microphone and the speaker and **nothing else** — no
  wake word, no models. Its entire install is:

  ```bash
  pip install -r requirements-edge.txt   # numpy, sounddevice, websockets
  sudo apt install libportaudio2
  ```

The two talk over a versioned WebSocket protocol, and the link may cross the
internet. All-in-one `python -m jarvis` is still the default; this is opt-in.

`python -m jarvis edge` is one edge. The other is firmware: an ESP32-S3 watch
(Waveshare ESP32-S3-Touch-AMOLED-2.06, repo `Nirvana77/esp32-s3-touch-amoled-2.06`)
speaks the same protocol, with push-to-talk on its button. It pairs itself,
declares its own tools (timers, find-my-watch, notifications), takes firmware
updates over the link and uploads its power log. The sections below cover each.

### The brain's two services

**`python -m jarvis serve` starts them for you.** They are separate processes,
not threads — a wedged model has to be killable on its own — but you do not
have to launch them: the brain spawns both, waits for them, restarts one that
dies, and shuts them down when it exits. A service already answering (under
systemd, or left by a brain that was killed) is *adopted* instead of started
twice, and an adopted one is left running when the brain stops.

So on a fresh box the whole thing is:

```bash
./jarvis-run -v serve
```

```
· services
  · jarvis-whisper: started
  · jarvis-voder: adopted
· speech recogniser: small.en on cuda (http://127.0.0.1:3461)
JARVIS brain ready — ws://0.0.0.0:8765, 1 device token(s), default mode 'byname'.
```

They run on the brain's own interpreter by default, which `requirements.txt`
has already equipped. Set `[whisper] python` / `[voder] python` to a separate
venv when you want one — which is the point on an NVIDIA box, where only the
transcription service should carry the CUDA wheels. `autostart = false` in
either section leaves that service alone entirely.

To run them yourself instead — under systemd, or on another schedule — each
runs in its own virtualenv and deliberately does not import `jarvis`:

```bash
python3 -m venv ~/.local/share/jarvis/whisper-venv
~/.local/share/jarvis/whisper-venv/bin/pip install -r services/whisper/requirements.txt
~/.local/share/jarvis/whisper-venv/bin/python services/whisper/serve.py \
    --model small.en --device cuda --port 3461

python3 -m venv ~/.local/share/jarvis/voder-venv
~/.local/share/jarvis/voder-venv/bin/pip install -r services/voder/requirements.txt
~/.local/share/jarvis/voder-venv/bin/python services/voder/serve.py \
    --voice-dir data/models/piper --port 3462
```

On an NVIDIA box also `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` into
the whisper venv — CTranslate2 loads them by bare name and does not depend on
them itself.

Both ship a systemd unit next to them. On a CPU-only brain use
`--model base.en` — measured here at 415 ms against `small.en`'s 1050 ms on the
same clips, with identical transcripts. The brain starts without either service
and says so out loud ("Cannot hear you: the transcription service is not
running"); text mode keeps working regardless.

`python check_setup.py` prints a ✓/✗ row per service from its `/healthz`.

### Talking to it

There is no wake word on this path — **the addressing modes are the wake
word**. The brain transcribes every segment, tells the edge what it `heard`
*before* deciding anything, and then gates on the transcript:

| mode | what reaches JARVIS | the mic |
|---|---|---|
| **ByName** (default) | "Jarvis, …", plus anything inside a conversation window | open |
| **Always** | everything said | open |
| **PushToTalk** | only what is said while the button is held | **off** otherwise |
| **Ignore** | nothing — input paused | open, discarded |

The mode commands work **in every mode, including Ignore**, so there is no
state you can reach that you cannot speak your way out of: *"Jarvis, pause
input"*, *"continue input"*, *"change input to always / by name / push to
talk"*, *"turn off the mic"*.

A sentence with a pause in it is one command, not two: fragments are merged for
`[addressing] hold_ms` after you stop talking, and the countdown pauses while
the edge can still hear you.

### Privacy, stated plainly

With no wake word, **every segment of speech the edge hears is transcribed on
your own brain machine.** ByName decides what reaches a skill, not what is
transcribed. The ways to actually stop that are **PushToTalk** (the microphone
is genuinely off until the button is held) and *"turn off the mic"*; Ignore
still listens and discards. Nothing is sent to a third party, and audio and
tokens are never logged.

### Tokens

Each device has its own pre-shared token, compared in constant time. They live
in `.env`, never in `config.toml`:

```
JARVIS_EDGE_TOKENS="livingroom:s3cret,kitchen:other"   # the brain's devices
JARVIS_EDGE_TOKEN="s3cret"                             # this edge's own
```

Or let a device **pair** and get its own token (the ESP32 watch does, with
`JARVIS_TOKEN ""`). It sends `pair` with an X25519 key instead of `hello`. Both
sides derive a key and a 6-digit code (`jarvis/remote/pairing.py`), and the
device shows the code. Nothing happens until you confirm it on the brain's
machine:

```bash
python -m jarvis pair 482913      # approve the device showing that code
python -m jarvis devices          # who may connect (.env and paired)
python -m jarvis devices --remove watch   # revoke a paired device's token
```

Approving takes the admin secret in `data/remote/admin.token`, made on first
use and readable by this user only. A device token never approves anything:
behind a tunnel, a request from 127.0.0.1 proves nothing. The brain then sends
a fresh token sealed with AES-256-GCM under the shared key, and keeps it in
`data/remote/tokens.json` (0600) next to `.env`'s. The code matching on both
sides is what proves nobody swapped the keys in between. An unconfirmed
request expires after 2 minutes (close 4005) and counts toward the auth
backoff. At most 4 wait at once.

### Firmware updates (OTA) for edges that flash themselves

An edge that reports its firmware version in `hello` (`fw`; the ESP32 watch
does) can be updated over the link. Copy the image `idf.py build` produced to
`data/firmware/<device_id>.bin`, e.g. `data/firmware/watch.bin`. The version
is read from the image itself. On its next connect, the edge is told about an
image whose version differs from what it runs (`event` `ota`: version, size,
sha256). It then fetches the image with `GET /firmware` on the same port and
tunnel, using `Authorization: Bearer <its token>` and `X-Jarvis-Device: <id>`.
The auth backoff is the same as for `hello`. Whether and when to flash is the
edge's call.

To push it without waiting for a reconnect, say **"update the watch"**. JARVIS
answers ("Updating the watch to <version>…", "already running …", "no new
firmware", or "isn't connected"). The `ota` event goes out once that answer has
finished playing, so the download does not start while JARVIS is still
speaking.

### Edge tools: things the edge can do

An edge can list its own tools in `hello` (`tools`: name, description, example
phrases, typed params). The watch has `find_watch`, `set_timer`,
`cancel_timers` and `notify`. The brain keeps the list in
`data/remote/tools/<device_id>.json` and registers each tool as a skill
(origin `edge`, `jarvis/skills/edge.py`). The classifier learns a tool from its
examples. A new list, after new firmware, is learned in the background and
swapped in at the next safe point, as a learned skill is. A tool's params are
pulled out by type (`jarvis/nlu/slots.py` `extract_typed`): `duration` ("an
hour and a half" → 5400 s), `number`, `text` (what follows "to" / "that" …),
`name` ("the **pizza** timer", "a timer called **pizza**").

When one is said, the brain sends `call` {id, tool, args} and speaks the
`say` of the edge's `result`. "Remind me in 20 minutes to take the pizza
out" → `set_timer(seconds=1200, label="take the pizza out")` → "I'll remind
you in 20 minutes to take the pizza out." A tool cannot take the name of a
builtin or learned skill. While the edge is away its tools still exist and
answer "The watch isn't connected."

How to write a tool on the watch side (fields, limits, examples that route):
`docs/edge-tools.md` in the watch repo (Nirvana77/esp32-s3-touch-amoled-2.06).

**The watch's power log** comes up the same way: `python -m jarvis power`
(`--day YYYY-MM-DD`, `--no-fetch`) asks the watch for what the brain lacks of its
SD-card log (`send_power_log`; the files arrive as `file` messages in
`data/remote/power/<device_id>/`) and prints the day: time per mode, how much of
it asleep, battery drain in mV/h per mode, restarts and drops. "How was the watch
battery today?" says the short version.

**Notifications** come from outside a conversation:

```bash
python -m jarvis notify "The build is done"        # on the brain's machine
make && python -m jarvis notify "Build done"       # at the end of a long job
```

This is `GET /notify?text=...` on the brain's port, with the device's token
(the same auth and backoff as `/firmware`). The edge's `notify` tool shows it.
If the edge is away it is queued (the last 20) and delivered when it connects.

### Over the internet, with a Cloudflare Tunnel

The recommended shape: no certificate, no open port, no port-forwarding.
`cloudflared` runs **on the brain machine** and dials out; Cloudflare terminates
TLS at the edge with a real certificate for your hostname.

```toml
# config.toml, on the brain
[server]
host = "127.0.0.1"                        # the tunnel is the only way in
port = 8765
trusted_proxy_header = "CF-Connecting-IP"
```

```yaml
# ~/.cloudflared/config.yml  — see services/cloudflared/config.example.yml
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
cloudflared tunnel run jarvis             # or: cloudflared service install
```

The edge then uses `server_url = "wss://jarvis.example.com"` — port 443, no
`:8765`, and `tls_ca` stays empty because Cloudflare's certificate is publicly
trusted.

**`trusted_proxy_header` is not optional here.** Through a tunnel every
connection reaches the brain from `127.0.0.1`, so the per-address login backoff
cannot tell your edge from anyone hammering your public hostname — without it, a
stranger's failed guesses lock out your own device. The header is believed only
when the connection came from loopback, i.e. from `cloudflared` on the same
machine.

**Running `cloudflared` in Docker?** Then it reaches the brain over the Docker
bridge rather than loopback, so name that address explicitly — otherwise the
header is (correctly) ignored and you are back to one bucket for every client:

```toml
[server]
host = "0.0.0.0"
allow_insecure = true                      # the LAN hop is now in the clear
trusted_proxy_header = "CF-Connecting-IP"
trusted_proxy_peers = ["172.17.0.0/16"]    # the container's subnet — see below
```

Which address to trust is easy to get backwards, and getting it wrong fails
*silently* — the header is ignored and the backoff quietly lumps every client
together again. The container **dials** the bridge gateway (`172.17.0.1`, or
`host.docker.internal` with `extra_hosts: ["host.docker.internal:host-gateway"]`),
but the address the brain **sees** is the container's own (`172.17.0.2`, and it
changes when the container is recreated). So `trusted_proxy_peers` names the
container's subnet, not the gateway. The brain logs `refused <device> from
<address>` — that address is the one to put in the list.

`network_mode: host` on the container avoids all of that. Anything listed in
`trusted_proxy_peers` can claim to be any client, so name the proxy's own
address rather than its subnet; a mistyped entry stops the brain at startup
rather than quietly trusting nothing.

**Two tunnel replicas will not give JARVIS failover.** Cloudflare load-balances
across replicas rather than ordering them primary/secondary, so a replica on
another machine *will* receive connections and must be able to reach the brain
across your LAN. It cannot help anyway — the brain runs on exactly one machine,
so a second replica only adds a path to a service that is already down. Serve
the JARVIS hostname from a replica on the brain machine, and let other
hostnames use the others.

**If you keep replicas on more than one machine** — a NAS and the brain box,
say — the ingress rule is stored centrally and every replica gets it, so it has
to name an address they can *all* reach: the brain's LAN address, not
`localhost`. That means the brain binds the LAN and the hop from any replica
that is not on the brain machine crosses it unencrypted:

```toml
[server]
host = "0.0.0.0"
allow_insecure = true                       # the LAN hop is in the clear
trusted_proxy_header = "CF-Connecting-IP"
trusted_proxy_peers = [
    "192.168.0.20",      # the NAS replica, by its LAN address
    "172.17.0.0/16",     # the brain box's own cloudflared container
]
```

with `service: http://<brain-lan-ip>:8765`. Firewall the port to just those
sources — the token is the authentication, but there is no reason to offer the
handshake to the whole LAN:

```bash
sudo ufw allow from 192.168.0.20 to any port 8765 proto tcp
sudo ufw allow from 172.17.0.0/16 to any port 8765 proto tcp
sudo ufw deny 8765
```

To encrypt that LAN hop too, give the brain a self-signed certificate
(`tls_cert`/`tls_key`), use `service: https://<brain-lan-ip>:8765` and
`originRequest: {noTLSVerify: true}` — the edge is unaffected either way, since
it only ever talks to Cloudflare.


Without a tunnel, `[server]` refuses to listen on a routable address unless you
set `tls_cert`/`tls_key` or explicitly set `allow_insecure = true`.

## Tests

```bash
python -m pytest
```

No test touches a real network, microphone, GPU or speaker, and the suite must
pass whatever your own `config.toml`, `data/` and `~/jarvis/knowledge` hold — a
test whose result depends on them is a bug (`tests/remote_harness.py`'s
`make_config` and `tests/knowledge_harness.py` point a config at tmp dirs).
