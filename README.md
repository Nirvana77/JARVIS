# J.A.R.V.I.S.

Just A Rather Very Intelligent System (J.A.R.V.I.S.) from Iron Man — a voice
assistant with an intent-classification NLP pipeline and a Claude-backed
free-form Q&A fallback.

## Install

```bash
python -m venv .venv
source .venv/bin/activate

pip install anthropic wikipedia pyttsx3 nltk python-dotenv SpeechRecognition openpyxl pyaudio tensorflow
python -c "import nltk; [nltk.download(p) for p in ('punkt', 'punkt_tab', 'wordnet', 'omw-1.4')]"
```

### System packages

`pyaudio` builds against PortAudio, so install the dev headers first:

```bash
sudo dnf install portaudio-devel      # Fedora
# sudo apt install portaudio19-dev    # Debian/Ubuntu
```

### Python version note

`tensorflow` (used by the intent classifier in `libs/brain.py` /
`libs/training.py`) has **no wheels for Python 3.14 yet**. Use a **Python 3.12**
virtualenv if you need the intent model. The Claude Q&A path and everything
else run fine on 3.14.

## Configure

Create a `.env` file in the project root:

```
language="en"
ANTHROPIC_API_KEY="sk-ant-..."
# ANTHROPIC_WORKSPACE_ID="wrkspc_..."   # required only for identity-linked API keys
# ANTHROPIC_MODEL="claude-sonnet-5"     # optional; default is claude-opus-5

# Optional, for the rebuilt assistant (see below):
# HF_TOKEN="hf_..."                     # authenticated model downloads
# JARVIS_EDGE_TOKENS="livingroom:s3cret"  # on the brain: one per edge device
# JARVIS_EDGE_TOKEN="s3cret"              # on an edge: its own
```

Secrets live only here. Everything else is in `config.toml` — start from
`config.example.toml`, which documents every setting.

If your key is **identity-linked** (calls fail with
`anthropic-workspace-id is required`), either add `ANTHROPIC_WORKSPACE_ID`
(find it in the Anthropic Console workspace URL) or issue a standard
workspace-scoped API key.

## Verify

```bash
python check_setup.py
```

Prints a ✓/✗/– report for dependencies, NLTK data, the API key (masked), and
whether the configured Claude model resolves.

## Run

```bash
python main.py           # Run JARVIS
python libs/training.py  # Train the NLP model manually
python libs/brain.py     # Type sentences to see raw intent classification
```

Always run from the repo root — `intents.json`, `JARVIS_model.keras`,
`words.pkl`, and `classes.pkl` are resolved relative to the working directory.

---

# The 2026 rebuild (`jarvis/`)

Everything above is the original code, kept working and untouched. The rebuilt
assistant lives in `jarvis/` and runs as `python -m jarvis`: a local voice loop
with an openwakeword wake word, faster-whisper for speech, an embedding-based
NLU, Piper for speech out, and a Claude-backed skill factory that writes new
skills on request. `PRD/jarvis-2026-rebuild.md` is the specification;
`config.example.toml` is a commented copy of every setting.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

sudo dnf install portaudio-devel      # Fedora — sounddevice needs PortAudio
# sudo apt install portaudio19-dev    # Debian/Ubuntu

cp config.example.toml config.toml    # then edit
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

Python 3.14 is fine for all of it (unlike the TensorFlow-era code above).

```bash
python -m jarvis                 # the voice loop — say "hey jarvis"
python -m jarvis text            # the same brain, typed in and printed out
python -m jarvis --selftest      # load NLU + persona, list skills, exit 0
python -m jarvis mic             # live microphone meter, for tuning the VAD
python -m jarvis serve           # the brain, waiting for an audio edge
python -m jarvis edge            # the audio satellite
```

`./jarvis-run <args>` does the same thing using the repo's own virtualenv, from
any directory.

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

Without a tunnel, `[server]` refuses to listen on a routable address unless you
set `tls_cert`/`tls_key` or explicitly set `allow_insecure = true`.

## Tests

```bash
python -m pytest
```

Covers `jarvis/` only; the legacy code above has no test suite.
