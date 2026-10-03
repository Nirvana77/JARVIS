# Milestone 3.5 — ESP32-S3 edge (a wearable push-to-talk satellite): plan

**Status:** Shipped, with differences — see `PRD/milestone-3.5-esp32-edge-outcome.md`. Kept as
written on 2026-09-28; the firmware ended up in its own repo, not `firmware/` here.
**Owner:** Kevin Lundell
**Branch:** `esp32-edge` off `develop`, to be merged back. Same workflow as
`milestone-3-remote-edge`.
**Hardware:** Waveshare ESP32-S3-Touch-AMOLED-2.06. The board kit (pin map,
schematic, datasheets, vendor demos, BSP source, an ESP-IDF + LVGL starter) is
at `~/git/esp32-s3-touch-amoled-2.06/`. Its `README.md` is the reference for
every pin and chip named below.

M3 split JARVIS into a **brain** (`python -m jarvis serve`) and an **edge**
(`python -m jarvis edge`), with a versioned JSON protocol
(`jarvis/remote/protocol.py`) between them. The edge was a Python program
written for a Raspberry Pi. This milestone adds a second edge implementation:
**firmware for the Waveshare board**. It is a watch-sized device with a
speaker, two microphones, a 410×502 AMOLED touchscreen, Wi-Fi and a battery.
Held up to the mouth, it becomes JARVIS's microphone and speaker.

The M3 protocol was designed to make this a client-only change. **The brain is
not modified.** If the brain turns out to need a change, that change is a
finding and is recorded in the outcome doc.

It is numbered 3.5 rather than 4 because M4 (misheard-command reasoning) is
already defined in `jarvis-2026-rebuild.md`. It is an addition to M3, the same
way 2.5 was an addition to M2.

---

## Scope

**In:** Push-to-talk on the **BOOT button (GPIO 0)**. Spoken answers on the
board's speaker. The status and the transcript on the AMOLED. Wi-Fi, and `wss://`
through the existing Cloudflare Tunnel. Reconnect, the offline earcon and
barge-in by button, all matching the Python edge's behaviour.

**Out, and deferred to phase B (below):** always-listening modes (ByName and
Always) on the board. These need the segmenter ported to C, and on a battery
device they also need a power-budget decision. **Out entirely:** echo
cancellation (M6), multi-edge on one brain, OTA updates and a touch-driven
settings UI.

---

## Key design decisions

### 1. The push-to-talk button is GPIO 0 (BOOT)

Almost every GPIO on this board is already in use (README, "GPIO map"). The two
physical buttons are BOOT (GPIO 0) and PWR. PWR goes to the AXP2101 and is only
readable through SYS_OUT on GPIO 10. **BOOT is the PTT button.**

- **Electrical:** active LOW, with the board's pull-up. The firmware also
  enables the internal pull-up, so an unpopulated or damaged external resistor
  still gives a defined level.
- **Debounce:** 20 ms, using the `espressif/button` component
  (`BUTTON_TYPE_GPIO`, `active_level = 0`). It sends events, not polls, so the
  press arrives as `BUTTON_PRESS_DOWN` and the release as `BUTTON_PRESS_UP`.
  This matches `gpiozero.Button`'s `when_pressed` / `when_released` in
  `jarvis/remote/edge.py`: the press takes effect immediately, and nothing waits
  for a hold time.
- **Strapping pin:** GPIO 0 selects download mode **at reset**. Holding the
  button while the board powers on or resets enters the ROM bootloader instead
  of JARVIS. This is by design, and it is also the recovery path (README
  gotcha 5). It goes in the firmware README so it is not reported as a bug. At
  runtime GPIO 0 is an ordinary input.
- **Kconfig:** `CONFIG_JARVIS_PTT_GPIO` (default `0`) and
  `CONFIG_JARVIS_PTT_ACTIVE_LEVEL` (default `0`).
- **Not the Python `ptt_gpio`.** In `config.toml`, `[edge] ptt_gpio = 0` means
  *no button*, because on a Pi GPIO 0 is an I2C ID-EEPROM pin that nobody wires
  a button to. The firmware has its own Kconfig and never reads that file, so
  the two meanings of 0 never meet. The firmware README says so, so that nobody
  "aligns" them.

**Button semantics** are exactly `Edge.on_press` / `Edge.on_release`:

| event | action |
|---|---|
| press, while an answer is playing | stop the I2S output at once, drop the queued `speech` parts, send `interrupt` (no round trip to the brain first) |
| press | open the microphone (start I2S RX and power up the ES7210), send `speaking{on:true}`, start filling a segment buffer, give a 15 ms haptic tick on the motor (GPIO 18) |
| press, while offline | all of the above, plus the offline earcon |
| release | close the microphone, send `speaking{on:false}`, and if at least 250 ms were captured send the segment as `audio{final:true, reason:"release"}` |
| held past 15 s | send what was captured with `reason:"maximum"`, keep the mic closed until the next press, and give a double haptic tick |

The mic is **physically off** whenever the button is up. I2S RX is stopped and
the ES7210 is put in standby, so the "demonstrably off" promise of M3
criterion 5 holds on this device in every mode, not only in PushToTalk.

### 2. Phase A has no segmenter, so the button is the only way in

Without a VAD, the board cannot tell speech from silence, so in phase A it
never listens unless the button is held. **This works in every addressing
mode, but is only natural in PushToTalk:** in ByName the brain would still want
the name. The device's mode is stored per `device_id` on the brain
(`data/remote/<device_id>.json`), so setting it once is enough. Hold the button
and say *"Jarvis, push to talk"*. Mode commands are matched before the gate
(`jarvis/remote/addressing.py`), so this works from any mode. The brain's
`default_mode` is not changed.

Silence at the start and end of the held segment is not trimmed in phase A.
Whisper's VAD filter already runs in `services/whisper/serve.py`, so the cost
is a few hundred milliseconds of base64, which is acceptable.

### 3. Audio

- **Format:** 16 kHz, s16le, mono, the only format the wire accepts. The ES7210
  and ES8311 both run at 16 kHz natively (`esp_codec_dev_sample_info_t`), so
  **no resampling happens on the device**.
- **Mic:** `bsp_audio_codec_microphone_init()`. The ES7210 delivers 2 mic
  channels. Phase A uses **mic 1 only** and discards the rest. Beamforming and
  AEC are left to M6. The channel is documented, because the wrong one sounds
  "quiet", not "broken".
- **Speaker:** `bsp_audio_codec_speaker_init()`, with PA_CTRL (GPIO 46)
  switched on only while something is playing. That saves the amplifier's idle
  current and removes its hiss between answers.
- **Speech rate:** on connect the board sends
  `control{setSpeech, on:true, sample_rate:16000}`. The voder synthesises at
  the brain's `[voder] sample_rate` (16000 by default). A `speech` part at any
  other rate is resampled on the board, the same way the Python `Player`
  handles a mismatch, and a warning is logged once.
- **Playback** follows `_play_loop`: parts in order, a 150 ms gap between
  them, and `control{playbackDone, id}` after the `final` part. Half-duplex is
  free here, because the mic is off whenever the button is up.
- **Offline earcon:** 660 Hz then 440 Hz, 90 ms each, 0.2 amplitude, so it is
  the same sound as `_earcon()`.

### 4. Wire protocol: the same one, protocol version 1

No new message types, and no change to `jarvis/remote/protocol.py`.

- **Transport:** `espressif/esp_websocket_client` over `wss://`. The server is
  verified with the ESP-IDF certificate bundle (`esp_crt_bundle_attach`), which
  covers Cloudflare's publicly trusted certificate. `CONFIG_JARVIS_TLS_CA`
  optionally embeds a PEM for a self-signed brain, the equivalent of
  `[edge] tls_ca`. Plain `ws://` is allowed for a LAN brain; the brain already
  refuses routable plain-text unless it is set to `allow_insecure`.
- **Outgoing audio** is one JSON text frame per segment, as specified. A full
  15 s segment is 480 KB of PCM and 640 KB of base64. The segment buffer and
  the frame are built in **PSRAM** (8 MB). The frame is sent **fragmented**:
  first the JSON prefix, then base64 in 4 KB continuation frames, then the
  suffix, using `esp_websocket_client_send_text_partial` /
  `_send_cont_msg` / `_send_fin`. The whole frame never exists in RAM at once,
  and the brain's `websockets` server reassembles it into one message.
- **Incoming `speech`** arrives as `WEBSOCKET_EVENT_DATA` chunks
  (`payload_offset` / `payload_len`). They are reassembled into a PSRAM buffer
  sized from `payload_len`, which is refused above the brain's own `max_size`
  (4 MB), then parsed.
- **JSON:** cJSON is vendored into the portable core (decision 6) so the same
  code builds on the host. The firmware does **not** also link ESP-IDF's
  `json` component, because that would define the same symbols twice. cJSON's
  hooks point at a PSRAM-preferring allocator.
- **Keepalive:** the brain pings every 20 s (`[server] ping_interval_s`).
  `esp_websocket_client` answers pings on its own. The client's own
  `ping_interval_sec` is also set to 20, so an idle watch on the far side of
  Cloudflare is not dropped.
- **Close codes** are shown on the screen:
  - 4001 *refused*: check the token. The same code with the reason "another
    device is connected" is shown as *busy: another edge is connected*, because
    the brain takes one edge at a time.
  - 4002 *firmware too old for this brain*.
  - 4004 *brain stopped*.
  - Anything else reconnects with backoff from 1 s up to
    `CONFIG_JARVIS_RECONNECT_MAX_S` (30 s). After a 4001 or 4002 the backoff
    starts at the maximum, so a wrong token does not hammer the login backoff.

### 5. Configuration and secrets

This follows the repo's convention that per-machine config stays untracked
(commit `bc0a21c`, "config.toml is per-machine").

- **Tracked:** `firmware/esp32-edge/sdkconfig.defaults` (target, PSRAM, flash
  size, the partition table, the certificate bundle) and
  `main/Kconfig.projbuild`. The Kconfig menu is *JARVIS edge*, with the Wi-Fi
  SSID, server URL, device id, PTT GPIO, active level and reconnect ceiling.
  **No secrets in either file.**
- **Untracked:** `sdkconfig` (generated) and `sdkconfig.defaults.local`, which
  is listed in `SDKCONFIG_DEFAULTS` after the tracked file. The Wi-Fi password
  and `CONFIG_JARVIS_EDGE_TOKEN` go there. Both files are added to
  `.gitignore`.
- **Brain side:** add the device to `JARVIS_EDGE_TOKENS` in the brain's `.env`,
  for example `JARVIS_EDGE_TOKENS="livingroom:…,wrist:…"`. The default
  `device_id` is `wrist`.

### 6. Code layout: a portable core and a thin ESP-IDF shell

The firmware is written so that everything that *decides* something can be
compiled and tested on the dev box. Only the code that *touches hardware*
needs the board.

```
firmware/esp32-edge/
├── CMakeLists.txt, sdkconfig.defaults, partitions.csv, README.md
├── main/
│   ├── main.c                 # boot: NVS, PMU, Wi-Fi, BSP, tasks
│   ├── Kconfig.projbuild
│   ├── idf_component.yml      # BSP ^2.0.0, esp_websocket_client, button, lvgl
│   ├── net.c                  # esp_websocket_client glue, fragmenting, reassembly
│   ├── audio.c                # I2S/codec open/close, PA_CTRL, the playback task
│   ├── button.c               # espressif/button on CONFIG_JARVIS_PTT_GPIO → edge_core events
│   └── ui.c                   # LVGL: state, heard, answer text, offline reason, battery
└── components/edge_core/      # NO ESP-IDF includes, only C11 + vendored cJSON
    ├── edge_core.h / edge.c   # the state machine: events in, actions out
    ├── proto.c                # C2S builders, S2C parser (mirrors protocol.py)
    ├── base64.c               # streaming encoder (fragment-friendly), strict decoder
    ├── earcon.c
    └── third_party/cJSON.{c,h}
```

`edge_core` is a pure state machine with the same shape as the Python
`Segmenter`: it has no clock, no device and no socket. It takes events
(`button_down`, `button_up`, `mic_frames`, `ws_open`, `ws_message`,
`ws_closed`, `playback_finished`, `tick(ms)`) and returns a list of actions
(`send_json`, `mic_open`, `mic_close`, `play`, `stop_playback`, `earcon`,
`haptic`, `ui_update`). The shell carries out the actions. This is what makes
decision 7 possible.

### 7. The screen

The screen has one job: when you look at your wrist, it answers three
questions at a glance, in this order:

1. **Can I talk to it right now?** This is the ring and its colour.
2. **What did it hear?** This is the `heard` line.
3. **What did it say?** This is the answer text.

Everything else is secondary. The screen is always LVGL on black, because
black pixels on AMOLED are off, which saves battery and burn-in (README
gotcha 3).

#### 7.1 Layout: 410 × 502 portrait, one screen, no pages

```
 0 ┌──────────────────────────────────────┐
   │ ● wrist         14:32        3.91 V ⚡│  status bar, 36 px, 20 px text
36 ├──────────────────────────────────────┤
   │                                      │
   │              ╭────────╮              │
   │            ╱            ╲            │  state ring, 150 px diameter,
   │           │   ◉  0:03    │           │  centred, 10 px stroke.
   │            ╲            ╱            │  Icon + one short label inside.
   │              ╰────────╯              │
   │             Listening                │  state label, 28 px
216├──────────────────────────────────────┤
   │ YOU                                  │  12 px caps, dim
   │ “search black holes”                 │  heard, 24 px, max 2 lines, …
   │                                      │
   │ JARVIS                               │
   │ A black hole is a region of          │  answer, 26 px, the rest of the
   │ spacetime where gravity is so        │  height, vertical scroll. The
   │ strong that nothing, not even        │  sentence being spoken is white,
   │ light, can escape it. ▌              │  the rest is dim.
   │                                      │
466├──────────────────────────────────────┤
   │           HOLD TO TALK               │  mode chip / toast line, 36 px
502└──────────────────────────────────────┘
```

- **Margins:** 24 px on both sides, because the panel's corners are rounded.
  Nothing important goes in the four corners.
- **Burn-in:** the status bar and the mode chip are the only static elements.
  They shift by ±2 px once a minute.

#### 7.2 The status bar

| slot | shows | source |
|---|---|---|
| left | link dot + `device_id`. The dot is green when connected, amber while connecting, grey when offline. | `ws_open` / `ws_closed`, Kconfig |
| centre | the time, `HH:MM`, 24-hour | set by SNTP once per connection, then kept by the PCF85063 RTC. Shows `--:--` until it has been set once. |
| right | battery voltage to 2 decimals, `⚡` while charging, and red below 3.50 V | AXP2101 through XPowersLib, sampled every 10 s. It shows the voltage, not a percentage (README gotcha 7). |

There is no Wi-Fi signal indicator. Weak Wi-Fi shows up as the link dot turning
amber, and the bars would only be noise.

#### 7.3 The ring: one state at a time, and local states win

The ring shows exactly one state. **States the board knows about locally take
priority over whatever the brain last said.** Pressing the button changes the
screen at once, not a round trip later.

| # | state | when | ring | icon + label | extra |
|---|---|---|---|---|---|
| 1 | **Joining Wi-Fi** | boot, or the AP is lost | grey, slow spin | wifi · *Joining <SSID>* | |
| 2 | **Connecting** | Wi-Fi is up, the socket is not | amber, slow spin | link · *Connecting* | the host (`jarvis.example.com`) in the answer area |
| 3 | **Offline** | the socket closed or was refused | grey, dashed | ✕ · short reason | the long reason and *retry in 12 s* in the answer area (7.5) |
| 4 | **Ready** | connected, `state.value = idle` | cyan at 30%, still | mic · *Ready* | the last exchange stays, dimmed |
| 5 | **Recording** | button held (local) | red. The stroke **fills clockwise** toward the 15 s cap, and its **thickness follows the live mic level**, so you can see it hearing you. It turns amber for the last 3 s. | ◉ · elapsed `0:03` | |
| 6 | **Sending** | released, waiting for `heard` (local) | cyan spinner | *Sending* | the previous exchange is cleared here, so what follows is the new turn |
| 7 | **Holding** | `state.value = listening` | cyan, breathing | ··· · *Go on* | the brain is holding a fragment (hold window) and more speech will be merged |
| 8 | **Thinking** | `state.value = thinking` | amber, spinning | *Thinking* | |
| 9 | **Working** | `state.value = acting` | amber, spinning faster | *Working* | a skill is running, such as a search |
| 10 | **Speaking** | `state.value = speaking`, or a `speech` part is playing | cyan, **pulsing with the playback level** | speaker · *Speaking* | the current sentence is highlighted (7.4) |
| 11 | **Stopped** | the button stopped playback (local) | cyan, 1 flash | *Stopped* | for 1 s, then state 5 because the button is still held |
| 12 | **Mic off** | `state.mic = false`, button up | grey, crossed out | mic-off · *Mic off* | the button still works (7.11) |

A `state` message that arrives during state 5 or 6 is stored and applied as
soon as the local state ends. It never interrupts the recording ring.

#### 7.4 The transcript area

- **YOU** is the last `heard.text`, in curly quotes, at most 2 lines and then
  truncated with an ellipsis. It appears the moment `heard` arrives, which is
  before the answer. If `heard.confidence` is present and below 0.5, the line
  is shown in dim italic, so a mishearing is visible before JARVIS acts on it.
- **JARVIS** is the answer. `speech` parts arrive with their own `text`, so the
  answer is built up part by part as they arrive. The part that is playing is
  full white and the rest is 50% grey. The view auto-scrolls to keep the
  current sentence in view. When no speech arrives (voder down or speech
  switched off), the `text` message is shown whole instead.
- **System lines** (`text` with `from = "system"`, such as *Input: hold to
  talk.*) are never added to the transcript. They go to the toast line (7.6).
- **History:** the current exchange only. The next press clears it
  (state 6). There is no scroll-back into earlier turns in this milestone.
- **Touch:** only the answer area responds to touch, with a vertical swipe to
  scroll a long answer. Any touch also wakes the screen (7.7). There are no
  buttons on the screen.

#### 7.5 Offline reasons: short on the ring, full text in the answer area

| cause | ring label | answer area |
|---|---|---|
| no AP / wrong Wi-Fi password | *No Wi-Fi* | *Can't join “<SSID>”. Retrying.* |
| DNS / TCP refused / timeout | *No brain* | *Can't reach <host>. Is `jarvis serve` running?* + retry countdown |
| TLS verification failed | *TLS error* | *Certificate not trusted for <host>.* |
| close 4001 | *Refused* | *The brain refused this device. Check `JARVIS_EDGE_TOKEN` and that “<device_id>” is in `JARVIS_EDGE_TOKENS`.* |
| close 4001, reason *another device is connected* | *Busy* | *Another edge is connected to the brain.* |
| close 4002 | *Update me* | *This firmware speaks protocol 1; the brain wants another.* |
| close 4004 | *Brain stopped* | *JARVIS was shut down.* + retry countdown |
| `error` with `fatal: true` | *Error* | the message |

#### 7.6 The toast line (bottom, 36 px)

When nothing else is showing, this line holds the **mode chip**: *HOLD TO
TALK*, *BY NAME*, *ALWAYS* or *PAUSED*, taken from `MODE_LABEL` and shown in
capitals. A toast replaces the chip for 3 s, then the chip returns:

- a system `text`, such as *Input: by name.*;
- an `error` with `fatal: false`, shown in red;
- *Not connected*, when a press happens while offline, together with the
  earcon;
- *Cut at 15 s*, when the recording reaches the cap.

#### 7.7 Brightness and sleep

| after | screen |
|---|---|
| any activity | 80% brightness |
| 15 s with nothing happening | 20% |
| 60 s | off. Wi-Fi and the socket stay up, so the first press has no connect delay. |
| a button press, a touch, a `heard` / `speech` / `text` from the brain, or going offline | back to 80% at once |

The button **never** has to wake the screen before it works. A press on a dark
screen starts recording and turns the screen on as it does so.

#### 7.8 Fonts and colours

- **Fonts:** LVGL's built-in Montserrat covers only ASCII, so curly quotes,
  `é`, `å`, `ä`, `ö`, `°` and `–` would render as boxes. The firmware converts
  **Inter** (OFL) with `lv_font_conv` at 12, 20, 24, 26 and 28 px, with
  Latin-1, General Punctuation and the degree sign, and puts the result in
  PSRAM. A glyph that is still missing renders as `?`, not as an empty box.
- **Colours**, as tokens in `ui.c`:

| token | value | used for |
|---|---|---|
| `bg` | `#000000` | everything |
| `fg` | `#EDEDED` | the current sentence, the heard line |
| `dim` | `#6B6B6B` | the labels, earlier sentences, the idle ring |
| `ready` | `#3FC8FF` | Ready, Sending, Holding, Speaking |
| `rec` | `#FF453A` | Recording, low battery, errors |
| `busy` | `#FFB020` | Thinking, Working, Connecting, the last 3 s of recording |

#### 7.9 Haptics (GPIO 18)

| event | pattern |
|---|---|
| press, as the mic opens | one 15 ms tick |
| the recording is cut at 15 s | two ticks |
| `heard` arrives | one light 8 ms tick, so you know it got you without looking |
| a press while offline | one long 120 ms buzz, together with the earcon |

#### 7.10 How the screen is tested

`ui_update` is an `edge_core` action carrying a plain struct: the ring state,
the labels, the heard text, the answer parts and the index of the current
part, the toast and the brightness. That makes **everything in 7.3–7.7 a host
test in `tests/test_esp32_ui_model.py`**, with no LVGL involved. For example:

- a press during Thinking shows Recording at once;
- a `state` that arrives during Recording is applied after the release;
- `heard` with confidence 0.3 is shown dim;
- a system `text` becomes a toast and not a transcript line;
- a close code maps to the right label and message;
- the screen goes to 20% after 15 s and off after 60 s on `tick`, and back to
  80% on `heard`.

`ui.c` only draws the struct. Rendering is checked on the board with one photo
per ring state in the outcome doc.

#### 7.11 Decided with the user (2026-09-28)

- **The answer area shows the spoken text.** The brain sends
  `speakable(answer)` in `text` and in each `speech` part: markdown stripped,
  code blocks replaced by *"code on the screen"*, URLs cut to their host, and
  at most 600 characters. The watch shows exactly that, with the current
  sentence highlighted. The brain is not changed. Showing the full text (code,
  links, long answers) would need an optional `display` field on `text`. That
  is a later, protocol-compatible addition and is not part of 3.5.
- **History is the current exchange only.** The next press clears it (7.4).
- **The button overrides "mic off".** A held button is consent, so push-to-talk
  records even when the brain has sent `mic{on:false}`. Without this, *"Jarvis,
  turn off the mic"* would lock the board: the button would open nothing, and
  *"turn on the mic"* could never be said. The brain does not check `mic_on`
  when audio arrives (it is only an instruction to the edge), so this needs no
  brain change. In phase A, *mic off* therefore has **no effect on this board**,
  because the board never listens without the button. In phase B it stops
  always-listening and nothing else. The screen still shows *Mic off* (state
  12) so the brain's setting is visible, and holding the button and saying
  *"turn on the mic"* clears it. This **differs from the Python edge**, where
  mic-off disables the button too, and the firmware README says so.
- **No raise-to-wake in 3.5.** Only a press or a touch wakes the screen. Waking
  on a wrist raise with the QMI8658 is deferred until the IMU's battery cost
  has been measured.

---

## Phase B (follow-up, not required to close 3.5)

**Always-listening on the board:** port `jarvis/audio/segment.py` to
`edge_core/segment.c`, reproducing its tuning (the minimum-statistics floor,
300 ms pre-roll, 700 ms hangover and the 15 s cap cut at a pause). Then enable
ByName and Always, with the hold-to-talk segmenter variant (`hold=True`) for
the button. The acceptance test is M3 criterion 8 across implementations:
**the same WAV gives the same segments from `segment.c` and `segment.py`**.
Before it ships, measure the battery cost of a continuously running ES7210 and
I2S RX, because the board's own README gives about 6 h in low-power use.

---

## Tests (written first, per CLAUDE.md)

All of these run under `python -m pytest` on the dev box, with no board, no
ESP-IDF and no network. `edge_core` is compiled with the system `cc` into a
shared library in a pytest `tmp_path` (in a session-scoped fixture) and driven
through `ctypes`. There is one new helper, `tests/edge_core_lib.py`. If there
is no `cc`, the tests **skip with that reason**. `cc` is present on the dev
box, so they are expected to run.

1. **`tests/test_esp32_protocol.py`**, the C builders against the Python
   validator:
   - every C2S message `proto.c` can build (`hello`, `audio` with every
     `reason`, `speaking`, `interrupt`, and `control` for `setSpeech`, `mic`,
     `playbackDone` and `clientLog`) passes `protocol.validate_c2s`;
   - `hello` carries `PROTOCOL_VERSION`;
   - a `device_id` that `_DEVICE_ID_RE` would reject is refused **by the C
     side** before sending;
   - the streaming base64 encoder, fed in random chunk sizes, produces exactly
     `protocol.encode_pcm`'s output;
   - a fragmented `audio` frame, reassembled, is byte-identical to the
     unfragmented one;
   - the S2C messages built by Python's constructors (`ready`, `heard`,
     `state`, `text`, `speech`, `event` and `error`) are parsed by the C side
     into the right fields;
   - `decode_pcm` junk (bad base64, an odd byte count) is rejected, not
     crashed on.
2. **`tests/test_esp32_ptt.py`**, the button state machine (decision 1),
   event by event:
   - with the button up, **no** `mic_open`, `speaking` or `audio` ever happens,
     in any mode and with any traffic from the brain (criterion 5);
   - a press opens the mic and sends `speaking{on:true}`;
   - a release closes the mic and sends `speaking{on:false}` **before**
     `audio{reason:"release"}`;
   - a hold under 250 ms sends no `audio`;
   - holding past 15 s gives `reason:"maximum"` once, and nothing more until
     the next press;
   - a press during playback emits `stop_playback` and drains the queued parts
     **before** `interrupt`, and never sends `playbackDone` for the interrupted
     answer;
   - the brain's `event{stopPlayback}` stops playback;
   - after `event{mic, on:false}`, a press **still** opens the mic and sends the
     segment (7.11);
   - a press while disconnected emits `earcon`, and a segment recorded while
     disconnected is dropped, not queued (see `_send_segment`);
   - `speech` parts play in `part` order with a 150 ms gap, and
     `playbackDone{id}` is sent after the `final` part only.
3. **`tests/test_esp32_session.py`**, connection behaviour:
   - on `ws_open` the board sends `hello` and then `setSpeech{sample_rate:16000}`;
   - the backoff doubles from 1 s to the ceiling and resets on a successful
     `ready`;
   - 4001 and 4002 start the backoff at the ceiling;
   - each close code maps to its UI text, including the *busy* reason;
   - a `speech` frame whose declared length is over 4 MB is refused without
     allocating the buffer.
4. **`tests/test_esp32_against_brain.py`**, the core against the **real**
   `RemoteServer`. It uses `tests/remote_harness.py`, the loopback WebSocket
   and the real orchestrator with fake STT and TTS, as `test_remote_link.py`
   does. A small Python relay carries `edge_core`'s `send_json` actions to the
   socket and the socket's frames back as `ws_message` events:
   - press, feed PCM, release: the brain transcribes (fake), answers, and the
     core receives `heard`, then the `speech` parts, then sends `playbackDone`;
   - a spoken *"Jarvis, push to talk"* from ByName changes the stored mode;
   - after *"turn off the mic"*, a held *"turn on the mic"* is still heard and
     clears it;
   - a second connection with the same `device_id` replaces the first;
   - no leaked threads (the existing `no_leaked_threads` guard).
5. **`tests/test_esp32_ui_model.py`**, the screen model (decision 7.10).
6. **`tests/test_edge_imports.py`** is unchanged. The Python edge's import
   graph is untouched.

**Not testable on the host** (checked on hardware, in the outcome doc):
the I2S and codec bring-up, the GPIO 0 electrical behaviour, the Wi-Fi and TLS
handshake, LVGL rendering and the battery.

---

## New files

- `PRD/milestone-3.5-esp32-edge.md` (this file) and, once shipped,
  `PRD/milestone-3.5-esp32-edge-outcome.md`
- `firmware/esp32-edge/**` (decision 6)
- `tests/edge_core_lib.py`, `tests/test_esp32_protocol.py`,
  `tests/test_esp32_ptt.py`, `tests/test_esp32_session.py`,
  `tests/test_esp32_ui_model.py`, `tests/test_esp32_against_brain.py`

## Modified files

- `.gitignore`: `firmware/esp32-edge/{build,managed_components,sdkconfig,sdkconfig.old,sdkconfig.defaults.local}`
- `PRD/jarvis-2026-rebuild.md`: add the M3.5 entry under "Phased delivery"
- `CLAUDE.md`: a short "The ESP32 edge" subsection under "The remote edge
  (M3)" with the build, flash and first-setup steps and the GPIO 0 strapping
  note
- `config.example.toml`: a comment by `ptt_gpio` saying that the firmware's
  button pin is configured in Kconfig, not here
- **No change** to `jarvis/`. If one turns out to be needed, it is recorded as
  a finding.

---

## Verification

`python -m pytest` is green, including every test above, and the thread counts
are reported per CLAUDE.md step 5. `python -m jarvis text` and `--selftest` are
unchanged (regression). The firmware builds cleanly with
`idf.py set-target esp32s3 && idf.py build` on ESP-IDF 5.4, with no warnings in
`edge_core`.

**Acceptance criteria**, on the real board against the real brain, over the
Cloudflare Tunnel:

1. With the device in PushToTalk, **holding BOOT** and saying *"search black
   holes"*, then releasing, produces the spoken answer from the board's
   speaker. `heard` appears on the screen before the answer starts.
2. With the button up, the brain logs **no** `speaking` or `audio` from the
   board for 5 minutes of talking next to it. The mic is off, not just ignored.
3. Holding BOOT and saying *"Jarvis, push to talk"* from ByName switches the
   device's mode, and the mode survives a brain restart.
4. **Pressing BOOT during a spoken answer** stops the sound within 100 ms, and
   the next held utterance is handled normally (M3 criterion 10, which could
   not be verified on hardware in M3).
5. Pressing BOOT with the brain down plays the earcon, and the screen gives the
   reason. When the brain comes back, the board reconnects without being
   touched.
6. Turning off the Wi-Fi access point mid-answer: the brain logs a clean session
   end, the board shows *offline*, and it reconnects by itself when the AP
   returns (M3 criterion 11, also unverified in M3).
7. A wrong token shows *refused: check token* and does not retry faster than
   every 30 s. A second edge already connected shows *busy*.
8. A 20 s hold is cut at 15 s with `reason:"maximum"`, and the brain handles it
   as one turn.
9. Resetting the board while holding BOOT enters download mode, as documented,
   and `idf.py flash` works from there.
10. On a charged 400 mAh cell, idle and connected with the screen dimmed, the
    board lasts at least 3 h. The measured figure goes in the outcome doc.

**Latency budget:**

| stage | target |
|---|---|
| button release → last byte of `audio` sent (5 s utterance, 213 KB base64) | < 400 ms |
| button press during playback → silence | < 100 ms |
| button release → first sound from the board | < 2 s plus the skill's own time (same as M3) |

## Open questions

- **Multi-edge.** The brain accepts one edge at a time, so the board and the
  Pi or laptop edge cannot both be connected. Having both is the natural next
  step (a room device and a wrist device). It needs a brain change and stays
  out of scope here, but this milestone will show how much it is wanted.
- **Keeping Wi-Fi up versus battery.** Phase A keeps Wi-Fi and the WebSocket
  connected all the time, so the first press has no connect latency. If
  criterion 10 fails, the alternative is to connect on press, which costs
  about 1–2 s of TLS handshake per turn. That trade-off gets measured, not
  guessed.
