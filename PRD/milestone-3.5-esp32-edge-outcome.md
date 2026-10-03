# Milestone 3.5 — the ESP32-S3 watch as an edge: outcome

**Written:** 2026-10-03, from the git history of 2026-09-28 → 10-02. The work
itself was done in many small branches (`edge-tools`, `edge-ota`,
`edge-standby-signal`, `pairing`, `power-upload`, `battery-current`, `ma-est`,
`force-update`, `timer-names`, `timer-cancel`, `notify-origin`, `power-modes`),
each merged into `develop`, without a milestone outcome of its own. This
document is that record.
**Plan:** `PRD/milestone-3.5-esp32-edge.md` (a draft, kept as written).
**Firmware:** `Nirvana77/esp32-s3-touch-amoled-2.06` (`jarvis-edge/`), not this repo.

---

## Where it differs from the plan

| Plan | What shipped |
|---|---|
| Firmware in `firmware/esp32-edge/` here, with a C core tested from pytest via `ctypes` | Firmware in its own repo, built by the `jarvis-builder` pod; the brain's side is tested here with fake edges |
| Protocol v1 unchanged, **the brain not modified** | Protocol v1 extended (all additions optional, an M3 edge is unaffected) and the brain extended for the watch, below |
| Push-to-talk on the BOOT button | Kept |
| `wss://` through the Cloudflare Tunnel | Kept; HAProxy (`jarvis-lb`) in front of a dev brain and a k8s pod brain |

## What the brain gained

| Feature | Where | Notes |
|---|---|---|
| **Standby signal** | `remote/server.py` | The edge is told about standby with a `state` before the standby line is spoken. |
| **Edge tools** | `skills/edge.py`, `protocol.validate_tools`, `nlu/slots.extract_typed` | An edge lists tools in `hello` (name, description, examples, typed params: `duration`, `number`, `text`, `name`). Stored in `data/remote/tools/<device>.json`, registered as `origin = "edge"` skills, learned in the background, called with `call` / `result`. A missing required param is asked for ("How long, sir?"). While the edge is away the tool answers that it isn't connected. |
| **OTA firmware** | `remote/firmware.py`, `skills/builtin/update_watch.py`, `force_update_watch.py` | `data/firmware/<device>.bin`; the version is read from the image; an `event ota` (version, size, sha256) on connect, fetched with `GET /firmware` + bearer token. "Update the watch" pushes it on demand; "force update" skips the version checks. An update the watch would refuse is not offered. |
| **Pairing** | `remote/pairing.py`, `python -m jarvis pair / devices` | `pair` with an X25519 key instead of `hello`; both sides derive a 6-digit code; approved on the brain with the admin secret in `data/remote/admin.token`; the token is sealed with AES-256-GCM and kept in `data/remote/tokens.json` (0600). Unconfirmed requests expire after 2 min; at most 4 wait. |
| **Power log** | `remote/powerlog.py`, `skills/builtin/watch_power.py`, `python -m jarvis power` | The watch's SD-card log arrives as `file` messages in `data/remote/power/<device>/`; the report gives time per mode, sleep, drain in mV/h, the estimated current per mode and the watch's own `ma_est`, restarts and drops, and the mode stretches (`YYYY-MM-DD-modes.csv`). |
| **Notifications** | `GET /notify`, `python -m jarvis notify` | Same auth as `/firmware`; queued (last 20) while the edge is away. |
| **Talking to the device that asked** | `core/orchestrator.py` | A background job's notices and questions go to the device that started it, are held until it is connected, and may be asked unprompted in standby. |

## Verified

Unit and loopback tests per feature: `tests/test_edge_tools.py`,
`test_remote_firmware.py`, `test_update_watch.py`, `test_pairing.py`,
`test_power_log.py`, plus the remote-link tests. Daily use on the real watch
over the internet. That use found two OTA problems that are still open:
known issues #8 and #9.

## Not done from the plan

- **Phase B**: a C port of the segmenter for ByName / Always on the watch.
  The watch is push-to-talk only.
- **Multi-device**: still one edge per brain (known issue #3). The watch and
  `python -m jarvis edge` cannot be connected at the same time.
