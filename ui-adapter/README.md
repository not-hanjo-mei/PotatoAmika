# orig-ui — the original OpenAmika Use page on the PotatoAmika scheduler

- `go/` — the page adapter as a single static Go binary (module
  `potatoamika/adapter`, Go 1.23, standard library only, CGO off; the
  WebSocket implementation is self-contained in `go/ws.go`). Default
  port 8372. Serves the original page and translates its endpoints to
  the scheduler (API.md) and the ASR gateway. `/api/tts_say` forwards
  to the scheduler's `POST /v1/say` (the one resident `tts_host`); the
  adapter spawns no TTS process of its own — standalone speech lives
  in the scheduler (409 while a turn is running, see API.md). The
  `/api/tts_say` semantics of the previous build are preserved: the
  response head and JSON keepalive whitespace go out immediately, so
  a long synthesis survives proxies that kill idle responses, and a
  repeated (tag, text) request is served from a result cache,
  byte-identical, in milliseconds.
  This replaces the previous zero-dependency Node implementation
  (`adapter.js`, removed in this round; it remains in git history).
  Same endpoints, same environment knobs; the HTTP acceptance suite
  passes 39/39 on both implementations and a full conversation round
  was compared event-for-event. Binary sizes: 8,812,626 B (aarch64),
  9,128,191 B (x86_64). Steady-state RSS ≈ 13 MB on the preview
  machine vs ≈ 65 MB for the Node build after long runs (dev machine:
  8.4 MB vs 42.5 MB). No JavaScript runtime is needed anywhere in the
  stack anymore.
- `patch.py` — builds `build/{use.html,use.js,common.js}` from the
  pristine originals in `tts-engine/refs-src/ui/`, applying exactly
  the removals in `REMOVALS.md` (asserted anchors; fails loudly if
  the originals drift).
- `build/` — the three patched page files. Every other page file is
  served byte-identical from a pristine copy.
- `REMOVALS.md` — what was cut from the page, where, and why; plus
  the one undecided item (camera) left for the user.
- `ws-client.js` — hand-rolled WebSocket acceptance client
  (ws/wss): one text turn, asserts commit → tts_audio (RIFF) →
  ai_surface_done → ai_end.

## Build

```sh
cd go
CGO_ENABLED=0 GOOS=linux GOARCH=arm64 go build -o adapter .
```

## Run

Installed on the phone by `deploy.sh` as `bin/adapter`, started with
every directory knob set explicitly. Standalone:

```sh
UI_PRISTINE=/path/to/pristine UI_PATCHED=/path/to/patched \
  MODELS_DIR=/path/to/models-live2d ./adapter
```

Directory defaults, when the variables are unset, are relative to
the adapter binary (`pristine/`, `patched/`, `models-live2d/`,
`static/`, `data/` next to it).

Env knobs: `PORT`, `UI_PRISTINE`, `UI_PATCHED`, `MODELS_DIR`,
`STATIC_DIR`, `DATA_DIR`, `SCHED`, `ASR_WS`. (The `TTS_*` /
`SAY_LD_LIBRARY_PATH` knobs of the long-gone private say host do not
exist.)
