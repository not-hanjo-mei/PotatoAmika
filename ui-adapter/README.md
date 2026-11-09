# orig-ui — the original OpenAmika Use page on the PotatoAmika scheduler

- `adapter.js` — zero-dependency Node service (default port 8372). Serves
  the original page and translates its endpoints to the scheduler
  (API.md) and the ASR gateway. `/api/tts_say` forwards to the
  scheduler's `POST /v1/say` (the one resident `tts_host`); the adapter
  spawns no TTS process of its own — an earlier build ran a private say
  host (~375 MB), which the phone build cannot afford, so standalone
  speech moved into the scheduler (409 while a turn is running, see
  API.md).
- `patch.py` — builds `build/{use.html,use.js,common.js}` from the
  pristine originals in `tts-engine/refs-src/ui/`, applying exactly the
  removals in `REMOVALS.md` (asserted anchors; fails loudly if the
  originals drift).
- `build/` — the three patched page files. Every other page file is
  served byte-identical from a pristine copy.
- `REMOVALS.md` — what was cut from the page, where, and why; plus the
  one undecided item (camera) left for the user.
- `ws-client.js` — hand-rolled WebSocket acceptance client
  (ws/wss): one text turn, asserts commit → tts_audio (RIFF) →
  ai_surface_done → ai_end.

## Run (preview machine, potato-screen)

Deployed at `/data/home/potato/orig-adapter/adapter.js` with page files
in `/data/home/potato/orig-ui/` (`pristine/` holds the untouched
originals incl. vendor; `_static/flow.html` is the original flow page)
and state in `/data/home/potato/orig-data/`:

```sh
cd /data/home/potato/orig-adapter
UI_PRISTINE=/data/home/potato/orig-ui/pristine \
  setsid nohup node adapter.js > adapter.log 2>&1 < /dev/null &
```

Port 8372 is published (public) via the InstaBox ports API:
https://8372-agq3ojt4iaxybp2mdpisjqj3wu.instabox.tech/use

Env knobs: `PORT`, `UI_PRISTINE`, `UI_PATCHED`, `MODELS_DIR`,
`STATIC_DIR`, `DATA_DIR`, `SCHED`, `ASR_WS`. (The `TTS_*` /
`SAY_LD_LIBRARY_PATH` knobs of the private say host are gone with it.)
