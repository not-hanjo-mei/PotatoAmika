# Original Use page — PotatoAmika variant: removal manifest

Method: the page is the original OpenAmika Use page (zeekk's repo,
`tts-engine/refs-src/ui/`). Three files carry removals — `use.html`,
`use.js`, `common.js` — produced by `patch.py` from the pristine
originals, so a diff against the originals shows exactly the cuts below
and nothing else. Every other page file (live2d-engine.js,
openamika.css, vendor libraries, about page) is served byte-identical
from the pristine tree. No surviving part was rewritten or restyled;
the two exceptions are marked † (a label and a renumber forced by a
removal).

Rule applied: a part was removed only if the capability behind it
definitively does not exist in the scheduler API / current backend and
is not planned for this phase. Anything merely unwired was wired
instead (see adapter.js header and the delivery report).

## Pages not served at all

- **Setup page** (`setup.html`/`setup.js`) and **Tools page**
  (`tools.html`/`tools.js`): not served (404), and their nav/footer
  links removed from `common.js`. Their function is configuring
  backends PotatoAmika does not have: brain providers (OpenRouter /
  Ollama), messenger bots, IoT devices, MCP tool servers, persona file
  editing. The two Setup sub-features that DO map to real capabilities
  (health view; avatar bank editing) are not lost: health lives in the
  nav light + popover on the Use page itself, and the banks endpoint
  (`GET/POST /api/avatar/models/{id}/banks`) is fully implemented in
  the adapter even though its original editor page is not served.
- The **About** page and the static **System flow** page are served
  unchanged.

## Removals (file · what · why)

- `use.html` — **Tool dialogue card (what she asks her helpers)** (-434 chars). Hermes/nanoMuse tool layer is disabled in PotatoAmika v1 by design; there are no helper dialogues to show.
- `use.html` — **Telemetry tiles (session cost / first voice / prompt cache) + per-round latency chart sections** (-1591 chars). Driven by the original's per-round telemetry spec (OpenRouter billing, surface/deep phase timings). The scheduler API has no such feed; local inference has no per-role cost accounting.
- `use.html` — **'Her helpers' jobs & approvals card** (-412 chars). Same missing tool layer: no helper jobs, no approvals queue exists in the backend.
- `use.html` — **Stage speech bubbles (her/helper tool-dialogue bubbles)** (-115 chars). These bubbles only render tool-dialogue lines (hers to helpers, helpers to her); with no tool layer they can never appear. Her own speech uses the HUD line and transcript, which stay.
- `use.html` — **Stage speech bubble (agent side)** (-128 chars). See bubbleHer removal.
- `use.html` — **Latency chart tooltip element** (-40 chars). Part of the removed telemetry chart.
- `use.html` — **Deep cue injection row (input + Inject button)** (-307 chars). Proactive cue turns do not exist in the scheduler API: every turn is a user utterance that enters history and memory extraction, so a 'cue she reacts to without you speaking' cannot be honoured. Queued context (the other row) is kept and wired.
- `use.html` — **Details summary label mentions the removed cue row** (-10 chars). Label of the block the cue row was removed from.
- `use.js` — **pollStatus() call on socket open** (-20 chars). Telemetry polling removed (see latency/status block).
- `use.js` — **scheduleToolPoll on ai_begin** (-111 chars). Tool-layer polling removed.
- `use.js` — **pollStatus/scheduleToolPoll calls on ai_end** (-123 chars). Telemetry + tool-layer polling removed.
- `use.js` — **handle() cases ai_timeline / ai_timeline_void / ai_call_receipt** (-240 chars). The scheduler API emits no timeline or call-receipt events.
- `use.js` — **Latency chart, prompt-cache tile, usage/status polling (pollStatus)** (-12010 chars). No telemetry feed in the scheduler API (see use.html telemetry removal).
- `use.js` — **Tool dialogue rendering, stage bubbles, helpers' jobs/approvals UI, pollTools** (-13458 chars). Tool layer disabled in v1; nothing to render or poll.
- `use.js` — **Deep cue Inject button handler** (-282 chars). See Deep cue removal in use.html.
- `use.js` — **setPowerUI line enabling the Inject button** (-35 chars). Removed with the Deep cue row.
- `use.js` — **commit handler's 'Cue' label branch** (-33 chars). Only the removed cue path produced input==="inject" commits.
- `use.js` — **Wizard step count 5 → 4** (-0 chars). Wizard provider step removed (below).
- `use.js` — **Wizard spoken line for the provider step** (-224 chars). Provider step removed (below).
- `use.js` — **Wizard keyStatus variable** (-18 chars). Only the removed provider step used it.
- `use.js` — **Wizard loadKeyStatus() (OpenRouter key status line)** (-351 chars). Only the removed provider step used it.
- `use.js` — **Wizard step 3: brain provider choice (OpenRouter key panel + Ollama panel)** (-6714 chars). Her brain is the local Minico via llama.cpp; there is no OpenRouter account and no Ollama in PotatoAmika, and the page cannot switch providers.
- `common.js` — **Nav link: Setup** (-53 chars). Setup page not served (see manifest note).
- `common.js` — **Nav link: Tools** (-53 chars). Tools page not served (see manifest note).
- `common.js` — **Footer link: Setup** (-35 chars). Setup page not served (see manifest note).
- `common.js` — **Footer link: Tools** (-35 chars). Tools page not served (see manifest note).

† The two non-pure-deletion edits: the wizard step count `TOTAL = 5`
became 4 (and the check step now reads `lines()[3]`), and the
transcript details summary that read "cue & context injection" now
reads "context injection". Both are direct consequences of removals
above, listed here so the diff has no surprises.

## Addition (not a removal)

- `use.html` — **Page title: `OpenAmika · Use` -> `PotatoAmika`**
  (2026-11-09, user direction). The preview is shared with others as
  PotatoAmika, so the browser tab carries that name instead of the
  upstream project's. Implemented in `patch.py` via `replace_once`
  next to the removals; nothing else on the page changes — a
  regenerated `use.html` differs from the previous build on the title
  line only.

Known leftover: the wizard's step-2 hint still says "Two quick things:
the brain she thinks with, and a check that her voice and hearing are
wired up." The brain half refers to the removed provider step. The
sentence was left untouched (it is surviving copy, not the removed
part itself); flagging it here rather than silently rewriting it.

CSS note: `openamika.css` is served byte-identical; classes that only
styled removed elements are inert.

## NOT removed — undecided, left for the user

- **Camera button + picture-in-picture + `/ws/camera`**: PotatoAmika's
  vision is the scheduler's one-shot `POST /v1/look` (ask about one
  photo; brain/embeddings unload and reload around it), which is not
  this page's continuous "she watches you" feed, and no page channel
  for it exists this phase. Left in place for now: `/api/vision`
  reports `enabled:false`, so the button produces the page's own
  "vision is disabled" toast instead of pretending to work.
  Recommendation: remove the camera UI like the other missing
  capabilities, or wire the button to a one-shot look — the user's
  call, deliberately not made here.
