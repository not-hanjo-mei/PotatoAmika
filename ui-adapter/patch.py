#!/usr/bin/env python3
"""Build the PotatoAmika variant of the original OpenAmika Use page.

Copies refs-src/ui/{use.html,use.js,common.js} and removes exactly the parts
whose backend capability PotatoAmika definitively does not have (per the
removal manifest). Everything else stays byte-identical to the original.
Each removal is asserted: the anchor must exist, and the removed chunk must
contain its marker string, or the build stops.
"""
import pathlib, re, sys

SRC = pathlib.Path.home() / "workspace/openamika-rebuild/tts-engine/refs-src/ui"
OUT = pathlib.Path.home() / "workspace/openamika-rebuild/orig-ui/build"
OUT.mkdir(parents=True, exist_ok=True)

manifest = []  # (file, title, marker, nchars, reason)

def cut_span(text, start_anchor, end_anchor, fname, title, marker, reason, keep_end=False):
    i = text.index(start_anchor)
    j = text.index(end_anchor, i + len(start_anchor))
    if not keep_end:
        j += len(end_anchor)
    chunk = text[i:j]
    assert marker in chunk, f"{fname}: marker {marker!r} not in removed chunk for {title}"
    manifest.append((fname, title, marker, len(chunk), reason))
    return text[:i] + text[j:]

def cut_line(text, line, fname, title, reason):
    needle = line + "\n"
    assert text.count(needle) == 1, f"{fname}: line not unique: {line!r}"
    manifest.append((fname, title, line.strip()[:60], len(needle), reason))
    return text.replace(needle, "")

def replace_once(text, old, new, fname, title, marker, reason):
    assert marker in old or marker in new
    assert text.count(old) == 1, f"{fname}: replace anchor not unique: {old[:60]!r}"
    manifest.append((fname, title, old.strip()[:60], len(old) - len(new), reason))
    return text.replace(old, new)

# ---------------------------------------------------------------- use.html
h = (SRC / "use.html").read_text(encoding="utf-8")
# Addition, not a removal (user direction, 2026-11-09): the browser tab
# carries the PotatoAmika name. Nothing else on the page changes.
h = replace_once(h, '<title>OpenAmika · Use</title>',
                 '<title>PotatoAmika</title>', "use.html",
                 "Page title: OpenAmika · Use -> PotatoAmika", "PotatoAmika",
                 "The preview is shared as PotatoAmika; the tab title must say so. The original title named the upstream project.")
h = cut_span(h, '<section class="tech-card rv" id="toolDialogCard"',
             '</section>', "use.html",
             "Tool dialogue card (what she asks her helpers)",
             'id="toolLog"',
             "Hermes/nanoMuse tool layer is disabled in PotatoAmika v1 by design; there are no helper dialogues to show.")
h = cut_span(h, '<!-- telemetry: Professional only -->',
             'id="latLegend"></div>\n      </section>', "use.html",
             "Telemetry tiles (session cost / first voice / prompt cache) + per-round latency chart sections",
             'id="latChart"',
             "Driven by the original's per-round telemetry spec (OpenRouter billing, surface/deep phase timings). The scheduler API has no such feed; local inference has no per-role cost accounting.")
h = cut_span(h, '<!-- her helpers: jobs in flight',
             '<div id="pendingBox"></div>\n  </section>', "use.html",
             "'Her helpers' jobs & approvals card",
             'id="jobsCard"',
             "Same missing tool layer: no helper jobs, no approvals queue exists in the backend.")
h = cut_span(h, '<div class="bubble her" id="bubbleHer"',
             '</div>', "use.html",
             "Stage speech bubbles (her/helper tool-dialogue bubbles)",
             'id="bubbleHer"',
             "These bubbles only render tool-dialogue lines (hers to helpers, helpers to her); with no tool layer they can never appear. Her own speech uses the HUD line and transcript, which stay.")
# bubbleAgent line follows the same pattern (single-line div)
h = cut_line(h, '        <div class="bubble agent" id="bubbleAgent" aria-live="polite"><span class="who"></span><span class="text"></span></div>',
             "use.html", "Stage speech bubble (agent side)", "See bubbleHer removal.")
h = cut_line(h, '<div class="viz-tip" id="vizTip"></div>',
             "use.html", "Latency chart tooltip element", "Part of the removed telemetry chart.")
h = cut_span(h, '        <div class="form-row">\n          <div class="field" style="flex:2"><label>Deep cue (she reacts without you speaking)</label>',
             '<button class="btn small" id="injectBtn" disabled>Inject</button>\n        </div>', "use.html",
             "Deep cue injection row (input + Inject button)",
             'id="injectInput"',
             "Proactive cue turns do not exist in the scheduler API: every turn is a user utterance that enters history and memory extraction, so a 'cue she reacts to without you speaking' cannot be honoured. Queued context (the other row) is kept and wired.")
h = replace_once(h, '<summary>Advanced: cue &amp; context injection</summary>',
                 '<summary>Advanced: context injection</summary>', "use.html",
                 "Details summary label mentions the removed cue row", "cue",
                 "Label of the block the cue row was removed from.")
(OUT / "use.html").write_text(h, encoding="utf-8")

# ---------------------------------------------------------------- use.js
j = (SRC / "use.js").read_text(encoding="utf-8")
j = replace_once(j, '      ws.send(JSON.stringify({ type: "settings", voice_output: $("voiceOutToggle").checked }));\n      pollStatus();',
                 '      ws.send(JSON.stringify({ type: "settings", voice_output: $("voiceOutToggle").checked }));', "use.js",
                 "pollStatus() call on socket open", "pollStatus", "Telemetry polling removed (see latency/status block).")
j = cut_line(j, '        if (msg.proactive) scheduleToolPoll(500);   // a helper\'s result is usually what she is about to relay',
             "use.js", "scheduleToolPoll on ai_begin", "Tool-layer polling removed.")
j = cut_span(j, '        pollStatus();\n        scheduleToolPoll(1500);',
             'the clerk dispatches right after she finishes',
             "use.js", "pollStatus/scheduleToolPoll calls on ai_end", "pollStatus",
             "Telemetry + tool-layer polling removed.")
j = cut_span(j, '      case "ai_timeline":', '      default:',
             "use.js", "handle() cases ai_timeline / ai_timeline_void / ai_call_receipt",
             'ai_call_receipt', "The scheduler API emits no timeline or call-receipt events.", keep_end=True)
j = cut_span(j, '  // ── latency chart (dataviz: fixed categorical order, validated palette) ──',
             '  // ── tool dialogue:', "use.js",
             "Latency chart, prompt-cache tile, usage/status polling (pollStatus)",
             'function pollStatus', "No telemetry feed in the scheduler API (see use.html telemetry removal).", keep_end=True)
j = cut_span(j, '  // ── tool dialogue:', '  // ── first run:',
             "use.js", "Tool dialogue rendering, stage bubbles, helpers' jobs/approvals UI, pollTools",
             'function pollTools', "Tool layer disabled in v1; nothing to render or poll.", keep_end=True)
j = cut_span(j, '  $("injectBtn").addEventListener("click", () => {',
             '  $("ctxBtn").addEventListener', "use.js",
             "Deep cue Inject button handler", "injectInput", "See Deep cue removal in use.html.", keep_end=True)
j = cut_line(j, '    $("injectBtn").disabled = !on;', "use.js",
             "setPowerUI line enabling the Inject button", "Removed with the Deep cue row.")
j = replace_once(j, 'chatAdd("you", msg.input === "inject" ? "Cue" : "You", msg.text);',
                 'chatAdd("you", "You", msg.text);', "use.js",
                 "commit handler's 'Cue' label branch", "inject",
                 "Only the removed cue path produced input===\"inject\" commits.")
j = replace_once(j, 'const TOTAL = 5;', 'const TOTAL = 4;', "use.js",
                 "Wizard step count 5 → 4", "TOTAL", "Wizard provider step removed (below).")
j = cut_line(j, '      `First, my brain. I think best through OpenRouter — just paste a key, it only costs a few dollars a month. I can also run entirely on your own computer with Ollama, but only with a big model and a big graphics card.`,',
             "use.js", "Wizard spoken line for the provider step", "Provider step removed (below).")
j = replace_once(j, 'let step = 0, keyStatus = null, lastHealth = null;',
                 'let step = 0, lastHealth = null;', "use.js",
                 "Wizard keyStatus variable", "keyStatus", "Only the removed provider step used it.")
j = cut_span(j, '    async function loadKeyStatus(line) {',
             '    async function runCheck', "use.js",
             "Wizard loadKeyStatus() (OpenRouter key status line)", "openrouter_key",
             "Only the removed provider step used it.", keep_end=True)
j = cut_span(j, '      } else if (step === 3) {',
             '      } else {\n        box.append(say(lines()[4]));', "use.js",
             "Wizard step 3: brain provider choice (OpenRouter key panel + Ollama panel)",
             'ollama-use', "Her brain is the local Minico via llama.cpp; there is no OpenRouter account and no Ollama in PotatoAmika, and the page cannot switch providers.")
j = j.replace('box.append(say(lines()[4]));', 'box.append(say(lines()[3]));')
(OUT / "use.js").write_text(j, encoding="utf-8")

# ---------------------------------------------------------------- common.js
c = (SRC / "common.js").read_text(encoding="utf-8")
c = cut_line(c, '        <a href="/setup" data-page="setup">Setup</a>', "common.js",
             "Nav link: Setup", "Setup page not served (see manifest note).")
c = cut_line(c, '        <a href="/tools" data-page="tools">Tools</a>', "common.js",
             "Nav link: Tools", "Tools page not served (see manifest note).")
c = cut_line(c, '        <a href="/setup">Setup</a>', "common.js",
             "Footer link: Setup", "Setup page not served (see manifest note).")
c = cut_line(c, '        <a href="/tools">Tools</a>', "common.js",
             "Footer link: Tools", "Tools page not served (see manifest note).")
(OUT / "common.js").write_text(c, encoding="utf-8")

for fname, title, marker, n, reason in manifest:
    print(f"{fname}\t{title}\t-{n} chars\t{reason}")
print(f"total removals: {len(manifest)}", file=sys.stderr)
