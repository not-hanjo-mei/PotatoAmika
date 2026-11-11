# PotatoAmika push manifest — scan record and file inventory

Prepared 2026-11-09 for the first push of PotatoAmika's own work into
this fork, in the flattened root layout the owner chose.

- **Repository:** `not-hanjo-mei/PotatoAmika` (fork of `zeekk0/OpenAmika`)
- **Target branch:** `main` (pushed directly, per the owner's decision)
- **Base head before this push:** `579586199843b4031a52519fe03ff9b586ab5e0b`
- **Method:** direct git push over SSH from the maintainer machine.
  Resulting commit SHAs are appended to the maintainer copy of this
  file after the push; this in-repo copy is the pre-push
  scan/inventory record.

## Push plan (logical commits, in order)

1. `engine/**` — "add the self-built TTS inference engine (C++, ORT-free, W4 packed format)"
2. `scheduler/**` — "add the Go scheduler (conversation, memory, tone, vision, /v1/say)"
3. `asr/**` — "add the ASR host (Moonshine + Silero VAD) and WebSocket gateway"
4. `quantize/**` — "add the weight conversion method (GPTQ variant included, not adopted)"
5. `ui-adapter/**` + `deploy/**` — "add the original Use page adapter and the phone deploy scripts"
6. `docs/**` (PotatoAmika additions only) + `PUSH-MANIFEST.md` + root `README.md` — "add engineering docs and the push manifest; prepend PotatoAmika notes to the root README (upstream text kept verbatim below)"
7. Deletion of the original Python implementation (142 files, listed
   under "Upstream Python implementation" below) — "removed the
   original Python implementation; kept UI sources and license
   notices"

Layout note: the work was first staged under a `potatoamika/`
directory; the owner chose a flat layout instead, so the six content
directories sit at the repository root and the staged README was
merged into the root README's PotatoAmika section. There is no
`potatoamika/` directory in the result. The root `README.md` change
is an edit of an existing upstream file: a PotatoAmika header
followed by the upstream text, byte-for-byte (verified
programmatically before the push). Our documents were merged into
upstream's `docs/` folder; the filename sets did not overlap, and no
upstream document was modified.

## Inventory (as staged for commits 1–6)

81 files, 4,837,829 bytes, listed in full below. This manifest is one
further file at the repository root and is not itself in the table.

| Path | Bytes |
|---|---|
| `README.md` | 22400 |
| `asr/CMakeLists.txt` | 860 |
| `asr/gateway/go.mod` | 30 |
| `asr/gateway/main.go` | 9384 |
| `asr/scripts/cer.py` | 1966 |
| `asr/scripts/export_vocab.py` | 963 |
| `asr/scripts/prep_tests.py` | 2717 |
| `asr/src/asr_host.cpp` | 29842 |
| `deploy/assemble_stage.py` | 5036 |
| `deploy/deploy.sh` | 18659 |
| `deploy/live2d-files.txt` | 34104 |
| `deploy/scheduler.json.tmpl` | 680 |
| `docs/API.md` | 8940 |
| `docs/ASR.md` | 17321 |
| `docs/BUDGET.md` | 3854 |
| `docs/CONVO-TEST.md` | 7420 |
| `docs/CONVO-TEST2.md` | 7244 |
| `docs/DEESS.md` | 18621 |
| `docs/FREE-CENSUS.md` | 11332 |
| `docs/ORT-FREE.md` | 37470 |
| `docs/SCHEDULER.md` | 29508 |
| `docs/STAGE1.md` | 25267 |
| `docs/STAGE2.md` | 15527 |
| `docs/STAGE3.md` | 33557 |
| `docs/VITS-W4.md` | 7022 |
| `engine/CMakeLists.txt` | 2240 |
| `engine/data/cmudict.tsv` | 3871142 |
| `engine/data/symbols.txt` | 3231 |
| `engine/src/cpu.h` | 3733 |
| `engine/src/deess.h` | 9655 |
| `engine/src/free_engine.cpp` | 163923 |
| `engine/src/free_engine.h` | 3635 |
| `engine/src/free_pool.h` | 2587 |
| `engine/src/g2p.cpp` | 16292 |
| `engine/src/g2p.h` | 1699 |
| `engine/src/json_min.h` | 5535 |
| `engine/src/kbench.cpp` | 2919 |
| `engine/src/main.cpp` | 50586 |
| `engine/src/npz.h` | 8746 |
| `engine/src/onnx_model.h` | 13161 |
| `engine/src/packed.h` | 13153 |
| `engine/src/packed_core.h` | 15499 |
| `engine/src/patch.h` | 4973 |
| `engine/src/rpk.h` | 2217 |
| `engine/src/test_softclip.cpp` | 1921 |
| `engine/src/tf_check.cpp` | 15166 |
| `engine/src/tf_check_free.cpp` | 11421 |
| `engine/src/vits_check.cpp` | 14359 |
| `engine/src/vits_check_free.cpp` | 13176 |
| `engine/src/vits_kernels.cpp` | 17100 |
| `engine/src/vits_kernels.h` | 3980 |
| `engine/src/vits_packed.h` | 10947 |
| `quantize/PTQ-PLUS.md` | 6279 |
| `quantize/README.md` | 3136 |
| `quantize/convert.py` | 379 |
| `quantize/dequant_wo.py` | 2512 |
| `quantize/gptq_pack.py` | 9031 |
| `quantize/make_packed_graphs.py` | 7842 |
| `quantize/make_packed_vits.py` | 12631 |
| `quantize/precompute_refs.py` | 4948 |
| `quantize/quant_wo.py` | 5230 |
| `quantize/vits_w4_hybrid.py` | 3095 |
| `quantize/vits_w4_pack.py` | 7216 |
| `scheduler/api.go` | 8602 |
| `scheduler/brain.go` | 12462 |
| `scheduler/config.go` | 6239 |
| `scheduler/go.mod` | 38 |
| `scheduler/main.go` | 4082 |
| `scheduler/main_test.go` | 1826 |
| `scheduler/memory.go` | 4700 |
| `scheduler/persona.go` | 4051 |
| `scheduler/servers.go` | 7187 |
| `scheduler/session.go` | 15575 |
| `scheduler/tools.go` | 1982 |
| `scheduler/tts.go` | 4315 |
| `scheduler/vision.go` | 5897 |
| `ui-adapter/README.md` | 1965 |
| `ui-adapter/REMOVALS.md` | 7405 |
| `ui-adapter/adapter.js` | 42678 |
| `ui-adapter/patch.py` | 10364 |
| `ui-adapter/ws-client.js` | 3442 |

## Red-line scan record

Every file in the inventory was scanned in the flattened tree before
the push, with the same criteria as the earlier staging round.
Methods and results:

| Check | Method | Result |
|---|---|---|
| Weight/model formats | extension blacklist over the added tree: `.onnx .gguf .rpk .ckpt .pth .npz .safetensors .moc3 .model3.json .wav .mp3 .bin .so .o .a .tar .zip .exe` | 0 hits |
| Binaries | `file` magic scan for ELF/Mach-O/PE | 0 hits (all files are text) |
| Symlinks | `find -type l` over the added tree | 0 |
| Encoding | every added file decoded as UTF-8 | all pass |
| Secrets | regex sweep for `ghp_`, `github_pat_`, `gho_`, `sk-`, `hf_`, `AKIA`, `PRIVATE KEY`, `Bearer <token>`, and `api_key/token/password/secret = "<value>"` assignments | 0 matches |
| Trial residue | grep for the preview trial's persona values and for stored-memory payloads | 0 (the only `packets` occurrences are the API field name in `scheduler/api.go` and `docs/API.md`) |
| Persona files | filename scan for the persona texts and `memory.json` | 0 — persona and memory files were never in the copy lists |
| Machine paths in code | grep for `/home/hatch`, `/data/home`, preview hostnames in `*.go *.js *.py *.sh *.cpp *.h *.tmpl *.mod` | 0 after the publication edits below |

Selection itself was allowlist-based: only the files in the inventory
were copied out of the development tree, each from an explicit list.

## Data-file provenance decisions (engine/data)

| File | Decision | Basis |
|---|---|---|
| `cmudict.tsv` | **Included** | CMU Pronouncing Dictionary export (word → ARPAbet); CMUdict is freely redistributable. |
| `symbols.txt` | **Included** | Phoneme symbol table following the GPT-SoVITS v2 symbol set; GPT-SoVITS is MIT. |
| `namedict.tsv` | **HELD — not pushed** | Supplemental English name-pronunciation dictionary (19,929 entries). Its source could not be verified to this release's standard, so it stays out and is named here, per the rule that unverifiable data does not ship. `g2p.cpp` treats it as a load-if-present dictionary. |
| `g2p_probe.json` | **HELD — not pushed** | Development probe artifact (sentences → phoneme-id sequences), not part of the release surface. |

Chinese G2P dictionary data is not part of this release at all: the
published engine data covers the English text path (see the root
README, "TTS release scope: English").

## Publication edits (staged copies only; originals untouched)

Applied to make the code publishable; each is a default-value or
path change, no logic changes:

- `quantize/convert.py` — the hardcoded developer working directory
  replaced by `TTS_ENGINE_DIR` (default `.`).
- `engine/src/main.cpp`, `engine/src/tf_check.cpp`,
  `engine/src/tf_check_free.cpp`, `engine/src/vits_check_free.cpp` —
  baked-in developer path as the default asset root replaced by `.`
  (all four tools already accept `--root`).
- `ui-adapter/adapter.js` — preview-machine default directories
  replaced by paths relative to the adapter file; `deploy.sh` sets
  all five environment variables explicitly, so installed behavior
  is unchanged.
- `deploy/assemble_stage.py` — source/stage roots made
  environment-overridable (`POTATO_SRC`, `POTATO_STAGE`) with neutral
  defaults.
- Flattening (this round) — the root README was composed from the
  staged header plus the staged `potatoamika/README.md`, with paths
  rewritten to the root layout. Two intentional non-changes:
  `scheduler/go.mod` keeps the module name `potatoamika/scheduler`
  (a module identifier, not a filesystem path), and `deploy.sh`
  keeps the phone deploy tarball's internal top-level name (an
  install-time format, unrelated to this repository's layout).

## Held for review — not pushed

- **`DISTRIBUTION.md`** (licensing/distribution memo) is held in the
  maintainer review area, not pushed. It contains an unsent draft
  permission-request email to MekaHime and internal deliberation
  about what may be published; the publishable conclusions are
  already reflected in the root README, and the memo itself should
  be reviewed (and the draft email removed) before any public
  posting.

## Excluded by the red lines / selection (named)

- All voice weights and derivatives: `tts-engine/weights/` (fp32
  `.ckpt`/`.pth`), `tts-engine/wo-packs/*.npz` (quantized packs),
  `tts-engine/graphs/`, `tts-engine/host/refpacks/` and all
  `phone-calib*/` payload packs (`.rpk` reference packs included).
- All GGUF models: `scheduler/models/`, vision model files.
- All ONNX models: `asr/models/`, `tts-engine/work/*.onnx`,
  `tts-engine/GenieData/` (~964 MB of Genie runtime data).
- Live2D models: `tts-engine/refs-src/avatar/models/` (174 + 70
  files) — never copied; only the hash list
  (`deploy/live2d-files.txt`) ships, so installers fetch the models
  themselves under MekaHime's license.
- Persona texts, every language version: the surface/thought/
  examples/scenes files under `tts-engine/refs-src/brain/` — never
  in the copy lists.
- User data: `scheduler/data/**/memory.json` (four memory stores).
- Build outputs and binaries: `scheduler/bin/`, `scheduler/scheduler`,
  `asr/gateway/asr_gateway*`, `tts-engine/host/build*/`, `dist-*/`,
  logits dumps, wav sample trees, `deploy/*.tar.part-*`.
- Development-only trees not in the selection: `ui-proto/` (frozen
  prototype, superseded by the original-page adapter),
  `quality-screen/`, `speedtest/`, `tts-research/`, `tts-engine/work/`
  experiment scripts other than the named conversion set,
  `github-issue-potatoamika.md` (stale, owner-held draft),
  `VISION-REF.md`, `scheduler/PERSONA-EN.md`,
  `scheduler/STABILITY-AB.md`, dev configs `scheduler/config.*.json`.

## Upstream Python implementation — removed in commit 7 of this push

Scope set by the owner: remove the original Python implementation
from this fork's `main`, as one commit ("removed the original Python
implementation; kept UI sources and license notices").

**142 files**, by directory: `brain/` 30, `servers/` 20, `speech/`
18, `tests/` 44, `config/` 8, `vision/` 2, `messengers/` 3,
`scripts/` 4 (`download_sensevoice.py`, `setup_core.py`,
`setup_tts.py`, `setup_windows.ps1`), root launchers and config
examples 8 (`.env.example`, `launch.py`, `Start_OpenAmika.bat`,
`Stop_OpenAmika.bat`, `Stop_OpenAmika.ps1`,
`Tailscale_Serve_OpenAmika.bat`, `requirements-dev.txt`,
`settings.example.json`), `.github/workflows/tests.yml` 1, `IoT/`
1 (`patch_herdsman_windows.py`), `avatar/` 1
(`retarget_motions.py`), `vendor/tts-engine/` 2 (`main.py`,
`requirements-tts.txt`). The commit itself carries the full file
list. Checked during planning: `avatar/` in the fork contains no
model data or weights (`avatar/models/` holds only a README).

Explicitly **kept** upstream (not Python implementation, or
attribution obligations): `LICENSE`, `THIRD_PARTY_NOTICES.md`, `ui/`
(page sources), `ui/vendor/` (JS libraries — `patch.py`'s input),
`vendor/README.md`, `vendor/tts-engine/LICENSE`,
`vendor/tts-engine/README.md`,
`vendor/tts-engine/THIRD_PARTY_NOTICES.md` (only that folder's two
Python files are in the deletion scope), `avatar/models/README.md`,
`docs/` upstream documents in full (license-related files such as
`docs/IOT_DISCLAIMER.md` included),
`scripts/MekaHime_Amika.manifest.json` (a download manifest, not a
script), the non-Python files under `IoT/` (driver installers and
`mosquitto.conf`, per the owner's "IoT Python parts" scoping),
`.gitignore`, `.gitattributes`, `README.ja.md`, `README.zh-CN.md`
and `README_IF_YOU_ARE_AI_AGENT.md` (upstream documents left
untouched — candidates for a later owner decision, like the
non-license docs).

---

# Round 2 — 2026-11-10: Go adapter, slim deploy, on-device status

Second sync of PotatoAmika's own work, following the first push
above (base head before this round: `cf62682`, the round-1 deletion
commit). Same method: direct git push over SSH from the maintainer
machine, logical commits, author `not-hanjo-mei`.

## Commits

1. `1645abfbb3354959b0adf714447646a670626418` — "rewrite the Use
   page adapter in Go as a single static binary; drop the Node
   build". `ui-adapter/go/**` added (8 files), `ui-adapter/README.md`
   rewritten for the Go build, `ui-adapter/adapter.js` removed (the
   Node implementation remains in git history at `cf62682`).
2. `13e2e4266c039b3a0344510d9a74041069752064` — "slim the deploy
   package: public models pulled at install time, Go adapter, four
   install fixes". `deploy/deploy.sh` replaced by the current slim
   version, `deploy/assemble_stage.py` updated, `deploy/build_slim.sh`
   added. `deploy/live2d-files.txt` and `deploy/scheduler.json.tmpl`
   are unchanged (verified identical to the current package's).
3. This commit — root `README.md` status update (our section only;
   the upstream text below the marker line is byte-identical to
   round 1, verified by hash before and after) and this manifest
   section.

## What changed and why

- **Adapter, Node → Go.** The Use page adapter was rewritten in Go
  (standard library only, CGO off, self-contained WebSocket) as a
  single static binary — aarch64 8,812,626 B, x86_64 9,128,191 B.
  Steady-state RSS ≈ 13 MB on the preview machine vs ≈ 65 MB for
  the Node build (dev machine 8.4 MB vs 42.5 MB). Behaviour is
  unchanged: same endpoints and environment knobs, HTTP acceptance
  suite 39/39 on both implementations, one conversation round
  compared event-for-event, and the `/api/tts_say` streaming
  keepalive + result-cache semantics (built for proxies that kill
  idle responses) preserved. Only the compiled binaries are
  deployed; this repository carries the source, never the binaries.
  The phone still runs the Node build installed earlier; swapping
  it on-device awaits the owner's nod.
- **Deploy package, slim generation.** The current package is
  `potatoamika-slim.tar` — 249,712,640 B, sha256
  `10e1d2790ff65685e3f7a00fbdf6988627d7fff145270f833278b0d4fb5198cd`,
  112 files (the tarball itself does not enter this repository, as
  before). The public models are no longer in the tar: brain and
  embeddings GGUFs and the three ASR models are downloaded by
  `deploy.sh` at install time from hash-pinned revisions (Minico
  pinned to `eee16357` — upstream `main` has since changed the
  file; a hash-matching brain GGUF already in the home directory is
  reused instead of downloaded), alongside the vision GGUFs and the
  Live2D models the installer already fetched. Four defects found
  during the first on-device install are fixed in the assembly
  sources (a missing TTS encoder graph in the file list, the vision
  step's missing directory, two pid-file writes, and the brain-GGUF
  reuse path now also checking `~/models/`). The adapter step now
  starts the Go binary, and the installer's nodejs step is gone —
  no JavaScript runtime is installed; onnxruntime remains (the ASR
  host and the ORT TTS fallback use it).
- **Status.** PotatoAmika was installed on the target Xiaomi Mi 4c
  on 2026-11-10 and passed the installer's full self-test there
  (conversation round trip, speech recognition, standalone
  synthesis, model listing); the brain runs at four threads. The
  root README's PotatoAmika section now says so.

## Round-2 delta inventory

| Path | Bytes | Change |
|---|---|---|
| `ui-adapter/go/avatar.go` | 9112 | added |
| `ui-adapter/go/dialog.go` | 13041 | added |
| `ui-adapter/go/go.mod` | 36 | added |
| `ui-adapter/go/main.go` | 14068 | added |
| `ui-adapter/go/say.go` | 4322 | added |
| `ui-adapter/go/server.go` | 21775 | added |
| `ui-adapter/go/tz.go` | 8688 | added |
| `ui-adapter/go/ws.go` | 5745 | added |
| `ui-adapter/README.md` | 2812 | rewritten |
| `ui-adapter/adapter.js` | 42678 | removed |
| `deploy/assemble_stage.py` | 5659 | updated |
| `deploy/build_slim.sh` | 2251 | added |
| `deploy/deploy.sh` | 24281 | replaced (slim version) |
| `README.md` | — | our section only (see above) |

## Publication edits (round 2, staged copies only; originals untouched)

- `ui-adapter/go/main.go` — the preview-machine default directories
  replaced by paths relative to the adapter binary (`pristine/`,
  `patched/`, `models-live2d/`, `static/`, `data/`), mirroring the
  Node build's staged defaults from round 1; `deploy.sh` sets all
  five environment variables explicitly, so installed behaviour is
  unchanged. This is the only deviation from the build-verified
  source: the maintainer machine has no Go toolchain (the binaries
  are cross-built on the build host), so the staged copy is shipped
  with the edit confined to that one default-path block.
- `deploy/assemble_stage.py` — source/stage roots made
  environment-overridable (`POTATO_SRC`, `POTATO_STAGE`) with
  neutral defaults, as in round 1 (the development copy had
  regressed to literal paths during the slim work).
- `deploy/build_slim.sh` — stage root made overridable via
  `POTATO_STAGE` with a `$HOME`-relative default.

## Red-line scan record (round 2)

Same criteria and methods as round 1, run over the round-2 delta
before the push and re-run over the remote tree after it:

| Check | Result |
|---|---|
| Weight/model formats (extension blacklist) | 0 hits |
| Binaries (`file` magic scan for ELF/Mach-O/PE over added files — the compiled adapter binaries in the development tree were not copied, source only) | 0 hits |
| Secrets (round-1 regex sweep) | 0 matches |
| Machine paths in code (`/home/hatch`, `/data/home`, preview/trial hostnames in `*.go *.py *.sh *.tmpl *.mod`) | 0 after the publication edits above |
| Persona filenames, `namedict.tsv`, `memory.json`, Live2D model files | 0 — none in the delta |
| Symlinks / UTF-8 | 0 / all pass |

---

## Round 3 — pre-push record (2026-11-11)

Base head before this push: `353bb49` (round 2). Scope: the TTS
kernel campaign of 2026-11-10/11, the deploy hardening that came out
of the on-device install, and the closing of the Mi 4c round.

Delta, by commit:

1. `engine/src/free_engine.cpp`, `engine/src/packed_core.h`,
   `engine/src/vits_kernels.cpp`, `engine/src/vits_kernels.h` —
   the five NEON kernel rounds (vits PackedConv NEON tile,
   register-window EXT taps for dilations 1/3/5, K11-d5 pos8
   window; NEON GEMV kept in source but gated off by default after
   measuring 24% slower than the compiler's output). The four
   check/host files (`main.cpp`, `tf_check.cpp`,
   `tf_check_free.cpp`, `vits_check_free.cpp`) were diffed against
   the development tree: their only delta remains the round-1
   publication edit, so they are unchanged in this round.
2. `deploy/deploy.sh`, `deploy/scheduler.json.tmpl` — package
   integrity guard in the extract step (a resident tarball whose
   hash differs from `PKG_SHA256` is skipped with a warning instead
   of being extracted over the live tree; this exact accident
   downgraded a live install on 2026-11-11), page-title self-heal,
   and post-render normalization of the decided thread counts
   (`tts_threads` 4, `brain_threads` 4; the template default for
   `brain_threads` is now 4 as well).
3. `docs/ORT-FREE.md` (sections 7–15: the kernel rounds and their
   verdicts), `ui-adapter/REMOVALS.md` (package-lineage addition),
   root `README.md` (status rewritten: the Mi 4c round is closed,
   failed on performance, development paused pending stronger
   hardware; the upstream tail is byte-identical to round 2,
   verified by hash from the marker line), and this manifest.

One publication edit in this round: a stale incident paragraph in
`docs/ORT-FREE.md` named the maintainer's private access domain and
a local recovery script that was never run; in the repo copy it is
replaced by the recorded outcome (the on-device full-graph re-run
was abandoned on memory grounds; correctness rests on the x86
bit-identical regression, tile-level maxdiff 0, and the QEMU
subset). The development tree keeps its original wording.

Diffed and unchanged this round: `scheduler/`, `asr/`,
`quantize/`, `ui-adapter/go/` (its only delta against the
development tree remains the round-2 publication edit),
`deploy/assemble_stage.py` and `deploy/build_slim.sh` (same).

Red-line scan over the round-3 delta: weight/model extensions 0,
ELF/Mach-O/PE magic 0 (all files text), secrets sweep 0, machine
paths in code 0, persona filenames / `namedict.tsv` / `memory.json`
/ Live2D files 0, symlinks 0, all files valid UTF-8.

# Round 4 — 2026-11-11: next target selected (Dimensity 8100)

## What changed and why

1. `docs/D8100-PORT.md` (new) — the port plan for the next hardware
   round: a locked-bootloader Dimensity 8100 handset, the whole
   stack in Termux userspace with NDK/bionic rebuilds, proot
   excluded per the standing project rule, a component-by-component
   build table, the risk order, and the unchanged success metric
   (same medium-sentence TTS RTF, target single digit).
2. Root `README.md` — status updated: the "paused pending stronger
   hardware" line is replaced by the selected next target, pointing
   at the plan. Both edits are inside the PotatoAmika section; the
   upstream tail is untouched.

No code changed in this round: the port does not exist yet. The
first de-risking step (a bionic test build of the TTS host) is in
progress locally and will land as its own round when it produces
something.

Red-line scan over the round-4 delta: no weights, models, persona
or Live2D content; no access domains or hostnames; no secrets;
both changed files are plain UTF-8 text.

# Round 5 — 2026-11-11: D8100 spike result

## What changed and why

1. `docs/ANDROID-SPIKE.md` (new) — the full record of the bionic
   test build of the TTS host: the free engine's core sources
   compile unmodified under the NDK, the only ORT coupling is one
   entry symbol in the host glue, a free-only binary links against
   bionic system libraries only, and the remaining production work
   is the host-glue split plus an on-device smoke test (about a
   day in total, estimated). Publication edits on the repo copy:
   two local scratch/toolchain paths generalized; nothing else
   altered.
2. `docs/D8100-PORT.md` — status updated and a spike-result section
   added: risk 1 (TTS host bionic build) is retired with the
   evidence above.

Still no production code in the repo from this round: the spike
ran on a scratch copy and the shipping sources are untouched.

# Round 6 — 2026-11-11: free-only build split (first port code)

## What changed and why

1. `engine/CMakeLists.txt` + `engine/src/main.cpp` — the
   `POTATO_FREE_ONLY` build option from the spike's remaining-
   work list: with the option on, the ORT engine path is
   compiled out of the TTS host and nothing links ONNX Runtime;
   default off, and the dual-engine build's regression numbers
   are unchanged (GPT 299/361, vits 0.959880). The repo copy of
   `main.cpp` keeps its published `--root` default ("."); only
   the free-only changes were applied to it, not the development
   tree's local default.
2. `docs/ANDROID-SPIKE.md` — the production split section
   appended (verification numbers, bionic binary inspection).
3. `docs/D8100-PORT.md` — closing update on the spike-result
   section: the split is done; only the on-device smoke test
   remains for the TTS host.

# Round 7 — 2026-11-12: Rubens verdicts and the Omni integration (first round on `rubens`)

## What changed and why

1. `engine/src/packed_core.h` + `engine/src/gemv_bench.cpp` (new) +
   `engine/CMakeLists.txt` — the SDOT GEMV promotion from
   RUBENS-OPT: with `POTATO_GEMV_I8` unset, SDOT CPUs now run mode B
   and mode A block-quantized at width 64 by default;
   `POTATO_GEMV_I8=0` remains the exact pre-promotion opt-out.
   `gemv_bench` is the tier's micro-bench, cited throughout
   RUBENS-OPT.md; it builds for the NDK target, unlike `kbench`.
2. `scheduler/` (`api.go`, `brain.go`, `config.go`, `servers.go`,
   `session.go`, `vision.go`) + `scheduler/frame_test.go` (new) —
   the LFM3-Omni integration: new config keys `brain_mmproj`
   (vision resident in the brain server; `/v1/look` answers in
   place, the unload/reload swap kept for the Minico fallback) and
   `brain_thinking` (`"lfm3"` sends `enable_thinking=false` and
   starts the server with `--reasoning off`); chat requests send
   `cache_prompt`; the persona's structure frame
   (status/tone/mode/reply) is extracted so only the reply field
   reaches TTS and history, in both the streaming and the
   whole-response paths, with `frame_test.go` covering the
   extraction.
3. `asr/gateway/main.go` — the gateway's backend is now the
   resident Omni brain: energy-VAD segmentation in the gateway,
   whole-segment WAV sent as `input_audio`. The `/ws/asr` contract
   toward the adapter is unchanged, and the legacy flags are
   accepted and ignored. `asr_host` and the Moonshine models stay
   in the tree as the fallback path.
4. `docs/` — new: `RUBENS-PROFILE.md`, `RUBENS-ACCEL.md`,
   `RUBENS-OPT.md` (including the 2026-11-12 on-device A/B
   results), `RUBENS-NPU-SPIKE.md` (including the on-device NNAPI
   verdict), `LFM3VL-SPIKE.md` (including Addendum 3: the
   6.2 tok/s generation figure corrected to a measured 13.5, and
   the five-config thread/pinning sweep — `brain_threads` 4
   unpinned is the device optimum), `OMNI-INTEGRATION.md`.
   Updated: `API.md` (resident-mode `/v1/look`), `ASR.md` (§9,
   gateway backend), `SCHEDULER.md` (Omni section).
5. `README.md` — status section only: the rubens verdicts replace
   the "next target selected" text. The upstream tail below the
   PotatoAmika header is byte-identical (marker-to-EOF sha256
   unchanged by this round).

## Publication edits (staged copies only; originals untouched)

- `docs/RUBENS-NPU-SPIKE.md` — the maintainer's private SSH domain
  in the reachability paragraph replaced by a generic description
  of the route.
- `docs/LFM3VL-SPIKE.md` — the same domain, in the follow-up
  addendum's route and probe lines, generalized the same way.
- `docs/RUBENS-ACCEL.md` — the device's Termux UID value dropped;
  the sentence keeps "Termux app UID".
- `docs/ASR.md`, `docs/SCHEDULER.md` — the cross-reference to
  OMNI-INTEGRATION.md normalized to the sibling filename for the
  flat `docs/` layout.
- `asr/gateway/main.go` — gofmt applied (comment reflow and field
  alignment only; no logic change).

## Held out / reverted

- `scheduler/config.dev.json` (Omni profile) and
  `scheduler/config.minico.json` (Minico fallback) stay
  unpublished, per round 1's selection excluding dev configs:
  both carry maintainer-machine absolute paths. The profiles ship
  as code (`config.go` keys and comments) and documentation
  (OMNI-INTEGRATION.md, SCHEDULER.md); the installed config
  surface remains `deploy/scheduler.json.tmpl`.
- NPU spike sources (`tts-engine/accel/`: `fold_wn.py`,
  `nnapi_runner.cpp`, run scripts) and the LFM3VL spike harness
  (`lfm3vl-spike/`) stay in the working tree this round; their
  reports ship. The converted models and the runner binary are
  red-line material regardless.
- `tts-engine/host/src/dec_probe.cpp` and its CMakeLists target —
  an NPU-spike throwaway that proved useless and is referenced by
  nothing shipped: reverted in the working tree itself (file
  deleted, target removed) rather than published.
- `tts-engine/host/src/vits_bench.cpp` — an unwired dev bench (no
  CMake target, no report citations): held out.
- No `deploy/` or `ui-adapter/` changes: the published copies
  already carry the round-2 slim and publication state, and the
  working-tree copies differ only by being behind it.
- `docs/ORT-FREE.md` and `docs/ANDROID-SPIKE.md` unchanged: their
  working-tree deltas are exactly the spots earlier rounds
  generalized in the published copies (the ORT-FREE paragraph
  still names the private access domain in the original; the
  published rewrite stands).

## Red-line scan record (round 7)

Scan over the round-7 delta (added lines and new files, round-1
method): weight/model extensions 0, ELF/Mach-O/PE magic 0 (all
files text), secrets sweep 0, private identifiers (access
domains, machine paths, device UID) 0 after the publication edits
above, persona filenames / `namedict.tsv` / `memory.json` 0,
symlinks 0, all files valid UTF-8. Sanity checks from the repo
copies: `engine/src/gemv_bench.cpp` compiles; scheduler `go vet`
clean and `go test` passes (frame tests included); the ASR
gateway builds.
