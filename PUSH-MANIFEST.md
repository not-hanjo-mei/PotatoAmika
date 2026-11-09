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
