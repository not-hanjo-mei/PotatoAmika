# PotatoAmika

**Status: the Dimensity 8100 round (branch `rubens`) has its verdicts — the stack runs in Android userspace, the TTS engine's SDOT GEMV tier is on by default, the NPU route is a measured dead end, and one Omni model now serves as brain, eyes and ears.**
PotatoAmika is an experiment, not finished software, so expect rough
edges, missing pieces and no support. The Mi 4c round closed as a
working port and a failed deployment (medium-sentence TTS at RTF
18.6–19.4 on the Snapdragon 808; the full record is
`docs/ORT-FREE.md`, sections 7–15). The same stack was then rebuilt
for a Redmi K50 (Dimensity 8100, locked bootloader, Termux
userspace — no container). The TTS host runs there with its SDOT
GEMV tier enabled by default on capable CPUs (mode B plus mode A
block-quantized at width 64); the on-device A/B against the
previous binary is performance-neutral (RTF 7.49–8.07 vs 7.54–7.86
at two threads), so the promotion stands on its numerics — record
in `docs/RUBENS-OPT.md`. The APU route was measured and closed:
NNAPI cannot run the 471-node vits decoder graph on this driver
stack (two accelerator shims fail at two different stages), so vits
stays on the hand-written CPU kernels (`docs/RUBENS-NPU-SPIKE.md`).
The brain changed shape: LFM3-Omni-2B passed the replacement gates
for chat, vision and speech recognition, and the scheduler now runs
it as one resident server — vision answers in place instead of the
unload/reload swap, the ASR gateway transcribes through it, and the
self-built TTS engine still does the speaking (Omni's built-in
voices are generic presets; the fine-tuned voice exists only here).
On-device sweep: `brain_threads` 4, unpinned, is optimal — prefill
22.7 tok/s, generation 13.5 tok/s, one ≈ 228 s cold persona prefill
per server start, then cached; pinning gains nothing and 6–8
threads collapse generation. Records: `docs/RUBENS-PROFILE.md`,
`docs/RUBENS-ACCEL.md`, `docs/LFM3VL-SPIKE.md`,
`docs/OMNI-INTEGRATION.md`.

**PotatoAmika** is a ground-up, fully local reimplementation of
[OpenAmika](https://github.com/zeekk0/OpenAmika)'s runtime. The original
targets Windows, an NVIDIA GPU and a Python stack of separate services.
This one targets a 2015 Xiaomi Mi 4c (Snapdragon 808, ~1.3 GiB of usable
RAM, Alpine Linux in a container): a 350M-parameter brain, a hand-written
C++ TTS engine with 4-bit weights, small local ears, and the original
Live2D page as the face. At runtime there is no Python, no Node and no
cloud call.

This repository is a fork of zeekk0/OpenAmika. Everything we built is at
the repository root, in the directories listed below. The original
Python implementation has been removed from this fork; what remains of
upstream is the Use page sources (`ui/`), the JS vendor libraries, the
license texts (`LICENSE`, `THIRD_PARTY_NOTICES.md`) and upstream's own
documents, which our documents now sit alongside in `docs/`.

## Layout

| Directory | What it is |
|---|---|
| `engine/` | Self-built TTS inference engine (C++17, no ONNX Runtime): static load-time execution plans for the five GPT-SoVITS graphs, a W4 weight-only packed format with fused dequantization, CPU-dispatched kernels (ARMv8.0 NEON baseline, x86-64 AVX2). Final on-device record: five NEON kernel rounds on the Mi 4c, medium-sentence RTF 18.6–19.4, ~262–320 MB resident. Design record: `docs/ORT-FREE.md`, `docs/STAGE*.md`. |
| `quantize/` | The conversion method: how the shipping weight packs are made from stock GPT-SoVITS fine-tune weights, locally. Start at `quantize/README.md`. |
| `scheduler/` | The orchestrator (Go, standard library only): owns the brain/embeddings llama.cpp servers and the TTS host; conversation turns, memory extraction and recall, tone classification, vision unload/reload for photo questions, and a standalone `POST /v1/say` synthesis endpoint. Contract: `docs/API.md`. |
| `asr/` | The ears: Moonshine Tiny (quantized) + Silero VAD in one C++ ONNX Runtime host with a streaming mode, plus a small Go WebSocket gateway. Record: `docs/ASR.md`. |
| `ui-adapter/` | Serves upstream's original Use page (`ui/` in this repo) against the scheduler. The adapter is a single static Go binary (`ui-adapter/go/`, standard library only; ~9 MB, steady RSS ~13 MB — the Node build it replaced used ~65 MB), so no JavaScript runtime exists anywhere in the stack. `patch.py` regenerates the three patched page files from the pristine `ui/` sources; what was removed from the page, and why, is listed in `REMOVALS.md`. Features this runtime definitively does not have are removed from the page, never faked. |
| `deploy/` | One-command install into the phone's container (`deploy.sh`). The package is slim: the public GGUF and ASR models are not in the tarball — `deploy.sh` downloads them at install time from hash-pinned revisions (the Minico brain GGUF is pinned to revision `eee16357`; upstream's `main` has since diverged), reuses a hash-matching brain GGUF already in the home directory when one is there, and fetches the Live2D models the same way (`live2d-files.txt`). Package assembly: `assemble_stage.py` + `build_slim.sh`. |
| `docs/` | PotatoAmika's engineering record: STAGE1–3 (TTS bring-up), ORT-FREE (dropping ONNX Runtime), VITS-W4 and DEESS (the voice output chain), SCHEDULER + API, ASR + FREE-CENSUS (why ASR stayed on ORT), BUDGET (the 1.3 GiB memory ledger), CONVO-TEST/2 (live conversation tests). These files sit alongside upstream's own documents in the same folder; upstream's files are unchanged. |

## What is NOT here — bring your own

No weights, models or persona files are in this repository, in any form:

- **Voice weights.** A GPT-SoVITS fine-tune (for Amika, MekaHime's
  `MekaHime_Amika` on Hugging Face) is the input to `quantize/`. The
  quantized packs that pipeline produces are derivatives of those
  weights and are likewise not distributed here — make your own from
  weights you have the right to use. MekaHime's weights and Live2D
  models are under MekaHime's personal-use license; use them only as
  that license allows.
- **Live2D models.** Also MekaHime's, same license. `deploy.sh` fetches
  them at install time using `deploy/live2d-files.txt` (per-file hash
  verification), or supply your own.
- **Persona files.** The character's persona texts are not included, in
  any language. The scheduler loads persona files from paths in its
  config; what you put there is yours.
- **GGUF models.** Brain (Minico-M1-1105), embeddings (Jina embeddings
  v5 text nano) and vision (LFM2.5-VL-450M) — exact files and sizes in
  `docs/BUDGET.md`. The installer downloads all three at install time
  from the pinned revisions in `deploy/deploy.sh`, or supply your own.
- **ASR models.** Moonshine Tiny + Silero VAD (both MIT) — see
  `docs/ASR.md`. The installer downloads them at install time from the
  pinned revisions in `deploy/deploy.sh`.

## TTS release scope: English

The engine source keeps its full G2P code, but this release ships data
for the **English** text path only:

- `engine/data/cmudict.tsv` — from the CMU Pronouncing Dictionary
  (freely redistributable; credit CMU).
- `engine/data/symbols.txt` — the phoneme symbol table, following the
  GPT-SoVITS v2 symbol set (GPT-SoVITS is MIT).

Chinese G2P dictionary data is **not** included, so out of the box the
engine speaks English text — which is also all PotatoAmika itself ever
speaks at runtime. One supplemental name dictionary used during
development (`namedict.tsv`) is withheld: its provenance could not be
verified to the standard this release requires. It is named in
`PUSH-MANIFEST.md`.

## License

The code added by PotatoAmika (`engine/`, `scheduler/`, `asr/`,
`quantize/`, `ui-adapter/`, `deploy/` and our documents in `docs/`) is
our own work in this fork of OpenAmika (MIT — see `LICENSE` and
`THIRD_PARTY_NOTICES.md` at the repository root, which remain in
force). Nothing here grants any rights to MekaHime's character, voice
weights, Live2D models or persona materials.

The original upstream README follows below, unchanged, for reference.

---

<div align="center">

<img src="docs/images/amika-avatar.png" alt="Amika" width="180">

# OpenAmika

**Hi! I'm Amika** — MekaHime's AI companion, open source

Amika is a Live2D anime girl who talks with you by voice and text in real time, remembers you, and lives with you
in your city. She runs on your own PC, with OpenRouter or fully local with Ollama. A hosted version runs at
[mekahime.com/amika](https://www.mekahime.com/amika).

[简体中文](README.zh-CN.md) | [日本語](README.ja.md) | **English**

[![Made with Love](https://img.shields.io/badge/Made%20with-Love-ff69b4)](https://www.mekahime.com/openamika)
[![GitHub](https://img.shields.io/badge/GitHub-zeekk0%2FOpenAmika-181717?logo=github)](https://github.com/zeekk0/OpenAmika)
[![Release](https://img.shields.io/github/v/release/zeekk0/OpenAmika?logo=github&color=d8b4e2)](https://github.com/zeekk0/OpenAmika/releases)
[![License: MIT](https://img.shields.io/badge/license-MIT-43b581)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20(Linux%20untested)-0078D6?logo=windows&logoColor=white)](#what-you-need)
[![Python 3.11](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3119/)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-MekaHime%2FMekaHime__Amika-ffcc4d)](https://huggingface.co/MekaHime/MekaHime_Amika)
[![Discord](https://img.shields.io/badge/Discord-MekaHime-5865F2?logo=discord&logoColor=white)](https://discord.gg/mekahime)
[![Website](https://img.shields.io/badge/website-mekahime.com-d8b4e2)](https://www.mekahime.com/openamika)
[![Try Online](https://img.shields.io/badge/Try%20Online-mekahime.com%2Famika-e58fb8)](https://www.mekahime.com/amika)
[![Tests](https://github.com/zeekk0/OpenAmika/actions/workflows/tests.yml/badge.svg)](https://github.com/zeekk0/OpenAmika/actions/workflows/tests.yml)
[![GitHub stars](https://img.shields.io/github/stars/zeekk0/OpenAmika?style=social)](https://github.com/zeekk0/OpenAmika/stargazers)

</div>

> 💖 If Amika makes your day better, give us a ⭐ on [GitHub](https://github.com/zeekk0/OpenAmika) — it helps more
> people find her! · Support the project: https://www.mekahime.com/donate

## What you need

- Windows 10 or 11 and an NVIDIA graphics card (RTX 20-series or newer, 8 GB of video memory or more; her camera
  eyes need 16 GB), with a recent driver
- [Python 3.11](https://www.python.org/downloads/release/python-3119/) (tick **Add python.exe to PATH** when you
  install it) and [Git](https://git-scm.com/download/win)
- About 45 GB of free disk. Setting her up downloads about 15–20 GB and takes 30–60 minutes
- An [OpenRouter](https://openrouter.ai) key for her language model (recommended: a few dollars a month), or
  [Ollama](https://ollama.com) with a model of 27B or more and a graphics card big enough for it (see
  [Which language model?](#which-language-model))

## Which language model?

Her ears, voice, memory and eyes always run on your PC. Her language model, the part that thinks and writes her
replies, runs one of two ways:

- **Semi-Local (recommended)**: through [OpenRouter](https://openrouter.ai) on your own key. In almost all cases
  this is the better choice: she uses the models she was tested with, answers fast, and it only costs a few dollars
  a month (about $1.50 if you talk 30 turns a day, about $5 at 100 turns a day).
- **Fully Local**: on your PC with [Ollama](https://ollama.com). Use a model of **27B parameters or more**; **70B or
  more** gives the full, reliable experience. Models smaller than 27B may be really bad: she loses her character,
  forgets what you told her and breaks her reply format. A local model is also slower: all her thinking takes turns
  on it, so after you interrupt her, or while she saves a memory or reflects, a reply can take 5–50 s.

A local model needs its own video memory on top of the ~6 GB her ears, memory and voice use (~10 GB with her eyes):
about 19 GB for a 27B model such as `gemma3:27b`, 24 GB for a 32B one such as `qwen3:32b`, and 50 GB for a 70B one
such as `qwen2.5:72b`. So one 32 GB card (RTX 5090) holds a 27B model next to all of her (a 32B one only with her
eyes off); a 24 GB card does not. A 70B model fits next to her only on a 72–96 GB workstation card. Otherwise run
Ollama on a second PC (the Setup page takes its address), or stay Semi-Local. Whichever model you run, pick one that
supports Chinese, since her prompts are written in it (`llama3.3:70b`, for one, does not). How to switch:
[Optional add-ons](#optional-add-ons).

## Set her up

Open PowerShell (a normal window, not as administrator) and run these lines one at a time. They put her in
`C:\OpenAmika`, a short folder that OneDrive does not sync:

```powershell
cd C:\
git clone https://github.com/zeekk0/OpenAmika.git
cd OpenAmika
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1   # her brain, ears, memory and eyes
envs\brain\Scripts\python.exe scripts\setup_tts.py                   # her voice, her Live2D model, the paper doll: type yes
.\Start_OpenAmika.bat                                                # start her
```

Her voice, her Live2D model and the paper doll are free under MekaHime's personal-use license (see
[Licenses](#licenses)): the `setup_tts.py` line shows it first and downloads her files only after you type `yes`.

Both setup commands end with a summary, one line per part: `OK` when it is done, `!!` with the fix when it is not,
and `--` for information only (for example: no OpenRouter key yet, your first visit asks for it). Running a command
again resumes and skips everything that is finished. When the `setup_tts.py` summary shows `OK voice MekaHime_Amika`
and `OK Live2D model Amika`, she has her voice and her body; `OK paper doll Host` is the stand-in for her helpers
([Her helpers](#her-helpers-hermes-and-nanomuse)).

## Talk to her

Double-click `Start_OpenAmika.bat` in the OpenAmika folder (in PowerShell: `.\Start_OpenAmika.bat`). One console
starts all of her, and your browser opens **http://localhost:53000** when she is ready: about a minute, a few more
the very first time. On your first visit she introduces herself and asks your name, your city and, if `brain\.env`
has none yet, your OpenRouter key. Then press the power button and talk, or type. Click the page once so the browser
lets her speak.

The light at the top right of every page is her health: green means everything works; click it to see what is not.
Double-click `Stop_OpenAmika.bat` (or press Ctrl+C in her console) to stop her. To check everything without starting
anything, run `envs\brain\Scripts\python.exe launch.py --check`: it ends with `result: ready`, or names what to fix.

To update her: stop her, run `git stash`, `git pull` and `git stash pop` in the OpenAmika folder (this keeps your
name, your city and your edits), then run the two setup commands again (they fetch only what changed).

## If something goes wrong

| what you see | what to do |
|---|---|
| `Python 3.11 was not found` | Install Python 3.11 (tick "Add python.exe to PATH"), open a new PowerShell window and run the setup again. |
| `!! voice MekaHime_Amika  not downloaded: her license was not accepted` | Run `envs\brain\Scripts\python.exe scripts\setup_tts.py` again and type `yes`. |
| `!! voice MekaHime_Amika  Hugging Face refused the download (HTTP 401)` | Her files are not public yet, or your `HF_TOKEN` is wrong. Set a read token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) in the same window (`$env:HF_TOKEN = "hf_..."`) and run the command again. |
| `download failed after 4 tries` or `Check the internet connection` | Run the same command again: it continues where it stopped. If huggingface.co cannot be reached from your network, set `$env:HF_ENDPOINT` to a Hugging Face mirror first; every file is still checked. |
| `does not see the GPU (CUDA)` | Update your NVIDIA driver and run the setup again. Without an NVIDIA card: `powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 --cpu`. |
| `The filename or extension is too long` | The folder's path is too long for Windows: move OpenAmika to a short folder such as `C:\OpenAmika` and run the setup there. |
| `settings.json not found` | Run `powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1` first. |
| `No Live2D model yet` on the stage, or `!! Live2D model` / `!! her voice` in `launch.py --check` | Run `envs\brain\Scripts\python.exe scripts\setup_tts.py`. |
| `no OpenRouter key yet` (`launch.py --check`, the setup summary) | Paste your key in the first-run wizard, or after `OPENROUTER_API_KEY=` in `brain\.env`. |
| `!! TTS (voice)  port 8002 answers but has no voice MekaHime_Amika` | Another voice server uses her port (OpenSakura's uses 8002 too). Stop it, or give her another port: change `8002` in `tts_url` and in `services.tts` (`--port`, `health`) in `settings.json`. |
| She answers in text but you hear nothing | Give her voice about 20 seconds after a start, then click the page once. The health light must show TTS green. |
| Her voice stutters or is very slow | Close other programs that use the graphics card, or turn her eyes off: `"vision": {"enabled": false}` in `settings.json`. |
| On Ollama she is out of character, forgets things, or her replies come out broken | The model is too small. Use one of 27B or more (70B or more for the full experience), or go back to OpenRouter: Setup → Brain → OpenRouter → "Switch back to the OpenRouter preset". |
| On Ollama she answers very slowly | If `ollama ps` shows a CPU/GPU split instead of `100% GPU`, the model does not fit in your video memory next to her. With `100% GPU`, slow replies after you interrupt her, or while she saves a memory or reflects, are normal for a local model. See [Which language model?](#which-language-model). |
| A helper's card on the Tools page is red, or she says she cannot reach Hermes or nanoMuse | Run `envs\brain\Scripts\python.exe launch.py --check` and look at its `agent` and `hermes key` lines; every message and its fix is in [docs/TOOLS.md](docs/TOOLS.md#when-it-does-not-work). |

Every part writes a log in `logs\`. For anything else, ask on [Discord](https://discord.gg/mekahime) or open an
[issue](https://github.com/zeekk0/OpenAmika/issues). Using an AI coding agent? Point it at
[README_IF_YOU_ARE_AI_AGENT.md](README_IF_YOU_ARE_AI_AGENT.md).

## Optional add-ons

- **Fully Local**: install [Ollama](https://ollama.com), run `ollama pull gemma3:27b` (17 GB; or another model of
  27B or more, see [Which language model?](#which-language-model)), then on the Setup page choose Brain → Ollama →
  Detect models → Use this model. No key needed. For Ollama on another PC, type its address there before Detect models.
- **Her helpers**: Hermes Agent or nanoMuse run errands for her, with the paper doll beside her on stage
  ([below](#her-helpers-hermes-and-nanomuse)). Any other MCP server works too, on the Tools page
  ([docs/TOOLS.md](docs/TOOLS.md)).
- **Smart home**: Zigbee plugs she can switch ([IoT/README.md](IoT/README.md)). It stays off until you read and
  accept the IoT safety statement ([docs/IOT_DISCLAIMER.md](docs/IOT_DISCLAIMER.md)) on the Tools page. She runs on
  language models and can make mistakes: never connect heaters, locks, valves or anything that could cause harm.
- **Text her** on Discord or Telegram ([docs/MESSENGERS.md](docs/MESSENGERS.md)).
- **Your phone**: `Tailscale_Serve_OpenAmika.bat` makes her reachable on your own devices over
  [Tailscale](https://tailscale.com).

## Her helpers: Hermes and nanoMuse

She can hand errands to a helper program on your PC, [Hermes Agent](https://hermes-agent.nousresearch.com) or
[nanoMuse](https://github.com/nano-muse/nanoMuse): it searches the web, uses a browser and works with files. She
decides whether to help, gives the helper the task after she has answered you, relays its questions, and tells you
the result in her own words. While a helper is switched on, MekaHime's paper doll stands at the bottom right of her
stage and plays the helper's side. Each helper runs its own model on OpenRouter, and the key she uses works for it.

**Hermes Agent**: install it from a PowerShell window. This form of its installer asks no questions; the lines
after it do the setup OpenAmika needs:

```powershell
& ([scriptblock]::Create((irm https://hermes-agent.nousresearch.com/install.ps1))) -NonInteractive
```

Then, in a new PowerShell window:

```powershell
hermes model                                    # choose OpenRouter, paste your key, then "Enter custom model name": z-ai/glm-5
$key = [guid]::NewGuid().ToString('N')          # a random key for its API server
hermes config set API_SERVER_ENABLED true
hermes config set API_SERVER_KEY $key
hermes gateway stop                             # a Hermes already running reads the key only when it starts again
```

**nanoMuse**: clone it into the `Nanomuse` folder, give it a `config.toml` and put your key in its own vault; the
commands and the file are in [docs/TOOLS.md](docs/TOOLS.md#set-up-nanomuse).

**Connect them.** In the OpenAmika folder, run `Copy-Item config\tools.example.json config\tools.json` (only if
there is no `config\tools.json` yet) and `notepad config\tools.json`. Set `"enabled": true` at the top and in the
helper's entry. For nanoMuse, `"NANOMUSE_LANGUAGE"` is the language it answers in (`"en"`; or `"ja"`, `"zh-CN"`).
For Hermes, copy its key with this line and paste it (Ctrl+V) as `"HERMES_API_KEY"`:

```powershell
(Select-String -Path "$env:LOCALAPPDATA\hermes\.env" -Pattern '^API_SERVER_KEY=(.+)').Matches[0].Groups[1].Value | Set-Clipboard
```

Save and start her (`.\Start_OpenAmika.bat`): she starts her helpers herself. On the Tools page the helper's card
turns green (`agent online`).

**Meet the paper doll.** Open the Use page: the doll slides onto her stage. Ask her, for example, "Can you have
Hermes check tomorrow's weather?". She answers, types the task in her speech bubble, and the doll takes over: it
thinks while the helper works, speaks the helper's steps in its own bubble, looks surprised when the helper has a
question or needs your OK (say yes or no, or click Approve on the Use page), and is happy when the result comes,
which she then tells you. The doll comes with `setup_tts.py` (`OK paper doll Host` in its summary).

Every step in detail, and what each message means: [docs/TOOLS.md](docs/TOOLS.md#her-helpers-set-up-hermes-agent-and-nanomuse).

## Licenses

- OpenAmika's code is **MIT** ([LICENSE](LICENSE)). Third-party parts keep their own licenses
  ([THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)); the Live2D Cubism Core in `ui/vendor/` is not MIT.
- **Amika herself is not MIT.** Her character, art, voice, Live2D model and their weights, and the paper doll that
  stands in for her helpers ([MekaHime/MekaHime_Amika](https://huggingface.co/MekaHime/MekaHime_Amika)), belong to
  MekaHime Inc. and are free under a personal-use license: personal use only, 18+, no redistribution or re-hosting,
  and no training other models on them. Two creator rules: earn on social media (videos, streams, ads, memberships,
  sponsorships) with no limit, crediting MekaHime.AI visibly; and fan merch up to US$1,000 in total, as an
  individual, made from fan art you recreated yourself (never an original image MekaHime published on its website
  or social media), with "MekaHime.AI" on the visible side and a small "Fan Art" label. Any other commercial use needs a
  license from MekaHime (hi@mekahime.com). The [LICENSE](https://huggingface.co/MekaHime/MekaHime_Amika/blob/main/LICENSE)
  on Hugging Face governs.
- Hermes Agent (MIT) and nanoMuse (GPL-3.0-or-later) are separate programs you install yourself; OpenAmika only
  talks to them.
- Built on [SenseVoice](https://github.com/FunAudioLLM/SenseVoice), [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS),
  [bge-m3](https://huggingface.co/BAAI/bge-m3), [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL),
  [pixi-live2d-display](https://github.com/guansss/pixi-live2d-display) with the
  [Live2D Cubism SDK](https://www.live2d.com/en/sdk/about/), [OpenRouter](https://openrouter.ai),
  [Ollama](https://ollama.com) and the [Model Context Protocol](https://modelcontextprotocol.io). Thank you.

## Support

Questions and bugs: [GitHub issues](https://github.com/zeekk0/OpenAmika/issues), hi@mekahime.com, or
[Discord](https://discord.gg/mekahime). Support the project: https://www.mekahime.com/donate

<p align="center">
  <a href="https://www.mekahime.com">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="docs/images/mekahime-logo-white.png">
      <img src="docs/images/mekahime-logo-card.png" alt="MekaHime" width="220">
    </picture>
  </a>
</p>
