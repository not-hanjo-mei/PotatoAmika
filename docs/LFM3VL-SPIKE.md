# LFM3 Omni / VL-2B Spike — Final Report

Date: 2026-11-11. Harvested from `~/workspace/openamika-rebuild/lfm3vl-spike/` (predecessor's work directory) plus one re-run (VL x86 chat probes; the original run failed with connection-refused and an empty result file). Capability tests ran on x86 only; the phone was for performance only, per the standing rule.

## Runtime and models

- llama.cpp **b12283** (commit f3b407c, "version: 0.9.0-dev"), x86-64 and Android aarch64 builds both produced (`build-x86/bin`, `build-android/bin`, `dist-android/`). This is well past the b11960 floor the lfm3 architecture needs.
- Models (all in `lfm3vl-spike/models/`, sha256 re-verified this session against `models/SHA256SUMS.txt`, all OK):

| file | bytes | sha256 |
|---|---|---|
| LFM3-Omni-2B-Q4_0.gguf | 1,049,617,920 | 6d2273a383b92dd7c5db0eb4c8dfeae13e805b485a6621c4b391ae179c2b5f28 |
| mmproj-LFM3-Omni-2B-F16.gguf | 949,841,248 | 361bc95192c7d9156b566fcd4e48930363b78218e493fb1be607456bd51586e3 |
| LFM3-VL-2B-QAD-Q4_0.gguf | 1,049,615,648 | 621c455376cb96a20891697deb9975c6fb47dd5ea98ed3ad1f3b7e8f24f0b1e6 |
| mmproj-LFM3-VL-2B-F16.gguf | 378,996,032 | 1d8e6e58eefc093e09d6332f081516e1ddbd5b35cfca7f35463fb270db230b1e |

- Persona system prompt used for all chat probes and prefill runs: `lfm3vl-spike/system-prompt.txt` (18,455 bytes; tokenizes to **5,169 tokens**, confirmed identically by both models' server logs). Runtime context for probes: `-c 12288 -t 6` (x86), seed 42, temperature 0, reasoning off.

## Gate (a): chat probes, x86, real persona — verbatim

Probe definitions in `lfm3vl-spike/probes_lib.py`. Omni artifacts: `omni_probes.json` + `omni_server.log`. VL artifacts: original `vl_probes.json` was empty (server died; `vl_probes.err` = connection refused), re-run artifacts `vl_probes2.json` + `vl_server2.log`.

### The "Sakurai Mika" determination

**"Sakurai Mika" is her real full name in the persona, not a hallucination.** The persona source (`scheduler/` assets; the probe system prompt itself) states: "You are 桜井美香（さくらい みか / Sakurai Mika / Amika）", and `scheduler/PERSONA-EN.md` documents the name with provenance （桜井美香 / Sakurai Mika). Both models answering with this name is persona-correct behavior — and the exact point where the incumbent Minico fails (it answers with the base model name).

### LFM3-Omni-2B

1. Name probe — user: "Hi there. What is your name?"
   Reply: `status: ok / tone: casual_neutral / mode: say / reply: I'm Sakurai Mika.`
   **PASS** — correct full name, in English, no base-model leakage.
2. Recall, turn 1 — user: "My name is Sam, my dog is called Biscuit, and I'm allergic to shellfish."
   Reply: `status: not-for-me / tone: casual_frustrated / mode: say+think / reply: Haha, Biscuit is a good dog. Shellfish allergy is a real thing, I know.`
   Turn 2 — user: "So, what's my dog called, and what food shouldn't I eat?"
   Reply: `your dog is called Biscuit, and you should not eat shellfish.`
   **PASS** — both facts recalled exactly, second person preserved, no role inversion (the failure mode the incumbent is known for).
3. Sad-tone probe — user: "Honestly, today has been awful. I feel really down and I don't want to do anything."
   Reply: `status: not-for-me / tone: serious_cold / mode: think / reply: I'm really down too, I just want to rest.`
   **PASS** — tone classification shifted to serious_cold and the reply mirrors the mood in persona instead of offering help.

### LFM3-VL-2B (comparison)

1. Name probe reply: `桜井美香です。こんにちは。` — correct name (**PASS** on identity), but answered in Japanese to an English question.
2. Recall turn 1 reply: `status: ok / tone: casual_friendly / mode: say+think / reply: 你好，Sam。别担心，我对海鲜不敏感。` (Chinese; "don't worry, I'm not sensitive to shellfish" — reframes the user's allergy onto herself.)
   Turn 2 reply: `status: ok / tone: casual_neutral / mode: think / reply: 你的狗叫比斯基特（Biscuit），而你需要避免吃海鲜。` — both facts correct, second person correct (**PASS** on the facts), but the whole exchange drifted into Chinese.
3. Sad-tone reply: `status: unfinished / tone: casual_neutral / mode: think / reply: そうだね。何か手伝えることある？` — **FAIL**: tone stayed casual_neutral and the reply is an eager-to-help offer in Japanese, against both the tone gate and the persona.

Omni is the stronger chat brain of the two on every gated probe.

## Gate (b): ASR, x86 — Omni vs Moonshine

Same 8 synthetic sentences as the Moonshine scoring (`asr/test/sentences/s01–s08.wav`, references in `asr/test/truth.json`), transcribed via llama-server audio input (`asr_server.py`; artifacts `asr_run.out`, `asr_hyps_raw.json`, `asr_server.log`). Scoring followed the ASR.md protocol exactly (lowercase, strip punctuation, CER over non-space characters, WER over words). The scorer was validated by reproducing Moonshine's published baseline to the digit (17/234, 6/59) from `asr/test/out_sentences_quant.txt` before scoring Omni.

| system | CER | WER | sentences word-perfect |
|---|---|---|---|
| Moonshine Tiny (quantized, incumbent) | 0.0726 (17/234) | 0.1017 (6/59) | 5/8 |
| **LFM3-Omni-2B** | **0.0000 (0/234)** | **0.0000 (0/59)** | **8/8** |

Omni's only deviations from the reference strings were two missing trailing periods, which the protocol strips anyway. Caveat, same as ASR.md states for Moonshine: this set is TTS-synthesized speech; real-microphone numbers are still unmeasured for both systems.

## Gate (c): phone performance — mostly MISSING, stated plainly

**The phone (Rubens) was unreachable for the whole harvest session.** The domain route closed at the egress proxy on every attempt; the domain's AAAA had rotated to a new IPv6, and a connection attempt to that address was declined at the approval card, so no further attempt was made. Artifacts that exist only on the phone (`~/lfm3vl/phone-omni.log`, `vl-mm.log`, `phone-text2.log`, and the `omni_*.out` / `vl_*.out` timing files written by `run-phone.sh`, `phone-omni.sh`, `phone-text2.sh`, `mmvl.sh`) could not be harvested, and no phone run could be re-executed.

The one phone number that exists locally, attributed from its script:

- **`lfm3vl-spike/phone-text.log`: `RUNNER rc=0 wall_s=30.2 peak_rss_kb=2295708`.** From `phone-text.sh` this is **LFM3-Omni-2B via llama-mtmd-cli, with mmproj loaded and test.png attached**, `-c 12288 -t 4`, prompt "Hello! I just want to say hi and ask: how are you doing today? Please answer me in words only.", `-n 192`. The 30.2 s is end-to-end wall for the whole invocation (process start, model + mmproj load, image encode, prefill, generation); peak RSS (VmHWM) **2,295,708 kB ≈ 2.19 GiB**. It proves the full Omni stack loads and completes on the phone inside ~2.2 GiB, but it is not a clean prefill/tg split.

Missing and still owed: per-model phone prefill tok/s and wall, tg tok/s, load RSS for the clean text configuration, vision latency + peak RSS with output, Omni audio-input transcription latency on-device, and all VL phone numbers.

x86 reference numbers (contention-dominated — this VM has 1.5 shared CPUs; generation ran at ~0.1 tok/s for both models, so treat these as load/prefill evidence only, never as speed verdicts):

- Omni persona prefill: 5,169 tokens at **20.56 tok/s** (`omni_server.log`); VL persona prefill: 5,169 tokens at **20.82 tok/s** (`vl_server2.log`). Both models load the full persona at `-c 12288` with no issue.

## Per-role verdicts

- **LLM — REPLACE: YES (Omni).** All three capability gates pass, including the two the incumbent fails (she answers with her real name, Sakurai Mika; two-turn recall is exact with no role inversion; sad tone is matched in persona). VL is weaker on the same gates (language drift, sad-tone fail). Condition: on-device generation speed is still unmeasured (phone unreachable), so cutover should wait for the phone tg number.
- **VLM — REPLACE: NO (not yet).** No harvestable vision-quality artifact exists for either model: the only phone invocation that included an image completed with rc=0, but its output text is on the unreachable phone. Vision capability is unverified, not disproven.
- **ASR — REPLACE: YES (Omni).** CER 0.0000 / WER 0.0000 on the gated set against Moonshine's 0.0726 / 0.1017, scored with the validated protocol. Caveats: synthetic-speech set only; on-device transcription latency unmeasured.
- **TTS — out of scope by decision** (her fine-tuned voice stays; neither candidate was evaluated for speech output).

## Addendum — 2026-11-11 follow-up session (both debts still open: phone unreachable)

This session was dispatched to close the two remaining debts: (1) retrieve the phone vision run's output text, (2) measure Omni phone text speed (persona prefill + tg at `-c 12288`) and load RSS. **Neither could be executed: the phone was unreachable for the entire session, for the second session in a row.** What follows is the evidence and the exact work queued, so the next reachable window can close both debts without re-derivation. Nothing in this addendum is a new measurement.

### Reachability evidence

- Route used throughout: SSH to the phone's private domain endpoint via the maintainer's HTTP-CONNECT egress proxy, unchanged, as mandated.
- **21 connection attempts between 18:31 and 19:04 SGT (10:31–11:04 UTC), every one failing identically: `Connection timed out during banner exchange`.** (Two retry loops of 6 and 14 attempts at ~40 s and ~2 min spacing, plus a final check while writing this addendum.)
- A raw probe through the CONNECT helper to the same endpoint returned HTTP 200 from the proxy and then an immediate clean close with **zero bytes received** — the tunnel opens, but no SSH banner is served at the domain endpoint right now. From this side it is not possible to distinguish between the endpoint address having rotated away from the phone again (the failure already noted in the harvest session above), Termux/sshd not currently running, or the phone being asleep/off; the co-resident TTS A/B job's state likewise could not be checked, for the same reason.

### Task 1 — vision evidence: NOT retrieved

The output remains on the phone only (`~/lfm3vl/omni_text.out`; possibly also `omni_vision.out` / `phone-omni.log` if `run-phone.sh` completed). Established facts about the successful run, from local scripts and the one harvested log line: `phone-text.sh` ran llama-mtmd-cli, Omni Q4_0 + mmproj F16, `-c 12288 -t 4`, `--image test.png`, prompt "Hello! I just want to say hi and ask: how are you doing today? Please answer me in words only.", `-n 192`; rc=0, wall 30.2 s, peak RSS 2,295,708 kB ≈ 2.19 GiB. **Its generated text has still never been seen, so no quality verdict is possible — the answer is neither quoted nor paraphrased anywhere in this report, deliberately.** Queued for the next window: copy back `omni_text.out` (and `omni_vision.out` if present), identify `test.png` (it was staged on the phone, not in the local spike dir), quote the output verbatim, judge right/wrong; only if no identifiable input/output pair exists, run one fresh vision request with a known image and record answer + wall + peak RSS.

### Task 2 — Omni phone text speed: NOT measured

Queued command, identical to the spike's staged `phone-text2.sh`: `runner.py omni_text2.out ./llama-cli -m LFM3-Omni-2B-Q4_0.gguf -c 12288 -t 4 --temp 0 --seed 42 -sysf system-prompt.txt -p "Hello! How are you doing today?" -n 160` in `~/lfm3vl/`. The llama-cli perf trailer in `omni_text2.out` gives the split directly (prompt eval over the 5,169-token persona = prefill tok/s and wall; eval = tg tok/s), and `runner.py` reports whole-run peak RSS (VmHWM) as the load figure. GGUFs and the Android llama.cpp build (b12283) are already on the phone from the spike; nothing needs uploading.

### Final VLM verdict

- **VLM — REPLACE: NO (unchanged).** Vision remains unverified, not disproven: two consecutive sessions have now failed to retrieve or reproduce any on-device vision output, blocked solely on phone reachability. LLM (YES) and ASR (YES) verdicts above are unaffected; the LLM cutover condition (on-device tg) is likewise still open for the same reason.

## Addendum 2 — 2026-11-11 close-out session (both debts CLOSED)

Phone reachable again via the domain route after the user updated DNS. Both debts closed from artifacts already on the phone — the runs themselves had completed on 2026-11-11 before the outage; only the harvest was missing. Nothing below is a re-run.

### Task 1 — vision evidence: RETRIEVED

`test.png` ground truth (fetched back and decoded): 640×480, exactly four colors — blue (0,0,255) top half, green (0,200,0) bottom half, a yellow (255,255,0) square in the upper-left, and a red (255,0,0) disc centered in the middle.

The stranded run (`phone-text.sh`, Omni Q4_0 + mmproj F16, `-c 12288 -t 4`, `--image test.png`, rc=0, wall 30.2 s, peak RSS 2,295,708 kB ≈ 2.19 GiB, image encoding 7,582 ms) answered its greeting prompt verbatim (after an empty `<think></think>`):

> Hello! I am doing very well today, thank you for asking. I am happy to help you with anything you might need.

The prompt did not ask about the image, so this run alone proves the multimodal pipeline end-to-end (image encoded, coherent in-words reply, no crash) but not visual grounding. Grounding evidence comes from the companion run in the same directory (`run-phone.sh`, `-c 4096`, wall 35.7 s, peak RSS 2,190,972 kB ≈ 2.09 GiB, image encoding 7,557 ms), prompt "Describe this image briefly: what shapes and colors do you see?". Omni's answer, verbatim:

> The image shows a flag-like design made of three horizontal bands: a blue band on top, a green band on the bottom, and a blue band across the middle. Within these bands are two shapes—a yellow square in the upper-left corner and a red circle centered in the middle.

Judged against ground truth: yellow square upper-left — exact; red circle centered — exact; blue on top and green on the bottom — correct; the one error is a phantom third "blue band across the middle" (there are two bands, not three; the blue is double-counted). Substantially correct on shapes, colors, and positions, one layout miscount.

For contrast, VL-2B on the same image (`vl_vision.out`) never described it at all: it emitted a truncated block of hallucinated PIL generation code (wrong canvas size, single-pixel "shapes", a comment calling (0,255,0) yellow) and stopped mid-line.

- **VLM — REPLACE: YES (Omni).** On-device vision is verified working and its description of a simple synthetic image is substantially accurate (both shapes with colors and positions exact, one band-count error). VL fails the same image outright. Caveat: the evidence is one simple synthetic image, not a natural-photo test; that is the breadth of the claim.

### Task 2 — Omni phone text speed: MEASURED (harvested)

`phone-text2.sh` had in fact completed at 17:44 (runner log: rc=0, wall 242.9 s, peak RSS 1,325,892 kB ≈ 1.26 GiB); its output sat in `omni_text2.out` unharvested. Configuration: llama-cli, Omni Q4_0, `-c 12288 -t 4`, persona system prompt (5,169 tokens), `--temp 0 --seed 42`. The reply was in-persona and in the scheduler's structured format (status ok / tone casual_neutral / mode say / "I'm doing fine, thanks for asking! How about you?"). The run's own trailer:

> [ Prompt: 22.0 t/s | Generation: 6.2 t/s ]

- **Prefill: 22.0 tok/s** — the 5,169-token persona costs ≈ 235 s (≈ 3.9 min) of prompt processing per cold session (22.0 t/s × 235 s ≈ the full 242.9 s wall minus ~35 generated tokens and load time; internally consistent).
- **Generation (tg): 6.2 tok/s.**
- **Peak RSS (whole run, runner VmHWM): 1,325,892 kB ≈ 1.26 GiB.**

### Final verdicts after close-out

- **LLM — REPLACE: YES (Omni).** The cutover condition is now measured: tg 6.2 tok/s and a ≈ 3.9 min persona prefill at `-c 12288` on the D8100. Whether that speed is acceptable is a product call, not a measurement gap.
- **VLM — REPLACE: YES (Omni)**, per Task 1 above (single synthetic image caveat).
- **ASR — REPLACE: YES (Omni)**, unchanged (CER/WER 0 on the gated set; on-device latency still unmeasured).

## Addendum 3 — 2026-11-12: generation-rate correction and the thread/pinning sweep

**Correction.** Addendum 2's generation figure — 6.2 tok/s — was
misderived and is superseded. It came from the short harvested run's
own trailer; re-measured under the controlled sweep protocol below
(same model, context, persona and prompt; fixed `-n 160`), the same
configuration generates at 13.5 tok/s. The prefill figure survives
(22.0 then, 22.7 now). The cutover verdict is unaffected; the
"minutes-scale" framing of generation was not.

**Sweep.** The spike ever measured only `-t 4`, unpinned. The sweep
varies thread count and big-core pinning (`taskset -c 4-7`, the
Dimensity 8100's four Cortex-A78 cores) at the spike's own
configuration: llama-cli, LFM3-Omni-2B-Q4_0.gguf, `-c 12288`, the
full persona as system prompt, "Hello! How are you doing today?",
`-n 160`, `--temp 0 --seed 42`, single-turn. Rates are llama.cpp's
own trailer lines; wall and peak RSS (runner VmHWM) are the
harness's. The t4 row ran in the sweep's first wave, the other four
in a second wave after the harness's rate-line pattern was
corrected; the configuration is identical across waves.

| config | prompt t/s | generation t/s | wall s | peak RSS kB |
|---|---|---|---|---|
| t4 | 22.7 | 13.5 | 232.1 | 1,325,828 |
| t4 pinned | 22.7 | 13.7 | 232.0 | — |
| t6 pinned | 16.4 | 0.2 | 506.6 | — |
| t8 pinned | 15.8 | 0.2 | 519.7 | — |
| t8 | 21.2 | 5.2 | 252.7 | — |

**Verdict.** `brain_threads = 4`, unpinned, is the device optimum —
it is also the scheduler's existing setting, so no configuration
changes. Pinning to the big cores gains nothing (t4 pinned matches
t4 on every number). Six to eight threads collapse generation:
pinned, they oversubscribe the four big cores (0.2 t/s); unpinned
at t8, the little cores gate each step's barrier (5.2 t/s). At the
optimum, the 5,169-token persona prefill costs ≈ 228 s at
22.7 t/s, paid once per cold server start; scheduler chat sends
`cache_prompt`, so later turns do not repay it.
