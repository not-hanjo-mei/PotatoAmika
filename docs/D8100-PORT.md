# Next target: Dimensity 8100 — Android userspace port plan

Status: selected 2026-11-11. The plan below was written before
the work started; its first de-risking step has since been done —
see the spike result at the end of this document.

## Why this device

The Mi 4c round closed on a compute verdict: the Snapdragon 808's
Cortex-A57 cores (ARMv8.0, no dot-product instructions) leave
medium-sentence TTS at RTF 18.6–19.4 after five kernel rounds, and
the brain at about 9 tok/s with a cold persona prefill of roughly
five minutes. The next round needs a SoC from the class the original
estimates assumed. The device now selected:

- MediaTek Dimensity 8100: 4× Cortex-A78 at 2.85 GHz + 4× Cortex-A55,
  dot-product instructions, LPDDR5 (about 51 GB/s theoretical).
- 8 GB RAM, 128 GB storage. The resident stack measured about
  0.9 GB on the Mi 4c, so memory is not a constraint here.
- Bootloader locked. This is the defining constraint of the round:
  no root, no Magisk, no Droidspaces container. The whole stack
  must run in Android userspace.

## The route: Termux, native builds, no proot

Standing project rule: proot-class runtimes are excluded on
performance grounds. A proot-hosted Alpine would reuse the existing
deploy package almost unchanged, which makes it the tempting
shortcut; it stays excluded.

Every native component is therefore rebuilt for bionic with the
Android NDK:

| Component | Mi 4c build | D8100 build |
|---|---|---|
| scheduler, ui-adapter, ASR gateway | static Go binaries | unchanged — static Go runs in Termux as-is |
| llama.cpp servers (brain, embeddings, vision) | Alpine package build (musl) | NDK build (upstream supports Android) |
| TTS host (self-built engine) | C++ built for musl | NDK build — the one component with no upstream to lean on; de-risked first, see below |
| ASR host | C++ against Alpine's ONNX Runtime package | C++ against the official ONNX Runtime Android build |

Models, weight packs, reference packs and the rest of the deploy
data are architecture- and libc-independent and carry over
unchanged.

## Risk order

1. **TTS host bionic build.** The engine is self-contained C++
   (the free-engine path uses no ONNX Runtime), so this is expected
   to be a build-system exercise rather than a code port — but it
   was the only component where that was a guess, so it was
   test-built first, before any device-side work. The guess held;
   see the spike result at the end of this document.
2. **ASR host** against ONNX Runtime's Android build: same library,
   different packaging.
3. **Android background execution.** Termux wake lock plus
   battery-exemption handling, so the stack survives as an
   always-on service. Userspace-only means no boot autostart;
   a manual start after a reboot is accepted for this round.

## Success metric

Unchanged from the Mi 4c round's closing note: re-measure the same
medium test sentence's TTS RTF on the new SoC; the target is a
single-digit number. The second number to beat is the brain's cold
prefill of the 5.6k-token persona (about five minutes on the
Mi 4c).

## Alternatives considered and parked

- **Snapdragon 855 handset** (Kryo 485 — the class the single-digit
  estimate was originally based on): the Droidspaces route needs a
  replacement kernel, and the unit's screen is broken, so flashing
  and recovery work would be done blind. Parked until the screen
  is repaired.
- **RK3566 SBC**: native Linux, zero porting, always-on — but four
  Cortex-A55 cores answer a different question. Its NPU would
  require an RKNN conversion of the TTS graphs plus an int8
  re-listen; that is a separate project, not a port.
- **Snapdragon 430 handset**: Cortex-A53 only, no dot product;
  weaker than the Mi 4c's big cores for this workload.
- **Snapdragon 8 Gen 2 handset**: the maintainer's daily phone,
  not a test device.

Licensing and publication rules are unchanged: no weights, Live2D
models or persona files in this repository, in any round.

## Spike result (2026-11-11): risk 1 retired

The TTS host was test-built for bionic on a scratch copy (NDK,
aarch64-linux-android, API 26); shipping sources untouched. The
free engine's core sources compile unmodified — the free path
never includes ONNX Runtime headers — and its runtime CPU
dispatch reads `getauxval(AT_HWCAP)`, which bionic provides, so
the dot-product kernels will be selected on the Cortex-A78
cores. The host's main file is the only ORT coupling (it
constructs the ORT environment before the engine branch); a
free-only binary linked against bionic system libraries alone,
using a trap stub for that single entry symbol. Stripped size
1,434,088 B against the musl phone build's 822,424 B, the delta
being the statically linked libc++. The binary has not run
anywhere yet: the build machine has no bionic runtime, so first
execution happens on the device, inside the smoke test.
Remaining tts_host work is mechanical: split the ORT path out
of the main file behind a build option (about half a day,
estimated), then the on-device smoke test (about half a day,
estimated). Full record: `docs/ANDROID-SPIKE.md`.

Update (2026-11-11, later): the split is done. `tts_host` now
has a `POTATO_FREE_ONLY` CMake option that compiles the ORT
path out (default off; the dual-engine build is unchanged and
its regression numbers did not move). The free-only bionic
binary was rebuilt and re-inspected: no ORT symbols, bionic
system libraries only, 1,099,728 B stripped. What remains for
the TTS host is only the on-device smoke test. Record:
`docs/ANDROID-SPIKE.md`, production split section.
