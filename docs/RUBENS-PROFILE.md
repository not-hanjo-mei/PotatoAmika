# Rubens profile — Dimensity 8100 (Redmi K50, Termux)

Date: 2026-11-11. Device: Redmi K50 (22041211AC), SoC MT6895Z/TCZA
(Dimensity 8100), Android 14, Termux userspace, free-only bionic
`tts_host` (sha256 7b4971e3…), root `~/amika-smoke`.

Measurement discipline: every number in sections 1–3 comes from ONE
detached on-device session (script `~/rubens-sweep.sh`, run
2026-11-11 ~15:58–16:07 +08), eight host invocations back-to-back in
the order listed. The short line produced the identical 60-token
sequence in all six of its runs and the medium line the identical
161-token sequence in both of its runs, so within-session comparisons
are clean. Cross-session variance on this phone is real: the same
short line at threads 4 measured 23,937 ms elapsed in the earlier
session (decode 297.7 ms/step) against 22,559 ms here (278.7 ms/step),
about 6% apart. Deltas under ~10% in this report are indicative, not
settled; the medium big-core run also ran last in the session, so
thermal order effects cannot be separated from its result.

CPU map (from `/sys/devices/system/cpu/cpu*/cpufreq/cpuinfo_max_freq`):
cpu0–3 = 2,000,000 kHz (Cortex-A55), cpu4–7 = 2,850,000 kHz
(Cortex-A78). `taskset` exists in Termux and pinning was accepted
(the pinned runs show the expected, very different behaviour).

## 1. Thread sweep — short line

Request (identical to the established baseline):
`{"text":"Hello, this is a test of the free engine.","tag":"neutral1","seed":42}`
60 steps, audio 2.28828 s in every run. All times ms.

| config | elapsed | encoder | first | decode | decode/step | vits | deess | rss MB |
|---|---|---|---|---|---|---|---|---|
| threads 2 | 19,104 | 113 | 1,524 | 11,505 | 191.8 | 5,830 | 111 | 219 |
| threads 4 | 22,559 | 125 | 1,170 | 16,722 | 278.7 | 4,412 | 109 | 221 |
| threads 6 | 28,039 | 122 | 1,873 | 21,474 | 357.9 | 4,442 | 109 | 223 |
| threads 8 | 26,737 | 127 | 2,523 | 19,412 | 323.5 | 4,537 | 109 | 225 |

Decode gets *slower* as threads are added (per-step 191.8 → 278.7 →
357.9 → 323.5). The decode GEMV is not compute-starved on this SoC;
extra threads buy synchronization and migration, not throughput.
Threads 2 is the best unpinned configuration measured, 15% faster
overall than the current threads-4 default. Vits is essentially flat
from 4 threads up (4,412–4,537); its threads-2 figure (5,830) is the
only outlier.

## 2. Affinity — short line, threads 4, taskset

| config | elapsed | encoder | first | decode | decode/step | vits | deess |
|---|---|---|---|---|---|---|---|
| pinned big (cpu4–7) | 19,573 | 96 | 1,121 | 13,883 | 231.4 | 4,348 | 107 |
| unpinned (from §1) | 22,559 | 125 | 1,170 | 16,722 | 278.7 | 4,412 | 109 |
| pinned little (cpu0–3) | 39,608 | 221 | 6,318 | 13,321 | 222.0 | 19,270 | 436 |

Two facts worth more than the headline deltas:

- Big-core pinning at threads 4 recovers 13% of total elapsed against
  unpinned threads 4 (decode/step −17%), but still loses to plain
  threads 2 (19,104 ms).
- Pinned to the *little* cores, decode (222.0 ms/step) still beat
  unpinned threads 4 (278.7 ms/step), while vits exploded 4.4×
  (19,270) and first 5.4× (6,318). Unpinned decode time on this
  device is dominated by where the scheduler happens to put the
  pool threads and by migration, not by raw core speed; vits and
  first are the segments that genuinely need the A78s.

## 3. Medium caliber — NEW CALIBER (verbatim ttsscan text not recovered)

The ttsscan medium sentence used in the Mi 4c tests could not be
recovered verbatim: it is not in `ORT-FREE.md`, not in the
calibration kit (whose four requests are different sentences), and
no ttsscan script or request file exists in the local trees — the
tool lived on the Mi 4c. The runs below therefore use a fixed
replacement sentence and are a **new caliber**, not comparable
step-for-step with the Mi 4c medium figures (96 steps, 3.75925 s):

`{"text":"The quick brown fox jumps over the lazy dog, and then it runs down to the river to drink some water before the sun goes down.","tag":"neutral1","temperature":0.85,"top_k":20,"top_p":0.9,"seed":42}`
(sampling parameters follow the ttsscan caliber). 161 steps, audio
6.41538 s in both runs.

| config | elapsed | RTF | decode | decode/step | vits | first | rss MB |
|---|---|---|---|---|---|---|---|
| threads 4 | 49,462 | 7.71 | 38,270 | 237.7 | 9,337 | 1,424 | 255 |
| threads 4, big-pinned | 53,785 | 8.38 | 42,195 | 262.1 | 9,766 | 1,403 | 255 |

Medium RTF at the current default (threads 4) is **7.71** — single
digit, against 18.6–19.4 for the same stack on the Mi 4c (different
sentence, so the honest statement is "about 2.5× better", not an
exact ratio). The big-pinned medium run was 8.7% *slower* than
unpinned, opposite to the short line; it ran last in the session,
so sustained-frequency/thermal effects on the A78 cluster are a
live suspect and pinning must not be treated as a settled win
until it survives a cold-session A/B.

## 4. Source audit — `tts-engine/host/src`

### (a) Does decode GEMV use the SDOT dot product? No.

- Call chain: `free_engine.cpp` `k_matmul` (line 230) →
  `packed::gemm_a_w4`/`gemm_b_w4` → for M==1, `gemv_a_w4` /
  `gemv_b_w4` in `packed_core.h`.
- Dispatch, `packed_core.h` lines 286–299 (mode A; mode B is the
  same shape at lines 368–374): the only choices are
  `gemv_a_w4_neon`, taken only when the environment variable
  `POTATO_NEON_GEMV=1` is set *and* `cpu::features().neon`, and
  `gemv_a_w4_scalar` otherwise. The header comment (lines 48–53)
  records why: the hand-written NEON GEMV measured ~24% *slower*
  than scalar on the A57, because clang -O2 already autovectorizes
  the scalar loop.
- `gemv_a_w4_scalar` (lines 255–285) unpacks the W4 nibbles to
  float and accumulates in fp32 — the multiply-accumulate is
  floating point end to end. The string `sdot` does not appear
  anywhere in `packed_core.h`; there is no `vdotq` or any int8
  arithmetic in the GEMV at all.
- `cpu.h` line 61 does detect the feature (`sdot` from
  `HWCAP_ASIMDDP`) and the host prints it (this device reports
  `sdot=1` at startup, `cpu.h` lines 93–95), but the GEMV dispatch
  never reads `features().sdot` — only `.neon`.
- Contrast inside the same binary: the vits side *does* use it.
  `vits_kernels.h` lines 62–72 (`active_tier()`) select
  `KernelTier::Sdot` when `features().sdot` is set, and
  `vits_kernels.cpp` registers SDOT int8 conv kernels. On this
  device, vits convolution runs int8 SDOT while GPT decode runs
  fp32 FMLA.

### (b) Does the thread pool set CPU affinity? No.

A tree-wide search of `host/src` for `sched_setaffinity`,
`pthread_setaffinity` and `CPU_SET` returns zero hits. The pool
(`packed_core.h` lines 76–137) spawns plain `std::thread`s
(`Pool::init`, lines 99–105) and splits each kernel's work into
static contiguous chunks (`chunk()`, lines 140–144). Placement is
entirely the Android scheduler's decision — which §2 shows is
currently costing real time.

### (c) Top 3 decode optimization candidates, ranked

1. **An SDOT int8 GEMV tier for W4 decode** (new kernel +
   dispatch on `features().sdot` in `packed_core.h`). Decode is
   60–77% of elapsed in every run above, and its GEMV is the only
   major kernel still doing fp32 FMLA over weights that are
   already stored as 4-bit ints; the vits SDOT tier is the
   in-tree proof of the pattern. SDOT performs 16 int8 MACs per
   instruction against 4 fp32 FMLA lanes, so a 1.5–2.5× GEMV gain
   is plausible (estimate), worth roughly −30–45% on the decode
   segment and −25–35% on total elapsed. Costs/risks: activations
   need per-group int8 quantization each step (small next to the
   K×D MACs), the trajectory will change (different arithmetic),
   and decode GEMV streams ~47 MB of packed weights per step, so
   memory traffic is the floor. This is the biggest build of the
   three.
2. **Thread discipline: threads 2 as the default on this SoC,
   plus optional big-core affinity in `Pool::init`.** Measured,
   not speculative, and no kernel work: threads 2 beat threads 4
   by 15% overall (decode/step −31%); big-pinning threads 4
   recovered 13%. The pinning half carries the §3 caveat — it
   regressed on the medium run, plausibly thermals — so the
   threads-2 default is the confident part (expected −13–18%
   total) and pinning wants one cold-session A/B before adoption.
3. **Finer-grained work splitting in `chunk()`** (smaller
   granules with an atomic counter instead of one static
   contiguous range per thread). Evidence: the little-core vits
   blowup (§2) shows what a badly-placed static chunk costs;
   threads 6/8 regressing (§1) shows the pool's current split
   does not scale on heterogeneous cores; the Mi 4c rounds
   recorded the same suspicion for PCONV. Expected gain is the
   hardest to bound — likely a few percent to ~10% across decode
   and vits at threads 4, and it is what would let higher thread
   counts stop hurting.

## 5. GPU / NPU inventory (presence only)

- Vulkan loader: `/system/lib64/libvulkan.so` present. GPU is
  Mali: `/vendor/lib64/egl/libGLES_mali.so` (plus MEOW libs).
  Termux side: no `libvulkan*` in `$PREFIX/lib` — no Termux
  Vulkan package installed.
- OpenCL: `/vendor/lib64/libOpenCL.so` present (vendor-side).
- NPU (MediaTek APU / Neuron), all in `/vendor/lib64`:
  `libneuron_runtime.5.so`, `libneuron_adapter_mgvi.so`,
  `libneuron_graph_delegate.mtk.so`, `libneuron_platform.so`,
  `libneuron_wrapper.so`, `libapusys.so`, `libapu_mdw.so`,
  `libapu_mdw_batch.so`, APUWare HAL libraries
  (`vendor.mediatek.hardware.apuware.apusys@2.0.so`, `@2.1.so`,
  `hmp@1.0.so`, `utils@2.0.so`, `libapu_mdw…` family).
- NNAPI: the runtime lives in the APEX on Android 14 —
  `/apex/com.android.neuralnetworks/` exists and its `lib64`
  contains `libneuralnetworks_sl_driver_mtk_prebuilt.so`;
  `/system/lib64` holds only `libneuralnetworks_packageinfo.so`.

Presence is not a Termux route: every one of these sits behind
the Android app sandbox. A Termux userspace process has no
supported path to the Neuron runtime or NNAPI (both expect an
APK context with the NDK/vendor SDK), and the vendor OpenCL is
in the same position the Mi 4c probe left it — the libraries
exist, a usable from-Termux compute path does not, yet.

## Bottom line

The port already runs at medium-caliber RTF 7.71 (threads 4) on
this device. The cheapest measured win is configuration, not
silicon: threads 2 (and possibly big-core pinning, pending a
cold A/B). The largest remaining lever is the one the audit
found unpressed: GPT decode still runs an fp32 GEMV on a chip
whose int8 dot product the vits half of the same binary already
uses. GPU/NPU offload has no userspace path from Termux today
and should not be planned around.
