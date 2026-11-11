# Rubens optimization — Dimensity 8100 (Redmi K50, Termux)

## Round 1: the SDOT GEMV knife (2026-11-11)

Target: GPT decode's W4 GEMV (`host/src/packed_core.h`), which ran
fp32 end to end — mode 0 (91% of packed MACs) unpacking nibbles to
float in `gemv_a_w4_scalar`, mode 1 (8%) dequantizing each row to an
fp32 temp in the `gemm_b_w4` block path — while `cpu.h` already
detected SDOT on this device (`sdot=1`) and nothing consumed it.

Correction to RUBENS-PROFILE.md §4(a), found while auditing for this
round: the profile says the vits side "registers SDOT int8 conv
kernels" and runs int8 SDOT. The source does not bear that out —
`active_tier()` in `vits_kernels.h` *labels* the tier `Sdot`, but no
code branches on it and no SDOT instruction existed anywhere in the
tree before this round; vits speed on Rubens is the NEON tile on
A78 cores. The GEMV tier below is the first SDOT arithmetic in the
codebase.

### Kernel design and dispatch

One quantization contract for both modes, defined by shared scalar
preludes in `packed_core.h` (`quantize_i8`, `gemv_a_i8_prelude`,
`gemv_b_i8_prelude`): the activation side is quantized to int8,
symmetric, scale = amax/127 per group; the W4 nibbles are consumed
as exact int8 values; weight scales and zero points stay in fp32
correction terms with the same algebra as the fp32 GEMV. Because
quantization happens once in shared code and the accumulation is
exact integer arithmetic (order-free, no overflow: worst case
2048 × 127 × 15 ≈ 3.9M), the NEON executor and the portable
executor are **bit-equal by construction** — verified on device,
max diff 0.000e+00 on every bench shape, both modes.

- Mode 1 (`gemv_b_w4_i8_neon`): the stored rows are the contraction,
  so this is the true dot-product form — per 64-wide group, X is
  quantized once per call, then each row runs
  `y += s[g] * (sx[g] * idot_g − z[g] * sumx[g])` with `idot_g` the
  exact integer dot (4 SDOTs per group). K%64 tail keeps the fp32
  formula, as in `gemv_b_w4_neon`.
- Mode 0 (`gemv_a_w4_i8_neon`): the stored layout contracts over
  rows, so no dot product exists along the packed bytes; the tier
  runs the fp32 kernel's rank-1 form in integers — per output group,
  the per-row factor `f[i] = X[i] * sc[i, g]` is quantized with ONE
  scale over the whole K, then `acc[j] += fq[i] * nib(i, j)`
  accumulates in int16 chunks of 16 rows (16 × 127 × 15 = 30480 <
  32767, overflow-free) promoted to int32. Remember this paragraph;
  the gate section is about it.
- SDOT is emitted via inline asm (`.arch armv8.2-a+dotprod` +
  `sdot`): this NDK clang's `arm_neon.h` does not declare
  `vsdotq_s32` at the project's baseline `-march=armv8-a` under any
  `-march` spelling (checked: `__ARM_FEATURE_DOTPROD` is defined,
  the intrinsic still is not declared). The instruction executes
  only behind the runtime `cpu::features().sdot` dispatch; all other
  codegen stays at the baseline ISA.
- Selection: `POTATO_GEMV_I8` — unset/`0` = off (the default, see
  the gate verdict), `1` = on where SDOT is reported, `sim` = force
  the portable integer executor on any arch (exists so the tier's
  numerics can run the standard x86 gates). `POTATO_GEMV_I8_ONLY=a|b`
  is a diagnostic split used for the gate attribution below.
  Dispatch sites: `gemv_a_w4` (mode 0, D a whole number of 64-wide
  groups) and `gemm_b_w4` (mode 1, M == 1, K ≥ 64), ahead of the
  existing NEON-GEMV and scalar floors, mirroring their style.
- The first graph's M>1 PACKED GEMM was NOT converted: prefill runs
  the block-dequant `gemm_a_w4`/`apply_tile` path, whose structure
  (fp32 weight tiles × X rows) does not reuse the GEMV machinery
  cheaply, and prefill is ~1.2–1.5 s of a 24–50 s request. Decode
  was the deliverable; noted for a later round.

### Micro-bench (device, `src/gemv_bench.cpp`, before any wiring)

Through the production dispatchers; per-step figures are the four
mode-0 shapes of one decoder layer ×24 plus mode 1 ×24. The bench
also cross-checked executors against a double-precision dequantized
reference: fp32 max err ~1e-05; i8 error is activation-quantization
noise, mode A RMS ≈ 3× mode B's (whole-K scale vs 64-wide groups) —
the bench saw the gate result coming before the gate did.

| variant (threads 4) | mode A ms/step | mode B ms/step | packed total |
|---|---|---|---|
| fp32 floor (production) | 104.1 | 35.1 | 139.2 |
| fp32 NEON GEMV (opt-in) | 90.2 | 12.3 | 102.5 |
| i8 portable ("sim") | 77.7 | 12.2 | 89.9 |
| **i8 NEON / SDOT** | **58.6** | **7.3** | **65.9** |

| variant (threads 2) | mode A ms/step | mode B ms/step | packed total |
|---|---|---|---|
| fp32 floor (production) | 108.3 | 14.9 | 123.2 |
| fp32 NEON GEMV (opt-in) | 83.3 | 5.5 | 88.8 |
| i8 portable ("sim") | 94.7 | 10.5 | 105.2 |
| **i8 NEON / SDOT** | **62.3** | **6.1** | **68.4** |

The kernel gate — SDOT must beat the scalar GEMV on device-class
hardware before integration — passed decisively: mode A 1.78×,
mode B 4.8× against the production floor at threads 4 (1.74× /
2.4× at threads 2). Per-shape mode A at t4, floor → SDOT (GMAC/s):
512×1536 0.67→1.21, 512×512 0.61→0.82, 512×2048 0.73→1.17,
2048×512 0.81→1.80.

### Correctness gates (x86; sim executor = the tier's numerics)

`tf_check_free --mode packed --ref host/logits_packed_x86.bin`,
established values in brackets.

| configuration | tf_agree [/361] | ref_argmax [/361] | max logit diff |
|---|---|---|---|
| baseline tree (unmodified) | 299 | 361 | 8.86917e-05 |
| new code, default env | 299 | 361 | 8.86917e-05 |
| i8 tier, both modes | **296** | **339** | 2.17202 |
| i8 tier, mode A only | **293** | **335** | 1.98015 |
| i8 tier, mode B only | 299 | **360** | 0.151795 |

`vits_check_free` on the new code: corr 0.959880, unchanged (the
tier does not touch the vits path).

**Verdict: the tier does not pass the gate and is not integrated.**
The full tier misses both counts. The attribution is clean: mode A
alone is worse than both modes together (293/335 vs 296/339) — its
single activation scale over the whole contraction (K = 512–2048)
is the dominant error, exactly the bench's error split; mode B's
64-wide groups are nearly clean. Mode B alone holds teacher
agreement at the established 299/361 and cuts the max logit diff
14× versus the full tier, but flips one ref argmax (360/361, a
near-tie step) — one short of the strict gate, so it is not enabled
by default either. The code stays in the tree as an opt-in
(`POTATO_GEMV_I8=1`, `ONLY` split included) with the fp32 paths
untouched as the production floor; the new-build default-env
regression is bit-identical to baseline (299/361/361,
8.86917e-05).

### Device A/B (for the decision, not an integration)

Because mode B alone held teacher agreement and missed the strict
count by one step, its end-to-end price was measured so the
trade-off is a decision with numbers, not a guess: one detached
session on the device, baseline `~/tts_host` (sha256 7b4971e3…)
vs the new build `~/tts_host_sdot` (sha256 e3960214…, stripped
1,121,360 B) run with `POTATO_GEMV_I8=1 POTATO_GEMV_I8_ONLY=b`,
interleaved per configuration. Short caliber = the baseline
request; medium caliber = the profile's new-caliber sentence,
ttsscan sampling, seed 42.

| caliber / threads | elapsed old → new | decode/step old → new | RTF old → new | RSS old → new (MB) |
|---|---|---|---|---|
| short, t2 | 19,728 → 21,036 | 195.9 → 212.0 | 8.62 → 10.20 | 219 → 219 |
| short, t4 | 23,603 → 22,654 | 292.6 → 268.3 | 10.31 → 10.98 | 221 → 222 |
| medium, t2 | 52,591 → 50,528 | 228.5 → 219.1 | 8.20 → 8.12 | 253 → 252 |
| medium, t4 | 67,268 → 61,576 | 343.7 → 322.4 | 10.49 → 9.89 | 255 → 254 |

Reading the table honestly:

- The sampled trajectories diverged (short: 60 → 63 steps, audio
  2.288 → 2.063 s; medium: 161 → 157 steps, 6.415 → 6.226 s) — the
  expected fingerprint of a one-argmax-flip class of numeric
  change. Short-caliber elapsed and RTF therefore compare
  different amounts of work and are weak evidence; decode/step
  within each interleaved pair is the clean comparator.
- Decode/step deltas: medium t2 −4.1%, medium t4 −6.2%, short t4
  −8.3% — the same magnitude as the bench's B-only projection
  (−27.8 ms/step of packed time at t4; measured −21 to −24 ms/step
  on a decode/step figure that also contains unchanged BINOP /
  attention / glue). Short t2 is the one contrary pair (+8.2% on
  a 60-step sample whose baseline already sat off the profile
  session's pace); it is reported, not explained away.
- Medium elapsed (the caliber that matters): −3.9% at t2, −8.5%
  at t4; medium RTF 8.20 → 8.12 (t2) and 10.49 → 9.89 (t4), each
  RTF computed on its own run's duration.
- This session ran slower overall than the profile session
  (medium t4 baseline 67.3 s here vs 49.5 s there): cross-session
  variance on this phone is large, and only the within-session
  pairs above are quoted.
- Threads: t2 beat t4 for BOTH binaries on both calibers in this
  session, confirming the profile's finding on the new code as
  well.

### Recommendation

1. **tts_threads for Rubens: 2** — see the A/B section; decode
   still scales negatively past 2 on this SoC, with or without the
   i8 tier.
2. Round 2, if the gate is to be met rather than renegotiated:
   block-quantize mode A's activations — split the contraction
   into ≤128-row blocks with a scale per block (the rank-1 form
   takes this directly: one int32 accumulator set per block, fp32
   combine per block). The evidence says group width is the error
   lever (64-wide: nearly clean; 512–2048-wide: 293/335), and the
   prize is the bench's packed total, 139 → 66 ms/step at t4
   against a measured PACKED segment of 174 ms/step — roughly a
   quarter of decode. Re-run the same gates; nothing ships on
   bench numbers alone.
3. Mode B at 360/361 is a policy question, not an engineering one:
   teacher agreement is intact and the flipped step is a near-tie
   against the previous build's dump, not against ground truth.
   It is left opt-in for the user's call.

## Round 2: mode B promoted on the ear's verdict; mode A block-quantized, still opt-in (2026-11-11)

### Part 1 — mode B promoted to default-on (SDOT only)

The round-1 samples went to the user's ear and came back
"差不多" — no audible difference from the fp32 floor. Per project
doctrine that verdict outranks the proxy gate mode B missed by one
near-tie ref argmax (360/361, teacher agreement intact at 299/361),
so mode B is promoted. The selection code in `packed_core.h` is
rewritten around it: `gemv_i8_mode()` now distinguishes unset (-1)
from explicit off (0), and a new `gemv_i8_enabled_for(which)` decides
per mode (`POTATO_GEMV_I8_ONLY` can only restrict further):

- unset (the new default): tier ON for mode B only, and only where
  the runtime reports SDOT; mode A stays on the fp32 floor, and so
  does everything on SDOT-less CPUs.
- `POTATO_GEMV_I8=0`: tier fully off — the exact round-1 default,
  kept as the opt-out.
- `POTATO_GEMV_I8=1`: both modes on where SDOT is reported.
- `POTATO_GEMV_I8=sim`: both modes, portable executor, any arch
  (the gate configuration).

x86 verification on the new code (`tf_check_free --mode packed`,
established reference; the baseline was re-confirmed on the round-1
binary first and reproduced identically):

| configuration | tf_agree [/361] | ref_argmax [/361] | max logit diff |
|---|---|---|---|
| new code, default env (tier off on x86) | 299 | 361 | 8.86917e-05 |
| new code, `POTATO_GEMV_I8=0` (opt-out) | 299 | 361 | 8.86917e-05 |
| new code, `sim ONLY=b` (the new default's numerics) | 299 | 360 | 0.151795 |

The opt-out reproduces the round-1 default exactly; the sim
mode-B-only run reproduces round 1's mode-B row exactly (299/360,
0.151795). `vits_check_free`: packed corr 0.959880, unchanged.

### Part 2 — mode A block quantization

Implemented as diagnosed: the contraction is split into blocks
with one activation scale per block (`gemv_a_i8_prelude_blocks`;
same f array and the same full-K fp32 zero-point correction as the
whole-K prelude — corr does not depend on the quantization). Both
executors accumulate an exact integer accumulator per block (the
NEON kernel keeps the whole-K kernel's int16 chunking inside the
block) and combine in fp32 in the identical written order — bias,
then per block `y[t] += sfb[b] * acc[t]`, then `y[t] -= corr` — so
NEON and portable stay bit-equal by construction (the device bench
cross-check of that claim is part of the pending device pass
below). The whole-K kernels are retained as `gemv_a_w4_i8_wk_*`,
reachable only from gemv_bench for A/B; production dispatch uses
the block form.

Bench error against the double-precision reference (portable
executors, synthetic outlier activations; RMS of the i8 error; the
fp32 floor's max error on the same shapes is 3.7e-06 / 1.3e-05 /
3.8e-06):

| shape | block @128 | block @64 | whole-K |
|---|---|---|---|
| K=512 D=1536 | 2.262e-02 | 1.718e-02 | 3.395e-02 |
| K=2048 D=512 | 8.172e-02 | 5.255e-02 | 1.114e-01 |
| K=520 D=128 | 1.610e-02 | 1.412e-02 | 1.557e-02 |

Gates (x86, sim executor; round-1 rows repeated for contrast):

| configuration | tf_agree [/361] | ref_argmax [/361] | max logit diff |
|---|---|---|---|
| i8 tier, mode A only, whole-K (round 1) | 293 | 335 | 1.98015 |
| i8 tier, mode A only, block @128 | 304 | 345 | 1.21978 |
| i8 tier, mode A only, block @64 | 298 | 351 | 0.794945 |
| i8 tier, both modes, whole-K (round 1) | 296 | 339 | 2.17202 |
| i8 tier, both modes, block @128 | 296 | 343 | 1.26397 |
| i8 tier, both modes, block @64 | 299 | 347 | 1.29845 |

**Verdict: mode A is NOT promoted.** The promotion rule was both
counts — teacher agreement at 299/361 AND ref_argmax at 361/361.
Blocking helps exactly as diagnosed (ref argmax 335 → 345 → 351 as
the width shrinks whole-K → 128 → 64; max diff 1.98 → 0.79), but it
saturates well short of 361, and the two counts start trading
against each other (@64 buys six ref steps for one teacher step
versus @128), so narrower blocks are not a free path to the strict
count either. Shipped width is 64 (best ref-argmax, max-diff and
bench RMS of the tested widths). Mode A remains available as
`POTATO_GEMV_I8=1` and goes to a listening round next, like mode B
did; it does not ship on numbers alone.

### Part 3 — device verification: PENDING (device unreachable)

The final binary is built and staged locally
(`build-ndk-free` tts_host, stripped 1,123,376 B, sha256
`71768cf5035c6258edad3f9b15c2960718297f38876284e6c45ed00404a87bbe`,
NEEDED libm/libdl/libc only, no ORT symbols), together with the
matching `gemv_bench` and a ready A/B runner (`/tmp/ab_r2.sh`:
previous production binary vs this build, short + new-caliber
medium, threads 2, two interleaved pairs). The device pass —
micro-bench (wk vs block NEON on device, per-step dispatch totals
at floor / default / both-on) and that A/B — could not run: the
Rubens SSH path was down for the whole session (banner-exchange
timeouts on the domain route, CONNECT refused on the previous raw
IPv6; ~45 minutes of retries over ~34 attempts). Nothing was
pushed to the phone. The moment the path is back, the pass is:
push `tts_host_r2` + `gemv_bench_r2`, verify sha256 on device, run
the bench variants, then the A/B — and this section gets its
numbers.

### Final recommended shipping configuration

- Binary: the free-only bionic `tts_host` built from this tree
  (`build-ndk-free`, POTATO_FREE_ONLY), stripped 1,123,376 B,
  sha256 `71768cf5035c6258edad3f9b15c2960718297f38876284e6c45ed00404a87bbe`
  (staged locally; device A/B confirmation pending, see Part 3).
- Run it with NO `POTATO_GEMV_I8` setting: mode B runs the SDOT
  tier by default on SDOT CPUs, mode A stays on the fp32 scalar
  floor, the fp32 floor serves SDOT-less CPUs, vits untouched.
  `tts_threads = 2` (round 1's settled Rubens default).
- `POTATO_GEMV_I8=0` restores the exact round-1 (pre-round-2)
  behavior; `POTATO_GEMV_I8=1` additionally enables block mode A
  (opt-in, pending its listening round).

## Round 2 addendum — mode A promoted on the ear's verdict (2026-11-11)

The round-2 listening pairs (`rubens-opt-round2-listen/`: the
shipped mode-B-only default vs the both-modes block-@64 candidate,
same five sentences, seed 42) went to the user's ear and came back
"俩mode 差不多" — no audible difference. That is the same verdict
standard that promoted mode B in Part 1, and per project doctrine
it outranks the proxy counts mode A missed (both modes @64:
299/347 against the strict 299/361 + 361/361 rule). Mode A
block-quant @64 is therefore promoted to the default.

The change is one line of selection logic in `packed_core.h` plus
its comments: in `gemv_i8_enabled_for`, the unset case now returns
`gemv_i8_have_sdot()` for both modes instead of mode B only. With
`POTATO_GEMV_I8` unset, an SDOT CPU now runs mode B and mode A
(block form, `kGemvA8Block = 64`) by default — the same effect as
`=1`. Everything else is untouched: `POTATO_GEMV_I8=0` remains the
exact round-1 opt-out, `=1` / `=sim` behave as before,
`POTATO_GEMV_I8_ONLY` can still only restrict, and SDOT-less CPUs
stay on the fp32 scalar floor for both modes.

x86 verification on the new code (`tf_check_free --mode packed`,
established reference; all four pre-change baselines were
reproduced on the round-2 code first and matched round 2's tables
exactly):

| configuration | tf_agree [/361] | ref_argmax [/361] | max logit diff |
|---|---|---|---|
| (a) `sim`, both modes @64 — the new default's numerics | 299 | 347 | 1.29845 |
| (b) `POTATO_GEMV_I8=0` (opt-out) | 299 | 361 | 8.86917e-05 |
| (c) default env (tier off on x86) | 299 | 361 | 8.86917e-05 |
| `sim ONLY=b` (mode B alone, unchanged) | 299 | 360 | 0.151795 |

Row (a) reproduces round 2's both-modes-@64 gate row exactly —
expected, since the promotion changes only which configuration is
the default, not any numerics. Rows (b) and (c) are unchanged from
round 2. `vits_check_free`: packed corr 0.959880, unchanged
(baked 0.959881, 226-layer worst corr 1.000000).

### Updated recommended shipping configuration

- Binary: the free-only bionic `tts_host` rebuilt from this tree
  (`build-ndk-free`, POTATO_FREE_ONLY), stripped 1,123,312 B,
  sha256
  `6211cf895aaf53744e8b52f9f435ef038f16f30178d066ab5c6cae63ad52a018`,
  NEEDED libm/libdl/libc only, no ORT symbols (zero `OrtGetApiBase`
  occurrences). This supersedes the round-2 binary quoted in
  Part 3 (1,123,376 B, `71768cf5…`).
- Run it with NO `POTATO_GEMV_I8` setting: on SDOT CPUs, mode B
  and mode A block-quant @64 both run the SDOT tier by default;
  the fp32 floor serves SDOT-less CPUs; vits untouched.
  `tts_threads = 2` (round 1's settled Rubens default).
- `POTATO_GEMV_I8=0` restores the exact round-1 behavior;
  `POTATO_GEMV_I8_ONLY=b` restores the round-2 (mode-B-only)
  default numerics without a rebuild.
- The device pass from Part 3 is still pending (nothing has been
  pushed to the phone) and now covers this binary and the new
  default: micro-bench plus the interleaved A/B at threads 2 will
  measure what the promotion is worth end to end.

## Round 2 addendum — device pass: NOT RUN (path died; mode A binary identified)

The mode A promotion addendum above landed while this close-out was
in flight, so the binary of record for the device pass is the
promoted build: `build-ndk-free/tts_host`, 1,123,312 B, sha256
`6211cf895aaf53744e8b52f9f435ef038f16f30178d066ab5c6cae63ad52a018`
(verified on the VM at the time of writing; supersedes the round-2
binary `71768cf5…`). The pass itself — push `tts_host_r2` +
`gemv_bench_r2`, on-device sha256 verify, micro-bench (default /
`POTATO_GEMV_I8=0` floor, threads 2 and 4), and the interleaved A/B
(old production binary vs new, threads 2, short + fox-medium, two
pairs) — could not run: the phone path, healthy ~19:51–20:10 for the
Omni harvest and the start of the NPU push, closed entirely from
~20:13 onward ("proxy closed during CONNECT" on every attempt,
including 40 watcher probes over ~50 minutes). Nothing for this item
was pushed; the phone's `~/tts_host` is still the round-1 production
binary. The two binaries are small (1.1 MB + 1.4 MB) and the staged
A/B protocol from Part 3 stands unchanged for the next window; the
transfers were never the problem for this item — only the closed
window was.

## Round 2 device pass — RESULTS (2026-11-12, supersedes NOT RUN above)

Ran on device via `ab_r3.sh` (detached), after the never-sleep fix
reopened the path. On-device sha256 verified before launch: new
`6211cf895aaf53744e8b52f9f435ef038f16f30178d066ab5c6cae63ad52a018`,
bench `e1845e7721044ba8063277d226d758ea76b17a6e4d6b97be800f107f3dbe1dee`,
the phone's incumbent `~/tts_host` (round-1 binary)
`7b4971e30abcea0c313077833830dd5f12ee5e8070adf6f74d174151701f1a5a`.
Two earlier script revisions produced no measurements (a wait
condition that fired early, then Termux environment defects: no
`/tmp`, freshly appended binaries hitting Text file busy, a
runner argv-order bug); v3 added smoke gates and the numbers below
are v3 only.

Micro-bench (`gemv_bench`, K=512 D=512 mode B shape): mode B SDOT
is worth 2.5× at threads 2 (1.80 vs 0.73 GMAC/s against the
`POTATO_GEMV_I8=0` floor) and 4× at threads 4 (0.69 vs 0.17).
Threads 4 is slower than threads 2 on every kernel line, both
modes — the settled `tts_threads = 2` default holds. Mode A's
kernel-level device figure is not cleanly separable in this bench
(its direct-call lines move between runs); the host A/B below is
the arbiter for mode A.

Host A/B, threads 2, interleaved pairs (second pair reversed).
The two binaries sample different token streams (expected — mode A
changes numerics; cross-config same-seed identity was never a
gate), so RTF and decode/step are the comparators, not totals.

| run | steps | audio s | elapsed ms | RTF | decode ms/step | vits ms |
|---|---|---|---|---|---|---|
| old short 1 | 60 | 2.288 | 17,989 | 7.86 | 189.9 | 4,952 |
| old short 2 | 60 | 2.288 | 17,263 | 7.54 | 177.8 | 4,932 |
| new short 1 | 75 | 2.929 | 23,396 | 7.99 | 209.7 | 6,036 |
| new short 2 | 75 | 2.929 | 23,637 | 8.07 | 212.9 | 5,931 |
| old fox 1 | 161 | 6.415 | 48,571 | 7.57 | 215.7 | 11,523 |
| old fox 2 | 161 | 6.415 | 49,427 | 7.70 | 221.1 | 11,585 |
| new fox 1 | 170 | 6.717 | 50,308 | 7.49 | 211.0 | 12,114 |
| new fox 2 | 170 | 6.717 | 51,332 | 7.64 | 216.4 | 12,250 |

Reading: on the fox sentence the new binary's decode/step is ~2%
faster (211.0/216.4 vs 215.7/221.1), inside same-binary repeat
noise (~2–3%). RTF overall: old 7.54–7.86, new 7.49–8.07 — the
promotion is **performance-neutral end to end on device**. RSS is
the same class (218–255 MB). Verdict: the promoted default stands
on its numerics (teacher-forced gate plus the user's ear), not on
a device speed win; the kernel-level mode B gain does not move
the whole pipeline, whose step cost sits mostly outside the GEMV
share. Shipping configuration unchanged: this binary, no env
switches, `tts_threads = 2`.
