# PotatoAmika benchmarks — before/after comparisons (Mi 4c / libra edition)

This is the Mi 4c (libra branch) edition of the benchmarks index,
compiled 2026-11-12. It carries the Mi 4c-era sections of the full
table: §1 (the TTS engine against stock GPT-SoVITS, against the
ORT-based path it replaced, and the shipped weight packs), §2 (TTS
on the Mi 4c across the five NEON kernel rounds) and the Mi 4c row
of §7 (whole-stack memory). The full cross-device table — the
Rubens (Dimensity 8100) sections, the Omni brain, KV cold start and
ASR — follows on the `rubens` branch. Like the full table, this
file is an index of numbers that already exist in the engineering
documents, not a new measurement round.

Rules applied throughout:

- A figure enters a table only as a same-calibre pair: same device or
  machine class, same sentence calibre, same seed and thread count,
  unless the row itself says otherwise. A number without its
  conditions does not go in.
- Cross-session and cross-machine numbers are quoted only where the
  source document quotes them, and are labelled as such. On the
  phones, same-session interleaved pairs are the evidence; absolute
  values swing widely between sessions.
- Where a comparison came out a tie, it says tie. Where a candidate
  was measured and rejected, it is listed as rejected, with its
  numbers.

## 1. TTS engine: self-built vs the ORT-based path, and the weight packs

The self-built ("free") engine runs the same five graphs as the
ONNX Runtime path it replaces, so the two are directly comparable on
the same host, same inputs, same teacher-forced protocol.

### Against stock GPT-SoVITS (the upstream project)

The full source record for this subsection is
`docs/BENCH-VS-STOCK.md` in this repository.

| metric | stock GPT-SoVITS | self-built (free engine + shipped W4 packs) | conditions | source |
|---|---|---|---|---|
| Overall RTF, 6 sentences | 2.96 | 5.80 (stock ≈1.96× faster overall; steady-state sentences S2–S6 run 2.3–2.7× slower on ours by wall clock) | x86 dev VM (tenant quota 1.5 cores / 6 GB), both sides locked to 2 threads, same 6 sentences and neutral1 reference, second of two runs each, RTF normalized to each side's own output duration | BENCH-VS-STOCK.md |
| Total wall clock, 6 sentences | 63.55 s | 121.64 s | same as above | BENCH-VS-STOCK.md |
| Peak RSS over synthesis (VmHWM) | 5,120.9 MB | 457.8 MB (≈11.2× less) | same as above | BENCH-VS-STOCK.md |
| RSS after load | ≈3,880 MB | ≈193 MB (after boot) | same as above | BENCH-VS-STOCK.md |
| Load time | ≈27–29 s | ≈1.1 s (≈27× faster) | same as above | BENCH-VS-STOCK.md |

Two qualifications from the source: the self-built engine was never
optimized for x86 (its kernels are width-4 portable kernels written
for A57/NEON), so the x86 speed loss is expected and is not the
target metric; and on the target phone class (≈1.28 GiB usable RAM)
stock GPT-SoVITS does not fit at all — its peak RSS alone is
≈5.1 GB — so no on-device comparison exists. The stock side ran
after a VM reboot under higher load than our side, which
disadvantages stock; the source records that the conclusion does
not flip on that difference (both sides' two runs are internally
consistent: RTF 5.87/5.80 ours, 3.04/2.96 stock).

### Runtime and memory (x86 development machine)

| metric | before (ORT path) | after (free engine) | conditions | source |
|---|---|---|---|---|
| GPT decode, 361 steps total | 65.9 s (≈183 ms/step) | ≈24.0 s (≈66 ms/step), ratio ≈0.36 | x86, same session back-to-back, high-load window (ORT's quiet baseline is 43–46 ms/step; both sides inflated, ratio is the evidence) | ORT-FREE.md §3 |
| GPT decode per step, host calibre | 34.5–37.1 ms | 48.1–50.6 ms (≈1.4×) | x86, quiet window, host `--threads 2`, 4-sentence mean band | ORT-FREE.md §5 |
| init (encoder + first), per line | 6.5–12.8 s | 4.0–9.1 s | x86, same loaded session as row 1 | ORT-FREE.md §3 |
| vits per token, packed | 146 ms (10.97 s per 75-token run) | 162–166 ms (12.15–12.48 s) | x86, acceptance-tool calibre, same feed, threads 4 | ORT-FREE.md §5 |
| vits per token, baked fp32 fallback | 69 ms (5.19 s) | 259 ms (19.44 s) | x86, same as above | ORT-FREE.md §5 |
| TTS host steady RSS, packed | 292–315 MB | 262–280 MB | x86 host, quiet window | ORT-FREE.md §5 |
| TTS host steady RSS | 1.83–1.88 GB (stage-2 host, fp32 graphs resident) | 270–318 MB (packed, no-arena) | x86 host, one greedy S0 request per rung | STAGE3.md §1, §3 |
| Numerical agreement vs ORT (teacher-forced) | — | 299/361 packed and baked, logits max diff 8.74e-05 packed / 5.53e-05 baked, argmax 361/361 | x86, full 361-step check | ORT-FREE.md §3, §5 |

Reading: on x86 the free engine wins memory and load time and loses
some per-step speed to ORT's MLAS kernels (AVX-512 width). That width
difference does not exist on the target A57, where both sides are
4-wide NEON; the device verdict is in §2 below. The baked fp32 path
is retained as a fallback for memory-rich machines, not a shipping
configuration.

### Shipped weight packs (quantization)

| metric | before | after | conditions | source |
|---|---|---|---|---|
| vits weight pack size | 62,844,080 B (W8 pack), full-graph corr 0.9680 vs fp32 | 43,523,616 B (hybrid pack E: enc_p + codebook at W8, rest W4), corr 0.9599 | VM, denoised evaluation, fixed neutral1 + S0 inputs, same protocol as the stage-1 baseline | VITS-W4.md |
| vits all-W4 pack | 37,170,298 B | corr 0.5199 — rejected, unusable | same as above | VITS-W4.md |
| GPT W4 pack agreement | — | 299/361 (82.83%) teacher-forced vs the fp32 reference | x86, established check protocol | ORT-FREE.md §3 |
| Total resident weight bytes (GPT + vits packs) | — | 87,123,996 B (GPT 44,173,824 + vits 42,950,172) | host BlobStore unique bytes | ORT-FREE.md §5, STAGE3.md |

## 2. TTS on the Mi 4c (Snapdragon 808 / Cortex-A57) — the five NEON rounds

All rows: device, medium sentence is "The weather looks really nice
today, so let us go outside together." unless noted, seed 42,
token sequences identical within each pair. `tts_threads` was raised
from 2 to 4 mid-series by user decision; each row states its count.

| step | before | after | conditions | source |
|---|---|---|---|---|
| vits NEON tile (PackedConv) | whole request 107.5 s; vits segment 79.8 s | 54.2 s; vits 31.7 s (2.52×) | short sentence, same session back-to-back, threads 2, ttsscan calibre | ORT-FREE.md §7 |
| vits NEON tile, medium RTF | ≈79.8 | ≈46.5 (≈1.7×) | medium, threads 2, ttsscan 4-request mean; per-run phone variance ±40%, same-session pairs are the evidence | ORT-FREE.md §7 |
| threads 2 → 4 | 142.1 s whole request | 83.0 s (1.71×) | medium, seed 42; the threads-2 run hit an environment trough (first segment 25.9 s outlier) — against the profile run's threads-2 total of 98.4 s as the most generous baseline, threads 4 still leads ≈1.2× | ORT-FREE.md §8 |
| EXT register window, dil=1 | whole request 82.2 s; vits 41.6 s; medium RTF 21.9 | 75.3 s (−8.4%); vits 34.2 s (1.22×); RTF 20.0 | same session back-to-back, threads 4 | ORT-FREE.md §8 |
| EXT window extended to dil=3/5 | whole request 74,974 ms; vits 33,773 ms; RTF 19.94 | 73,713 ms; vits 31,329 ms (−7.2%); RTF 19.61 | same session, threads 4 | ORT-FREE.md §9 |
| K11 dil=5 8-position window (final knife, promoted) | whole-request mean 73,076 ms | 71,413 ms (−2.3%); vits segment two-pair mean −1,082 ms (−3.5%) | two alternating same-session pairs, medium, threads 4 | ORT-FREE.md §13 |
| Final production vs previous build | RTF 19.6 / 19.3; whole request 73,611 / 72,541 ms | RTF 19.4 / 18.6; whole request 72,903 / 69,923 ms | medium (96 steps, 3.75925 s audio), two same-session pairs, threads 4 | ORT-FREE.md §15 |
| Endpoints of the series | medium RTF ≈79.8 (generic kernels, threads 2) | medium RTF 18.6–19.4 (final production, threads 4) | not a single A/B: thread count changed mid-series; first device profile had the whole medium request at 424.8 s with PackedConv at 365.3 s (86.0%) | ORT-FREE.md §7, §15 |
| TTS host RSS on device | — | 261 MB | threads-4 profile run, medium sentence | ORT-FREE.md §9 |

Rejected on measurement in the same series (kept here so the record
is complete): a resident fp32 dequantization cache for `first`
(+304 MB, does not fit); NEON dequantization (ceiling ≈0.25% of a
request); a span-walk replacement for decode BINOP (8,476 ms vs
walk_par's 7,443 ms on the same call batch — slower, removed); a
group-panel GEMV weight layout (208.4 vs 195.0 ms/step — slower, not
adopted). Source: ORT-FREE.md §11, §12, §14. The series closes with
the remaining gap to real time (≈19×) attributed to the A57 itself,
not to a kernel that is still on the table.

## 7. Whole-stack memory

| metric | before | after | conditions | source |
|---|---|---|---|---|
| Mi 4c resident stack (Minico era, 2026-11-09 budget) | original estimate ≈850 MB | measured total ≈1,019–1,075 MB against ≈1,311 MB usable: brain ≈405 MB (12k), embeddings ≈150 MB, TTS host 262–280 MB, ASR host 154–178 MB, scheduler ≈8 MB, UI adapter 38–54 MB | the budget document mixes measured and device-derived rows and says so per row; per-component phone RSS had not been calibrated yet at that date | BUDGET.md |
