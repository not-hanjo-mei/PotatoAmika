# Rubens NPU Spike — vits decoder on the APU via NNAPI

Date: 2026-11-11. Device: Redmi K50 (rubens, Dimensity 8100 / MT6895Z), Android 14.
Question: can the TTS vits graph run on the MediaTek APU through NNAPI at fp16,
what gets refused, what does it cost vs CPU, and does the output stay faithful?
Background probes (device enumeration, fp16/fp32 acceptance matrix):
`RUBENS-ACCEL.md`. All conversion artifacts under `tts-engine/accel/npu/`.

## TL;DR

- The vits **decoder subgraph** (z + ge → waveform; 93 convs + 5 transposed
  convs, the dominant compute share of vits) converts cleanly to TFLite with
  **pure builtin ops** after folding the weight-norm chains offline, and is
  numerically exact against the ONNX fp32 reference on the VM
  (corr 1.00000000, relRMS 0.00000).
- The **full vits graph does not fit NNAPI's shape**: it is built on
  data-dependent ops (Shape ×216, ConstantOfShape ×106), has no TFLite/NNAPI
  equivalent for LayerNormalization (×24), and carries an in-graph
  RandomNormalLike noise source. A TFLite conversion of the full graph exists
  (onnx2tf produced one), but it is a CPU artifact, not an NPU candidate.
- On-device NNAPI latency/correlation: **PENDING — phone SSH path down**
  (see "On-device status").

## Graph identification

Two fp32 variants of vits exist in the tree; they are not interchangeable:

| variant | file | nodes | note |
|---|---|---|---|
| canonical, unfolded | `work/vits_noiseless.onnx` | 8,190 | weight-norm computed at runtime (95 ReduceL2/Div/Mul chains in the decoder alone); reproduces the stored stage-1 reference `work/memopt/vits_fz.npy` at corr 1.0000000000 |
| weight-norm folded | `work/wo_graphs/w4/vits_wnfold.onnx` | 7,666 | the free engine's baked/packed graphs derive from this family |

This spike uses the **canonical** graph as the reference of truth.

Test segment (a real synthesis segment, from `work/memopt/vits_feed.npz`):
text_seq [1,31], pred_semantic [1,1,75], ge [1,1024,1], ge_advanced [1,512,1]
→ 96,000 samples @ 32 kHz (3.0 s of audio). Decoder boundary tensors taken
from a full-graph ORT run of the canonical graph: z = `/vq_model/Mul_2_output_0`
[1,192,150], ge [1,1024,1] (files `accel/npu/ref/z_real.bin`, `ge_real.bin`;
verified byte-identical to the full graph's own tensors).

## Conversion

Toolchain (VM, venv at `tts-engine/venv`): tensorflow-cpu 2.21.0,
tf_keras 2.21.0, onnx2tf 2.6.9, onnx 1.20.1, onnxruntime 1.26.0,
numpy 2.2.6, ai-edge-litert 2.1.2.

### Decoder subgraph — converted, builtin-only, exact

1. Extracted the decoder (z, ge → audio) from the canonical graph:
   634 nodes, of which 95 are weight-norm chains feeding Conv/ConvTranspose
   weights (two operand arrangements: `Mul(Div(g,‖v‖), v)` for Conv,
   `Mul(Div(v,‖v‖), g)` for ConvTranspose).
2. **Weight-norm fold** (`accel/npu/fold_wn.py`): baked
   w = g·v/‖v‖₂ per output channel into single initializers, chains deleted.
   Result: 254 nodes — Conv 93, LeakyRelu 86, Add 56, ConvTranspose 5,
   Div 5, Gather 2, Tanh 1, Constant 6. Fold fidelity vs the unfolded
   subgraph in ORT: max abs diff 1.5e-05, corr 0.99999999997; vs the full
   graph's reference audio: corr 1.0 (0.99999999994).
   (Without this fold, onnx2tf's tf_converter backend dies on a Keras
   broadcasting bug at the first dilated resblock, and the flatbuffer_direct
   backend lowers the 30 dilated (3/5) resblock convs and all 5
   ConvTransposes to unusable custom ops ONNX_CONV / ONNX_CONVTRANSPOSE.
   The unfolded canonical decoder does **not** convert; the folded one does.)
3. onnx2tf flatbuffer_direct on the folded decoder (input shapes pinned in
   the graph): `accel/npu/dec_fold_conv/vits_dec_folded_float32.tflite`
   (135,421,012 B, sha256 fdf61dea…) and `..._float16.tflite`
   (67,784,200 B, sha256 ace01fef…).
   Op census of the TFLite model — **all builtin**: CONV_2D 93,
   TRANSPOSE_CONV 5, LEAKY_RELU 86, ADD 61, MUL 5, TANH 1, GATHER 2, plus
   layout ops (TRANSPOSE 105, RESHAPE 110, STRIDED_SLICE 3). No custom ops.
4. VM check (LiteRT CPU, fp32 model, z_real/ge_real): output corr
   **1.00000000**, relRMS **0.00000** vs the ORT folded-decoder output
   (96,000 samples). The conversion is numerically transparent.

### Full graph — conversion exists, NNAPI does not fit

onnx2tf did produce a full-graph TFLite from the folded variant
(`accel/npu/full_conv/vits_wnfold_float16.tflite`, 126 MB). It is not an
NNAPI candidate. Static audit of the full graph against NNAPI's op set:

- **Blockers**: Shape (×216, data-dependent length logic), ConstantOfShape
  (×106), LayerNormalization (×24 — no TFLite builtin, no NNAPI op),
  RandomNormalLike (×1 — noise must leave the graph and become an input).
- **Risky**: Pad in reflect mode (×96; NNAPI PAD has no reflect),
  dynamic Slice/Gather, batched MatMul (no NNAPI BATCH_MATMUL),
  ConvTranspose (spotty MTK coverage), Where/Equal in the length logic.
- **Fine**: Conv, Softmax, Split, Cast, elementwise Add/Mul/Sub/Div/Tanh/
  Sigmoid/LeakyRelu.

Splitting the graph at the decoder boundary (everything upstream of z stays
on CPU — it is the cheap part) is the natural shape, and is what this spike
tests.

## The 0.5199 ghost — resolved

An earlier session recorded "folded-variant full graph scores 0.5199 against
the reference while the free engine scores 0.9599" and could not explain it.
Root cause found this session: the file `accel/npu/ref/audio_ref_denoise.npy`
used as the reference in that comparison is **not** the reference audio —
it correlates 0.5199 with `work/memopt/vits_fz.npy` itself (it appears to be
a W4-pack output, matching the known all-W4 score of 0.5199). With the true
reference, the chain is fully consistent: canonical full graph → fz corr
1.0000000000; decoder subgraph fed the full graph's own z/ge → full-graph
audio corr 1.0 at every probed stage (conv_pre, all five upsampling stages,
byte-exact max diff 0.0); folded decoder → corr 1.0 as above. No graph
discrepancy remains.

## Android runtime build

- TensorFlow **v2.20.0** source (`accel/npu/tf-src`; master is unusable —
  its lite CMakeLists references converter paths that moved). NDK r27.2,
  arm64-v8a, android-26, Release, XNNPACK/GPU off, NNAPI on, host flatc from
  a local flatbuffers build (`accel/npu/fb-build`).
- Two upstream build breaks had to be worked around (details in the spike
  notes): protobuf's FetchContent full-history clone through this VM's
  egress proxy (replaced with a depth-1 fetch of the pinned commit
  90b73ac3… via `FETCHCONTENT_SOURCE_DIR_PROTOBUF`), an unguarded
  `target_compile_options(xnnpack-delegate)` with XNNPACK off, and
  TF_MAJOR/MINOR/PATCH_VERSION defines that upstream only supplies via
  Bazel (added as compile definitions: 2/20/0, empty suffix).
  These are build-script patches in the scratch tree only; no engine code
  was touched.
- Result: `libtensorflow-lite.a` (static). `benchmark_model` was not used:
  its build dies at host-protoc codegen (Error 127, cross-build), and the
  custom runner covers the need.
- Runner: `accel/npu/nnapi_runner.cpp` (TFLite C API + NNAPI delegate C
  API; binds a named accelerator, allow_fp16, unlimited partitions; times
  Invoke, dumps raw fp32 output). Note the TF 2.20 C API differs from older
  docs: options are a plain struct from `TfLiteNnapiDelegateOptionsDefault()`,
  deletion is `TfLiteNnapiDelegateDelete()`. Built binary: 4,670,424 B
  stripped, sha256 98675b33…, NEEDED = libdl/libm/liblog/libc only.

## On-device status

**Blocked at the time of writing.** The phone's SSH path
(the maintainer's private SSH route: the VM's HTTP-CONNECT proxy plus Termux sshd)
has been failing for ~80 minutes at the time of writing (last attempt
10:59 UTC): first "proxy closed during CONNECT" while this VM's egress was
saturated by the protobuf clone, then persistent "Connection timed out
during banner exchange" — TCP connects through, the phone's sshd never
answers. That signature matches Termux being frozen by Android (app not in
foreground / doze) rather than a network fault; the device was reachable
and idle at 09:41 UTC. One phone-side action should unblock it: bring
Termux to the foreground (and keep it there during the runs). Nothing has been pushed yet;
the full bundle (runner, both TFLite models split into parallel-push
chunks, inputs, `run_spike.sh`) is staged locally and one command away
(`accel/npu/push_spike.sh run`). The runs queued on the phone, in order:
CPU fp32 TFLite (comparator), NNAPI fp16 bound to mtk-neuron_shim, NNAPI
fp16 bound to mtk-mdla_shim (the APU), NNAPI auto — each with delegate
partition logging on the bound runs.

Comparison basis (planned, stated per the brief): the decoder has no
free-engine segment timing on the phone for this exact segment, so latency
is NPU vs **same-model, same-input TFLite CPU on the phone**, with the
known whole-vits context that vits costs ≈9.2–9.8 s per ~150-frame segment
in the free engine on this device (SDOT A/B session, t4) — of which the
decoder stack is the dominant share.

## Verdict

PENDING the on-device runs. Everything up to the phone is green: the
decoder — the part of vits that costs the time — is convertible,
builtin-only, and exact at fp32; fp16 fidelity and APU latency are the two
numbers still missing, and they are exactly what NNAPI acceptance on this
SoC will decide.

## Close-out attempt — 2026-11-11 (path died mid-push; verdict still PENDING)

The phone path came back after the user's DNS update (new AAAA
2409:8a34:…, China Mobile) and was healthy from ~19:51 to ~20:10 UTC+8
(all of Task 1's harvest went through it). The staged
`accel/npu/push_spike.sh run` was then launched as briefed — and this
path cannot carry its transfer pattern:

- The script's 8 MiB single-stream `ssh 'cat >'` transfers wedge:
  after ~14 minutes the six parallel part streams had delivered
  1.1 MB total, individual files frozen mid-write (parts16 paa 256 KiB,
  pab 128 KiB, pac 160 KiB, pad 192 KiB, pae/paf 160 KiB each, still
  growing at ~14 KB/s aggregate in bursts). Note for the record: the
  earlier ~3 MB/s figure for this path was measured on the OLD
  (240e, China Telecom) endpoint; on the new Mobile endpoint the
  sustained rate is two orders of magnitude worse and streams wedge
  rather than merely running slow.
- A replacement resumable pusher (512 KiB chunks, skip-if-size-matches,
  per-chunk size verification) fared no better: the window closed
  entirely while it ran — from ~20:13 to the end of the session
  (~13:03 UTC), every connection failed at "proxy closed during
  CONNECT", including 40 watcher probes at ~85 s spacing. Zero chunks
  were verified delivered. The phone side was meanwhile alive (the
  user connected from their own machine and got the phone's host-key
  prompt, fingerprint matching the record).
- Nothing was corrupted: the phone holds only the partial
  `~/npu-spike/parts16/*` fragments listed above plus empty
  `parts32/`, `c16/`, `c32/` staging dirs. The models on the phone are
  NOT complete and `run_spike.sh` was never launched.

**Verdict remains PENDING** — unchanged from above: the decoder is
convertible, builtin-only, and exact at fp32 on the VM; the two
deciding numbers (APU latency, fp16 correlation) still do not exist.

What the next window needs (no re-derivation): a transfer that fits
this path — small chunks with phone-side append-resume
(`~/npu_resume.sh`, written locally, not yet installed on the phone)
or the user's own LAN/USB copy of `dec_fp16.tflite` (67,784,200 B,
sha256 ace01fef…) and `dec_fp32.tflite` (135,421,012 B, sha256
fdf61dea…) into `~/npu-spike/`, then `bash run_spike.sh` (the runner,
inputs, and run script are small and push in seconds once the models
are there), then the correlation of `results/out_*.bin` against
`accel/npu/ref/dec_out_folded_ort.npy` on the VM.

## On-device results (2026-11-12, supersedes the PENDING verdict above)

The transfer blocker cleared when the user copied both models to
`~/npu-spike/` over USB/LAN himself (hashes verified on-device by
finish.sh before any run) and set the phone to never sleep — the
overnight tunnel failures were the screen-off freeze of Termux, not
the path. The small files went over the tunnel with a 256 KiB
append-resume pusher (whole-stream pushes wedge and truncate on this
endpoint; the runner took 17 appends over ~22 min).

Measured on the phone (Redmi K50, Termux, detached runs):

- **TFLite CPU, fp32 model, 4 threads: invoke avg 3,133.7 ms** over
  3 iters for the decoder subgraph (z + ge in, 96,000-sample waveform
  out, output rms 0.10194). This is the banked on-device baseline;
  output in `~/npu-spike/results/out_cpu.bin`.
- **fp16 model + NNAPI delegate: dead at interpreter creation**, all
  three modes (mtk-neuron_shim, mtk-mdla_shim, auto). The delegate
  (allow_fp16=1) does not claim the fp16 CONV_2D nodes; CPU kernel
  Prepare then rejects the float16 tensor type (conv.cc:360). This
  was the fp16 model's first real invoke anywhere — VM-side
  validation only ever invoked the fp32 model.
- **fp32 model + NNAPI delegate (allow_fp16=1)**: neuron_shim claims
  468/471 nodes (7 partitions), compiles, feeds inputs, then the
  warmup invoke fails: `ANEURALNETWORKS_OP_FAILED ... while running
  burst computation`, delegate node failed to invoke — the failing
  op is not identified and the SL driver has no diagnostic
  callbacks. auto mode is byte-identical in behaviour. mdla_shim
  (the APU proper) claims only 350/471 nodes (22 partitions) and
  fails earlier: `ANEURALNETWORKS_BAD_DATA` while completing NNAPI
  compilation, delegate rolls back, interpreter creation fails.

**Verdict: NNAPI is a dead end for this graph on this driver stack.**
Not a format problem — both model formats were tried, the delegate
options were the standard ones, and two different shims fail at two
different stages (compile vs execution). The door that RUBENS-ACCEL.md
found open (trivial graphs compile and execute on all shims) does not
extend to this 471-node decoder graph. Revival condition, if anyone
wants it: bisect the graph to find the op that fails burst execution
on neuron_shim — that needs subgraph files ferried to the phone and
has no guaranteed payoff; not done.

Practical consequence: vits stays on our own CPU NEON/SDOT kernels.
The TFLite CPU number (3.13 s, decoder subgraph only, fp32, generic
kernels) also reads as a sanity anchor for how much headroom generic
runtimes leave on this SoC — it is not a slice of our packed engine's
pipeline and is not compared against it directly.
