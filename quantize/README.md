# Weight conversion: fine-tune weights in, shipping packs out

This is the "method" half of PotatoAmika's distribution story. The
quantized packs themselves are derivatives of MekaHime's fine-tune
weights and are **not** distributed; these scripts are, so anyone with
weights they have the right to use can reproduce the packs locally.
Everything runs offline on a development machine.

Input: a GPT-SoVITS (v2ProPlus-generation) fine-tune — one `.ckpt`
(GPT) and one `.pth` (SoVITS) — plus its reference clips.
Output: the packed weight pools and reference packs the engine in
`../engine/` loads. Total for the Amika voice: ~91 MB on disk, from
328 MB of fp32 originals.

## Pipeline

0. **`convert.py`** — turns the `.ckpt` + `.pth` into fp32 ONNX graphs
   using Genie's converter. Set `TTS_ENGINE_DIR` to a working
   directory holding `weights/` (the two files) and `graphs/`
   (output). Note: Genie-distributed `.pth` files can carry a 2-byte
   container header (the file starts with the ASCII bytes `06`);
   strip it before other tools will read the file. Graph-level
   validation of the export is covered in `../docs/STAGE1.md`.
1. **`quant_wo.py`** — weight-only quantization of the GPT graphs.
   Shipping configuration: W4, group-64 asymmetric, true half-byte
   storage (activations are never quantized; the engine dequantizes
   in-kernel while streaming weights, which is where the speed on
   bandwidth-starved hardware comes from). W8 output is also
   supported.
2. **`vits_w4_pack.py` / `vits_w4_hybrid.py`** — the SoVITS (vits)
   side. The shipping configuration is the *hybrid* pack produced by
   the latter: `enc_p` and the quantizer codebook at W8, everything
   else W4. A full-W4 vits measured unusable (denoising correlation
   0.52); the hybrid holds 0.9599 against the fp32 graph. Record:
   `../docs/VITS-W4.md`.
3. **`make_packed_graphs.py` / `make_packed_vits.py`** — build the
   packed graphs and content-addressed weight pools the engine loads;
   decoder tensors shared by the GPT graphs are stored once.
4. **`precompute_refs.py`** — precomputes the reference packs (`.rpk`)
   from the reference clips. The prompt encoder runs here, offline,
   so the runtime never loads it.
5. **`dequant_wo.py`** — inspection helper: dequantize a pack and
   compare it against its source tensors when checking a build.

Measured anchor points for the Amika weights (details in the docs):
GPT token agreement of the shipping W4 pack against the fp32 graphs
is 82.83% (argmax 361/361 on the probe set); the user's listening
verdict on the stage-1 samples was "no difference heard".

## Not adopted: the GPTQ variant

**`gptq_pack.py`** implements a GPTQ-style error-compensated W4 pack
(GPTQ update + MSE-optimal clipping bounds). It measured *better*
than the plain pack — token agreement 89.47% vs 82.83% — and was
still **not adopted**: the shipping configuration is the plain
`quant_wo.py` output, by the project owner's decision. The script
and its record (`PTQ-PLUS.md`) are kept so the comparison is
reproducible, not because they are the shipping path. Do not mistake
them for it.
