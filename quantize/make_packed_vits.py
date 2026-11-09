#!/usr/bin/env python3
"""Build the packed-residency vits graph (Stage 3 S10, step 2).

Reads work/folded/vits_wnfold.onnx + wo-packs/manifest_vits_w4hybrid.json
(the shipping E hybrid pack) and rewrites every node that consumes a
quantized weight into a potato-domain op that reads the packed bytes at
runtime:

    Conv(weight)          -> potato::PackedConv
    ConvTranspose(weight) -> potato::PackedConvTranspose
    Gather(table)         -> potato::PackedGather

fp32-residual entries (biases, norms, the two emb_rel tables) stay baked
fp32 initializers, exactly as the baked path restores them -- the packed
path changes WHERE the quantized weights live, not their values.

Outputs (host/packed_graphs/):
    vits_wnfold.onnx        shipping packed graph
    vits_wnfold.onnx.json   sidecar: packed graph inputs (pool filter)

With --probes DIR, additionally writes validation variants into DIR:
    vits_packed_noiseless_probes.onnx   noise Constant zeroed, every
                                        rewritten node's output exposed
    vits_bakedE_noiseless_probes.onnx   same treatment for the baked E
                                        graph (work/vits_w4_variants/)
The comparison tool (vits_check) runs the deterministic noiseless pair.
"""
import argparse
import json
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FOLDED = os.path.join(ROOT, "work", "folded", "vits_wnfold.onnx")
MANIFEST = os.path.join(ROOT, "wo-packs", "manifest_vits_w4hybrid.json")
PACK = os.path.join(ROOT, "wo-packs", "sovits_vits_w4hybrid.npz")
OUTDIR = os.path.join(HERE, "packed_graphs")
NOISE_OUT = "/vq_model/Constant_7_output_0"


def attrs(node):
    out = {}
    for a in node.attribute:
        if a.type == onnx.AttributeProto.INTS:
            out[a.name] = list(a.ints)
        elif a.type == onnx.AttributeProto.INT:
            out[a.name] = a.i
        elif a.type == onnx.AttributeProto.STRING:
            out[a.name] = a.s.decode()
        elif a.type == onnx.AttributeProto.FLOAT:
            out[a.name] = a.f
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", metavar="DIR", default=None)
    args = ap.parse_args()

    man = json.load(open(MANIFEST))["sovits_vits"]["w4hybrid"]
    entries = {e["name"]: e for e in man["manifest"]}
    store = np.load(PACK)
    model = onnx.load(FOLDED)
    g = model.graph

    def scheme_of(name):
        e = entries.get(name)
        return None if e is None else e["scheme"]

    def is_quant(name):
        s = scheme_of(name)
        return s is not None and (s.startswith("w4") or s.startswith("w8"))

    # ---- consumer inventory: every use of a quantized tensor must be a
    # weight-position use of Conv / ConvTranspose / Gather -- with one
    # deliberate exception: the emb_rel_k/v tables (24 tensors, 83 KB
    # fp32 total) are consumed by Pad as DATA, not as linear weights.
    # For those, the packed graph bakes the pack-restored fp32 values
    # (bit-identical to what the baked path computes at load); writing a
    # PackedPad op for 83 KB is not worth a new kernel path.
    consumers = {}
    for node in g.node:
        for idx, inp in enumerate(node.input):
            if is_quant(inp):
                consumers.setdefault(inp, []).append((node, idx))
    pad_baked = set()
    for name, uses in consumers.items():
        if all(node.op_type == "Pad" and idx == 0 for node, idx in uses):
            pad_baked.add(name)
            continue
        for node, idx in uses:
            ok = (node.op_type in ("Conv", "ConvTranspose") and idx == 1) or (
                node.op_type == "Gather" and idx == 0
            )
            if not ok:
                sys.exit(f"unsupported consumer: {name} -> {node.op_type} input {idx}")

    def restore_entry(name):
        e = entries[name]
        if e["scheme"].startswith("w8"):
            q = store[name].astype(np.float32)
            return (q.reshape(e["shape"][0], -1) * store[name + "@scale"][:, None].astype(np.float32)).reshape(e["shape"])
        if e["scheme"].startswith("w4"):
            rows = e["shape"][0]
            cols = int(np.prod(e["shape"])) // rows
            pk = store[name]
            q = np.empty((rows, cols // 64, 64), dtype=np.uint8)
            flat = pk.reshape(rows, -1)
            q[:, :, 0::2] = flat >> 4
            q[:, :, 1::2] = flat & 0xF
            return ((q.astype(np.float32) - store[name + "@zero"][:, :, None].astype(np.float32))
                    * store[name + "@scale"][:, :, None]).reshape(e["shape"])
        return store[name]

    packed_inputs = {}   # graph input name -> shape
    sidecar_tensors = {}  # bare tensor name -> sidecar record (GPT format)
    rewritten_out = {}   # original node output names (for probes)

    def add_packed_inputs(tname):
        e = entries[tname]
        w4 = e["scheme"].startswith("w4")
        pk = store[tname]
        sc = store[tname + "@scale"]
        packed_inputs[tname + "__pk"] = list(pk.shape)
        packed_inputs[tname + "__sc"] = list(sc.shape)
        ze_shape = None
        if w4:
            ze = store[tname + "@zero"]
            packed_inputs[tname + "__ze"] = list(ze.shape)
            ze_shape = list(ze.shape)
        rows = e["shape"][0]
        cols = int(np.prod(e["shape"])) // rows
        sidecar_tensors[tname] = {
            "tensor": tname, "mode": 0 if w4 else 1, "rows": rows, "cols": cols,
            "scheme": e["scheme"], "pk_shape": list(pk.shape), "pk_dtype": "uint8",
            "sc_shape": list(sc.shape), "ze_shape": ze_shape,
        }
        return (0 if w4 else 1), rows, cols

    new_nodes = []
    rewritten_tensors = set()
    for node in g.node:
        wpos = None
        for idx, inp in enumerate(node.input):
            if is_quant(inp) and inp not in pad_baked:
                wpos = (idx, inp)
                break
        if wpos is None:
            new_nodes.append(node)
            continue
        idx, wname = wpos
        at = attrs(node)
        mode, rows, cols = add_packed_inputs(wname)
        rewritten_tensors.add(wname)
        ins = [wname + "__pk", wname + "__sc"] + ([wname + "__ze"] if mode == 0 else [])
        if node.op_type in ("Conv", "ConvTranspose"):
            kshape = at.get("kernel_shape")
            if not kshape or len(kshape) != 1:
                sys.exit(f"{node.op_type} {node.name}: kernel_shape {kshape}, only 1D supported")
            if at.get("group", 1) != 1:
                sys.exit(f"{node.op_type} {node.name}: group != 1 not supported")
            strides = at.get("strides", [1])
            pads = at.get("pads", [0, 0])
            dil = at.get("dilations", [1])
            if len(strides) != 1 or len(pads) != 2 or len(dil) != 1:
                sys.exit(f"{node.op_type} {node.name}: bad strides/pads/dilations {strides} {pads} {dil}")
            bias = ""
            if len(node.input) > 2 and node.input[2]:
                bias = node.input[2]
                if is_quant(bias):
                    sys.exit(f"{node.op_type} {node.name}: quantized bias not supported")
            # Fixed 5-slot layout [X, pk, sc, ze, bias] with "" placeholders
            # for absent optionals, matching the op schema's input count
            # (same convention as the GPT packed graphs).
            ze_in = ins[2] if mode == 0 else ""
            op_inputs = [node.input[0], ins[0], ins[1], ze_in, bias]
            op = "PackedConv" if node.op_type == "Conv" else "PackedConvTranspose"
            nattrs = {
                "mode": mode, "w_rows": rows, "w_cols": cols, "k": kshape[0],
                "stride": strides[0], "pad_l": pads[0], "pad_r": pads[1],
                "dilation": dil[0],
                "has_zero": 1 if mode == 0 else 0, "has_bias": 1 if bias else 0,
            }
            if node.op_type == "ConvTranspose":
                if at.get("auto_pad", "NOTSET") not in ("NOTSET",):
                    sys.exit(f"ConvTranspose {node.name}: auto_pad {at.get('auto_pad')} not supported")
                nattrs["out_pad"] = at.get("output_padding", [0])[0]
            nn = helper.make_node(op, op_inputs, list(node.output),
                                  name=node.name + "_packed", domain="potato", **nattrs)
        elif node.op_type == "Gather":
            axis = at.get("axis", 0)
            if axis != 0:
                sys.exit(f"Gather {node.name}: axis {axis} not supported")
            ze_in = ins[2] if mode == 0 else ""
            nn = helper.make_node("PackedGather", [ins[0], ins[1], ze_in, node.input[1]], list(node.output),
                                  name=node.name + "_packed", domain="potato",
                                  mode=mode, w_rows=rows, w_cols=cols,
                                  has_zero=1 if mode == 0 else 0)
        else:  # pragma: no cover - inventory check above already exits
            sys.exit(f"unexpected op {node.op_type}")
        for o in node.output:
            rewritten_out[o] = node.op_type
        new_nodes.append(nn)

    del g.node[:]
    g.node.extend(new_nodes)

    # Drop rewritten initializers; keep residuals + everything else.
    # Pad-consumed tables get their initializer REPLACED by the
    # pack-restored values (the folded graph still holds the pre-quant
    # fp32 originals, which are NOT what the shipping baked path uses).
    keep_inits = []
    for t in g.initializer:
        if t.name in rewritten_tensors:
            continue
        if t.name in pad_baked:
            keep_inits.append(onnx.numpy_helper.from_array(restore_entry(t.name), name=t.name))
        else:
            keep_inits.append(t)
    del g.initializer[:]
    g.initializer.extend(keep_inits)
    print(f"  pad-baked tables (pack-restored): {len(pad_baked)}")

    # Packed buffers become graph inputs (fed from the WeightPool).
    existing_inputs = {i.name for i in g.input}
    for name, shape in packed_inputs.items():
        if name in existing_inputs:
            continue
        dt = TensorProto.FLOAT if name.endswith("__sc") else TensorProto.UINT8
        g.input.append(helper.make_tensor_value_info(name, dt, shape))

    os.makedirs(OUTDIR, exist_ok=True)
    out_path = os.path.join(OUTDIR, "vits_wnfold.onnx")
    onnx.save(model, out_path)
    # Same sidecar format as make_packed_graphs.py: "packed_inputs" is a
    # list of per-tensor records keyed by the BARE tensor name (the pool
    # filter matches manifest entry names against "tensor").
    sidecar = {"packed_inputs": [sidecar_tensors[n] for n in sorted(sidecar_tensors)]}
    with open(out_path + ".json", "w") as f:
        json.dump(sidecar, f, indent=1, sort_keys=True)
    print(f"packed vits graph: {out_path}")
    print(f"  rewritten tensors: {len(rewritten_tensors)}, packed inputs: {len(packed_inputs)}")
    dead = [n for n in entries if is_quant(n) and n not in rewritten_tensors and n not in consumers]
    if dead:
        print(f"  note: {len(dead)} quantized manifest entries have no consumer: {dead[:4]}")

    if args.probes:
        os.makedirs(args.probes, exist_ok=True)

        def noiseless_probes(m, extra_out_names):
            gg = m.graph
            for node in gg.node:
                if node.op_type == "Constant" and NOISE_OUT in list(node.output):
                    t = node.attribute[0].t
                    arr = np.frombuffer(t.raw_data, dtype=np.float32)
                    assert np.count_nonzero(arr) > 0
                    t.raw_data = np.zeros(arr.shape, dtype=np.float32).tobytes()
            have = {o.name for o in gg.output}
            for name in extra_out_names:
                if name not in have:
                    gg.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, None))
            return m

        p = os.path.join(args.probes, "vits_packed_noiseless_probes.onnx")
        onnx.save(noiseless_probes(onnx.load(out_path), sorted(rewritten_out)), p)
        print(f"  probes: {p} ({len(rewritten_out)} rewritten outputs)")

        baked = onnx.load(os.path.join(ROOT, "work", "vits_w4_variants", "vits_w4hybrid.onnx"))
        baked_outs = set()
        for node in baked.graph.node:
            for o in node.output:
                if o in rewritten_out:
                    baked_outs.add(o)
        q = os.path.join(args.probes, "vits_bakedE_noiseless_probes.onnx")
        onnx.save(noiseless_probes(baked, sorted(baked_outs)), q)
        print(f"  probes: {q} ({len(baked_outs)} matching outputs)")


if __name__ == "__main__":
    main()
