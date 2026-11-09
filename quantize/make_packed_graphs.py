"""Stage 3: rewrite the two GPT decoder graphs for packed-resident weights.

For each decoder graph (first stage / stage), every MatMul or Gemm whose
weight initializer comes from the W4 pack is replaced by a custom op
    potato::PackedMatMul(X, <name>__pk, <name>__sc[, <name>__ze][, bias])
and the weight initializer is dropped; packed bytes / scales / zeros become
graph inputs, fed at runtime from one shared pool (the two graphs' decoder
weights are byte-identical, so the pool dedupes them).

Modes (attribute `mode`):
  0 = MatMul layout, W4:  W stored (d_in, d_out), groups of 64 along d_out
  1 = Gemm transB=1, W4:  W stored (d_out, d_in), groups of 64 along d_in
  2 = MatMul layout, W8:  W stored (d_in, d_out), per-row (d_in) scales
  3 = Gemm transB=1, W8:  (not present in these graphs; kernel supports it)

All other pack entries (embedding Gather, fp32 residuals, biases, norms) are
dequantized with the stage-1 reference code and baked back into the graph as
fp32 initializers, so the output graph is self-contained apart from the
packed MatMul inputs. Non-pack initializers pass through untouched.

Inputs:  work/folded/<tpl> + wo-packs/manifest.json (section, scheme w4)
Outputs: host/packed_graphs/<tpl> and host/packed_graphs/<tpl>.json sidecars
         listing the packed inputs (name, tensor, scheme, rows, cols, mode).
"""
import json
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "wo-packs"))
import dequant_wo  # noqa: E402

OUT_DIR = os.path.join(BASE, "host", "packed_graphs")
os.makedirs(OUT_DIR, exist_ok=True)

JOBS = [
    ("gpt_first", "t2s_first_stage_decoder_fp32.onnx"),
    ("gpt_stage", "t2s_stage_decoder_fp32.onnx"),
]


def main():
    manifest = json.load(open(os.path.join(BASE, "wo-packs", "manifest.json")))
    for section, tpl in JOBS:
        info = manifest[section]["w4"]
        entries = {e["name"]: e for e in info["manifest"]}
        pack_path = os.path.join(BASE, "wo-packs", info["pack_file"])
        restored = dequant_wo.restore(pack_path, info["manifest"])
        store = np.load(pack_path)

        model = onnx.load(os.path.join(BASE, "work", "folded", tpl))
        g = model.graph
        inits = {t.name: t for t in g.initializer}

        # Which pack tensors feed a MatMul/Gemm as weight (input[1])?
        rewrite = {}  # tensor name -> (node, kind)
        for node in g.node:
            if node.op_type in ("MatMul", "Gemm") and len(node.input) > 1 and node.input[1] in entries:
                e = entries[node.input[1]]
                assert e["scheme"].startswith(("w4", "w8")), (node.name, e["scheme"])
                rewrite[node.input[1]] = node

        packed_inputs = []
        new_nodes = []
        for node in g.node:
            if node.op_type in ("MatMul", "Gemm") and len(node.input) > 1 and node.input[1] in rewrite:
                wname = node.input[1]
                e = entries[wname]
                rows, cols = e["shape"]
                is_w4 = e["scheme"].startswith("w4")
                if node.op_type == "Gemm":
                    attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
                    assert attrs.get("alpha", 1.0) == 1.0 and attrs.get("beta", 1.0) == 1.0
                    assert attrs.get("transB", 0) == 1 and attrs.get("transA", 0) == 0
                    mode = 1 if is_w4 else 3
                else:
                    mode = 0 if is_w4 else 2
                pk = store[wname]
                sc = store[wname + "@scale"]
                inputs = [node.input[0], wname + "__pk", wname + "__sc"]
                if is_w4:
                    inputs.append(wname + "__ze")
                if node.op_type == "Gemm" and len(node.input) > 2 and node.input[2]:
                    inputs.append(node.input[2])  # bias stays an initializer input
                nattrs = [
                    helper.make_attribute("mode", mode),
                    helper.make_attribute("rows", rows),
                    helper.make_attribute("cols", cols),
                    helper.make_attribute("has_zero", 1 if is_w4 else 0),
                    helper.make_attribute("has_bias", 1 if len(inputs) > (4 if is_w4 else 3) else 0),
                ]
                new_nodes.append(helper.make_node("PackedMatMul", inputs, list(node.output),
                                                  name=node.name + "_packed", domain="potato",
                                                  **{a.name: helper.get_attribute_value(a) for a in nattrs}))
                packed_inputs.append({
                    "tensor": wname, "mode": mode, "rows": rows, "cols": cols,
                    "scheme": e["scheme"],
                    "pk_shape": list(pk.shape), "pk_dtype": str(pk.dtype),
                    "sc_shape": list(sc.shape),
                    "ze_shape": list(store[wname + "@zero"].shape) if is_w4 else None,
                })
            else:
                new_nodes.append(node)
        del g.node[:]
        g.node.extend(new_nodes)

        # Rebuild initializer list: drop rewritten weights; bake the restored
        # fp32 values for every remaining pack entry; keep the rest as-is.
        keep = []
        for t in g.initializer:
            if t.name in rewrite:
                continue
            if t.name in restored:
                keep.append(numpy_helper.from_array(np.ascontiguousarray(restored[t.name], dtype=np.float32), name=t.name))
            else:
                keep.append(t)
        del g.initializer[:]
        g.initializer.extend(keep)

        # Declare the packed tensors as graph inputs.
        existing_inputs = {i.name for i in g.input}
        for pi in packed_inputs:
            wname = pi["tensor"]
            for suffix, shape, dtype in (
                # pk is always declared UINT8: for w8 entries the int8 payload
                # is fed byte-identically and reinterpreted by the kernel.
                ("__pk", pi["pk_shape"], TensorProto.UINT8),
                ("__sc", pi["sc_shape"], TensorProto.FLOAT),
                ("__ze", pi["ze_shape"], TensorProto.UINT8),
            ):
                if shape is None:
                    continue
                nm = wname + suffix
                if nm not in existing_inputs:
                    g.input.append(helper.make_tensor_value_info(nm, dtype, shape))

        # Register the custom domain.
        have = {(o.domain, o.version) for o in model.opset_import}
        if ("potato", 1) not in have:
            model.opset_import.append(helper.make_opsetid("potato", 1))

        # NOTE: the source templates themselves fail onnx.checker (a graph
        # input value_info without type) while ORT loads them fine, so instead
        # of check_model we assert the rewrite's own invariants here and let
        # session creation be the structural test.
        pm = [n for n in g.node if n.op_type == "PackedMatMul"]
        assert len(pm) == len(rewrite) == len(packed_inputs)
        init_names = {t.name for t in g.initializer}
        assert not (init_names & set(rewrite)), "rewritten weight still an initializer"
        in_names = {i.name for i in g.input}
        for pi in packed_inputs:
            assert pi["tensor"] + "__pk" in in_names and pi["tensor"] + "__sc" in in_names
        out_path = os.path.join(OUT_DIR, tpl)
        onnx.save(model, out_path)
        side = {"graph": tpl, "section": section, "packed_inputs": packed_inputs}
        json.dump(side, open(out_path + ".json", "w"), indent=1)
        print(f"{tpl}: rewrote {len(rewrite)} MatMul/Gemm, packed inputs {len(packed_inputs)}, "
              f"size {os.path.getsize(out_path)} B")


if __name__ == "__main__":
    main()
