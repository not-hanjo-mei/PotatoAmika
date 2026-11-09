"""Reference dequantizer for the weight-only packs in this directory.

Reads manifest.json + the .npz packs, restores fp32 weights, and (with --check)
compares them against the source fp32 graphs to prove the only error in a pack
is the quantization itself. Usage:
    python dequant_wo.py --check <dir-of-folded-fp32-graphs>
"""
import json, os, sys
import numpy as np

GROUP = 64

def dequant_w8(q, scale):
    flat = q.reshape(q.shape[0], -1).astype(np.float32)
    return (flat * scale[:, None]).reshape(q.shape)

def dequant_w4(packed, scale, zero, shape):
    rows, cols = shape
    q = np.empty((rows, cols // GROUP, GROUP), dtype=np.uint8)
    q[:, :, 0::2] = packed >> 4
    q[:, :, 1::2] = packed & 0xF
    return ((q.astype(np.float32) - zero[:, :, None].astype(np.float32)) * scale[:, :, None]).reshape(shape)

def restore(pack_npz, manifest_entries):
    store = np.load(pack_npz)
    out = {}
    for e in manifest_entries:
        name = e['name']
        if e['scheme'] == 'fp32-residual':
            out[name] = store[name]
        elif e['scheme'].startswith('w8'):
            out[name] = dequant_w8(store[name], store[name + '@scale'])
        elif e['scheme'].startswith('w4'):
            out[name] = dequant_w4(store[name], store[name + '@scale'], store[name + '@zero'], e['shape'])
    return out

if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__))
    manifest = json.load(open(os.path.join(here, 'manifest.json')))
    src_dir = sys.argv[sys.argv.index('--check') + 1] if '--check' in sys.argv else None
    import onnx
    from onnx import numpy_helper
    for tag, schemes in manifest.items():
        for scheme, info in schemes.items():
            restored = restore(info['pack_file'], info['manifest'])
            if src_dir:
                src_name = {'gpt_encoder': 't2s_encoder_fp32.onnx', 'gpt_first': 't2s_first_stage_decoder_fp32.onnx',
                            'gpt_stage': 't2s_stage_decoder_fp32.onnx', 'sovits_vits': 'vits_wnfold.onnx'}[tag]
                m = onnx.load(os.path.join(src_dir, src_name))
                worst = 0.0
                for t in m.graph.initializer:
                    if t.name in restored:
                        worst = max(worst, float(np.max(np.abs(restored[t.name] - numpy_helper.to_array(t)))))
                print(f'{tag} {scheme}: restored {len(restored)} tensors, max err vs source = {worst:.6g}')
            else:
                print(f'{tag} {scheme}: restored {len(restored)} tensors')
