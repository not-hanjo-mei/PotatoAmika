"""Practical hybrid pack (variant E): enc_p and the quantizer codebook at W8
per-channel, everything else exactly as the W4 pack (group-64 asymmetric,
W8 fallback for non-divisible shapes, fp32 residuals). Built because variant D
showed flow/dec tolerate W4 while enc_p + codebook are the two blockers.
Emits a real pack + manifest (same formats as the other packs) and a baked
measurement graph."""
import os, sys, json
import numpy as np, onnx
from onnx import numpy_helper

BASE = os.path.expanduser('~/workspace/openamika-rebuild/tts-engine')
sys.path.insert(0, f'{BASE}/work')
from vits_w4_pack import pack_w4, pack_w8, dequant_w4, dequant_w8, module_of, SRC, GROUP  # noqa

m = onnx.load(SRC)
weights, residuals = {}, {}
for t in m.graph.initializer:
    arr = numpy_helper.to_array(t)
    if arr.dtype == np.float32 and arr.ndim >= 2:
        weights[t.name] = arr
    elif arr.dtype == np.float32:
        residuals[t.name] = arr

store, manifest, restored = {}, [], {}
for name, w in weights.items():
    mod = module_of(name)
    is_codebook = 'quantizer' in name
    w2 = w.reshape(w.shape[0], -1)
    if mod == 'enc_p' or is_codebook:
        q, scale = pack_w8(w2)
        store[name] = q; store[name + '@scale'] = scale
        rec = dequant_w8(q, scale, w.shape)
        scheme = 'w8-per-channel-symmetric'
    elif w2.shape[1] % GROUP == 0:
        packed, scale, zero = pack_w4(w2)
        store[name] = packed; store[name + '@scale'] = scale; store[name + '@zero'] = zero
        rec = dequant_w4(packed, scale, zero, w2.shape).reshape(w.shape)
        scheme = 'w4-group64-asymmetric' if w.ndim == 2 else 'w4-group64-asymmetric-flat'
    else:
        q, scale = pack_w8(w2)
        store[name] = q; store[name + '@scale'] = scale
        rec = dequant_w8(q, scale, w.shape)
        scheme = 'w8-fallback (shape not W4-packable)'
    restored[name] = rec
    manifest.append({'name': name, 'shape': list(w.shape), 'scheme': scheme,
                     'module': mod, 'restore_max_err': float(np.max(np.abs(rec - w)))})
for name, w in residuals.items():
    store[name] = w
    manifest.append({'name': name, 'shape': list(w.shape), 'scheme': 'fp32-residual',
                     'module': module_of(name), 'restore_max_err': 0.0})

pack_file = f'{BASE}/wo-packs/sovits_vits_w4hybrid.npz'
np.savez(pack_file, **store)
info = {'pack_file': pack_file, 'pack_bytes': os.path.getsize(pack_file),
        'max_restore_err': max(e['restore_max_err'] for e in manifest),
        'tensors_packed': len(weights), 'tensors_residual': len(residuals), 'manifest': manifest}
json.dump({'sovits_vits': {'w4hybrid': info}}, open(f'{BASE}/wo-packs/manifest_vits_w4hybrid.json', 'w'), indent=1)
print('PACK', pack_file, info['pack_bytes'], 'max restore err', info['max_restore_err'])

mm = onnx.load(SRC)
repl = 0
for t in mm.graph.initializer:
    if t.name in restored:
        t.CopyFrom(numpy_helper.from_array(restored[t.name], name=t.name)); repl += 1
out = f'{BASE}/work/vits_w4_variants/vits_w4hybrid.onnx'
onnx.save(mm, out)
print('BAKE', out, 'replaced', repl)
print('HYBRID_DONE')
