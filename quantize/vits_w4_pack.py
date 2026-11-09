"""SoVITS vits W4 weight-only pack, same scheme as the GPT packs (quant_wo.py):
group-64 asymmetric along the input axis, nibble-packed; tensors whose flattened
input axis is not divisible by 64 fall back to W8 per-channel symmetric (annotated
in the manifest); 1D tensors stay fp32-residual.

Difference vs the GPT tool: the GPT tool only packs 2D tensors. vits weights are
mostly 3D conv weights, so each tensor is viewed as (shape[0], -1) -- output
channels x flattened input axis -- and grouped along that axis. For 2D tensors
this is exactly the GPT format (bit-identical layout).
Also bakes dequantized measurement graphs: variant A (all W4) always;
B (enc_p from the stage-1 W8 pack, rest W4) and C (enc_p fp32, rest W4) are
only baked/evaluated if the caller passes --bake-bc (gated on A's correlation).
"""
import os, sys, json
import numpy as np
import onnx
from onnx import numpy_helper

BASE = os.path.expanduser('~/workspace/openamika-rebuild/tts-engine')
PACK_DIR = f'{BASE}/wo-packs'
SRC = f'{BASE}/work/folded/vits_wnfold.onnx'
GROUP = 64

# --- identical math to work/quant_wo.py (do not change) ---
def pack_w8(w2):
    flat = w2
    scale = np.max(np.abs(flat), axis=1) / 127.0
    scale = np.where(scale == 0, 1.0, scale)
    q = np.round(flat / scale[:, None]).clip(-127, 127).astype(np.int8)
    return q, scale.astype(np.float32)

def dequant_w8(q, scale, shape):
    return (q.astype(np.float32) * scale[:, None]).reshape(shape)

def pack_w4(w2):
    rows, cols = w2.shape
    wg = w2.reshape(rows, cols // GROUP, GROUP)
    mn = wg.min(axis=2); mx = wg.max(axis=2)
    scale = np.where(mx > mn, (mx - mn) / 15.0, 1.0).astype(np.float32)
    zero = np.round(-mn / scale).clip(0, 15).astype(np.uint8)
    q = np.round(wg / scale[:, :, None] + zero[:, :, None].astype(np.float32)).clip(0, 15).astype(np.uint8)
    packed = (q[:, :, 0::2] << 4) | q[:, :, 1::2]
    return packed, scale, zero

def dequant_w4(packed, scale, zero, shape2):
    rows, cols = shape2
    q = np.empty((rows, cols // GROUP, GROUP), dtype=np.uint8)
    q[:, :, 0::2] = packed >> 4
    q[:, :, 1::2] = packed & 0xF
    return ((q.astype(np.float32) - zero[:, :, None].astype(np.float32)) * scale[:, :, None])

def module_of(name):
    if '.enc_p.' in name or '/enc_p/' in name: return 'enc_p'
    if '.flow.' in name or '/flow/' in name: return 'flow'
    if '.dec.' in name or '/dec/' in name: return 'dec'
    return 'other'

def main():
    bake_bc = '--bake-bc' in sys.argv
    m = onnx.load(SRC)
    weights, residuals = {}, {}
    for t in m.graph.initializer:
        arr = numpy_helper.to_array(t)
        if arr.dtype == np.float32 and arr.ndim >= 2:
            weights[t.name] = arr
        elif arr.dtype == np.float32:
            residuals[t.name] = arr

    store, manifest = {}, []
    restored_w4 = {}
    pack_stats = {}
    for name, w in weights.items():
        w2 = w.reshape(w.shape[0], -1)
        if w2.shape[1] % GROUP == 0:
            packed, scale, zero = pack_w4(w2)
            store[name] = packed; store[name + '@scale'] = scale; store[name + '@zero'] = zero
            rec2 = dequant_w4(packed, scale, zero, w2.shape)
            scheme = 'w4-group64-asymmetric' if w.ndim == 2 else 'w4-group64-asymmetric-flat'
        else:
            q, scale = pack_w8(w2)
            store[name] = q; store[name + '@scale'] = scale
            rec2 = dequant_w8(q, scale, w2.shape)
            scheme = 'w8-fallback (shape not W4-packable)'
        rec = rec2.reshape(w.shape)
        restored_w4[name] = rec
        err = float(np.max(np.abs(rec - w)))
        mod = module_of(name)
        st = pack_stats.setdefault(mod, {'tensors': 0, 'max_err': 0.0})
        st['tensors'] += 1; st['max_err'] = max(st['max_err'], err)
        manifest.append({'name': name, 'shape': list(w.shape), 'scheme': scheme,
                         'module': mod, 'restore_max_err': err})
    for name, w in residuals.items():
        store[name] = w
        manifest.append({'name': name, 'shape': list(w.shape), 'scheme': 'fp32-residual',
                         'module': module_of(name), 'restore_max_err': 0.0})

    pack_file = f'{PACK_DIR}/sovits_vits_w4.npz'
    np.savez(pack_file, **store)
    pack_bytes = os.path.getsize(pack_file)
    global_max = max(e['restore_max_err'] for e in manifest)
    info = {'pack_file': pack_file, 'pack_bytes': pack_bytes,
            'max_restore_err': global_max,
            'tensors_packed': len(weights), 'tensors_residual': len(residuals),
            'per_module_max_err': pack_stats, 'manifest': manifest}
    man = {'sovits_vits': {'w4': info}}
    json.dump(man, open(f'{PACK_DIR}/manifest_vits_w4.json', 'w'), indent=1)
    print('PACK', pack_file, pack_bytes, 'global max restore err', global_max)
    for mod, st in sorted(pack_stats.items()):
        print(' module', mod, st)

    # --- independent restore check: reload npz + manifest only ---
    store2 = np.load(pack_file)
    worst, worst_name = 0.0, None
    for e in manifest:
        name = e['name']
        if e['scheme'] == 'fp32-residual':
            rec = store2[name]
            src = residuals[name]
        else:
            src = weights[name]
            w2shape = (src.shape[0], src.size // src.shape[0])
            if e['scheme'].startswith('w4'):
                rec = dequant_w4(store2[name], store2[name + '@scale'], store2[name + '@zero'], w2shape).reshape(src.shape)
            else:
                rec = dequant_w8(store2[name], store2[name + '@scale'], src.shape)
        err = float(np.max(np.abs(rec - src)))
        if err > worst: worst, worst_name = err, name
    print('CHECK independent restore: global max err', worst, 'worst tensor', worst_name)
    assert abs(worst - global_max) < 1e-12, 'restore check disagrees with pack-time errors'

    def bake(restored, out_path):
        mm = onnx.load(SRC)
        repl = 0
        for t in mm.graph.initializer:
            if t.name in restored:
                t.CopyFrom(numpy_helper.from_array(restored[t.name], name=t.name)); repl += 1
        onnx.save(mm, out_path)
        print('BAKE', out_path, 'replaced', repl)
        return out_path

    os.makedirs(f'{BASE}/work/wo_graphs/w4', exist_ok=True)
    bake(restored_w4, f'{BASE}/work/wo_graphs/w4/vits_wnfold.onnx')

    if bake_bc:
        # enc_p restored from the stage-1 W8 pack for variant B
        sys.path.insert(0, PACK_DIR)
        import dequant_wo
        man1 = json.load(open(f'{PACK_DIR}/manifest.json'))
        w8_restored = dequant_wo.restore(man1['sovits_vits']['w8']['pack_file'],
                                         man1['sovits_vits']['w8']['manifest'])
        os.makedirs(f'{BASE}/work/vits_w4_variants', exist_ok=True)
        rb = dict(restored_w4)
        for name in rb:
            if module_of(name) == 'enc_p' and name in w8_restored:
                rb[name] = w8_restored[name]
        bake(rb, f'{BASE}/work/vits_w4_variants/vits_w4_encpw8.onnx')
        rc = dict(restored_w4)
        for name in rc:
            if module_of(name) == 'enc_p':
                rc[name] = weights[name]
        bake(rc, f'{BASE}/work/vits_w4_variants/vits_w4_encpfp32.onnx')
    print('PACK_DONE')

if __name__ == '__main__':
    main()
