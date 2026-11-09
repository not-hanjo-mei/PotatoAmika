"""Weight-only packs: GPT W8 (per-channel) + W4 (group 64, asymmetric, nibble-packed);
SoVITS vits W8. Emits packs + manifest + dequant-baked measurement graphs + round-trip errors."""
import os, json
import numpy as np
import onnx
from onnx import numpy_helper

BASE = os.path.expanduser('~/workspace/openamika-rebuild/tts-engine')
PACK_DIR = f'{BASE}/wo-packs'
os.makedirs(PACK_DIR, exist_ok=True)
GROUP = 64

def pack_w8(w):
    """per-output-channel symmetric int8."""
    flat = w.reshape(w.shape[0], -1)
    scale = np.max(np.abs(flat), axis=1) / 127.0
    scale = np.where(scale == 0, 1.0, scale)
    q = np.round(flat / scale[:, None]).clip(-127, 127).astype(np.int8).reshape(w.shape)
    return q, scale.astype(np.float32)

def dequant_w8(q, scale):
    flat = q.reshape(q.shape[0], -1).astype(np.float32)
    return (flat * scale[:, None]).reshape(q.shape)

def pack_w4(w):
    """group-wise asymmetric along axis 1, group=64; nibble-packed along last axis."""
    assert w.ndim == 2, 'W4 group packing defined for 2D weights'
    rows, cols = w.shape
    assert cols % GROUP == 0, f'cols {cols} not divisible by group {GROUP}'
    wg = w.reshape(rows, cols // GROUP, GROUP)
    mn = wg.min(axis=2); mx = wg.max(axis=2)
    scale = np.where(mx > mn, (mx - mn) / 15.0, 1.0).astype(np.float32)
    zero = np.round(-mn / scale).clip(0, 15).astype(np.uint8)
    q = np.round(wg / scale[:, :, None] + zero[:, :, None].astype(np.float32)).clip(0, 15).astype(np.uint8)
    packed = (q[:, :, 0::2] << 4) | q[:, :, 1::2]
    return packed, scale, zero

def dequant_w4(packed, scale, zero, shape):
    rows, cols = shape
    q = np.empty((rows, cols // GROUP, GROUP), dtype=np.uint8)
    q[:, :, 0::2] = packed >> 4
    q[:, :, 1::2] = packed & 0xF
    w = (q.astype(np.float32) - zero[:, :, None].astype(np.float32)) * scale[:, :, None]
    return w.reshape(shape)

def bake(graph_path, restored, out_path):
    m = onnx.load(graph_path)
    repl = 0
    for t in m.graph.initializer:
        if t.name in restored:
            arr = restored[t.name]
            t.CopyFrom(numpy_helper.from_array(arr, name=t.name))
            repl += 1
    onnx.save(m, out_path)
    return repl

def process(graph_path, tag, schemes):
    m = onnx.load(graph_path)
    weights = {}
    residuals = {}
    for t in m.graph.initializer:
        arr = numpy_helper.to_array(t)
        if arr.dtype == np.float32 and arr.ndim >= 2:
            weights[t.name] = arr
        elif arr.dtype == np.float32:
            residuals[t.name] = arr
    out = {}
    for scheme in schemes:
        store, manifest, restored = {}, [], {}
        max_err = 0.0
        for name, w in weights.items():
            if scheme == 'w8':
                q, scale = pack_w8(w)
                store[name] = q; store[name + '@scale'] = scale
                rec = dequant_w8(q, scale)
                manifest.append({'name': name, 'shape': list(w.shape), 'scheme': 'w8-per-channel-symmetric'})
            else:
                if w.ndim == 2 and w.shape[1] % GROUP == 0:
                    packed, scale, zero = pack_w4(w)
                    store[name] = packed; store[name + '@scale'] = scale; store[name + '@zero'] = zero
                    rec = dequant_w4(packed, scale, zero, w.shape)
                    manifest.append({'name': name, 'shape': list(w.shape), 'scheme': f'w4-group{GROUP}-asymmetric'})
                else:
                    q, scale = pack_w8(w)
                    store[name] = q; store[name + '@scale'] = scale
                    rec = dequant_w8(q, scale)
                    manifest.append({'name': name, 'shape': list(w.shape), 'scheme': 'w8-fallback (shape not W4-packable)'})
            restored[name] = rec
            max_err = max(max_err, float(np.max(np.abs(rec - w))))
        for name, w in residuals.items():
            store[name] = w
            restored[name] = w
            manifest.append({'name': name, 'shape': list(w.shape), 'scheme': 'fp32-residual'})
        pack_file = f'{PACK_DIR}/{tag}_{scheme}.npz'
        np.savez(pack_file, **store)
        gdir = f'{BASE}/work/wo_graphs/{scheme}'
        os.makedirs(gdir, exist_ok=True)
        out_file = f'{gdir}/{os.path.basename(graph_path)}'
        repl = bake(graph_path, restored, out_file)
        out[scheme] = {'pack_file': pack_file, 'pack_bytes': os.path.getsize(pack_file),
                       'graph_file': out_file, 'graph_bytes': os.path.getsize(out_file),
                       'max_restore_err': max_err, 'tensors_packed': len(weights),
                       'tensors_residual': len(residuals), 'baked': repl, 'manifest': manifest}
        print(tag, scheme, 'pack', out[scheme]['pack_bytes'], 'restore max err', max_err, flush=True)
    return out

report = {}
GPT = [('gpt_encoder', f'{BASE}/work/folded/t2s_encoder_fp32.onnx'),
       ('gpt_first', f'{BASE}/work/folded/t2s_first_stage_decoder_fp32.onnx'),
       ('gpt_stage', f'{BASE}/work/folded/t2s_stage_decoder_fp32.onnx')]
for tag, path in GPT:
    report[tag] = process(path, tag, ['w8', 'w4'])
report['sovits_vits'] = process(f'{BASE}/work/folded/vits_wnfold.onnx', 'sovits_vits', ['w8'])
json.dump(report, open(f'{PACK_DIR}/manifest.json', 'w'), indent=1)
print('WO_DONE')
