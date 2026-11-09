"""W4-plus packs: activation-aware PTQ for the GPT graphs.
- Case B (Gemm transB=1, stored (out,in), groups along in): textbook GPTQ,
  columns processed in order, error feedback via Cholesky of inv(H),
  group scale/zero from MSE-optimal clip search (diag(H)-weighted).
- Case A (MatMul, stored (in,out), groups along out per input row): GPTQ
  transposed - input rows processed in order, the whole row quantized at its
  step (its group segments live inside the row), feedback to later rows.
- Tensors without a MatMul/Gemm consumer (embeddings): MSE-clip RTN.
Non-W4 entries (fp32-residual, w8-fallback) are copied verbatim from the
current W4 packs. Pack format byte-compatible with wo-packs (dequant_wo.py).
Outputs: wo-packs/gpt_*_w4plus.npz, wo-packs/manifest_w4plus.json,
work/wo_graphs/w4plus/*.onnx, work/ptqplus/pack_report.json."""
import os, sys, json, time
import numpy as np
import onnx
from onnx import numpy_helper

BASE = os.path.expanduser('~/workspace/openamika-rebuild/tts-engine')
W = f'{BASE}/work'
PACK_DIR = f'{BASE}/wo-packs'
PTQ = f'{W}/ptqplus'
GROUP = 64
SHRINKS = np.round(np.arange(1.0, 0.49, -0.025), 4)  # 1.00 .. 0.50

sys.path.insert(0, PACK_DIR)
import dequant_wo as dq

man = json.load(open(f'{PACK_DIR}/manifest.json'))

def clip_search(block, wts):
    """block: (..., 64) values of one or more quantization units (last axis is
    the group); wts: (64,) per-position weights or None.
    Returns scale, zero shaped block.shape[:-1], minimizing weighted MSE over
    a symmetric shrink grid of the [min,max] range."""
    mn = block.min(axis=-1); mx = block.max(axis=-1); rng = mx - mn
    best_err = np.full(mn.shape, np.inf)
    best_scale = np.ones(mn.shape); best_zero = np.zeros(mn.shape)
    for f in SHRINKS:
        lo = mn + (1 - f) / 2 * rng; hi = mx - (1 - f) / 2 * rng
        scale = np.where(hi > lo, (hi - lo) / 15.0, 1.0)
        zero = np.clip(np.round(-lo / scale), 0, 15)
        q = np.clip(np.round(block / scale[..., None] + zero[..., None]), 0, 15)
        rec = (q - zero[..., None]) * scale[..., None]
        err = (block - rec) ** 2
        if wts is not None:
            err = err * wts
        err = err.sum(axis=-1)
        better = err < best_err
        best_err = np.where(better, err, best_err)
        best_scale = np.where(better, scale, best_scale)
        best_zero = np.where(better, zero, best_zero)
    return best_scale, best_zero

def hinv_of(H):
    H = (H + H.T) / 2
    dead = np.diag(H) == 0
    damp = 0.01 * float(np.mean(np.diag(H)[~dead])) if (~dead).any() else 1.0
    H = H.copy(); H[np.diag_indices_from(H)] += damp
    Hinv = np.linalg.cholesky(np.linalg.inv(H)).T  # upper, inv(H) = Hinv^T Hinv
    return Hinv, dead, damp

def gptq_caseB(Wm, H):
    """Wm stored (out, in); standard GPTQ along axis 1."""
    out, inn = Wm.shape
    W = Wm.astype(np.float64).copy()
    Hinv, dead, damp = hinv_of(H)
    if dead.any():
        W[:, dead] = 0
    diagH = np.maximum(np.diag(H), 0)
    qint = np.zeros((out, inn), dtype=np.uint8)
    scales = np.zeros((out, inn // GROUP), dtype=np.float64)
    zeros_ = np.zeros((out, inn // GROUP), dtype=np.float64)
    for i in range(inn):
        if i % GROUP == 0:
            s, z = clip_search(W[:, i:i + GROUP], diagH[i:i + GROUP])
            scales[:, i // GROUP] = s; zeros_[:, i // GROUP] = z
        s = scales[:, i // GROUP]; z = zeros_[:, i // GROUP]
        q = np.clip(np.round(W[:, i] / s + z), 0, 15)
        qint[:, i] = q.astype(np.uint8)
        qv = (q - z) * s
        err = (W[:, i] - qv) / Hinv[i, i]
        if i + 1 < inn:
            W[:, i + 1:] -= np.outer(err, Hinv[i, i + 1:])
    return qint, scales, zeros_, damp

def gptq_caseA(Wm, H):
    """Wm stored (in, out); GPTQ along axis 0 (input rows)."""
    inn, out = Wm.shape
    W = Wm.astype(np.float64).copy()
    Hinv, dead, damp = hinv_of(H)
    if dead.any():
        W[dead, :] = 0
    qint = np.zeros((inn, out), dtype=np.uint8)
    scales = np.zeros((inn, out // GROUP), dtype=np.float64)
    zeros_ = np.zeros((inn, out // GROUP), dtype=np.float64)
    for i in range(inn):
        row = W[i]
        s, z = clip_search(row.reshape(out // GROUP, GROUP), None)
        scales[i] = s; zeros_[i] = z
        sfull = np.repeat(s, GROUP); zfull = np.repeat(z, GROUP)
        q = np.clip(np.round(row / sfull + zfull), 0, 15)
        qint[i] = q.astype(np.uint8)
        qv = (q - zfull) * sfull
        err = (row - qv) / Hinv[i, i]
        if i + 1 < inn:
            W[i + 1:, :] -= Hinv[i, i + 1:][:, None] * err[None, :]
    return qint, scales, zeros_, damp

def rtn_clip(Wm):
    rows, cols = Wm.shape
    seg = Wm.reshape(rows, cols // GROUP, GROUP).astype(np.float64)
    s, z = clip_search(seg, None)
    q = np.clip(np.round(seg / s[:, :, None] + z[:, :, None]), 0, 15).astype(np.uint8)
    return q.reshape(rows, cols), s, z

def pack_q(qint):
    rows, cols = qint.shape
    q = qint.reshape(rows, cols // GROUP, GROUP)
    return ((q[:, :, 0::2] << 4) | q[:, :, 1::2]).astype(np.uint8)

GRAPHS = {'gpt_encoder': 't2s_encoder_fp32.onnx',
           'gpt_first': 't2s_first_stage_decoder_fp32.onnx',
           'gpt_stage': 't2s_stage_decoder_fp32.onnx'}

def process(tag):
    t0 = time.time()
    m = onnx.load(f'{W}/folded/{GRAPHS[tag]}')
    inits = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}
    entries = man[tag]['w4']['manifest']
    targets = json.load(open(f'{PTQ}/targets_{tag}.json'))
    info = targets['info']
    Hz = np.load(f'{PTQ}/H_{tag}.npz')
    cur = np.load(f'{PACK_DIR}/{tag}_w4.npz')
    store, mentries, restored = {}, [], {}
    stats = {'gptqA': 0, 'gptqB': 0, 'rtn': 0, 'copied': 0, 'damps': []}
    for e in entries:
        name, scheme = e['name'], e['scheme']
        if not scheme.startswith('w4'):
            # copy verbatim from current pack (residual / w8-fallback)
            store[name] = cur[name]
            if name + '@scale' in cur:
                store[name + '@scale'] = cur[name + '@scale']
            restored[name] = dq.restore(f'{PACK_DIR}/{tag}_w4.npz', [e])[name]
            mentries.append(e); stats['copied'] += 1
            continue
        w = inits[name]
        if name in info:
            H = Hz[name]; rows = int(Hz[name + '@rows'])
            if info[name]['case'] == 'B':
                qint, s, z, damp = gptq_caseB(w, H); stats['gptqB'] += 1
            else:
                qint, s, z, damp = gptq_caseA(w, H); stats['gptqA'] += 1
            stats['damps'].append(float(damp))
            print(tag, name, info[name]['case'], 'rows', rows, f'{time.time()-t0:.0f}s', flush=True)
        else:
            qint, s, z = rtn_clip(w); stats['rtn'] += 1
            print(tag, name, 'RTN-clip (no linear consumer)', flush=True)
        store[name] = pack_q(qint)
        store[name + '@scale'] = s.astype(np.float32)
        store[name + '@zero'] = z.astype(np.uint8)
        mentries.append({'name': name, 'shape': list(w.shape), 'scheme': scheme})
        rec = dq.dequant_w4(store[name], store[name + '@scale'], store[name + '@zero'], list(w.shape))
        restored[name] = rec
    pack_file = f'{PACK_DIR}/{tag}_w4plus.npz'
    np.savez(pack_file, **store)
    # restore check vs folded source
    worst, worst_name = 0.0, None
    for name, rec in restored.items():
        err = float(np.max(np.abs(rec - inits[name]))) if name in inits else 0.0
        if err > worst: worst, worst_name = err, name
    # bake graph
    gdir = f'{W}/wo_graphs/w4plus'; os.makedirs(gdir, exist_ok=True)
    out_file = f'{gdir}/{GRAPHS[tag]}'
    repl = 0
    for t in m.graph.initializer:
        if t.name in restored:
            t.CopyFrom(numpy_helper.from_array(restored[t.name].astype(np.float32), name=t.name))
            repl += 1
    onnx.save(m, out_file)
    info_out = {'pack_file': pack_file, 'pack_bytes': os.path.getsize(pack_file),
                'graph_file': out_file, 'max_restore_err': worst, 'worst_tensor': worst_name,
                'baked': repl, 'manifest': mentries, 'stats': stats}
    print(tag, 'pack bytes', info_out['pack_bytes'], 'restore max err', worst, 'stats', stats, flush=True)
    return info_out

if __name__ == '__main__':
    # per-tag invocation supported (python gptq_pack.py gpt_first) so a crash
    # only loses the in-flight graph; manifest_w4plus.json is updated per tag.
    tags = sys.argv[1:] or ['gpt_encoder', 'gpt_first', 'gpt_stage']
    mf_path = f'{PACK_DIR}/manifest_w4plus.json'
    report = json.load(open(mf_path)) if os.path.exists(mf_path) else {}
    for tag in tags:
        report[tag] = {'w4plus': process(tag)}
        json.dump(report, open(mf_path, 'w'), indent=1)
    if all(t in report for t in ['gpt_encoder', 'gpt_first', 'gpt_stage']):
        tot = sum(report[t]['w4plus']['pack_bytes'] for t in ['gpt_encoder', 'gpt_first', 'gpt_stage'])
        print('TOTAL w4plus bytes:', tot, 'vs current w4 92735008, ratio', tot / 92735008)
        json.dump({'total_bytes': tot}, open(f'{PTQ}/pack_report.json', 'w'))
    print('GPTQ_PACK_DONE', tags)
