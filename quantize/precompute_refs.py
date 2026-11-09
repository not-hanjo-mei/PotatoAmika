"""Offline reference feature precompute for all 69 emotion references.

Mirrors stage-1 inputs.py (Genie ReferenceAudio frontend) and additionally
computes ge / ge_advanced with the fp32 prompt_encoder, so the C++ host needs
no HuBERT / RoBERTa / speaker-encoder / prompt_encoder at runtime.

Output: host/refpacks/<tag>.rpk + host/refpacks/index.json
RPK1 format (little-endian):
  char[4] magic 'RPK1'; u32 tensor_count
  per tensor: u16 name_len, name bytes, u8 dtype (0=f32, 1=i64), u8 ndim,
              u32 dims[ndim], u64 data_bytes, raw C-order data
Tensors: ref_seq i64[1,L], ref_bert f32[L,1024], ssl_content f32[1,768,T],
         ge f32[1,1024,1], ge_advanced f32[1,512,1]
"""
import os, sys, json, struct
BASE = os.path.expanduser('~/workspace/openamika-rebuild/tts-engine')
os.environ.setdefault('GENIE_DATA_DIR', f'{BASE}/GenieData')
sys.path.insert(0, f'{BASE}/venv/lib/python3.12/site-packages')
import numpy as np
import onnxruntime as ort
from genie_tts.Audio.ReferenceAudio import ReferenceAudio
from genie_tts.ModelManager import model_manager

OUT = f'{BASE}/host/refpacks'
os.makedirs(OUT, exist_ok=True)
emo = json.load(open(f'{BASE}/refs/emotions.json'))

so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
prompt_sess = ort.InferenceSession(f'{BASE}/work/fp32_full/prompt_encoder_fp32.onnx',
                                   sess_options=so, providers=['CPUExecutionProvider'])

def write_rpk(path, tensors):
    with open(path, 'wb') as f:
        f.write(b'RPK1')
        f.write(struct.pack('<I', len(tensors)))
        for name, arr in tensors:
            arr = np.ascontiguousarray(arr)
            dt = 0 if arr.dtype == np.float32 else 1
            assert arr.dtype in (np.float32, np.int64), (name, arr.dtype)
            nb = name.encode()
            f.write(struct.pack('<H', len(nb))); f.write(nb)
            f.write(struct.pack('<BB', dt, arr.ndim))
            for d in arr.shape: f.write(struct.pack('<I', int(d)))
            data = arr.tobytes()
            f.write(struct.pack('<Q', len(data))); f.write(data)

index = {}
if os.path.exists(f'{OUT}/index.json'):
    index = json.load(open(f'{OUT}/index.json'))

def rpk_shapes(path):
    import struct as _st
    shapes = {}
    with open(path, 'rb') as f:
        buf = f.read()
    assert buf[:4] == b'RPK1'
    pos = 4
    (cnt,) = _st.unpack_from('<I', buf, pos); pos += 4
    for _ in range(cnt):
        (nl,) = _st.unpack_from('<H', buf, pos); pos += 2
        name = buf[pos:pos+nl].decode(); pos += nl
        nd = buf[pos+1]; pos += 2
        dims = _st.unpack_from(f'<{nd}I', buf, pos); pos += 4 * nd
        (nb,) = _st.unpack_from('<Q', buf, pos); pos += 8 + nb
        shapes[name] = dims
    return shapes

# Seed index entries for refpacks already on disk (crash/OOM resume support).
for fn in os.listdir(OUT):
    if fn.endswith('.rpk'):
        tag = fn[:-4]
        if tag in emo and tag not in index:
            sh = rpk_shapes(f'{OUT}/{fn}')
            index[tag] = {'file': fn, 'audio': emo[tag]['audio'], 'text': emo[tag]['text'],
                          'ref_seq_len': int(sh['ref_seq'][1]), 'ssl_len': int(sh['ssl_content'][2])}
only = os.environ.get('TAGS_ONLY')
tags = sorted(k for k in emo.keys() if not k.startswith('_'))
if only:
    tags = [t for t in tags if t in only.split(',')]
failed = []
for i, tag in enumerate(tags):
    if os.environ.get('SKIP_EXISTING') and os.path.exists(f'{OUT}/{tag}.rpk'):
        print(f'[skip] {tag}', flush=True)
        continue
    info = emo[tag]
    try:
        ra = ReferenceAudio(f'{BASE}/refs/{info["audio"]}', info['text'], 'chinese')
        if not model_manager.load_sv_model():
            raise RuntimeError('SV model failed to load')
        sv = model_manager.speaker_verification_model.run(None, {'waveform': ra.audio_16k})[0]
        ge, ge_adv = prompt_sess.run(None, {'ref_audio': ra.audio_32k, 'sv_emb': sv})
        write_rpk(f'{OUT}/{tag}.rpk', [
            ('ref_seq', ra.phonemes_seq.astype(np.int64)),
            ('ref_bert', ra.text_bert.astype(np.float32)),
            ('ssl_content', ra.ssl_content.astype(np.float32)),
            ('ge', ge.astype(np.float32)),
            ('ge_advanced', ge_adv.astype(np.float32)),
        ])
        index[tag] = {'file': f'{tag}.rpk', 'audio': info['audio'], 'text': info['text'],
                      'ref_seq_len': int(ra.phonemes_seq.shape[1]),
                      'ssl_len': int(ra.ssl_content.shape[2])}
        print(f'[{i+1}/{len(tags)}] {tag} seq={ra.phonemes_seq.shape[1]} ssl={ra.ssl_content.shape[2]}', flush=True)
    except Exception as e:
        failed.append(tag)
        print(f'[FAIL] {tag}: {type(e).__name__}: {e}', flush=True)

if failed:
    index['_failed'] = failed
json.dump(index, open(f'{OUT}/index.json', 'w'), ensure_ascii=False, indent=1)
print('REFPACKS_DONE', len(index) - (1 if '_failed' in index else 0), 'failed:', failed)
