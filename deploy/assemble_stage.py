#!/usr/bin/env python3
"""Assemble the PotatoAmika deploy staging tree (potatoamika/) from sources,
then verify every file that existed in the previous package against the old
MANIFEST.md table (sha256 + size). Expected diffs: bin/scheduler (new /v1/say
build), ui/* (removed), orig-ui/* (new), config tmpl + hello.pcm (handled
separately, passed in as arguments if present)."""
import hashlib, os, shutil, sys

ROOT = os.environ.get("POTATO_SRC", ".")  # run from the openamika-rebuild root, or set POTATO_SRC
STAGE = os.environ.get("POTATO_STAGE", "stage-potato/potatoamika")

def cp(src, dst_rel):
    dst = os.path.join(STAGE, dst_rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)

def cp_tree(src_dir, dst_rel, names=None):
    for dirpath, _, files in os.walk(src_dir):
        for f in files:
            full = os.path.join(dirpath, f)
            rel = os.path.relpath(full, src_dir)
            if names is not None and rel not in names:
                continue
            cp(full, os.path.join(dst_rel, rel))

if os.path.exists(STAGE):
    shutil.rmtree(STAGE)
os.makedirs(STAGE)

# binaries
cp(f"{ROOT}/scheduler/bin/scheduler-aarch64", "bin/scheduler")
cp(f"{ROOT}/tts-engine/host/dist-aarch64-free/tts_host", "bin/tts_host")
cp(f"{ROOT}/asr/build-aarch64/asr_host", "bin/asr_host")
cp(f"{ROOT}/asr/gateway/asr_gateway-arm64", "bin/asr_gateway")
# persona (8)
for f in ["表层.txt", "表层_en.txt", "思考.txt", "思考_en.txt", "例子.txt", "例子_en.txt", "scenes.json", "scenes_en.json"]:
    cp(f"{ROOT}/tts-engine/refs-src/brain/{f}", f"persona/{f}")
# models
cp(f"{ROOT}/scheduler/models/Minico-M1-1105-Q4_K_M.gguf", "models/Minico-M1-1105-Q4_K_M.gguf")
cp(f"{ROOT}/scheduler/models/v5-nano-retrieval-Q4_K_M.gguf", "models/v5-nano-retrieval-Q4_K_M.gguf")
# tts tree from phone-calib-all layout
cal = f"{ROOT}/tts-engine/phone-calib-all"
cp_tree(f"{cal}/wo-packs", "tts/wo-packs")
cp_tree(f"{cal}/host/packed_graphs", "tts/host/packed_graphs")
cp_tree(f"{cal}/host/data", "tts/host/data")
cp_tree(f"{cal}/host/refpacks", "tts/host/refpacks")
# asr models subset
for f in ["onnx_decoder_model_merged_quantized.onnx", "onnx_encoder_model_quantized.onnx", "silero_vad.onnx", "vocab.tsv"]:
    cp(f"{ROOT}/asr/models/{f}", f"asr-models/{f}")
# orig-ui: adapter + patched + pristine subset (mirrors the preview machine) + static + docs
cp(f"{ROOT}/orig-ui/adapter.js", "orig-ui/adapter.js")
cp(f"{ROOT}/orig-ui/REMOVALS.md", "orig-ui/REMOVALS.md")
cp(f"{ROOT}/orig-ui/live2d-files.txt", "orig-ui/live2d-files.txt")
for f in ["use.html", "use.js", "common.js"]:
    cp(f"{ROOT}/orig-ui/build/{f}", f"orig-ui/patched/{f}")
pristine = ["about.html", "about.js", "assets/mekahime-logo.webp", "live2d-engine.js",
            "openamika.css", "vendor/cubism4.min.js", "vendor/live2dcubismcore.min.js",
            "vendor/pixi-legacy.min.js"]
for rel in pristine:
    cp(f"{ROOT}/tts-engine/refs-src/ui/{rel}", f"orig-ui/pristine/{rel}")
cp(f"{ROOT}/tts-engine/refs-src/speech/static/flow.html", "orig-ui/static/flow.html")
# separately-resolved inputs
if len(sys.argv) > 1 and sys.argv[1] != "-":
    cp(sys.argv[1], "config/scheduler.json.tmpl")
if len(sys.argv) > 2 and sys.argv[2] != "-":
    cp(sys.argv[2], "test/hello.pcm")

# ---- verify against old MANIFEST table ----
old = {}
in_table = False
for line in open(f"{ROOT}/deploy/MANIFEST.md"):
    if line.startswith("## 附表"):
        in_table = True
    if in_table and line.startswith("| ") and " | " in line:
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if len(parts) == 3 and parts[1].isdigit():
            old[parts[0]] = (int(parts[1]), parts[2])

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

new = {}
for dirpath, _, files in os.walk(STAGE):
    for f in files:
        full = os.path.join(dirpath, f)
        rel = os.path.relpath(full, STAGE)
        new[rel] = (os.path.getsize(full), sha(full))

bad, changed, removed, added = [], [], [], []
for rel, (sz, h) in sorted(new.items()):
    if rel in old:
        if old[rel] != (sz, h):
            changed.append(rel)
    else:
        added.append(rel)
for rel in old:
    if rel not in new:
        removed.append(rel)
print(f"staged files: {len(new)}  (old package: {len(old)})")
print(f"verified identical to old MANIFEST: {sum(1 for r in new if r in old and old[r] == new[r])}")
print("CHANGED vs old:", changed)
print("REMOVED vs old:", removed)
print("ADDED vs old:", added)
# write the new hashes file (same format: '<sha256>  <relpath>', sorted)
with open(os.path.join(STAGE, "hashes.sha256"), "w") as fh:
    for rel in sorted(new):
        fh.write(f"{new[rel][1]}  {rel}\n")
print("hashes.sha256 written:", len(new), "entries (+ itself not listed, matching previous convention check needed)")
total = sum(sz for sz, _ in new.values())
print("tree bytes (excl hashes.sha256):", total)
