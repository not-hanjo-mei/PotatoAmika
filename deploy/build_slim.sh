#!/bin/sh
# Build the slim deploy package (potatoamika-slim.tar) from the assembly
# sources: run assemble_stage.py, strip the files the slim package does not
# carry (public models pulled at install time + the unreferenced gen-1
# fallback packs), regenerate hashes.sha256, tar, and sync PKG_SHA256 in
# the slim deploy.sh next to the tar.
#
# Usage: sh build_slim.sh [out_dir]     (default: ~/workspace/phone/slim)
# The two separately-resolved assembly inputs (scheduler.json.tmpl,
# hello.pcm) are taken from the previous slim tar — they are package
# content, byte-verified by its hashes.sha256.
set -eu
ROOT="$HOME/workspace/openamika-rebuild"
OUT="${1:-$HOME/workspace/phone/slim}"
mkdir -p "$HOME/workspace/.tmp"
WORK="$(mktemp -d "$HOME/workspace/.tmp/slim-build.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

mkdir -p "$OUT" "$WORK/ref"
tar --no-same-owner -xf "$OUT/potatoamika-slim.tar" -C "$WORK/ref" \
  potatoamika/config/scheduler.json.tmpl potatoamika/test/hello.pcm

python3 "$ROOT/deploy/assemble_stage.py" \
  "$WORK/ref/potatoamika/config/scheduler.json.tmpl" \
  "$WORK/ref/potatoamika/test/hello.pcm"

STAGE="${POTATO_STAGE:-$HOME/workspace/stage-potato/potatoamika}"
TREE="$WORK/tree"
mkdir -p "$TREE"
cp -a --no-preserve=ownership "$STAGE" "$TREE/potatoamika"
cd "$TREE/potatoamika"

# --- slim strip: exactly the files README-slim.md lists as install-pulled ---
rm -f models/Minico-M1-1105-Q4_K_M.gguf models/v5-nano-retrieval-Q4_K_M.gguf
rm -f asr-models/onnx_encoder_model_quantized.onnx \
      asr-models/onnx_decoder_model_merged_quantized.onnx \
      asr-models/silero_vad.onnx
rm -rf tts/wo-packs/fallback-gen1

# --- regenerate hashes.sha256 (same format as assemble_stage.py) ---
find . -type f ! -name hashes.sha256 | sed 's|^\./||' | LC_ALL=C sort | \
while IFS= read -r f; do printf '%s  %s\n' "$(sha256sum "$f" | cut -d' ' -f1)" "$f"; done \
  > hashes.sha256

cd "$TREE"
tar cf "$WORK/potatoamika-slim.tar" potatoamika
SHA="$(sha256sum "$WORK/potatoamika-slim.tar" | cut -d' ' -f1)"
SIZE="$(stat -c %s "$WORK/potatoamika-slim.tar")"
cp "$WORK/potatoamika-slim.tar" "$OUT/potatoamika-slim.tar"
sed -i "s|^PKG_SHA256=\"[0-9a-f]*\"|PKG_SHA256=\"$SHA\"|" "$OUT/deploy.sh"
echo "potatoamika-slim.tar: $SIZE B sha256 $SHA"
