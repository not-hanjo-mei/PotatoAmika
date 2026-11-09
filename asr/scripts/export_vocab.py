#!/usr/bin/env python3
"""Export Moonshine tokenizer.json BPE vocab to a flat TSV the C++ host can read.

Each line: <token_id> TAB <hex of the token string's UTF-8 bytes>.
Hex keeps the TSV free of tabs/newlines/escape issues; the host converts back.
Only the 32000 BPE entries are exported (special/added tokens are handled by id
in the host: 0=<unk>, 1=<s>=start, 2=</s>=eos, >=32000 skipped).
"""
import json
import os
import sys

base = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "models")
src = os.path.join(base, "tokenizer.json")
dst = os.path.join(base, "vocab.tsv")

with open(src, encoding="utf-8") as f:
    tok = json.load(f)
vocab = tok["model"]["vocab"]  # token string -> id
inv = {i: t for t, i in vocab.items()}
with open(dst, "w", encoding="utf-8") as f:
    for i in sorted(inv):
        f.write(f"{i}\t{inv[i].encode('utf-8').hex()}\n")
print(f"wrote {dst}: {len(inv)} entries")
