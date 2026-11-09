#!/usr/bin/env python3
"""Character/word error rate of asr_host output vs ground truth.

Usage: cer.py <truth.json> <asr output lines file> [--sentences]
Normalization: lowercase, drop punctuation, collapse whitespace.
Per-sentence mode pairs line i of the output with sentence i (use with the
individual sentence files, transcribed one per run, or with spliced output
when segment count matches).
"""
import json
import re
import sys


def norm(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9' ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def lev(a, b):
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[n]


def main():
    truth = json.load(open(sys.argv[1]))
    lines = [ln.strip() for ln in open(sys.argv[2]) if ln.strip()]
    # strip "[start end] " prefix
    hyps = [re.sub(r"^\[[\d.]+ [\d.]+\]\s*", "", ln) for ln in lines]
    refs = truth["sentences"][:len(hyps)] if "--spliced" not in sys.argv else truth["sentences"][:6]
    if len(hyps) != len(refs):
        print(f"COUNT MISMATCH: {len(hyps)} hyp lines vs {len(refs)} refs")
    tot_c = tot_w = ref_c = ref_w = 0
    for i, (r, h) in enumerate(zip(refs, hyps)):
        rn, hn = norm(r), norm(h)
        ce = lev(rn.replace(" ", ""), hn.replace(" ", ""))
        we = lev(rn.split(), hn.split())
        rc, rw = len(rn.replace(" ", "")), len(rn.split())
        tot_c += ce; tot_w += we; ref_c += rc; ref_w += rw
        print(f"s{i+1}: CER {ce}/{rc}={ce/max(rc,1):.3f}  WER {we}/{rw}={we/max(rw,1):.3f}")
        print(f"  ref: {rn}")
        print(f"  hyp: {hn}")
    print(f"TOTAL: CER {tot_c}/{ref_c} = {tot_c/max(ref_c,1):.4f}   "
          f"WER {tot_w}/{ref_w} = {tot_w/max(ref_w,1):.4f}")


if __name__ == "__main__":
    main()
