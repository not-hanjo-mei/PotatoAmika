#!/usr/bin/env python3
"""Prepare ASR test audio from tts_host output.

Inputs : test/raw32k/req001..008.wav (32 kHz, synthesized by tts_host) and the
         fixed sentence list below (ground truth).
Outputs: test/sentences/s01..s08.wav  (16 kHz, one sentence each)
         test/spliced.wav             (sentences 1-6 joined by silences of
                                         300/500/800/1200/2000 ms)
         test/truth.json              (sentence texts + splice boundaries)
Resampling 32k -> 16k is done by ffmpeg (sinc resampler). Run with the
tts-engine venv python (needs numpy + soundfile).
"""
import json
import os
import subprocess
import sys

import numpy as np
import soundfile as sf

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test")
SENTENCES = [
    "Hello, how are you today?",
    "The quick brown fox jumps over the lazy dog.",
    "Please turn off the lights before you leave.",
    "I would like a cup of coffee, please.",
    "The weather is nice and sunny this morning.",
    "Can you tell me what time it is?",
    "She sells seashells by the seashore.",
    "Thank you very much for your help.",
]
GAPS_MS = [300, 500, 800, 1200, 2000]  # between sentences 1-2 .. 5-6


def main():
    os.makedirs(os.path.join(BASE, "sentences"), exist_ok=True)
    wavs = []
    for i in range(1, 9):
        src = os.path.join(BASE, "raw32k", f"req{i:03d}.wav")
        dst = os.path.join(BASE, "sentences", f"s{i:02d}.wav")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src,
                        "-ar", "16000", "-ac", "1", dst], check=True)
        data, sr = sf.read(dst, dtype="float32")
        assert sr == 16000
        wavs.append(data)
        print(f"s{i:02d}: {len(data)/16000:.3f}s  {SENTENCES[i-1]}")
    # splice the first six with the gap ladder
    parts, bounds, t = [], [], 0.0
    for idx in range(6):
        start = t
        parts.append(wavs[idx])
        t += len(wavs[idx]) / 16000.0
        bounds.append({"sent": idx + 1, "start_s": round(start, 3),
                       "end_s": round(t, 3)})
        if idx < 5:
            gap = GAPS_MS[idx]
            parts.append(np.zeros(int(gap * 16), dtype=np.float32))
            t += gap / 1000.0
    out = np.concatenate(parts)
    sf.write(os.path.join(BASE, "spliced.wav"), out, 16000, subtype="PCM_16")
    truth = {"sentences": SENTENCES, "gaps_ms": GAPS_MS, "bounds": bounds,
             "spliced_total_s": round(len(out) / 16000.0, 3)}
    with open(os.path.join(BASE, "truth.json"), "w") as f:
        json.dump(truth, f, indent=1, ensure_ascii=False)
    print("spliced:", truth["spliced_total_s"], "s, bounds:", bounds)


if __name__ == "__main__":
    main()
