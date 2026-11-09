// Unit check for deess::softclip against the samples-fun provenance
// experiment: holy_big_fat_ass_angry_take1.wav must give d = 1.468,
// RMS -17.15 dB in and out, crest 15.40 -> 13.11 dB.
#include "deess.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: test_softclip in.wav\n"); return 2; }
    std::ifstream f(argv[1], std::ios::binary);
    std::vector<char> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    // minimal WAV scan for the data chunk (16-bit PCM or float32)
    size_t pos = 12;
    int fmt = 1, bits = 16;
    const char* data = nullptr;
    size_t dlen = 0;
    while (pos + 8 <= buf.size()) {
        if (!memcmp(&buf[pos], "fmt ", 4)) { fmt = *(int16_t*)&buf[pos + 8]; bits = *(int16_t*)&buf[pos + 22]; }
        if (!memcmp(&buf[pos], "data", 4)) { data = &buf[pos + 8]; dlen = *(uint32_t*)&buf[pos + 4]; break; }
        pos += 8 + *(uint32_t*)&buf[pos + 4];
    }
    if (!data) { fprintf(stderr, "no data chunk\n"); return 2; }
    std::vector<float> x;
    if (fmt == 1 && bits == 16) {
        size_t n = dlen / 2;
        x.resize(n);
        for (size_t i = 0; i < n; i++) { int16_t v; memcpy(&v, data + 2 * i, 2); x[i] = v / 32768.0f; }
    } else if (fmt == 3 && bits == 32) {
        size_t n = dlen / 4;
        x.resize(n);
        memcpy(x.data(), data, dlen);
    } else { fprintf(stderr, "unsupported fmt %d bits %d\n", fmt, bits); return 2; }
    auto st = deess::softclip(x.data(), x.size());
    auto db = [](double v) { return 20.0 * std::log10(v); };
    printf("n=%zu drive=%.3f peak %.5f -> %.5f  rms %.2f -> %.2f dB  crest %.2f -> %.2f dB\n",
           x.size(), st.drive, st.peak_in, st.peak_out, db(st.rms_in), db(st.rms_out),
           db(st.peak_in / st.rms_in), db(st.peak_out / st.rms_out));
    return 0;
}
