// Fast kernels for the vits packed ops (see vits_packed.h for the
// contract).
//
// Structure shared by all conv paths:
//   * The weight is dequantized ONCE per node into a scratch buffer
//     (rows = output channels); packed bytes are never re-read per tile.
//   * Work is tiled over output positions: a task owns an LT-wide slice
//     of positions for ALL channels, so its X slice is streamed from
//     DRAM once and reused from L2 by every channel block. (The naive
//     per-channel-block streaming re-reads X once per block and is
//     DRAM-bound; it did not scale past ~1.5 threads.)
//   * The accumulation tile is 4 channels x 32 positions.
//
// Dispatch is by RUNTIME CPU feature detection (cpu.h): on x86-64 the
// tile inner loop has an AVX2+FMA intrinsics implementation selected
// when cpu::features().avx2 holds; AVX2 is the desktop baseline tier
// and AVX-512 is never required. Everywhere else (and on x86 without
// AVX2) the portable C++ tile runs -- it is the scalar floor, written
// so aarch64 compilers can emit NEON for its inner loop. The generic
// scalar transpose remains as the fallback for dilation != 1. Future
// hand-written tiers (NEON intrinsics, SDOT int8) register at the same
// dispatch points; nothing in the pack format or graphs depends on it.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <tuple>
#include <fcntl.h>
#include <unistd.h>
#include <vector>

#include "cpu.h"
#include "vits_kernels.h"

#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace packed {
namespace {

// Fill xv[0..tn) with X row values at src = tt0*stride + base + tt*stride,
// zero outside [0, L).
inline void gather_xv(const float* xi, int64_t L, int64_t stride, int64_t base, int64_t tt0,
                      int64_t tn, float* xv) {
    if (stride == 1) {
        int64_t src0 = tt0 + base;
        for (int64_t tt = 0; tt < tn; tt++) {
            int64_t s = src0 + tt;
            xv[tt] = (s >= 0 && s < L) ? xi[s] : 0.0f;
        }
    } else {
        for (int64_t tt = 0; tt < tn; tt++) {
            int64_t s = (tt0 + tt) * stride + base;
            xv[tt] = (s >= 0 && s < L) ? xi[s] : 0.0f;
        }
    }
}

// Portable tile: positions [lt0, lt0+lt_len) of batch row Xn, output
// channels [o0, o0+ob) of Yn. Handles any stride/dilation.
void conv_tile_generic(const float* Xn, int64_t L, const float* wbuf, int64_t C, int64_t Ci,
                       int64_t K, int64_t stride, int64_t pad_l, int64_t dil, const float* bias,
                       float* Yn, int64_t Lout, int64_t lt0, int64_t lt_len, int64_t o0, int64_t ob) {
    const int64_t TT = 32;
    float win[32 + 64];
    float xv[32];
    const bool window_ok = stride == 1 && (K - 1) * dil <= 64;
    for (int64_t tt0 = lt0; tt0 < lt0 + lt_len; tt0 += TT) {
        int64_t tn = std::min(TT, lt0 + lt_len - tt0);
        float acc[4][32];
        for (int64_t o = 0; o < ob; o++)
            for (int64_t tt = 0; tt < tn; tt++) acc[o][tt] = bias ? bias[o0 + o] : 0.0f;
        if (window_ok) {
            const int64_t wn = tn + (K - 1) * dil;
            for (int64_t i = 0; i < Ci; i++) {
                const float* xi = Xn + (size_t)i * L;
                int64_t src0 = tt0 - pad_l;
                for (int64_t w = 0; w < wn; w++) {
                    int64_t s = src0 + w;
                    win[w] = (s >= 0 && s < L) ? xi[s] : 0.0f;
                }
                for (int64_t kk = 0; kk < K; kk++) {
                    const float* xvp = win + kk * dil;
                    int64_t c = i * K + kk;
                    for (int64_t o = 0; o < ob; o++) {
                        float wgt = wbuf[(size_t)(o0 + o) * C + c];
                        float* a = acc[o];
                        for (int64_t tt = 0; tt < tn; tt++) a[tt] += wgt * xvp[tt];
                    }
                }
            }
        } else {
            for (int64_t c = 0; c < C; c++) {
                gather_xv(Xn + (size_t)(c / K) * L, L, stride, -pad_l + (c % K) * dil, tt0, tn, xv);
                for (int64_t o = 0; o < ob; o++) {
                    float wgt = wbuf[(size_t)(o0 + o) * C + c];
                    float* a = acc[o];
                    for (int64_t tt = 0; tt < tn; tt++) a[tt] += wgt * xv[tt];
                }
            }
        }
        for (int64_t o = 0; o < ob; o++) {
            float* y = Yn + (size_t)(o0 + o) * Lout + tt0;
            for (int64_t tt = 0; tt < tn; tt++) y[tt] = acc[o][tt];
        }
    }
}

#if POTATO_PACKED_NEON
// NEON tile (aarch64, ARMv8.0 baseline). Same tiling contract and
// preconditions as the AVX2 tile: stride == 1 and (K-1)*dil <= 64,
// task = LT positions x up-to-4 output channels, weights pre-
// dequantized in wbuf. Difference vs AVX2: A57 has 32 q registers
// but half the FMLA width, so the accumulation tile is 4 channels x
// 16 positions (16 q accumulators), stepping 16 positions at a time.
//
// Why hand-written here when the GPT GEMV lost to autovectorization:
// the generic tile's inner loop is indexed by the RUNTIME channel
// count, so clang -O2 compiled it to a 4-wide FMLA with the
// accumulator loaded from and stored to the stack on every step
// (whole-object vector-fmla count: 4, two of them in this tile's two
// paths). Here the channel dim is unrolled through the OB template
// and the accumulators never leave registers.
//
// Dilation costs nothing extra: x vectors load from win + kk*dil at
// unit stride, exactly like the AVX2 tile; dil only shifts offsets.
template <int OB>
inline void neon_pos16(const float* Xn, int64_t L, const float* wbuf, int64_t C, int64_t Ci,
                       int64_t K, int64_t pad_l, int64_t dil, const float* bias, float* Yn,
                       int64_t Lout, int64_t tt0, int64_t tn, int64_t o0) {
    float32x4_t acc[OB][4];
    for (int o = 0; o < OB; o++) {
        float32x4_t b = vdupq_n_f32(bias ? bias[o0 + o] : 0.0f);
        for (int j = 0; j < 4; j++) acc[o][j] = b;
    }
    float win[16 + 64];
    const int64_t wn = 16 + (K - 1) * dil;
    for (int64_t i = 0; i < Ci; i++) {
        const float* xi = Xn + (size_t)i * L;
        const int64_t src0 = tt0 - pad_l;
        if (src0 >= 0 && src0 + wn <= L) {
            // Interior block: the whole window is inside the row.
            // Vector copy -- the scalar bounds-checked fill this
            // replaces costs ~5 instructions per element and was
            // about half the tile's dynamic instruction count.
            int64_t w = 0;
            for (; w + 4 <= wn; w += 4) vst1q_f32(win + w, vld1q_f32(xi + src0 + w));
            for (; w < wn; w++) win[w] = xi[src0 + w];
        } else {
            for (int64_t w = 0; w < wn; w++) {
                int64_t s = src0 + w;
                win[w] = (s >= 0 && s < L) ? xi[s] : 0.0f;
            }
        }
        const float* wrow = wbuf + (size_t)o0 * C + (size_t)i * K;
        for (int64_t kk = 0; kk < K; kk++) {
            const float* xvp = win + kk * dil;
            float32x4_t x0 = vld1q_f32(xvp);
            float32x4_t x1 = vld1q_f32(xvp + 4);
            float32x4_t x2 = vld1q_f32(xvp + 8);
            float32x4_t x3 = vld1q_f32(xvp + 12);
            const float* wp = wrow + kk;
            for (int o = 0; o < OB; o++) {
                float wgt = wp[(size_t)o * C];
                acc[o][0] = vfmaq_n_f32(acc[o][0], x0, wgt);
                acc[o][1] = vfmaq_n_f32(acc[o][1], x1, wgt);
                acc[o][2] = vfmaq_n_f32(acc[o][2], x2, wgt);
                acc[o][3] = vfmaq_n_f32(acc[o][3], x3, wgt);
            }
        }
    }
    if (tn == 16) {
        for (int o = 0; o < OB; o++) {
            float* y = Yn + (size_t)(o0 + o) * Lout + tt0;
            vst1q_f32(y, acc[o][0]);
            vst1q_f32(y + 4, acc[o][1]);
            vst1q_f32(y + 8, acc[o][2]);
            vst1q_f32(y + 12, acc[o][3]);
        }
    } else {
        alignas(16) float spill[OB][16];
        for (int o = 0; o < OB; o++) {
            vst1q_f32(spill[o], acc[o][0]);
            vst1q_f32(spill[o] + 4, acc[o][1]);
            vst1q_f32(spill[o] + 8, acc[o][2]);
            vst1q_f32(spill[o] + 12, acc[o][3]);
            float* y = Yn + (size_t)(o0 + o) * Lout + tt0;
            for (int64_t tt = 0; tt < tn; tt++) y[tt] = spill[o][tt];
        }
    }
}

// Register-window variant for dil == 1 (2026-11-10, knife 3).
// neon_pos16 reloads its four position vectors from the stack win[]
// buffer on every tap (K x 4 L1 loads per input channel) and pays a
// store-to-load round trip for the fill. Here the whole window
// (16 + K - 1 <= 26 floats for K <= 11) is loaded ONCE per input
// channel into q registers and each tap's vectors are formed with
// EXT (vextq_f32, 1-cycle, no memory pipe). Accumulation order is
// unchanged -- (i, kk) ascending, one vfmaq_n per (o, group) -- and
// the x values are the same floats, so results are bit-identical to
// neon_pos16; only how the vectors are produced differs. K and the
// tap index are template parameters because EXT lane indices must
// be compile-time constants. Caller guarantees the block is a full
// 16 wide and the window plus two spare floats lie inside the row
// (the register loads span the window rounded up to a whole number
// of vectors); edge blocks keep the win[] path.
//
// Generalized to dil == 3 and dil == 5 (2026-11-10, knife 4): the
// win[] window is a contiguous input segment regardless of
// dilation, and tap kk reads it at offset kk*DIL, so the same
// register window works with DIL as a template parameter and the
// EXT picks taken at offsets kk*DIL + {0,4,8,12}. The window grows
// with dilation (16 + (K-1)*DIL floats), so the register budget
// caps the covered set: dil=3 all K in {3,7,11} (NW <= 12), dil=5
// only K in {3,7} (K=11 would need NW=17, plus 16 accumulators
// that is the whole register file before temporaries). The caller
// additionally requires the rounded-up window (NW*4 floats) to lie
// fully inside the row, which is up to one float stricter than the
// dil=1 guard; blocks failing it keep the win[] path, as do all
// edge and tail blocks.
template <int IDX>
inline float32x4_t neon_ext_pick(const float32x4_t* w) {
    constexpr int v = IDX >> 2, r = IDX & 3;
    if constexpr (r == 0) {
        return w[v];
    } else {
        return vextq_f32(w[v], w[v + 1], r);
    }
}

template <int K, int DIL, int OB, int KK>
inline void neon_ext_tap(const float32x4_t* w, const float* wrow, int64_t C, float32x4_t (*acc)[4]) {
    if constexpr (KK < K) {
        float32x4_t x[4];
        x[0] = neon_ext_pick<KK * DIL>(w);
        x[1] = neon_ext_pick<KK * DIL + 4>(w);
        x[2] = neon_ext_pick<KK * DIL + 8>(w);
        x[3] = neon_ext_pick<KK * DIL + 12>(w);
        const float* wp = wrow + KK;
        for (int o = 0; o < OB; o++) {
            const float wgt = wp[(size_t)o * C];
            acc[o][0] = vfmaq_n_f32(acc[o][0], x[0], wgt);
            acc[o][1] = vfmaq_n_f32(acc[o][1], x[1], wgt);
            acc[o][2] = vfmaq_n_f32(acc[o][2], x[2], wgt);
            acc[o][3] = vfmaq_n_f32(acc[o][3], x[3], wgt);
        }
        neon_ext_tap<K, DIL, OB, KK + 1>(w, wrow, C, acc);
    }
}

template <int K, int DIL, int OB>
inline void neon_pos16_ext(const float* Xn, int64_t L, const float* wbuf, int64_t C, int64_t Ci,
                           int64_t pad_l, const float* bias, float* Yn, int64_t Lout,
                           int64_t tt0, int64_t o0) {
    // Window registers: the contiguous input segment of
    // 16 + (K-1)*DIL floats rounded up to whole vectors.
    // DIL=1: K=3 -> 5, K=7 -> 6, K=11 -> 7 (knife 3).
    // DIL=3: K=3 -> 6, K=7 -> 9, K=11 -> 12. DIL=5: K=3 -> 7,
    // K=7 -> 12 (K=11 would need 17 and is never instantiated:
    // 17 window + 16 accumulator registers cannot stay resident).
    constexpr int NW = (16 + (K - 1) * DIL + 3) / 4;
    float32x4_t acc[OB][4];
    for (int o = 0; o < OB; o++) {
        float32x4_t b = vdupq_n_f32(bias ? bias[o0 + o] : 0.0f);
        for (int j = 0; j < 4; j++) acc[o][j] = b;
    }
    for (int64_t i = 0; i < Ci; i++) {
        const float* xi = Xn + (size_t)i * L + (tt0 - pad_l);
        float32x4_t w[NW];
        for (int j = 0; j < NW; j++) w[j] = vld1q_f32(xi + 4 * j);
        const float* wrow = wbuf + (size_t)o0 * C + (size_t)i * K;
        neon_ext_tap<K, DIL, OB, 0>(w, wrow, C, acc);
    }
    for (int o = 0; o < OB; o++) {
        float* y = Yn + (size_t)(o0 + o) * Lout + tt0;
        vst1q_f32(y, acc[o][0]);
        vst1q_f32(y + 4, acc[o][1]);
        vst1q_f32(y + 8, acc[o][2]);
        vst1q_f32(y + 12, acc[o][3]);
    }
}

// 8-position register window for the one family the 16-wide
// window cannot hold: K=11, dil=5 needs NW=17 window registers at
// 16 positions (17 + 16 accumulators exceeds the register file,
// knife 4). At 8 positions the window is 8 + (K-1)*DIL = 58 floats
// (NW=15) and the accumulators are OB*2, which fits. Per-element
// accumulation order is unchanged -- (i, kk) ascending, one
// vfmaq_n per (o, group) -- so results are bit-identical to the
// win[] path; only how the x vectors are produced differs. The
// caller runs two halves per 16-wide block and guarantees each
// half's rounded-up window (NW*4 floats) lies inside the row.
template <int K, int DIL, int OB, int KK>
inline void neon_ext_tap8(const float32x4_t* w, const float* wrow, int64_t C, float32x4_t (*acc)[2]) {
    if constexpr (KK < K) {
        float32x4_t x[2];
        x[0] = neon_ext_pick<KK * DIL>(w);
        x[1] = neon_ext_pick<KK * DIL + 4>(w);
        const float* wp = wrow + KK;
        for (int o = 0; o < OB; o++) {
            const float wgt = wp[(size_t)o * C];
            acc[o][0] = vfmaq_n_f32(acc[o][0], x[0], wgt);
            acc[o][1] = vfmaq_n_f32(acc[o][1], x[1], wgt);
        }
        neon_ext_tap8<K, DIL, OB, KK + 1>(w, wrow, C, acc);
    }
}

template <int K, int DIL, int OB>
inline void neon_pos8_ext(const float* Xn, int64_t L, const float* wbuf, int64_t C, int64_t Ci,
                          int64_t pad_l, const float* bias, float* Yn, int64_t Lout,
                          int64_t tt0, int64_t o0) {
    constexpr int NW = (8 + (K - 1) * DIL + 3) / 4;
    float32x4_t acc[OB][2];
    for (int o = 0; o < OB; o++) {
        float32x4_t b = vdupq_n_f32(bias ? bias[o0 + o] : 0.0f);
        acc[o][0] = b;
        acc[o][1] = b;
    }
    for (int64_t i = 0; i < Ci; i++) {
        const float* xi = Xn + (size_t)i * L + (tt0 - pad_l);
        float32x4_t w[NW];
        for (int j = 0; j < NW; j++) w[j] = vld1q_f32(xi + 4 * j);
        const float* wrow = wbuf + (size_t)o0 * C + (size_t)i * K;
        neon_ext_tap8<K, DIL, OB, 0>(w, wrow, C, acc);
    }
    for (int o = 0; o < OB; o++) {
        float* y = Yn + (size_t)(o0 + o) * Lout + tt0;
        vst1q_f32(y, acc[o][0]);
        vst1q_f32(y + 4, acc[o][1]);
    }
}

template <int OB>
inline void neon_block(const float* Xn, int64_t L, const float* wbuf, int64_t C, int64_t Ci,
                       int64_t K, int64_t pad_l, int64_t dil, const float* bias, float* Yn,
                       int64_t Lout, int64_t tt0, int64_t tn, int64_t o0) {
    if (dil == 1 && tn == 16) {
        const int64_t src0 = tt0 - pad_l;
        // Register loads span wn rounded up to whole vectors, which
        // reads at most 2 floats past the window; require those to
        // be inside the row as well.
        if (src0 >= 0 && src0 + (16 + K - 1) + 2 <= L) {
            if (K == 3) { neon_pos16_ext<3, 1, OB>(Xn, L, wbuf, C, Ci, pad_l, bias, Yn, Lout, tt0, o0); return; }
            if (K == 7) { neon_pos16_ext<7, 1, OB>(Xn, L, wbuf, C, Ci, pad_l, bias, Yn, Lout, tt0, o0); return; }
            if (K == 11) { neon_pos16_ext<11, 1, OB>(Xn, L, wbuf, C, Ci, pad_l, bias, Yn, Lout, tt0, o0); return; }
        }
    }
    if ((dil == 3 || dil == 5) && tn == 16) {
        // Knife 4: register window for the dilated families. The
        // rounded-up window (NW*4 floats) must lie fully inside the
        // row. K=11 dil=5 is excluded (NW=17 register pressure).
        const int64_t src0 = tt0 - pad_l;
        const int64_t nw4 = ((16 + (K - 1) * dil + 3) / 4) * 4;
        if (src0 >= 0 && src0 + nw4 <= L) {
            if (dil == 3) {
                if (K == 3) { neon_pos16_ext<3, 3, OB>(Xn, L, wbuf, C, Ci, pad_l, bias, Yn, Lout, tt0, o0); return; }
                if (K == 7) { neon_pos16_ext<7, 3, OB>(Xn, L, wbuf, C, Ci, pad_l, bias, Yn, Lout, tt0, o0); return; }
                if (K == 11) { neon_pos16_ext<11, 3, OB>(Xn, L, wbuf, C, Ci, pad_l, bias, Yn, Lout, tt0, o0); return; }
            } else {
                if (K == 3) { neon_pos16_ext<3, 5, OB>(Xn, L, wbuf, C, Ci, pad_l, bias, Yn, Lout, tt0, o0); return; }
                if (K == 7) { neon_pos16_ext<7, 5, OB>(Xn, L, wbuf, C, Ci, pad_l, bias, Yn, Lout, tt0, o0); return; }
            }
        }
    }
    if (dil == 5 && K == 11 && tn == 16) {
        // Knife 5: K=11 dil=5 via two 8-position register windows
        // (NW=15 each). Both halves' rounded windows (60 floats,
        // the second starting 8 later) must lie inside the row;
        // anything else keeps the win[] path.
        const int64_t src0 = tt0 - pad_l;
        if (src0 >= 0 && src0 + 8 + 60 <= L) {
            neon_pos8_ext<11, 5, OB>(Xn, L, wbuf, C, Ci, pad_l, bias, Yn, Lout, tt0, o0);
            neon_pos8_ext<11, 5, OB>(Xn, L, wbuf, C, Ci, pad_l, bias, Yn, Lout, tt0 + 8, o0);
            return;
        }
    }
    neon_pos16<OB>(Xn, L, wbuf, C, Ci, K, pad_l, dil, bias, Yn, Lout, tt0, tn, o0);
}

void conv_tile_neon(const float* Xn, int64_t L, const float* wbuf, int64_t C, int64_t Ci,
                    int64_t K, int64_t pad_l, int64_t dil, const float* bias, float* Yn,
                    int64_t Lout, int64_t lt0, int64_t lt_len, int64_t o0, int64_t ob) {
    for (int64_t tt0 = lt0; tt0 < lt0 + lt_len; tt0 += 16) {
        int64_t tn = std::min<int64_t>(16, lt0 + lt_len - tt0);
        switch (ob) {
            case 4: neon_block<4>(Xn, L, wbuf, C, Ci, K, pad_l, dil, bias, Yn, Lout, tt0, tn, o0); break;
            case 3: neon_block<3>(Xn, L, wbuf, C, Ci, K, pad_l, dil, bias, Yn, Lout, tt0, tn, o0); break;
            case 2: neon_block<2>(Xn, L, wbuf, C, Ci, K, pad_l, dil, bias, Yn, Lout, tt0, tn, o0); break;
            default: neon_block<1>(Xn, L, wbuf, C, Ci, K, pad_l, dil, bias, Yn, Lout, tt0, tn, o0); break;
        }
    }
}
#endif  // POTATO_PACKED_NEON

#if defined(__x86_64__)
// AVX2+FMA tile. Precondition (checked by the dispatcher): stride == 1
// and (K-1)*dil <= 64. Accumulators are 4 channels x 4 ymm = the full
// register file minus the position vector and broadcasts.
__attribute__((target("avx2,fma"))) void conv_tile_avx2(
    const float* Xn, int64_t L, const float* wbuf, int64_t C, int64_t Ci, int64_t K, int64_t pad_l,
    int64_t dil, const float* bias, float* Yn, int64_t Lout, int64_t lt0, int64_t lt_len,
    int64_t o0, int64_t ob) {
    alignas(32) float win[32 + 64];
    alignas(32) float spill[4][32];
    for (int64_t tt0 = lt0; tt0 < lt0 + lt_len; tt0 += 32) {
        int64_t tn = std::min<int64_t>(32, lt0 + lt_len - tt0);
        __m256 acc[4][4];
        for (int64_t o = 0; o < 4; o++) {
            __m256 b = _mm256_set1_ps(o < ob && bias ? bias[o0 + o] : 0.0f);
            for (int j = 0; j < 4; j++) acc[o][j] = b;
        }
        const int64_t wn = tn + (K - 1) * dil;
        for (int64_t i = 0; i < Ci; i++) {
            const float* xi = Xn + (size_t)i * L;
            int64_t src0 = tt0 - pad_l;
            for (int64_t w = 0; w < wn; w++) {
                int64_t s = src0 + w;
                win[w] = (s >= 0 && s < L) ? xi[s] : 0.0f;
            }
            for (int64_t kk = 0; kk < K; kk++) {
                const float* xvp = win + kk * dil;
                __m256 x0 = _mm256_loadu_ps(xvp);
                __m256 x1 = _mm256_loadu_ps(xvp + 8);
                __m256 x2 = _mm256_loadu_ps(xvp + 16);
                __m256 x3 = _mm256_loadu_ps(xvp + 24);
                int64_t c = i * K + kk;
                for (int64_t o = 0; o < ob; o++) {
                    __m256 w = _mm256_set1_ps(wbuf[(size_t)(o0 + o) * C + c]);
                    acc[o][0] = _mm256_fmadd_ps(w, x0, acc[o][0]);
                    acc[o][1] = _mm256_fmadd_ps(w, x1, acc[o][1]);
                    acc[o][2] = _mm256_fmadd_ps(w, x2, acc[o][2]);
                    acc[o][3] = _mm256_fmadd_ps(w, x3, acc[o][3]);
                }
            }
        }
        for (int64_t o = 0; o < ob; o++) {
            float* y = Yn + (size_t)(o0 + o) * Lout + tt0;
            if (tn == 32) {
                _mm256_storeu_ps(y, acc[o][0]);
                _mm256_storeu_ps(y + 8, acc[o][1]);
                _mm256_storeu_ps(y + 16, acc[o][2]);
                _mm256_storeu_ps(y + 24, acc[o][3]);
            } else {
                _mm256_store_ps(spill[o], acc[o][0]);
                _mm256_store_ps(spill[o] + 8, acc[o][1]);
                _mm256_store_ps(spill[o] + 16, acc[o][2]);
                _mm256_store_ps(spill[o] + 24, acc[o][3]);
                for (int64_t tt = 0; tt < tn; tt++) y[tt] = spill[o][tt];
            }
        }
    }
}
#endif

// Generic scalar ConvTranspose (gather form). Fallback for dilation != 1;
// not used on the shipping graph's nodes (all dilations are 1 there).
void conv_transpose1d_scalar_impl(const float* X, int64_t N, int64_t Ci, int64_t L,
                                  const uint8_t* pk, const float* sc, const uint8_t* ze, int mode,
                                  int64_t O, int64_t K, int64_t stride, int64_t pad_l, int64_t dil,
                                  int64_t Tout, const float* bias, float* Y) {
    const int64_t W = O * K;
    std::vector<float> rb((size_t)W);
    for (int64_t n = 0; n < N; n++) {
        for (int64_t o = 0; o < O; o++)
            for (int64_t t = 0; t < Tout; t++)
                Y[((size_t)n * O + o) * Tout + t] = bias ? bias[o] : 0.0f;
        for (int64_t i = 0; i < Ci; i++) {
            dequant_row(pk, sc, ze, mode, i, W, rb.data());
            const float* xi = X + ((size_t)n * Ci + i) * L;
            for (int64_t s = 0; s < L; s++) {
                float xv = xi[s];
                if (xv == 0.0f) continue;
                for (int64_t kk = 0; kk < K; kk++) {
                    int64_t t = s * stride - pad_l + kk * dil;
                    if (t < 0 || t >= Tout) continue;
                    const float* wrow = rb.data() + kk;  // W[i][o][kk] at o*K+kk
                    float* Yn = Y + (size_t)n * O * Tout + t;
                    for (int64_t o = 0; o < O; o++) Yn[(size_t)o * Tout] += xv * wrow[(size_t)o * K];
                }
            }
        }
    }
}

// ---- PCONV shape census (diagnostics) ----
// Per executed conv shape: call count, output length, MAC, and the
// dequant-vs-tile time split. Enabled by env PCONV_CENSUS=<path>; the
// table is written at process exit (atexit). Zero cost when unset
// (one cached getenv). Transpose phase convs appear with phase=1 and
// their executed (Ci, O, Kp) geometry.
struct PconvKey {
    int64_t Ci, O, K, stride, dil;
    int mode, phase;
    bool operator<(const PconvKey& o) const {
        return std::tie(Ci, O, K, stride, dil, mode, phase) <
               std::tie(o.Ci, o.O, o.K, o.stride, o.dil, o.mode, o.phase);
    }
};
struct PconvVal {
    long long calls = 0, mac = 0, dq_us = 0, cp_us = 0, sum_lout = 0;
};
// Leaked on purpose: pconv_dump runs from atexit, and a function-local
// static map would be destroyed BEFORE the dump if its construction
// (first pconv_add) registered its destructor after the dump's atexit
// registration -- the dump then traverses a destroyed tree (observed:
// _Rb_tree_increment spinning in __cxa_finalize). Never destroy.
std::map<PconvKey, PconvVal>& pconv_map() {
    static auto* m = new std::map<PconvKey, PconvVal>();
    return *m;
}
std::mutex& pconv_mu() {
    static auto* m = new std::mutex();
    return *m;
}
const char* pconv_path() {
    static const char* p = std::getenv("PCONV_CENSUS");
    return p;
}
void pconv_dump() {
    const char* path = pconv_path();
    if (!path) return;
    std::string out;
    char buf[512];
    long long tot_cp = 0, tot_dq = 0;
    for (auto& kv : pconv_map()) { tot_cp += kv.second.cp_us; tot_dq += kv.second.dq_us; }
    std::vector<std::pair<PconvKey, PconvVal>> rows(pconv_map().begin(), pconv_map().end());
    std::sort(rows.begin(), rows.end(),
              [](const std::pair<PconvKey, PconvVal>& x, const std::pair<PconvKey, PconvVal>& y) {
                  return x.second.cp_us > y.second.cp_us;
              });
    out += "# Ci O K stride dil mode phase | calls sum_lout mac dq_ms cp_ms cp_share%\n";
    for (auto& kv : rows) {
        const PconvKey& k = kv.first;
        const PconvVal& v = kv.second;
        std::snprintf(buf, sizeof buf, "%lld %lld %lld %lld %lld %d %d | %lld %lld %lld %.1f %.1f %.2f\n",
                      (long long)k.Ci, (long long)k.O, (long long)k.K, (long long)k.stride,
                      (long long)k.dil, k.mode, k.phase, v.calls, v.sum_lout, v.mac, v.dq_us / 1e3,
                      v.cp_us / 1e3, tot_cp ? 100.0 * (double)v.cp_us / (double)tot_cp : 0.0);
        out += buf;
    }
    std::snprintf(buf, sizeof buf, "# totals: cp_ms %.1f dq_ms %.1f\n", tot_cp / 1e3, tot_dq / 1e3);
    out += buf;
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd >= 0) { (void)!write(fd, out.data(), out.size()); close(fd); }
}
void pconv_add(const PconvKey& k, long long mac, long long dq_us, long long cp_us, int64_t lout) {
    if (!pconv_path()) return;
    static std::once_flag once;
    std::call_once(once, [] { std::atexit(pconv_dump); });
    std::lock_guard<std::mutex> lk(pconv_mu());
    PconvVal& v = pconv_map()[k];
    v.calls++;
    v.mac += mac;
    v.dq_us += dq_us;
    v.cp_us += cp_us;
    v.sum_lout += lout;
}

// Tile choice: bench override > env POTATO_VITS_TILE > auto (NEON
// tier when compiled in and detected). Returns 1 for the NEON tile,
// 0 for the generic tile; the AVX2 dispatch on x86 is separate and
// precedes this.
std::atomic<int>& vits_tile_ovr() {
    static std::atomic<int> v{-1};
    return v;
}
int vits_tile_pick() {
    int ov = vits_tile_ovr().load(std::memory_order_relaxed);
    if (ov >= 0) return ov;
    static const int env = [] {
        const char* e = std::getenv("POTATO_VITS_TILE");
        if (e && !std::strcmp(e, "generic")) return 0;
        if (e && !std::strcmp(e, "neon")) return 1;
        return -1;
    }();
    if (env >= 0) return env;
#if POTATO_PACKED_NEON
    return cpu::features().neon ? 1 : 0;
#else
    return 0;
#endif
}

}  // namespace

void vits_tile_override(int v) { vits_tile_ovr().store(v, std::memory_order_relaxed); }

void conv1d_fast_impl(const float* X, int64_t N, int64_t Ci, int64_t L,
                      const uint8_t* pk, const float* sc, const uint8_t* ze, int mode,
                      int64_t O, int64_t K, int64_t stride, int64_t pad_l, int64_t dil,
                      int64_t Lout, const float* bias, float* Y,
                      const float* w_override, int64_t w_cols_override) {
    const int64_t OB4 = 4, LT = 1024;
    const int64_t C = w_override ? w_cols_override : Ci * K;
    // Dequantize the whole weight ONCE per node into wbuf (rows = output
    // channels); the compute tiles over output positions (see header).
    const float* wbuf = w_override;
    std::vector<float> wbuf_owned;
    long long dq_us = 0;
    if (!w_override) {
        auto dq0 = std::chrono::steady_clock::now();
        wbuf_owned.resize((size_t)O * C);
        kernel_pool().run([&](int idx, int nt) {
            chunk(idx, nt, O, [&](int64_t r0, int64_t r1) {
                for (int64_t r = r0; r < r1; r++)
                    dequant_row(pk, sc, ze, mode, r, C, wbuf_owned.data() + (size_t)r * C);
            });
        });
        wbuf = wbuf_owned.data();
        dq_us = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - dq0).count();
        vc_stats().dequant_us += dq_us;
    }
    vc_stats().calls++;
    auto cp0 = std::chrono::steady_clock::now();
    const bool window_ok = stride == 1 && (K - 1) * dil <= 64;
    const int64_t ntiles = (Lout + LT - 1) / LT;
    const int64_t nblocks = (O + OB4 - 1) / OB4;
    // Dynamic task fetch (atomic counter) over (batch, position tile,
    // 4-channel block): on a shared VM the pool workers wake and run at
    // uneven speeds, and a static contiguous partition makes every node
    // as slow as its most stolen worker (measured: ~3 GMAC/s in-graph
    // vs ~15 cold in isolation). With fetch-and-add, fast workers simply
    // take more units; a unit is ~0.1-1 ms of work.
    std::atomic<int64_t> next_task{0};
    const int64_t total_tasks = N * ntiles * nblocks;
    kernel_pool().run([&](int, int) {
        while (true) {
            int64_t task = next_task.fetch_add(1, std::memory_order_relaxed);
            if (task >= total_tasks) break;
            {
                int64_t blk = task % nblocks;
                int64_t tile = (task / nblocks) % ntiles;
                int64_t n = task / (nblocks * ntiles);
                int64_t lt0 = tile * LT, lt_len = std::min(LT, Lout - lt0);
                const float* Xn = X + (size_t)n * Ci * L;
                float* Yn = Y + (size_t)n * O * Lout;
                {
                    int64_t o0 = blk * OB4, ob = std::min(OB4, O - o0);
#if defined(__x86_64__)
                    if (window_ok && cpu::features().avx2) {
                        conv_tile_avx2(Xn, L, wbuf, C, Ci, K, pad_l, dil, bias, Yn, Lout, lt0, lt_len,
                                       o0, ob);
                        continue;
                    }
#endif
#if defined(__aarch64__)
                    if (window_ok && vits_tile_pick() == 1) {
                        conv_tile_neon(Xn, L, wbuf, C, Ci, K, pad_l, dil, bias, Yn, Lout, lt0,
                                       lt_len, o0, ob);
                        continue;
                    }
#endif
                    conv_tile_generic(Xn, L, wbuf, C, Ci, K, stride, pad_l, dil, bias, Yn, Lout, lt0,
                                      lt_len, o0, ob);
                }
            }
        }
    });
    long long cus = std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - cp0).count();
    vc_stats().compute_us += cus;
    pconv_add({Ci, O, K, stride, dil, mode, w_override ? 1 : 0}, (long long)N * O * Lout * C,
              dq_us, cus, Lout);
    vc_stats().total_mac += O * Lout * C;
    long long ms = cus / 1000;
    int b = ms < 1 ? 0 : ms < 5 ? 1 : ms < 20 ? 2 : ms < 50 ? 3 : ms < 200 ? 4 : 5;
    vc_stats().hist[b]++;
    long long cur = vc_stats().max_call_us.load();
    while (cus > cur && !vc_stats().max_call_us.compare_exchange_weak(cur, cus)) {
    }
}

// ConvTranspose with dilation 1 via phase decomposition: output position
// t = p + stride*m (phase p) receives taps kk == p + pad_l (mod stride),
// which is a stride-1 conv over X with Kp = #{such kk} taps. Each phase
// runs through conv1d_fast_impl with pre-dequantized phase rows, into a
// scratch row, then scatters into Y at the phase offset.
void conv_transpose1d_fast_impl(const float* X, int64_t N, int64_t Ci, int64_t L,
                                const uint8_t* pk, const float* sc, const uint8_t* ze, int mode,
                                int64_t O, int64_t K, int64_t stride, int64_t pad_l, int64_t dil,
                                int64_t Tout, const float* bias, float* Y) {
    if (dil != 1) {
        conv_transpose1d_scalar_impl(X, N, Ci, L, pk, sc, ze, mode, O, K, stride, pad_l, dil, Tout,
                                     bias, Y);
        return;
    }
    const int64_t W = O * K;
    // Dequantize the whole weight once: wfull[i*W + o*K + kk].
    std::vector<float> wfull((size_t)Ci * W);
    for (int64_t i = 0; i < Ci; i++) dequant_row(pk, sc, ze, mode, i, W, wfull.data() + (size_t)i * W);
    for (int64_t p = 0; p < stride; p++) {
        if (p >= Tout) break;
        // Taps for this phase, kk DESCENDING so the conv's implied input
        // shift increases by exactly 1 per tap.
        int64_t kks[64];
        int64_t Kp = 0;
        bool overflow = false;
        for (int64_t kk = K - 1; kk >= 0; kk--) {
            int64_t rem = (p + pad_l - kk) % stride;
            if (rem < 0) rem += stride;
            if (rem == 0) {
                if (Kp >= 64) { overflow = true; break; }
                kks[Kp++] = kk;
            }
        }
        if (overflow) {
            conv_transpose1d_scalar_impl(X, N, Ci, L, pk, sc, ze, mode, O, K, stride, pad_l, dil,
                                         Tout, bias, Y);
            return;
        }
        if (Kp == 0) continue;
        int64_t shift0 = (p + pad_l - kks[0]) / stride;  // exact: rem == 0
        int64_t Lm = (Tout - 1 - p) / stride + 1;
        // Phase weight rows: prow[o * (Ci*Kp) + i*Kp + j] = W[i][o][kks[j]].
        std::vector<float> prow((size_t)O * Ci * Kp);
        for (int64_t o = 0; o < O; o++)
            for (int64_t i = 0; i < Ci; i++) {
                const float* wrow = wfull.data() + (size_t)i * W + (size_t)o * K;
                float* dst = prow.data() + ((size_t)o * Ci + i) * Kp;
                for (int64_t j = 0; j < Kp; j++) dst[j] = wrow[kks[j]];
            }
        std::vector<float> tmp((size_t)N * O * Lm);
        conv1d_fast_impl(X, N, Ci, L, nullptr, nullptr, nullptr, 1, O, Kp, 1, -shift0, 1, Lm, bias,
                         tmp.data(), prow.data(), Ci * Kp);
        for (int64_t n = 0; n < N; n++)
            for (int64_t o = 0; o < O; o++) {
                const float* src = tmp.data() + ((size_t)n * O + o) * Lm;
                float* dst = Y + ((size_t)n * O + o) * Tout + p;
                for (int64_t m = 0; m < Lm; m++) dst[(size_t)m * stride] = src[m];
            }
    }
}

void conv1d_fast(const float* X, int64_t N, int64_t Ci, int64_t L,
                 const uint8_t* pk, const float* sc, const uint8_t* ze, int mode,
                 int64_t O, int64_t K, int64_t stride, int64_t pad_l, int64_t dil,
                 int64_t Lout, const float* bias, float* Y) {
    conv1d_fast_impl(X, N, Ci, L, pk, sc, ze, mode, O, K, stride, pad_l, dil, Lout, bias, Y,
                     nullptr, 0);
}

void conv_transpose1d_fast(const float* X, int64_t N, int64_t Ci, int64_t L,
                           const uint8_t* pk, const float* sc, const uint8_t* ze, int mode,
                           int64_t O, int64_t K, int64_t stride, int64_t pad_l, int64_t dil,
                           int64_t Tout, const float* bias, float* Y) {
    conv_transpose1d_fast_impl(X, N, Ci, L, pk, sc, ze, mode, O, K, stride, pad_l, dil, Tout, bias,
                               Y);
}

// Gather over a packed table (axis 0): out[..., :] = dequant(table[idx]).
// Body moved verbatim from vits_packed.h (ORT-free phase 2) so the free
// engine links the same definition as the ORT custom op.
void gather_packed(const uint8_t* pk, const float* sc, const uint8_t* ze, int mode,
                   int64_t rows, int64_t cols, const int64_t* idx, int64_t nidx, float* Y) {
    (void)rows;
    kernel_pool().run([&](int widx, int nt) {
        chunk(widx, nt, nidx, [&](int64_t i0, int64_t i1) {
            for (int64_t i = i0; i < i1; i++)
                dequant_row(pk, sc, ze, (int)mode, idx[i], cols, Y + (size_t)i * cols);
        });
    });
}

}  // namespace packed
