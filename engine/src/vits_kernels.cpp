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
#include <cstring>
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

}  // namespace

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
        vc_stats().dequant_us += std::chrono::duration_cast<std::chrono::microseconds>(
                                     std::chrono::steady_clock::now() - dq0).count();
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
                    conv_tile_generic(Xn, L, wbuf, C, Ci, K, stride, pad_l, dil, bias, Yn, Lout, lt0,
                                      lt_len, o0, ob);
                }
            }
        }
    });
    long long cus = std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - cp0).count();
    vc_stats().compute_us += cus;
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
