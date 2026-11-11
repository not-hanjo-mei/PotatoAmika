// Packed-resident decoder weights + fused dequant GEMV/GEMM (Stage 3).
//
// The two GPT decoder graphs (first stage + stage) load their packed W4/W8
// weight buffers through a WeightPool; a custom ORT op
// (potato::PackedMatMul) computes X @ dequant(W) reading the packed bytes
// directly, so decoder weights are never baked to fp32 at load. Each graph
// gets its own pool of NAME -> tensor descriptors (the same name may map
// to different bytes in another graph's pack: the GPTQ W4-plus packs are
// calibrated per graph and differ), while the payload bytes themselves
// live in a shared content-addressed BlobStore (see below): sharing is by
// memcmp-verified content, never by name, so it cannot serve one graph's
// tensors to the other.
//
// Pack layouts (see wo-packs/dequant_wo.py; tensor viewed as (rows, cols)):
//   w4 group-64 asymmetric: pk uint8 (rows, cols/2), nibble high-first;
//     scale fp32 (rows, cols/64), zero uint8 (rows, cols/64);
//     w = (nib - zero) * scale, groups run along cols.
//   w8 per-row symmetric:   pk int8 bytes (rows, cols), scale fp32 (rows,);
//     w = q * scale[row].
// Op modes (graph attribute `mode`):
//   0: MatMul layout W4: W stored (d_in, d_out), groups along d_out.
//   1: Gemm transB=1 W4: W stored (d_out, d_in), groups along d_in.
//   2: MatMul layout W8.  3: Gemm transB=1 W8.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "cpu.h"
#include "json_min.h"
#include "npz.h"

// NEON tier (aarch64): selected at RUNTIME via cpu::features().neon, per
// the cpu.h portability contract; the scalar implementations remain the
// floor and the fallback for shapes the vector path does not cover.
// POTATO_NO_NEON at compile time forces the scalar path everywhere; it
// exists to A/B-validate the vector kernels on the same target build.
//
// The GEMV tier specifically is OPT-IN at runtime (POTATO_NEON_GEMV=1):
// measured on the A57 target (kbench, 2026-11-10) the hand-written
// NEON GEMV is ~24% SLOWER than the scalar path, because clang -O2
// already autovectorizes the scalar GEMV loop (17 FMLA inside) and the
// explicit version loses. Scalar is therefore the aarch64 default for
// GEMV until a faster variant exists. The vits conv tile is a
// different story (its scalar form does NOT autovectorize) and stays
// default-on; see vits_kernels.cpp.
#if defined(__aarch64__) && !defined(POTATO_NO_NEON)
#define POTATO_PACKED_NEON 1
#include <arm_neon.h>
#else
#define POTATO_PACKED_NEON 0
#endif

#if POTATO_PACKED_NEON
inline bool neon_gemv_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("POTATO_NEON_GEMV");
        return e && e[0] == '1';
    }();
    return on;
}
#endif

namespace packed {

// ---------------- tiny thread pool for the kernels ----------------
struct Pool {
    int n = 1;
    std::vector<std::thread> ts;
    std::mutex m;
    std::condition_variable cv, done_cv;
    std::function<void(int, int)> job;
    int njobs = 0, finished = 0;
    bool stop = false;
    // Without this the static pool's implicit destructor destroys the
    // condition_variable while workers are parked in wait() and process
    // exit hangs forever (found via kbench: main sat in pthread_cond_destroy).
    ~Pool() {
        {
            std::lock_guard<std::mutex> lk(m);
            stop = true;
        }
        cv.notify_all();
        for (auto& t : ts)
            if (t.joinable()) t.join();
    }
    void init(int n_) {
        if (n_ > 1 && ts.empty()) {
            n = n_;
            for (int i = 1; i < n; i++) ts.emplace_back([this, i] { worker(i); });
        }
    }
    void worker(int idx) {
        int seen = 0;
        std::unique_lock<std::mutex> lk(m);
        while (!stop) {
            cv.wait(lk, [&] { return stop || njobs > seen; });
            if (stop) return;
            seen = njobs;
            auto j = job;
            lk.unlock();
            j(idx, n);
            lk.lock();
            if (++finished == n - 1) done_cv.notify_one();
        }
    }
    // Run fn(idx, nthreads) on the pool plus the calling thread (idx 0).
    void run(const std::function<void(int, int)>& fn) {
        if (n <= 1) { fn(0, 1); return; }
        {
            std::lock_guard<std::mutex> lk(m);
            job = fn; finished = 0; njobs++;
        }
        cv.notify_all();
        fn(0, n);
        std::unique_lock<std::mutex> lk(m);
        done_cv.wait(lk, [&] { return finished == n - 1; });
    }
};

inline Pool& kernel_pool() { static Pool p; return p; }
inline void set_kernel_threads(int n) { kernel_pool().init(n); }

// Partition [0, total) into n contiguous chunks; call f(begin, end) for ours.
template <typename F>
inline void chunk(int idx, int n, int64_t total, F&& f) {
    int64_t b = total * idx / n, e = total * (idx + 1) / n;
    if (e > b) f(b, e);
}

// ---------------- kernels ----------------
// X: (M, K) row-major fp32. Y: (M, D) row-major fp32, accumulators start at
// bias (or 0). pk/sc/ze describe W in its stored (rows, cols) view.

// ---------------- NEON kernels (aarch64, ARMv8.0 baseline) ----------------
#if POTATO_PACKED_NEON

// Expand 16 packed bytes (32 nibbles, high nibble first) into 8
// float32x4_t in element order. The shift/and split produces separate
// high/low nibble planes; vzipq interleaves them back into the packed
// layout's (2b, 2b+1) element order; widening then runs u8 -> u16 ->
// u32 -> fp32. Plain ARMv8.0 ASIMD only: no dot product, no fp16
// arithmetic (the A57 target has neither).
inline void expand_w4_32(const uint8_t* p, float32x4_t* out) {
    uint8x16_t v = vld1q_u8(p);
    uint8x16x2_t z = vzipq_u8(vshrq_n_u8(v, 4), vandq_u8(v, vdupq_n_u8(0x0F)));
    for (int h = 0; h < 2; h++) {
        uint16x8_t w0 = vmovl_u8(vget_low_u8(z.val[h]));
        uint16x8_t w1 = vmovl_u8(vget_high_u8(z.val[h]));
        out[4 * h + 0] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(w0)));
        out[4 * h + 1] = vcvtq_f32_u32(vmovl_u16(vget_high_u16(w0)));
        out[4 * h + 2] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(w1)));
        out[4 * h + 3] = vcvtq_f32_u32(vmovl_u16(vget_high_u16(w1)));
    }
}

// mode 0, M == 1, NEON twin of gemv_a_w4_scalar below: identical group
// decomposition and zero-point correction algebra. The only numeric
// difference is fused multiply-add (one rounding) where the scalar
// loop multiplies and adds separately -- fp32 reassociation noise, the
// same class as the established cross-build logits differences.
inline void gemv_a_w4_neon(const float* X, int64_t K, int64_t D,
                           const uint8_t* pk, const float* sc, const uint8_t* ze,
                           const float* bias, float* Y) {
    const int64_t ng = D / 64;
    kernel_pool().run([&](int idx, int nt) {
        chunk(idx, nt, ng, [&](int64_t g0, int64_t g1) {
            for (int64_t g = g0; g < g1; g++) {
                float32x4_t acc[16];
                const float* bsrc = bias ? bias + g * 64 : nullptr;
                for (int t = 0; t < 16; t++)
                    acc[t] = bsrc ? vld1q_f32(bsrc + 4 * t) : vdupq_n_f32(0.0f);
                float corr = 0.0f;
                for (int64_t i = 0; i < K; i++) {
                    float f = X[i] * sc[(size_t)i * ng + g];
                    corr += f * (float)ze[(size_t)i * ng + g];
                    const uint8_t* prow = pk + (size_t)i * (D / 2) + g * 32;
                    for (int c = 0; c < 2; c++) {
                        float32x4_t w[8];
                        expand_w4_32(prow + 16 * c, w);
                        for (int t = 0; t < 8; t++)
                            acc[8 * c + t] = vfmaq_n_f32(acc[8 * c + t], w[t], f);
                    }
                }
                float32x4_t cv = vdupq_n_f32(corr);
                float* y = Y + g * 64;
                for (int t = 0; t < 16; t++) vst1q_f32(y + 4 * t, vsubq_f32(acc[t], cv));
            }
        });
    });
}

// mode 1, M == 1: fused GEMV for the transB layout (W stored (D, K),
// groups along K). Folds the zero point per group instead of
// dequantizing the row into an fp32 temp buffer:
//   y[j] = bias[j] + sum_g s[j,g] * (dot_g(X, nib[j,g]) - z[j,g] * sumx_g)
// with sumx_g = sum of X over group g, computed once per call and
// shared by all D rows. The packed bytes are read exactly once.
inline void gemv_b_w4_neon(const float* X, int64_t K, int64_t D,
                           const uint8_t* pk, const float* sc, const uint8_t* ze,
                           const float* bias, float* Y) {
    const int64_t ng = K / 64;
    float sumx_buf[64];
    std::vector<float> sumx_heap;
    float* sumx = sumx_buf;
    if (ng > 64) { sumx_heap.assign((size_t)ng, 0.0f); sumx = sumx_heap.data(); }
    for (int64_t g = 0; g < ng; g++) {
        float s = 0.0f;
        const float* xg = X + g * 64;
        for (int t = 0; t < 64; t++) s += xg[t];
        sumx[g] = s;
    }
    kernel_pool().run([&](int idx, int nt) {
        chunk(idx, nt, D, [&](int64_t j0, int64_t j1) {
            for (int64_t j = j0; j < j1; j++) {
                const uint8_t* prow = pk + (size_t)j * (K / 2);
                const float* srow = sc + (size_t)j * ng;
                const uint8_t* zrow = ze + (size_t)j * ng;
                float y = bias ? bias[j] : 0.0f;
                for (int64_t g = 0; g < ng; g++) {
                    const float* xg = X + g * 64;
                    float32x4_t a[4] = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                                        vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
                    for (int c = 0; c < 2; c++) {
                        float32x4_t w[8];
                        expand_w4_32(prow + g * 32 + 16 * c, w);
                        for (int t = 0; t < 8; t++)
                            a[t % 4] = vfmaq_f32(a[t % 4], w[t], vld1q_f32(xg + 32 * c + 4 * t));
                    }
                    float d = vaddvq_f32(vaddq_f32(vaddq_f32(a[0], a[1]),
                                                   vaddq_f32(a[2], a[3])));
                    y += srow[g] * (d - (float)zrow[g] * sumx[g]);
                }
                for (int64_t i = ng * 64; i < K; i++) {
                    uint8_t byte = prow[i / 2];
                    int nib = (i % 2 == 0) ? (byte >> 4) : (byte & 0xF);
                    y += X[i] * ((float)nib - (float)zrow[i / 64]) * srow[i / 64];
                }
                Y[j] = y;
            }
        });
    });
}
#endif  // POTATO_PACKED_NEON

// mode 0, M == 1: fully fused GEMV. Reads the packed bytes exactly once,
// never materializes fp32 weights. Per (row i, group g) the factor f = x_i *
// s[i,g] is constant over the group's 64 outputs, and the zero-point term
// collapses into a per-group scalar correction:
//   y[j] = sum_i f(i, g(j)) * nib(i, j) - sum_i f(i, g(j)) * z[i, g(j)].
inline void gemv_a_w4_scalar(const float* X, int64_t K, int64_t D,
                             const uint8_t* pk, const float* sc, const uint8_t* ze,
                             const float* bias, float* Y) {
    const int64_t ng = D / 64;
    kernel_pool().run([&](int idx, int nt) {
        chunk(idx, nt, ng, [&](int64_t g0, int64_t g1) {
            for (int64_t g = g0; g < g1; g++) {
                float corr = 0.0f;
                float* __restrict__ acc = Y + g * 64;
                for (int64_t jb = 0; jb < 64; jb++) acc[jb] = bias ? bias[g * 64 + jb] : 0.0f;
                for (int64_t i = 0; i < K; i++) {
                    float f = X[i] * sc[(size_t)i * ng + g];
                    corr += f * (float)ze[(size_t)i * ng + g];
                    const uint8_t* prow = pk + (size_t)i * (D / 2) + g * 32;
                    for (int64_t b = 0; b < 32; b++) {
                        uint8_t byte = prow[b];
                        acc[2 * b] += f * (float)(byte >> 4);
                        acc[2 * b + 1] += f * (float)(byte & 0xF);
                    }
                }
                for (int64_t jb = 0; jb < 64; jb++) acc[jb] -= corr;
            }
        });
    });
}

// Dispatch: NEON tier only when explicitly enabled (POTATO_NEON_GEMV=1,
// see the note at POTATO_PACKED_NEON) and the shape is a whole number
// of 64-wide groups; the scalar floor otherwise (and always on x86,
// which never compiles the NEON branch).
inline void gemv_a_w4(const float* X, int64_t K, int64_t D,
                      const uint8_t* pk, const float* sc, const uint8_t* ze,
                      const float* bias, float* Y) {
#if POTATO_PACKED_NEON
    if (neon_gemv_enabled() && cpu::features().neon && D >= 64 && D % 64 == 0) {
        gemv_a_w4_neon(X, K, D, pk, sc, ze, bias, Y);
        return;
    }
#endif
    gemv_a_w4_scalar(X, K, D, pk, sc, ze, bias, Y);
}

// Row-tile apply for the block-dequantized GEMM: acc[m][jb] += X[m][i] *
// Wd[i][jb] over all i. Kept as a small standalone function so the compiler
// vectorizes the innermost loop (it does not inside the nested lambdas).
__attribute__((noinline)) inline void apply_tile(
    const float* __restrict__ X, int64_t K, int64_t m0, int64_t mw,
    const float* __restrict__ Wd, int64_t BD, int64_t bw,
    float* __restrict__ acc) {
    for (int64_t i = 0; i < K; i++) {
        const float* __restrict__ wd = Wd + (size_t)i * BD;
        for (int64_t m = 0; m < mw; m++) {
            float xv = X[(size_t)(m0 + m) * K + i];
            float* __restrict__ a = acc + (size_t)m * BD;
            for (int64_t jb = 0; jb < bw; jb++) a[jb] += xv * wd[jb];
        }
    }
}

// mode 0: W stored (K, D); groups along D.
inline void gemm_a_w4(const float* X, int64_t M, int64_t K, int64_t D,
                      const uint8_t* pk, const float* sc, const uint8_t* ze,
                      const float* bias, float* Y) {
    if (M == 1) { gemv_a_w4(X, K, D, pk, sc, ze, bias, Y); return; }
    const int64_t ng = D / 64;
    const int64_t BD = 128;  // output block
    const int64_t MT = 32;   // row tile: keeps the dequantized block hot in L2
    int64_t nblocks = (D + BD - 1) / BD;
    kernel_pool().run([&](int idx, int nt) {
        thread_local std::vector<float> Wd, acc;
        if (Wd.size() < (size_t)K * BD) Wd.resize((size_t)K * BD);
        if (acc.size() < (size_t)MT * BD) acc.resize((size_t)MT * BD);
        chunk(idx, nt, nblocks, [&](int64_t b0, int64_t b1) {
            for (int64_t blk = b0; blk < b1; blk++) {
                int64_t j0 = blk * BD;
                int64_t bw = std::min(BD, D - j0);
                for (int64_t i = 0; i < K; i++) {
                    const uint8_t* prow = pk + (size_t)i * (D / 2);
                    const float* srow = sc + (size_t)i * ng;
                    const uint8_t* zrow = ze + (size_t)i * ng;
                    float* wd = &Wd[(size_t)i * BD];
                    for (int64_t jb = 0; jb < bw; jb++) {
                        int64_t j = j0 + jb;
                        uint8_t byte = prow[j / 2];
                        int nib = (j % 2 == 0) ? (byte >> 4) : (byte & 0xF);
                        wd[jb] = ((float)nib - (float)zrow[j / 64]) * srow[j / 64];
                    }
                }
                for (int64_t m0 = 0; m0 < M; m0 += MT) {
                    int64_t mw = std::min(MT, M - m0);
                    for (int64_t m = 0; m < mw; m++)
                        for (int64_t jb = 0; jb < bw; jb++) acc[(size_t)m * BD + jb] = bias ? bias[j0 + jb] : 0.0f;
                    apply_tile(X, K, m0, mw, Wd.data(), BD, bw, acc.data());
                    for (int64_t m = 0; m < mw; m++) {
                        float* y = Y + (size_t)(m0 + m) * D + j0;
                        const float* a = &acc[(size_t)m * BD];
                        for (int64_t jb = 0; jb < bw; jb++) y[jb] = a[jb];
                    }
                }
            }
        });
    });
}

// mode 1: W stored (D, K); groups along K (the contraction dim).
inline void gemm_b_w4(const float* X, int64_t M, int64_t K, int64_t D,
                      const uint8_t* pk, const float* sc, const uint8_t* ze,
                      const float* bias, float* Y) {
#if POTATO_PACKED_NEON
    // Decode steps are M == 1: take the fused NEON GEMV (no fp32 temp
    // row) when the tier is explicitly enabled (POTATO_NEON_GEMV=1;
    // default off, it measured slower than scalar on A57). Prefill
    // (M > 1) keeps the block path below on every arch.
    if (M == 1 && K >= 64 && neon_gemv_enabled() && cpu::features().neon) {
        gemv_b_w4_neon(X, K, D, pk, sc, ze, bias, Y);
        return;
    }
#endif
    const int64_t ng = K / 64;
    kernel_pool().run([&](int idx, int nt) {
        std::vector<float> wd(K);
        chunk(idx, nt, D, [&](int64_t j0, int64_t j1) {
            for (int64_t j = j0; j < j1; j++) {
                const uint8_t* prow = pk + (size_t)j * (K / 2);
                const float* srow = sc + (size_t)j * ng;
                const uint8_t* zrow = ze + (size_t)j * ng;
                for (int64_t i = 0; i < K; i++) {
                    uint8_t byte = prow[i / 2];
                    int nib = (i % 2 == 0) ? (byte >> 4) : (byte & 0xF);
                    wd[i] = ((float)nib - (float)zrow[i / 64]) * srow[i / 64];
                }
                for (int64_t m = 0; m < M; m++) {
                    const float* __restrict__ x = X + (size_t)m * K;
                    const float* __restrict__ wdp = wd.data();
                    float a = bias ? bias[j] : 0.0f;
                    for (int64_t i = 0; i < K; i++) a += x[i] * wdp[i];
                    Y[(size_t)m * D + j] = a;
                }
            }
        });
    });
}

// mode 2: W stored (K, D) int8; scale per stored row (= per input dim).
inline void gemm_a_w8(const float* X, int64_t M, int64_t K, int64_t D,
                      const uint8_t* pk, const float* sc, const float* bias, float* Y) {
    const int64_t BD = 256;
    int64_t nblocks = (D + BD - 1) / BD;
    kernel_pool().run([&](int idx, int nt) {
        thread_local std::vector<float> acc, f;
        if (acc.size() < (size_t)std::max<int64_t>(M, 1) * BD) acc.resize((size_t)std::max<int64_t>(M, 1) * BD);
        if ((int64_t)f.size() < M) f.resize((size_t)M);
        chunk(idx, nt, nblocks, [&](int64_t b0, int64_t b1) {
            for (int64_t blk = b0; blk < b1; blk++) {
                int64_t j0 = blk * BD;
                int64_t bw = std::min(BD, D - j0);
                for (int64_t m = 0; m < M; m++)
                    for (int64_t jb = 0; jb < bw; jb++) acc[(size_t)m * BD + jb] = bias ? bias[j0 + jb] : 0.0f;
                for (int64_t i = 0; i < K; i++) {
                    for (int64_t m = 0; m < M; m++) f[m] = X[(size_t)m * K + i] * sc[i];
                    const int8_t* qrow = (const int8_t*)(pk + (size_t)i * D) + j0;
                    for (int64_t m = 0; m < M; m++) {
                        float fv = f[m];
                        float* a = &acc[(size_t)m * BD];
                        for (int64_t jb = 0; jb < bw; jb++) a[jb] += fv * (float)qrow[jb];
                    }
                }
                for (int64_t m = 0; m < M; m++) {
                    float* y = Y + (size_t)m * D + j0;
                    for (int64_t jb = 0; jb < bw; jb++) y[jb] = acc[(size_t)m * BD + jb];
                }
            }
        });
    });
}

// mode 3: W stored (D, K) int8; scale per stored row (= per output dim).
inline void gemm_b_w8(const float* X, int64_t M, int64_t K, int64_t D,
                      const uint8_t* pk, const float* sc, const float* bias, float* Y) {
    kernel_pool().run([&](int idx, int nt) {
        chunk(idx, nt, D, [&](int64_t j0, int64_t j1) {
            for (int64_t j = j0; j < j1; j++) {
                const int8_t* qrow = (const int8_t*)(pk + (size_t)j * K);
                for (int64_t m = 0; m < M; m++) {
                    const float* x = X + (size_t)m * K;
                    float a = 0.0f;
                    for (int64_t i = 0; i < K; i++) a += x[i] * (float)qrow[i];
                    Y[(size_t)m * D + j] = a * sc[j] + (bias ? bias[j] : 0.0f);
                }
            }
        });
    });
}


// ---------------- content-addressed blob store (shared) ----------------
// Physical storage for packed payloads, keyed by CONTENT: an FNV-1a hash
// over the pk|scale|zero bytes, always verified by a full memcmp before a
// hit is accepted (a hash collision can only cost a duplicate copy, never
// a wrong tensor). Descriptors stay per-pool and per-name (see WeightPool
// below); only the bytes live here.
//
// Why sharing is safe, including for packs that are NOT byte-identical:
// the S8 bug was a NAME-keyed descriptor map with first-writer-wins
// semantics, which silently served one graph's tensors to the other when
// per-graph-calibrated packs diverged. Content addressing cannot do
// that: two payloads share storage only after a byte-for-byte memcmp, so
// identical tensors (the two GPT decoder graphs under the gen-1 packs)
// occupy one physical copy whoever loaded them first, and differing
// tensors simply never share. One store can therefore back every pool in
// the process (both GPT graphs and the vits graph).
class BlobStore {
  public:
    struct Canon {
        const uint8_t* pk = nullptr;
        const float* sc = nullptr;
        const uint8_t* ze = nullptr;
    };
    Canon intern(const NpyTensor& pk, const NpyTensor& sc, const NpyTensor* ze) {
        uint64_t h = 1469598103934665603ULL;
        h = fnv(pk.data.data(), pk.data.size(), h);
        h = fnv(sc.data.data(), sc.data.size(), h);
        if (ze) h = fnv(ze->data.data(), ze->data.size(), h);
        auto& bucket = canon_[h];
        for (auto& c : bucket) {
            if ((bool)ze != c.has_ze) continue;
            if (blob_eq(c.pk, pk) && blob_eq_f(c.sc, sc) && (!ze || blob_eq(c.ze_t, *ze))) return c.ref;
        }
        Canon nc;
        blobs_.push_back(pk.data);
        nc.pk = blobs_.back().data();
        blobs_f_.push_back(std::vector<float>(sc.data.size() / 4));
        memcpy(blobs_f_.back().data(), sc.data.data(), sc.data.size());
        nc.sc = blobs_f_.back().data();
        if (ze) {
            blobs_.push_back(ze->data);
            nc.ze = blobs_.back().data();
        }
        unique_bytes_ += pk.data.size() + sc.data.size() + (ze ? ze->data.size() : 0);
        bucket.push_back({nc.pk, nc.sc, ze ? nc.ze : nullptr, (bool)ze, nc});
        return nc;
    }
    size_t unique_bytes() const { return unique_bytes_; }

  private:
    static uint64_t fnv(const uint8_t* d, size_t n, uint64_t h) {
        for (size_t i = 0; i < n; i++) { h ^= d[i]; h *= 1099511628211ULL; }
        return h;
    }
    bool blob_eq(const uint8_t* p, const NpyTensor& t) const {
        return p && memcmp(p, t.data.data(), t.data.size()) == 0;
    }
    bool blob_eq_f(const float* p, const NpyTensor& t) const {
        return p && memcmp(p, t.data.data(), t.data.size()) == 0;
    }
    struct BucketEntry { const uint8_t* pk; const float* sc; const uint8_t* ze_t; bool has_ze; Canon ref; };
    std::vector<std::vector<uint8_t>> blobs_;
    std::vector<std::vector<float>> blobs_f_;
    std::unordered_map<uint64_t, std::vector<BucketEntry>> canon_;
    size_t unique_bytes_ = 0;
};


}  // namespace packed
