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
#include <cmath>
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

// i8 GEMV tier selection, POTATO_GEMV_I8:
//   "0": tier OFF everywhere — the round-1 default behavior, kept as
//        the exact opt-out for the round-2 default below.
//   unset: the default since round 2, extended by its addendum
//        (RUBENS-OPT.md). The tier is ON for BOTH modes, and only
//        where the runtime reports SDOT; mode 0 (mode A) runs its
//        block-quantized form at the shipped width 64 (see
//        kGemvA8Block). Both promotions rest on the user's listening
//        verdicts, per project doctrine that the ear outranks the
//        proxy gates: mode B was promoted in round 2 after the
//        round-1 A/B samples (rubens-opt-round1-listen/) were judged
//        indistinguishable from the fp32 floor ("差不多") despite
//        missing the strict count by one near-tie ref argmax
//        (360/361, teacher agreement intact at 299/361); mode A
//        followed in the addendum after the round-2 pairs
//        (rubens-opt-round2-listen/) came back with the same verdict
//        ("俩mode 差不多") at its both-modes gate row of 299/347.
//        Unset now has the same effect as "1"; everywhere without
//        SDOT the fp32 GEMV paths remain the floor.
//   "1": tier on for both modes where the runtime reports SDOT
//        (aarch64); the fp32 GEMV paths remain the floor elsewhere.
//   "sim": force the portable integer executor on ANY arch. The tier's
//          accumulation is exact integer arithmetic, so the portable
//          executor is bit-equal to the NEON one by construction;
//          "sim" exists to run the tier's numerics through the
//          standard x86 correctness gates, where no SDOT silicon is
//          available, and to isolate semantics from silicon on device.
// Returns: -1 unset, 0 off, 1 on, 2 sim.
inline int gemv_i8_mode() {
    static const int m = [] {
        const char* e = std::getenv("POTATO_GEMV_I8");
        if (!e || !e[0]) return -1;
        if (e[0] == '0') return 0;
        if (e[0] == 's') return 2;
        return 1;
    }();
    return m;
}

// Diagnostic split, POTATO_GEMV_I8_ONLY=a|b: restrict the tier to one
// GEMV mode. Exists to attribute the tier's numeric cost per mode in
// gate runs; unset (both modes) is the only production setting.
inline int gemv_i8_only() {
    static const int m = [] {
        const char* e = std::getenv("POTATO_GEMV_I8_ONLY");
        if (!e || !e[0]) return 0;
        if (e[0] == 'a') return 1;
        if (e[0] == 'b') return 2;
        return 0;
    }();
    return m;
}

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

// ---------------- i8 GEMV tier: shared quantization ----------------
// The i8 tier quantizes the ACTIVATION side to int8 (per 64-wide group,
// symmetric, scale = amax/127) and consumes the W4 nibbles as exact
// int8 values in integer accumulators; the weight scales and zero
// points stay in fp32 correction terms, exactly as in the fp32 GEMV
// algebra. Quantization is done ONCE in the shared scalar helpers
// below -- never inside the vector kernels -- so the NEON executor and
// the portable executor consume identical integers and, the integer
// accumulation being exact and order-free, produce bit-equal outputs.
inline float quantize_i8(const float* f, int64_t n, int8_t* q) {
    float amax = 0.0f;
    for (int64_t i = 0; i < n; i++) amax = std::max(amax, std::fabs(f[i]));
    if (amax == 0.0f) {
        for (int64_t i = 0; i < n; i++) q[i] = 0;
        return 1.0f;
    }
    float sf = amax / 127.0f;
    float inv = 1.0f / sf;
    for (int64_t i = 0; i < n; i++) {
        float v = std::nearbyint(f[i] * inv);
        v = std::max(-127.0f, std::min(127.0f, v));
        q[i] = (int8_t)v;
    }
    return sf;
}

// Mode 0 prelude for output group g: f[i] = X[i] * sc[i*ng + g],
// corr = sum_i f[i] * ze[i*ng + g], fq = quantize_i8(f). Returns the
// quantization scale; corr is returned via corr_out.
inline float gemv_a_i8_prelude(const float* X, int64_t K, int64_t ng, int64_t g,
                               const float* sc, const uint8_t* ze,
                               float* fbuf, int8_t* fq, float* corr_out) {
    float corr = 0.0f;
    for (int64_t i = 0; i < K; i++) {
        float f = X[i] * sc[(size_t)i * ng + g];
        fbuf[i] = f;
        corr += f * (float)ze[(size_t)i * ng + g];
    }
    *corr_out = corr;
    return quantize_i8(fbuf, K, fq);
}

// Mode 0 block quantization (round 2): the contraction is split
// into blocks of kGemvA8Block rows and f is quantized with one scale
// per block (sfb), instead of the whole-K single scale above. Round 1
// diagnosed group width as the error lever (mode B's 64-wide groups
// nearly clean; the whole-K scale cost 293/335 at the gates). Width
// 64 is shipped: at the round-2 gates it beat width 128 on ref
// argmax (351 vs 345), max logit diff (0.79 vs 1.22) and bench RMS
// on every shape, at a one-step teacher-agreement cost (298 vs
// 304) — neither width reached the strict 361/361, but the block
// form is the tier's mode 0 and runs by default on SDOT CPUs since
// the round-2 addendum's listening verdict (see
// gemv_i8_enabled_for); the whole-K form is retained below as
// gemv_a_w4_i8_wk_* for gemv_bench A/B.
inline constexpr int64_t kGemvA8Block = 64;

// Block prelude for output group g: same f and the same full-K fp32
// corr as gemv_a_i8_prelude (corr does not depend on the
// quantization), then f is quantized per block into fq with scales
// sfb[0..nb). Returns nb, the number of blocks.
inline int64_t gemv_a_i8_prelude_blocks(const float* X, int64_t K, int64_t ng, int64_t g,
                                        const float* sc, const uint8_t* ze,
                                        float* fbuf, int8_t* fq, float* sfb, float* corr_out) {
    float corr = 0.0f;
    for (int64_t i = 0; i < K; i++) {
        float f = X[i] * sc[(size_t)i * ng + g];
        fbuf[i] = f;
        corr += f * (float)ze[(size_t)i * ng + g];
    }
    *corr_out = corr;
    int64_t nb = 0;
    for (int64_t i0 = 0; i0 < K; i0 += kGemvA8Block) {
        int64_t n = std::min(kGemvA8Block, K - i0);
        sfb[nb++] = quantize_i8(fbuf + i0, n, fq + i0);
    }
    return nb;
}

// Mode 1 prelude (shared by all rows of the call): per group g,
// sumx[g] = sum of X over the group (fp32, feeds the zero-point
// correction) and xq[g] = quantize_i8 of the group, scale sx[g].
inline void gemv_b_i8_prelude(const float* X, int64_t ng, float* sumx, float* sx, int8_t* xq) {
    for (int64_t g = 0; g < ng; g++) {
        const float* xg = X + g * 64;
        float s = 0.0f;
        for (int t = 0; t < 64; t++) s += xg[t];
        sumx[g] = s;
        sx[g] = quantize_i8(xg, 64, xq + (size_t)g * 64);
    }
}

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

// Expand 16 packed bytes (32 nibbles, high nibble first) into 2
// int8x16_t in element order: the int8 twin of expand_w4_32's front
// half (same zip interleave, no widening to fp32).
inline void expand_w4_32_i8(const uint8_t* p, int8x16_t* out) {
    uint8x16_t v = vld1q_u8(p);
    uint8x16x2_t z = vzipq_u8(vshrq_n_u8(v, 4), vandq_u8(v, vdupq_n_u8(0x0F)));
    out[0] = vreinterpretq_s8_u8(z.val[0]);
    out[1] = vreinterpretq_s8_u8(z.val[1]);
}

// SDOT (FEAT_DotProd): acc[l] += sum_{m=0..3} a[4l+m] * b[4l+m],
// signed bytes. This clang's arm_neon.h does not declare vsdotq_s32
// at the project's baseline -march=armv8-a under any -march spelling,
// so the instruction is emitted directly. The .arch directive raises
// the assembler's accepted ISA for the rest of the translation unit's
// inline-asm validation only; codegen everywhere else stays governed
// by the baseline flags, and this instruction executes solely behind
// the runtime cpu::features().sdot dispatch.
inline int32x4_t sdotq_i8(int32x4_t acc, int8x16_t a, int8x16_t b) {
    asm(".arch armv8.2-a+dotprod\n\tsdot %0.4s, %1.16b, %2.16b"
        : "+w"(acc) : "w"(a), "w"(b));
    return acc;
}

// Mode 0, M == 1, i8 tier, WHOLE-K form (round 1) — retained for
// gemv_bench A/B against the block form; NOT dispatched in
// production. The stored layout contracts over rows, so no dot
// product exists along the packed bytes; the tier instead runs the
// fp32 kernel's rank-1 form in integers: per output group g the
// per-row factor f[i] = X[i] * sc[i, g] is quantized once (shared
// prelude) with ONE scale over the whole contraction, then
// acc[j] += fq[i] * nib(i, j) accumulates exactly. Partial products
// are held in int16 for 16 rows at a time (16 * 127 * 15 = 30480 <
// 32767, so the chunk cannot overflow) and promoted to int32 between
// chunks. The fp32 epilogue expression is identical to the portable
// executor's.
inline void gemv_a_w4_i8_wk_neon(const float* X, int64_t K, int64_t D,
                              const uint8_t* pk, const float* sc, const uint8_t* ze,
                              const float* bias, float* Y) {
    const int64_t ng = D / 64;
    kernel_pool().run([&](int idx, int nt) {
        thread_local std::vector<float> fbuf;
        thread_local std::vector<int8_t> fq;
        if ((int64_t)fbuf.size() < K) { fbuf.resize((size_t)K); fq.resize((size_t)K); }
        chunk(idx, nt, ng, [&](int64_t g0, int64_t g1) {
            for (int64_t g = g0; g < g1; g++) {
                float corr;
                float sf = gemv_a_i8_prelude(X, K, ng, g, sc, ze, fbuf.data(), fq.data(), &corr);
                int32_t tmp[64] = {};
                int64_t i = 0;
                for (; i + 16 <= K; i += 16) {
                    int16x8_t a16[8];
                    for (int h = 0; h < 8; h++) a16[h] = vdupq_n_s16(0);
                    for (int64_t r = 0; r < 16; r++) {
                        const uint8_t* prow = pk + (size_t)(i + r) * (D / 2) + g * 32;
                        int8x16_t w[4];
                        expand_w4_32_i8(prow, w);
                        expand_w4_32_i8(prow + 16, w + 2);
                        int8x8_t qd = vdup_n_s8(fq[i + r]);
                        a16[0] = vmlal_s8(a16[0], vget_low_s8(w[0]), qd);
                        a16[1] = vmlal_s8(a16[1], vget_high_s8(w[0]), qd);
                        a16[2] = vmlal_s8(a16[2], vget_low_s8(w[1]), qd);
                        a16[3] = vmlal_s8(a16[3], vget_high_s8(w[1]), qd);
                        a16[4] = vmlal_s8(a16[4], vget_low_s8(w[2]), qd);
                        a16[5] = vmlal_s8(a16[5], vget_high_s8(w[2]), qd);
                        a16[6] = vmlal_s8(a16[6], vget_low_s8(w[3]), qd);
                        a16[7] = vmlal_s8(a16[7], vget_high_s8(w[3]), qd);
                    }
                    for (int h = 0; h < 8; h++) {
                        int32x4_t lo = vaddq_s32(vld1q_s32(tmp + 8 * h),
                                                 vmovl_s16(vget_low_s16(a16[h])));
                        int32x4_t hi = vaddq_s32(vld1q_s32(tmp + 8 * h + 4),
                                                 vmovl_s16(vget_high_s16(a16[h])));
                        vst1q_s32(tmp + 8 * h, lo);
                        vst1q_s32(tmp + 8 * h + 4, hi);
                    }
                }
                for (; i < K; i++) {
                    int32_t q = fq[i];
                    if (q == 0) continue;
                    const uint8_t* prow = pk + (size_t)i * (D / 2) + g * 32;
                    for (int64_t b = 0; b < 32; b++) {
                        uint8_t byte = prow[b];
                        tmp[2 * b] += q * (byte >> 4);
                        tmp[2 * b + 1] += q * (byte & 0xF);
                    }
                }
                float* y = Y + g * 64;
                const float* bsrc = bias ? bias + g * 64 : nullptr;
                for (int64_t t = 0; t < 64; t++)
                    y[t] = (bsrc ? bsrc[t] : 0.0f) + sf * (float)tmp[t] - corr;
            }
        });
    });
}

// Mode 0, M == 1, i8 tier, BLOCK form (round 2) — the tier's
// mode 0 (default-on where SDOT, see gemv_i8_enabled_for). Same
// rank-1 integer
// form as the whole-K kernel above, but
// the contraction is accumulated per block of kGemvA8Block rows with
// the block's own activation scale (shared block prelude): per
// block, an exact integer tmp[64] is accumulated exactly as in the
// whole-K kernel (same int16 chunking), then combined into y in fp32
// before the next block starts. The fp32 combine sequence — bias,
// then y[t] += sfb[b] * tmp[t] block by block in order, then
// y[t] -= corr — is written identically in the portable executor, so
// the two stay bit-equal.
inline void gemv_a_w4_i8_neon(const float* X, int64_t K, int64_t D,
                              const uint8_t* pk, const float* sc, const uint8_t* ze,
                              const float* bias, float* Y) {
    const int64_t ng = D / 64;
    const int64_t nb_max = (K + kGemvA8Block - 1) / kGemvA8Block;
    kernel_pool().run([&](int idx, int nt) {
        thread_local std::vector<float> fbuf, sfb;
        thread_local std::vector<int8_t> fq;
        if ((int64_t)fbuf.size() < K) { fbuf.resize((size_t)K); fq.resize((size_t)K); }
        if ((int64_t)sfb.size() < nb_max) sfb.resize((size_t)nb_max);
        chunk(idx, nt, ng, [&](int64_t g0, int64_t g1) {
            for (int64_t g = g0; g < g1; g++) {
                float corr;
                gemv_a_i8_prelude_blocks(X, K, ng, g, sc, ze, fbuf.data(), fq.data(),
                                         sfb.data(), &corr);
                float* y = Y + g * 64;
                const float* bsrc = bias ? bias + g * 64 : nullptr;
                for (int64_t t = 0; t < 64; t++) y[t] = bsrc ? bsrc[t] : 0.0f;
                for (int64_t b = 0, i0 = 0; i0 < K; b++, i0 += kGemvA8Block) {
                    const int64_t i1 = std::min(K, i0 + kGemvA8Block);
                    int32_t tmp[64] = {};
                    int64_t i = i0;
                    for (; i + 16 <= i1; i += 16) {
                        int16x8_t a16[8];
                        for (int h = 0; h < 8; h++) a16[h] = vdupq_n_s16(0);
                        for (int64_t r = 0; r < 16; r++) {
                            const uint8_t* prow = pk + (size_t)(i + r) * (D / 2) + g * 32;
                            int8x16_t w[4];
                            expand_w4_32_i8(prow, w);
                            expand_w4_32_i8(prow + 16, w + 2);
                            int8x8_t qd = vdup_n_s8(fq[i + r]);
                            a16[0] = vmlal_s8(a16[0], vget_low_s8(w[0]), qd);
                            a16[1] = vmlal_s8(a16[1], vget_high_s8(w[0]), qd);
                            a16[2] = vmlal_s8(a16[2], vget_low_s8(w[1]), qd);
                            a16[3] = vmlal_s8(a16[3], vget_high_s8(w[1]), qd);
                            a16[4] = vmlal_s8(a16[4], vget_low_s8(w[2]), qd);
                            a16[5] = vmlal_s8(a16[5], vget_high_s8(w[2]), qd);
                            a16[6] = vmlal_s8(a16[6], vget_low_s8(w[3]), qd);
                            a16[7] = vmlal_s8(a16[7], vget_high_s8(w[3]), qd);
                        }
                        for (int h = 0; h < 8; h++) {
                            int32x4_t lo = vaddq_s32(vld1q_s32(tmp + 8 * h),
                                                     vmovl_s16(vget_low_s16(a16[h])));
                            int32x4_t hi = vaddq_s32(vld1q_s32(tmp + 8 * h + 4),
                                                     vmovl_s16(vget_high_s16(a16[h])));
                            vst1q_s32(tmp + 8 * h, lo);
                            vst1q_s32(tmp + 8 * h + 4, hi);
                        }
                    }
                    for (; i < i1; i++) {
                        int32_t q = fq[i];
                        if (q == 0) continue;
                        const uint8_t* prow = pk + (size_t)i * (D / 2) + g * 32;
                        for (int64_t bb = 0; bb < 32; bb++) {
                            uint8_t byte = prow[bb];
                            tmp[2 * bb] += q * (byte >> 4);
                            tmp[2 * bb + 1] += q * (byte & 0xF);
                        }
                    }
                    const float sf = sfb[b];
                    for (int64_t t = 0; t < 64; t++) y[t] += sf * (float)tmp[t];
                }
                for (int64_t t = 0; t < 64; t++) y[t] -= corr;
            }
        });
    });
}

// Mode 1, M == 1, i8 tier: the stored rows ARE the contraction, so
// this is the true dot-product form. Per group, the shared prelude
// quantizes X once; each row then runs
//   y += s[g] * (sx[g] * idot_g - z[g] * sumx[g])
// with idot_g the exact integer dot of xq with the group's nibbles
// (4 SDOTs per group). The K % 64 tail uses the fp32 formula, as in
// gemv_b_w4_neon.
inline void gemv_b_w4_i8_neon(const float* X, int64_t K, int64_t D,
                              const uint8_t* pk, const float* sc, const uint8_t* ze,
                              const float* bias, float* Y) {
    const int64_t ng = K / 64;
    float sumx_buf[64], sx_buf[64];
    std::vector<float> sumx_heap, sx_heap;
    std::vector<int8_t> xq_heap;
    float* sumx = sumx_buf;
    float* sx = sx_buf;
    int8_t* xq = nullptr;
    if (ng > 64) {
        sumx_heap.assign((size_t)ng, 0.0f);
        sx_heap.assign((size_t)ng, 0.0f);
        sumx = sumx_heap.data();
        sx = sx_heap.data();
    }
    xq_heap.assign((size_t)ng * 64, 0);
    xq = xq_heap.data();
    gemv_b_i8_prelude(X, ng, sumx, sx, xq);
    kernel_pool().run([&](int idx, int nt) {
        chunk(idx, nt, D, [&](int64_t j0, int64_t j1) {
            for (int64_t j = j0; j < j1; j++) {
                const uint8_t* prow = pk + (size_t)j * (K / 2);
                const float* srow = sc + (size_t)j * ng;
                const uint8_t* zrow = ze + (size_t)j * ng;
                float y = bias ? bias[j] : 0.0f;
                for (int64_t g = 0; g < ng; g++) {
                    const int8_t* xg = xq + (size_t)g * 64;
                    const uint8_t* pg = prow + g * 32;
                    int8x16_t w[4];
                    expand_w4_32_i8(pg, w);
                    expand_w4_32_i8(pg + 16, w + 2);
                    int32x4_t a4 = vdupq_n_s32(0);
                    a4 = sdotq_i8(a4, vld1q_s8(xg), w[0]);
                    a4 = sdotq_i8(a4, vld1q_s8(xg + 16), w[1]);
                    a4 = sdotq_i8(a4, vld1q_s8(xg + 32), w[2]);
                    a4 = sdotq_i8(a4, vld1q_s8(xg + 48), w[3]);
                    int32_t idot = vaddvq_s32(a4);
                    y += srow[g] * (sx[g] * (float)idot - (float)zrow[g] * sumx[g]);
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

// ---------------- i8 GEMV tier: portable executors ----------------
// These are the tier's semantic definition: same shared preludes, same
// fp32 epilogue expressions, exact integer accumulation. They run
// under POTATO_GEMV_I8=sim on any arch and are bit-equal to the NEON
// executors by construction; they are NOT a production fallback (on
// SDOT-less silicon the fp32 GEMV below stays the fastest floor).
// Mode 0 portable, WHOLE-K form (round 1) — gemv_bench A/B only,
// not dispatched; the block form below is the tier's mode 0.
inline void gemv_a_w4_i8_wk_portable(const float* X, int64_t K, int64_t D,
                                     const uint8_t* pk, const float* sc, const uint8_t* ze,
                                     const float* bias, float* Y) {
    const int64_t ng = D / 64;
    kernel_pool().run([&](int idx, int nt) {
        thread_local std::vector<float> fbuf;
        thread_local std::vector<int8_t> fq;
        if ((int64_t)fbuf.size() < K) { fbuf.resize((size_t)K); fq.resize((size_t)K); }
        chunk(idx, nt, ng, [&](int64_t g0, int64_t g1) {
            for (int64_t g = g0; g < g1; g++) {
                float corr;
                float sf = gemv_a_i8_prelude(X, K, ng, g, sc, ze, fbuf.data(), fq.data(), &corr);
                int32_t acc[64] = {};
                for (int64_t i = 0; i < K; i++) {
                    int32_t q = fq[i];
                    if (q == 0) continue;
                    const uint8_t* prow = pk + (size_t)i * (D / 2) + g * 32;
                    for (int64_t b = 0; b < 32; b++) {
                        uint8_t byte = prow[b];
                        acc[2 * b] += q * (byte >> 4);
                        acc[2 * b + 1] += q * (byte & 0xF);
                    }
                }
                float* y = Y + g * 64;
                const float* bsrc = bias ? bias + g * 64 : nullptr;
                for (int64_t t = 0; t < 64; t++)
                    y[t] = (bsrc ? bsrc[t] : 0.0f) + sf * (float)acc[t] - corr;
            }
        });
    });
}

// Mode 0 portable, BLOCK form (round 2) — the tier's mode 0 and
// its semantic definition: same shared block prelude
// and the same fp32 combine sequence as the NEON executor (bias,
// then per block y[t] += sfb[b] * acc[t], then y[t] -= corr), exact
// integer accumulation per block, so the two are bit-equal.
inline void gemv_a_w4_i8_portable(const float* X, int64_t K, int64_t D,
                                  const uint8_t* pk, const float* sc, const uint8_t* ze,
                                  const float* bias, float* Y) {
    const int64_t ng = D / 64;
    const int64_t nb_max = (K + kGemvA8Block - 1) / kGemvA8Block;
    kernel_pool().run([&](int idx, int nt) {
        thread_local std::vector<float> fbuf, sfb;
        thread_local std::vector<int8_t> fq;
        if ((int64_t)fbuf.size() < K) { fbuf.resize((size_t)K); fq.resize((size_t)K); }
        if ((int64_t)sfb.size() < nb_max) sfb.resize((size_t)nb_max);
        chunk(idx, nt, ng, [&](int64_t g0, int64_t g1) {
            for (int64_t g = g0; g < g1; g++) {
                float corr;
                gemv_a_i8_prelude_blocks(X, K, ng, g, sc, ze, fbuf.data(), fq.data(),
                                         sfb.data(), &corr);
                float* y = Y + g * 64;
                const float* bsrc = bias ? bias + g * 64 : nullptr;
                for (int64_t t = 0; t < 64; t++) y[t] = bsrc ? bsrc[t] : 0.0f;
                for (int64_t b = 0, i0 = 0; i0 < K; b++, i0 += kGemvA8Block) {
                    const int64_t i1 = std::min(K, i0 + kGemvA8Block);
                    int32_t acc[64] = {};
                    for (int64_t i = i0; i < i1; i++) {
                        int32_t q = fq[i];
                        if (q == 0) continue;
                        const uint8_t* prow = pk + (size_t)i * (D / 2) + g * 32;
                        for (int64_t bb = 0; bb < 32; bb++) {
                            uint8_t byte = prow[bb];
                            acc[2 * bb] += q * (byte >> 4);
                            acc[2 * bb + 1] += q * (byte & 0xF);
                        }
                    }
                    const float sf = sfb[b];
                    for (int64_t t = 0; t < 64; t++) y[t] += sf * (float)acc[t];
                }
                for (int64_t t = 0; t < 64; t++) y[t] -= corr;
            }
        });
    });
}

inline void gemv_b_w4_i8_portable(const float* X, int64_t K, int64_t D,
                                  const uint8_t* pk, const float* sc, const uint8_t* ze,
                                  const float* bias, float* Y) {
    const int64_t ng = K / 64;
    std::vector<float> sumx((size_t)ng), sx((size_t)ng);
    std::vector<int8_t> xq((size_t)ng * 64);
    gemv_b_i8_prelude(X, ng, sumx.data(), sx.data(), xq.data());
    kernel_pool().run([&](int idx, int nt) {
        chunk(idx, nt, D, [&](int64_t j0, int64_t j1) {
            for (int64_t j = j0; j < j1; j++) {
                const uint8_t* prow = pk + (size_t)j * (K / 2);
                const float* srow = sc + (size_t)j * ng;
                const uint8_t* zrow = ze + (size_t)j * ng;
                float y = bias ? bias[j] : 0.0f;
                for (int64_t g = 0; g < ng; g++) {
                    const int8_t* xg = xq.data() + (size_t)g * 64;
                    const uint8_t* pg = prow + g * 32;
                    int32_t idot = 0;
                    for (int64_t b = 0; b < 32; b++) {
                        uint8_t byte = pg[b];
                        idot += (int32_t)xg[2 * b] * (byte >> 4) +
                                (int32_t)xg[2 * b + 1] * (byte & 0xF);
                    }
                    y += srow[g] * (sx[g] * (float)idot - (float)zrow[g] * sumx[g]);
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

// Tier entry points: NEON executor when SDOT silicon is present and
// the mode is not forced to "sim"; the portable executor otherwise
// (reachable only when the tier is enabled, see gemv_i8_enabled_for).
inline void gemv_a_w4_i8(const float* X, int64_t K, int64_t D,
                         const uint8_t* pk, const float* sc, const uint8_t* ze,
                         const float* bias, float* Y) {
#if POTATO_PACKED_NEON
    if (gemv_i8_mode() != 2 && cpu::features().sdot) {
        gemv_a_w4_i8_neon(X, K, D, pk, sc, ze, bias, Y);
        return;
    }
#endif
    gemv_a_w4_i8_portable(X, K, D, pk, sc, ze, bias, Y);
}

inline void gemv_b_w4_i8(const float* X, int64_t K, int64_t D,
                         const uint8_t* pk, const float* sc, const uint8_t* ze,
                         const float* bias, float* Y) {
#if POTATO_PACKED_NEON
    if (gemv_i8_mode() != 2 && cpu::features().sdot) {
        gemv_b_w4_i8_neon(X, K, D, pk, sc, ze, bias, Y);
        return;
    }
#endif
    gemv_b_w4_i8_portable(X, K, D, pk, sc, ze, bias, Y);
}

// Whether SDOT silicon is available for the NEON executor.
inline bool gemv_i8_have_sdot() {
#if POTATO_PACKED_NEON
    return cpu::features().sdot;
#else
    return false;
#endif
}

// Per-mode tier enable: which = 0 for mode A (gemv_a_w4), 1 for
// mode B (the gemm_b_w4 M == 1 path). POTATO_GEMV_I8_ONLY can only
// restrict further (turn a mode off), never enable one.
//   env "0":   both off (round-1 default; the opt-out).
//   env "sim": both on, portable executor, any arch (gate runs).
//   env "1":   both on where SDOT is reported.
//   unset:     both on where SDOT is reported — mode B since
//              round 2, mode A (block form @64) since its addendum,
//              each promoted on the user's listening verdict (see
//              gemv_i8_mode), not on the proxy counts: mode A's
//              block form improved the gates (ref argmax 335
//              whole-K -> 345 @128 -> 351 @64 for mode A alone;
//              both modes @64 scored 299/347, RUBENS-OPT.md
//              round 2) without reaching the strict 361/361.
inline bool gemv_i8_enabled_for(int which) {
    int only = gemv_i8_only();
    if (only == 1 && which != 0) return false;
    if (only == 2 && which != 1) return false;
    int m = gemv_i8_mode();
    if (m == 0) return false;
    if (m == 2) return true;
    if (m == 1) return gemv_i8_have_sdot();
    return gemv_i8_have_sdot();  // unset: both modes where SDOT
}

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

// Dispatch: the i8 tier first when enabled for mode A (see
// gemv_i8_enabled_for) and the shape is a whole number of 64-wide
// groups; then the NEON tier only when explicitly enabled
// (POTATO_NEON_GEMV=1, see the note at POTATO_PACKED_NEON); the
// scalar floor otherwise (and always on x86, which never compiles
// the NEON branch).
inline void gemv_a_w4(const float* X, int64_t K, int64_t D,
                      const uint8_t* pk, const float* sc, const uint8_t* ze,
                      const float* bias, float* Y) {
    if (gemv_i8_enabled_for(0) && D >= 64 && D % 64 == 0) {
        gemv_a_w4_i8(X, K, D, pk, sc, ze, bias, Y);
        return;
    }
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
    // Decode steps are M == 1: take the i8 GEMV when its tier is
    // enabled for mode B (see gemv_i8_enabled_for), else the fused
    // NEON GEMV when that tier is explicitly enabled
    // (POTATO_NEON_GEMV=1; default off, it measured slower than
    // scalar on A57). Prefill (M > 1) keeps the block path below on
    // every arch.
    if (M == 1 && K >= 64 && gemv_i8_enabled_for(1)) {
        gemv_b_w4_i8(X, K, D, pk, sc, ze, bias, Y);
        return;
    }
#if POTATO_PACKED_NEON
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
