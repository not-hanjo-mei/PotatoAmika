// ORT-free declarations for the vits packed kernels (vits_kernels.cpp):
// dequantization, kernel stats, tier dispatch, and the conv /
// conv-transpose / gather entry points. Split out of vits_packed.h in
// ORT-free phase 2 (same treatment packed.h got in phase 1) so the
// free engine and its validation tools link the kernels without the
// ORT glue header. vits_packed.h includes this header and adds the
// OrtCustomOp plumbing; the math contract is unchanged (see the
// vits_packed.h header comment).
#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "cpu.h"
#include "packed_core.h"

namespace packed {

// Dequantize one packed row (w_cols values) into dst.
// mode 0 = W4 group-64 asymmetric, mode 1 = W8 per-row symmetric.
inline void dequant_row(const uint8_t* pk, const float* sc, const uint8_t* ze, int mode,
                        int64_t row, int64_t w_cols, float* dst) {
    if (mode == 1) {
        const int8_t* q = (const int8_t*)pk + (size_t)row * w_cols;
        float s = sc[row];
        for (int64_t c = 0; c < w_cols; c++) dst[c] = (float)q[c] * s;
    } else {
        const int64_t ng = w_cols / 64;
        const uint8_t* p = pk + (size_t)row * ng * 32;
        const float* srow = sc + (size_t)row * ng;
        const uint8_t* zrow = ze + (size_t)row * ng;
        for (int64_t g = 0; g < ng; g++) {
            float s = srow[g];
            float z = (float)zrow[g];
            const uint8_t* pb = p + (size_t)g * 32;
            float* d = dst + (size_t)g * 64;
            for (int j = 0; j < 32; j++) {
                uint8_t byte = pb[j];
                d[2 * j] = ((float)(byte >> 4) - z) * s;
                d[2 * j + 1] = ((float)(byte & 0xF) - z) * s;
            }
        }
    }
}

// Optional kernel timing accumulators (diagnostics; filled in
// vits_kernels.cpp, read by validation tools).
struct VcStats {
    std::atomic<long long> dequant_us{0}, compute_us{0}, ct_prep_us{0}, calls{0}, ct_calls{0};
    // compute-call duration histogram (ms buckets) + max + MAC-weighted rate
    std::atomic<long long> hist[6]{{0}, {0}, {0}, {0}, {0}, {0}};
    std::atomic<long long> max_call_us{0};
    std::atomic<long long> total_mac{0};
};
inline VcStats& vc_stats() { static VcStats s; return s; }

// Kernel tiers, from runtime feature detection. Wider tiers register in
// vits_kernels.cpp; the scalar implementation is always the floor.
enum class KernelTier { Scalar, Avx2, Neon, Sdot };
inline KernelTier active_tier() {
    static const KernelTier t = [] {
#if defined(__x86_64__)
        if (cpu::features().avx2) return KernelTier::Avx2;
#elif defined(__aarch64__)
        if (cpu::features().sdot) return KernelTier::Sdot;
        if (cpu::features().neon) return KernelTier::Neon;
#endif
        return KernelTier::Scalar;
    }();
    return t;
}

// Fast kernels (vits_kernels.cpp). Conv1d: X (N, Ci, L) -> Y (N, O, Lout),
// group 1, weight rows = output channels. ConvTranspose1d: X (N, Ci, L)
// -> Y (N, O, Tout), weight rows = input channels.
void conv1d_fast(const float* X, int64_t N, int64_t Ci, int64_t L,
                 const uint8_t* pk, const float* sc, const uint8_t* ze, int mode,
                 int64_t O, int64_t K, int64_t stride, int64_t pad_l, int64_t dil,
                 int64_t Lout, const float* bias, float* Y);
void conv_transpose1d_fast(const float* X, int64_t N, int64_t Ci, int64_t L,
                           const uint8_t* pk, const float* sc, const uint8_t* ze, int mode,
                           int64_t O, int64_t K, int64_t stride, int64_t pad_l, int64_t dil,
                           int64_t Tout, const float* bias, float* Y);

// Bench/diagnostics hook: force the conv tile choice (-1 = auto,
// 0 = generic, 1 = NEON; the NEON choice still requires the runtime
// NEON feature and the window precondition). Env POTATO_VITS_TILE=
// generic|neon is the process-level equivalent. Production code
// never calls this; it exists so one binary can A/B both tiles.
void vits_tile_override(int v);

// Gather over a packed table (axis 0): out[..., :] = dequant(table[idx]).
void gather_packed(const uint8_t* pk, const float* sc, const uint8_t* ze, int mode,
                   int64_t rows, int64_t cols, const int64_t* idx, int64_t nidx, float* Y);

}  // namespace packed
