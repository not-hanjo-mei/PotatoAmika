// Micro-benchmark + numeric cross-check for the W4 GEMV tiers
// (packed_core.h), no ORT, no engine. Timing goes through the public
// dispatchers, so the POTATO_GEMV_I8 / POTATO_NEON_GEMV environment
// selects the variant exactly as in production:
//   POTATO_GEMV_I8=0                    fp32 floor (scalar A / block B)
//   POTATO_GEMV_I8=0 POTATO_NEON_GEMV=1 fp32 NEON GEMV (aarch64)
//   POTATO_GEMV_I8=sim                  portable integer i8 executor
//   (unset, aarch64 + SDOT)             NEON i8 executor (round-2 default)
// The correctness section calls the executors directly and compares
// against a double-precision dequantized reference; the portable and
// NEON i8 executors must agree bit-exactly (max diff 0). For mode A
// it reports the whole-K (round 1) and block (round 2) forms side by
// side, and the timing section A/Bs them through direct calls.
#include "packed_core.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace packed;
using Clk = std::chrono::steady_clock;

static double secs(Clk::time_point a, Clk::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

// Activation-like vector: N(0,1) with ~1% large outliers (the i8
// quantization stress case).
static void fill_x(std::vector<float>& X, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : X) v = nd(rng);
    for (size_t i = 0; i < X.size(); i += 97) X[i] *= 8.0f;
}

struct PackA {  // mode 0: W stored (K, D), groups along D
    std::vector<uint8_t> pk;
    std::vector<float> sc;
    std::vector<uint8_t> ze;
};
static PackA make_pack_a(int64_t K, int64_t D, std::mt19937& rng) {
    PackA p;
    p.pk.resize((size_t)K * D / 2);
    p.sc.resize((size_t)K * (D / 64));
    p.ze.resize((size_t)K * (D / 64));
    for (auto& b : p.pk) b = (uint8_t)rng();
    for (auto& v : p.sc) v = 0.005f + 0.01f * std::uniform_real_distribution<float>(0, 1)(rng);
    for (auto& b : p.ze) b = (uint8_t)(rng() % 16);
    return p;
}
struct PackB {  // mode 1: W stored (D, K), groups along K
    std::vector<uint8_t> pk;
    std::vector<float> sc;
    std::vector<uint8_t> ze;
};
static PackB make_pack_b(int64_t K, int64_t D, std::mt19937& rng) {
    PackB p;
    p.pk.resize((size_t)D * K / 2);
    p.sc.resize((size_t)D * (K / 64));
    p.ze.resize((size_t)D * (K / 64));
    for (auto& b : p.pk) b = (uint8_t)rng();
    for (auto& v : p.sc) v = 0.005f + 0.01f * std::uniform_real_distribution<float>(0, 1)(rng);
    for (auto& b : p.ze) b = (uint8_t)(rng() % 16);
    return p;
}

static void err_stats(const std::vector<float>& a, const std::vector<double>& ref,
                      double* maxabs, double* rms) {
    double m = 0, s = 0;
    for (size_t i = 0; i < a.size(); i++) {
        double d = (double)a[i] - ref[i];
        m = std::max(m, std::fabs(d));
        s += d * d;
    }
    *maxabs = m;
    *rms = std::sqrt(s / (double)a.size());
}

int main(int argc, char** argv) {
    int threads = argc > 1 ? std::atoi(argv[1]) : 4;
    set_kernel_threads(threads);
    std::mt19937 rng(7);
    const cpu::Features& F = cpu::features();
    std::printf("gemv_bench: arch=%s neon=%d sdot=%d threads=%d POTATO_GEMV_I8=%s POTATO_NEON_GEMV=%s\n",
                F.arch, (int)F.neon, (int)F.sdot, threads,
                std::getenv("POTATO_GEMV_I8") ? std::getenv("POTATO_GEMV_I8") : "(unset)",
                std::getenv("POTATO_NEON_GEMV") ? std::getenv("POTATO_NEON_GEMV") : "(unset)");

    // ---------------- correctness (direct executor calls) ----------
    {
        struct S { int64_t K, D; };
        for (auto s : {S{512, 1536}, S{2048, 512}, S{520, 128}}) {
            PackA p = make_pack_a(s.K, s.D, rng);
            std::vector<float> X((size_t)s.K), bias((size_t)s.D, 0.25f);
            fill_x(X, rng);
            std::vector<double> ref((size_t)s.D, 0.0);
            const int64_t ng = s.D / 64;
            for (int64_t i = 0; i < s.K; i++)
                for (int64_t j = 0; j < s.D; j++) {
                    uint8_t byte = p.pk[(size_t)i * (s.D / 2) + j / 2];
                    int nib = (j % 2 == 0) ? (byte >> 4) : (byte & 0xF);
                    ref[j] += (double)X[i] * ((double)nib - p.ze[(size_t)i * ng + j / 64]) *
                              p.sc[(size_t)i * ng + j / 64];
                }
            for (int64_t j = 0; j < s.D; j++) ref[j] += bias[j];
            std::vector<float> y0((size_t)s.D), y1((size_t)s.D), y2((size_t)s.D), yw((size_t)s.D);
            gemv_a_w4_scalar(X.data(), s.K, s.D, p.pk.data(), p.sc.data(), p.ze.data(), bias.data(), y0.data());
            gemv_a_w4_i8_portable(X.data(), s.K, s.D, p.pk.data(), p.sc.data(), p.ze.data(), bias.data(), y1.data());
            gemv_a_w4_i8_wk_portable(X.data(), s.K, s.D, p.pk.data(), p.sc.data(), p.ze.data(), bias.data(), yw.data());
            double m0, r0, m1, r1, mw, rw;
            err_stats(y0, ref, &m0, &r0);
            err_stats(y1, ref, &m1, &r1);
            err_stats(yw, ref, &mw, &rw);
            std::printf("check A K=%lld D=%lld: fp32 maxerr %.3e | i8-block maxerr %.3e rms %.3e | i8-wk maxerr %.3e rms %.3e",
                        (long long)s.K, (long long)s.D, m0, m1, r1, mw, rw);
#if POTATO_PACKED_NEON
            if (F.sdot) {
                gemv_a_w4_i8_neon(X.data(), s.K, s.D, p.pk.data(), p.sc.data(), p.ze.data(), bias.data(), y2.data());
                double md = 0;
                for (int64_t j = 0; j < s.D; j++) md = std::max(md, (double)std::fabs(y2[j] - y1[j]));
                std::printf(" | i8-block-neon vs sim maxdiff %.3e", md);
                gemv_a_w4_i8_wk_neon(X.data(), s.K, s.D, p.pk.data(), p.sc.data(), p.ze.data(), bias.data(), y2.data());
                md = 0;
                for (int64_t j = 0; j < s.D; j++) md = std::max(md, (double)std::fabs(y2[j] - yw[j]));
                std::printf(" | i8-wk-neon vs sim maxdiff %.3e", md);
            }
#endif
            std::printf("\n");
        }
        for (auto s : {S{512, 512}, S{2048, 256}}) {
            PackB p = make_pack_b(s.K, s.D, rng);
            std::vector<float> X((size_t)s.K), bias((size_t)s.D, 0.25f);
            fill_x(X, rng);
            std::vector<double> ref((size_t)s.D, 0.0);
            const int64_t ng = s.K / 64;
            for (int64_t j = 0; j < s.D; j++)
                for (int64_t i = 0; i < s.K; i++) {
                    uint8_t byte = p.pk[(size_t)j * (s.K / 2) + i / 2];
                    int nib = (i % 2 == 0) ? (byte >> 4) : (byte & 0xF);
                    ref[j] += (double)X[i] * ((double)nib - p.ze[(size_t)j * ng + i / 64]) *
                              p.sc[(size_t)j * ng + i / 64];
                }
            for (int64_t j = 0; j < s.D; j++) ref[j] += bias[j];
            std::vector<float> y1((size_t)s.D), y2((size_t)s.D);
            gemv_b_w4_i8_portable(X.data(), s.K, s.D, p.pk.data(), p.sc.data(), p.ze.data(), bias.data(), y1.data());
            double m1, r1;
            err_stats(y1, ref, &m1, &r1);
            std::printf("check B K=%lld D=%lld: i8-sim maxerr %.3e rms %.3e",
                        (long long)s.K, (long long)s.D, m1, r1);
#if POTATO_PACKED_NEON
            if (F.sdot) {
                gemv_b_w4_i8_neon(X.data(), s.K, s.D, p.pk.data(), p.sc.data(), p.ze.data(), bias.data(), y2.data());
                double md = 0;
                for (int64_t j = 0; j < s.D; j++) md = std::max(md, (double)std::fabs(y2[j] - y1[j]));
                std::printf(" | i8-neon vs sim maxdiff %.3e", md);
            }
#endif
            std::printf("\n");
        }
    }

    // ---------------- timing through the dispatchers ----------------
    {
        struct S { int64_t K, D; };
        std::vector<S> shapes = {{512, 1536}, {512, 512}, {512, 2048}, {2048, 512}};
        double layer_ms = 0;
        for (auto s : shapes) {
            PackA p = make_pack_a(s.K, s.D, rng);
            std::vector<float> X((size_t)s.K), Y((size_t)s.D), bias((size_t)s.D, 0.0f);
            fill_x(X, rng);
            gemv_a_w4(X.data(), s.K, s.D, p.pk.data(), p.sc.data(), p.ze.data(), bias.data(), Y.data());
            auto t0 = Clk::now();
            int reps = 200;
            for (int r = 0; r < reps; r++)
                gemv_a_w4(X.data(), s.K, s.D, p.pk.data(), p.sc.data(), p.ze.data(), bias.data(), Y.data());
            double dt = secs(t0, Clk::now()) / reps;
            layer_ms += dt * 1e3;
            std::printf("gemv_a_w4 K=%lld D=%lld: %.3f ms/call, %.2f GMAC/s\n",
                        (long long)s.K, (long long)s.D, dt * 1e3,
                        (double)s.K * s.D / 1e9 / dt);
        }
        std::printf("gemv_a x4 shapes (one layer) = %.2f ms -> x24 layers = %.1f ms/step\n",
                    layer_ms, layer_ms * 24);
    }
    // -------- direct A/B: mode A whole-K vs block (round 2) -------
    {
        int64_t K = 512, D = 1536;
        PackA p = make_pack_a(K, D, rng);
        std::vector<float> X((size_t)K), Y((size_t)D), bias((size_t)D, 0.0f);
        fill_x(X, rng);
        auto timeit = [&](const char* name, auto&& fn) {
            fn();
            auto t0 = Clk::now();
            int reps = 200;
            for (int r = 0; r < reps; r++) fn();
            double dt = secs(t0, Clk::now()) / reps;
            std::printf("direct A K=%lld D=%lld %-16s: %.3f ms/call, %.2f GMAC/s\n",
                        (long long)K, (long long)D, name, dt * 1e3,
                        (double)K * D / 1e9 / dt);
        };
        timeit("wk-portable", [&] { gemv_a_w4_i8_wk_portable(X.data(), K, D, p.pk.data(), p.sc.data(), p.ze.data(), bias.data(), Y.data()); });
        timeit("block-portable", [&] { gemv_a_w4_i8_portable(X.data(), K, D, p.pk.data(), p.sc.data(), p.ze.data(), bias.data(), Y.data()); });
#if POTATO_PACKED_NEON
        if (F.sdot) {
            timeit("wk-neon", [&] { gemv_a_w4_i8_wk_neon(X.data(), K, D, p.pk.data(), p.sc.data(), p.ze.data(), bias.data(), Y.data()); });
            timeit("block-neon", [&] { gemv_a_w4_i8_neon(X.data(), K, D, p.pk.data(), p.sc.data(), p.ze.data(), bias.data(), Y.data()); });
        }
#endif
    }
    {
        int64_t K = 512, D = 512;
        PackB p = make_pack_b(K, D, rng);
        std::vector<float> X((size_t)K), Y((size_t)D), bias((size_t)D, 0.0f);
        fill_x(X, rng);
        gemm_b_w4(X.data(), 1, K, D, p.pk.data(), p.sc.data(), p.ze.data(), bias.data(), Y.data());
        auto t0 = Clk::now();
        int reps = 200;
        for (int r = 0; r < reps; r++)
            gemm_b_w4(X.data(), 1, K, D, p.pk.data(), p.sc.data(), p.ze.data(), bias.data(), Y.data());
        double dt = secs(t0, Clk::now()) / reps;
        std::printf("gemm_b_w4 M=1 K=512 D=512: %.3f ms/call, %.2f GMAC/s -> x24 = %.1f ms/step\n",
                    dt * 1e3, (double)K * D / 1e9 / dt, dt * 1e3 * 24);
    }
    return 0;
}
