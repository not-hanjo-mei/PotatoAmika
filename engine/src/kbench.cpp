// Micro-benchmark for the packed kernels in isolation (no ORT): fc1-sized
// mode-0 GEMM at prefill M, the fused GEMV at M=1, mode-1 Gemm. x86 numbers
// are relative-only evidence.
#include "packed.h"

#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

using namespace packed;
using Clk = std::chrono::steady_clock;

static double secs(Clk::time_point a, Clk::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

int main(int argc, char** argv) {
    int threads = argc > 1 ? std::atoi(argv[1]) : 4;
    set_kernel_threads(threads);
    std::mt19937 rng(7);
    auto rf = [&] { return std::uniform_real_distribution<float>(-1, 1)(rng); };

    // fc1 shape: K=512, D=2048, w4 groups of 64 along D
    {
        int64_t K = 512, D = 2048, M = 280;
        std::vector<uint8_t> pk((size_t)K * D / 2);
        std::vector<float> sc((size_t)K * (D / 64));
        std::vector<uint8_t> ze((size_t)K * (D / 64));
        for (auto& b : pk) b = (uint8_t)rng();
        for (auto& v : sc) v = 0.01f;
        for (auto& b : ze) b = 8;
        std::vector<float> X((size_t)M * K), Y((size_t)M * D), bias(D, 0.0f);
        for (auto& v : X) v = rf();
        gemm_a_w4(X.data(), M, K, D, pk.data(), sc.data(), ze.data(), bias.data(), Y.data());
        auto t0 = Clk::now();
        int reps = 20;
        for (int r = 0; r < reps; r++)
            gemm_a_w4(X.data(), M, K, D, pk.data(), sc.data(), ze.data(), bias.data(), Y.data());
        double dt = secs(t0, Clk::now()) / reps;
        double gmac = (double)M * K * D / 1e9;
        printf("gemm_a_w4 fc1 M=280: %.1f ms/call, %.2f GMAC/s (threads=%d)\n", dt * 1e3, gmac / dt, threads);
    }
    // GEMV M=1, all four big shapes of one decoder layer
    {
        struct Shape { int64_t K, D; };
        std::vector<Shape> shapes = {{512, 1536}, {512, 512}, {512, 2048}, {2048, 512}};
        double total_ms = 0;
        for (auto s : shapes) {
            std::vector<uint8_t> pk((size_t)s.K * s.D / 2);
            std::vector<float> sc((size_t)s.K * (s.D / 64));
            std::vector<uint8_t> ze((size_t)s.K * (s.D / 64));
            for (auto& b : pk) b = (uint8_t)rng();
            for (auto& v : sc) v = 0.01f;
            for (auto& b : ze) b = 8;
            std::vector<float> X((size_t)s.K), Y((size_t)s.D), bias((size_t)s.D, 0.0f);
            for (auto& v : X) v = rf();
            gemv_a_w4(X.data(), s.K, s.D, pk.data(), sc.data(), ze.data(), bias.data(), Y.data());
            auto t0 = Clk::now();
            int reps = 200;
            for (int r = 0; r < reps; r++)
                gemv_a_w4(X.data(), s.K, s.D, pk.data(), sc.data(), ze.data(), bias.data(), Y.data());
            total_ms += secs(t0, Clk::now()) / reps * 1e3;
        }
        printf("gemv_a_w4 x4 mode-0 ops (one layer, M=1): %.2f ms -> x24 layers = %.1f ms/step\n",
               total_ms, total_ms * 24);
    }
    return 0;
}
