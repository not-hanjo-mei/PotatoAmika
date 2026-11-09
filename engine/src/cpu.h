// Runtime CPU feature detection + kernel tier dispatch (Stage 3 S10).
//
// Portability contract for the packed kernels:
//   - Implementations are selected at RUNTIME from detected features,
//     never from compile-time target assumptions. The scalar
//     implementation is the floor and is always present; a missing
//     feature falls back to it automatically.
//   - x86-64 desktop tier: AVX2 (detected via CPUID + XCR0, i.e. the OS
//     must have AVX state enabled). AVX-512 is detected for the log
//     only; no kernel tier may require it.
//   - aarch64 tiers: NEON is the armv8-a baseline; SDOT (ASIMDDP), fp16
//     arithmetic (ASIMDHP) and i8mm (HWCAP2) are detected for future
//     tiers -- on SDOT parts int8 compute stops being a slowdown and
//     becomes an optional speed path (see STAGE3.md S10 device table).
//   - Nothing in the weight-pack format, the manifests, or the graph
//     layout depends on any of this; features only choose among
//     interchangeable kernel implementations.
#pragma once

#include <cstdint>
#include <cstdio>

#if defined(__aarch64__)
#include <sys/auxv.h>
#ifndef HWCAP_ASIMD
#define HWCAP_ASIMD (1 << 1)
#endif
#ifndef HWCAP_ASIMDHP
#define HWCAP_ASIMDHP (1 << 10)
#endif
#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1 << 20)
#endif
#ifndef HWCAP2_I8MM
#define HWCAP2_I8MM (1 << 13)
#endif
#endif

namespace cpu {

struct Features {
    bool sdot = false;    // aarch64: SDOT/UDOT (dot product)
    bool fp16 = false;    // aarch64: fp16 arithmetic
    bool i8mm = false;    // aarch64: int8 matrix multiply
    bool neon = false;    // aarch64: ASIMD baseline
    bool avx2 = false;    // x86-64: AVX2 usable (CPU + OS state)
    bool fma = false;     // x86-64: FMA3
    bool avx512 = false;  // x86-64: AVX-512F usable (log only, never required)
    const char* arch = "unknown";
};

inline const Features& features() {
    static const Features f = [] {
        Features r;
#if defined(__aarch64__)
        r.arch = "aarch64";
        unsigned long hw = getauxval(AT_HWCAP);
        unsigned long hw2 = getauxval(AT_HWCAP2);
        r.neon = (hw & HWCAP_ASIMD) != 0;
        r.fp16 = (hw & HWCAP_ASIMDHP) != 0;
        r.sdot = (hw & HWCAP_ASIMDDP) != 0;
        r.i8mm = (hw2 & HWCAP2_I8MM) != 0;
#elif defined(__x86_64__)
        r.arch = "x86-64";
        unsigned int a = 0, b = 0, c = 0, d = 0;
        auto cpuid = [](unsigned leaf, unsigned sub, unsigned& a, unsigned& b, unsigned& c, unsigned& d) {
            __asm__ volatile("cpuid" : "=a"(a), "=b"(b), "=c"(c), "=d"(d) : "a"(leaf), "c"(sub));
        };
        cpuid(1, 0, a, b, c, d);
        bool osxsave = ((c >> 27) & 1) != 0;
        bool avx = ((c >> 28) & 1) != 0;
        r.fma = ((c >> 12) & 1) != 0;
        unsigned long long xcr0 = 0;
        if (osxsave) {
            unsigned lo = 0, hi = 0;
            __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
            xcr0 = lo;
        }
        bool os_avx = avx && osxsave && ((xcr0 & 0x6) == 0x6);
        cpuid(7, 0, a, b, c, d);
        r.avx2 = os_avx && (((b >> 5) & 1) != 0);
        // AVX-512 additionally needs opmask/ZMM OS state (xcr0 bits 5,6,7).
        r.avx512 = os_avx && (((b >> 16) & 1) != 0) && ((xcr0 & 0xE6) == 0xE6);
#endif
        return r;
    }();
    return f;
}

inline void log_features(const char* tag) {
    const Features& f = features();
    std::fprintf(stderr,
                 "[cpu] %s arch=%s neon=%d sdot=%d fp16=%d i8mm=%d avx2=%d fma=%d avx512=%d "
                 "(kernel floor: scalar; tiers are runtime-selected)\n",
                 tag, f.arch, (int)f.neon, (int)f.sdot, (int)f.fp16, (int)f.i8mm, (int)f.avx2,
                 (int)f.fma, (int)f.avx512);
}

}  // namespace cpu
