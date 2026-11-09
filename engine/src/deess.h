// deess2air output stage (DEESS.md v2, user-approved by listening):
// split-band dynamic de-esser (slope 2.0, cap 10 dB) followed by a static
// air shelf (+3 dB plateau at 16 kHz). Faithful port of work/deess.py +
// work/deess2.py: zero-phase FFT magnitude split (order-6 Butterworth HP
// magnitude at 5 kHz, lo = x - hi complement), power envelopes via truncated
// exponential kernel (tau 3 ms), ratio detector R = 10log10(env(hi^2)/env(x^2)),
// GR target = clip(2.0 * (R + 5 dB), 0, 10 dB), attack 0.5 ms / release 50 ms
// smoothing in the dB domain on a 0.5 ms grid with linear interpolation to
// samples. All state is per-call: nothing carries across sentences.
// The only deviation from the numpy prototype: FFTs run on the next power of
// two (zero-padded) instead of the exact padded length; the magnitude
// response applied is the same continuous curve. Verified numerically
// against the prototype's own output files (see STAGE2.md).
#pragma once
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

namespace deess {

using cd = std::complex<double>;

inline void fft(std::vector<cd>& a, bool inverse) {
    size_t n = a.size();
    for (size_t i = 1, j = 0; i < n; i++) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        double ang = 2.0 * M_PI / (double)len * (inverse ? 1.0 : -1.0);
        cd wlen(std::cos(ang), std::sin(ang));
        for (size_t i = 0; i < n; i += len) {
            cd w(1.0);
            for (size_t k = 0; k < len / 2; k++) {
                cd u = a[i + k], v = a[i + k + len / 2] * w;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
    if (inverse) for (auto& v : a) v /= (double)n;
}

inline size_t next_pow2(size_t n) {
    size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

// Real FFT helper: returns the full spectrum (size = pow2 >= n) of a real signal.
inline std::vector<cd> rfft_pow2(const double* x, size_t n, size_t nfft) {
    std::vector<cd> a(nfft);
    for (size_t i = 0; i < n; i++) a[i] = x[i];
    fft(a, false);
    return a;
}
inline std::vector<double> irfft_pow2(std::vector<cd> a, size_t n_out) {
    fft(a, true);
    std::vector<double> out(n_out);
    for (size_t i = 0; i < n_out; i++) out[i] = a[i].real();
    return out;
}

// Reflection padding exactly as the prototype: left = x[pad..1], right = x[n-2..n-pad-1].
inline std::vector<double> reflect_pad(const double* x, size_t n, size_t pad) {
    if (pad == 0 || n < 2) return std::vector<double>(x, x + n);
    size_t p = std::min(pad, n - 1);
    std::vector<double> xp;
    xp.reserve(n + 2 * pad);
    for (size_t i = 0; i < pad; i++) xp.push_back(x[p - i]);          // x[pad], x[pad-1], ..., x[1] (wraps if pad > n-1: clamp p)
    // NOTE: prototype indexes x[pad:0:-1]; for pad >= n this would wrap in numpy
    // slicing semantics differently; sentences are always >> pad, and p clamp
    // only guards pathological inputs.
    xp.insert(xp.end(), x, x + n);
    for (size_t i = 0; i < pad; i++) xp.push_back(x[n - 2 - (i % (n - 1))]);
    return xp;
}

// Zero-phase magnitude filtering of the padded signal; returns cropped result.
inline std::vector<double> fft_mag_filter(const double* x, size_t n, int sr, size_t pad,
                                          double (*gain)(double freq_hz, void* ctx), void* ctx) {
    auto xp = reflect_pad(x, n, pad);
    size_t np = xp.size();
    size_t nfft = next_pow2(np);
    auto A = rfft_pow2(xp.data(), np, nfft);
    for (size_t i = 0; i < nfft; i++) {
        double f = (i <= nfft / 2) ? (double)i * sr / (double)nfft : (double)(nfft - i) * sr / (double)nfft;
        A[i] *= gain(f, ctx);
    }
    auto y = irfft_pow2(std::move(A), np);
    return std::vector<double>(y.begin() + (ptrdiff_t)pad, y.begin() + (ptrdiff_t)(pad + n));
}

struct SplitCtx { double fc; int order; };
inline double hp_mag(double f, void* ctx) {
    auto* c = (SplitCtx*)ctx;
    if (f <= 0) return 0.0;
    return 1.0 / std::sqrt(1.0 + std::pow(c->fc / f, 2.0 * c->order));
}

// Causal exponential power envelope: FFT convolution with the truncated,
// sum-normalized kernel k[j] = exp(-j/tau), j < 6*tau+1 (prototype-identical).
inline std::vector<double> exp_env_pow(const double* sig, size_t n, double tau_ms, int sr) {
    double tau = tau_ms / 1000.0 * sr;
    size_t L = (size_t)(6.0 * tau) + 1;
    std::vector<double> kern(L);
    double sum = 0;
    for (size_t j = 0; j < L; j++) { kern[j] = std::exp(-(double)j / tau); sum += kern[j]; }
    for (auto& v : kern) v /= sum;
    size_t nfft = next_pow2(n + L);
    std::vector<cd> A(nfft), K(nfft);
    for (size_t i = 0; i < n; i++) A[i] = sig[i] * sig[i];
    for (size_t j = 0; j < L; j++) K[j] = kern[j];
    fft(A, false); fft(K, false);
    for (size_t i = 0; i < nfft; i++) A[i] *= K[i];
    fft(A, true);
    std::vector<double> out(n);
    for (size_t i = 0; i < n; i++) out[i] = std::max(A[i].real(), 0.0);
    return out;
}

struct ShelfCtx { double f0, f1, gain_db; };
inline double shelf_mag(double f, void* ctx) {
    auto* c = (ShelfCtx*)ctx;
    if (f <= c->f0) return 1.0;
    if (f >= c->f1) return std::pow(10.0, c->gain_db / 20.0);
    double t = (f - c->f0) / (c->f1 - c->f0);
    double s = 0.5 * (1.0 - std::cos(M_PI * t));
    return std::pow(10.0, c->gain_db * s / 20.0);
}

// Full deess2air chain on one sentence. in/out may alias. sr in Hz.
inline void deess2air(const float* in, size_t n, int sr, float* out) {
    if (n == 0) return;
    std::vector<double> x(in, in + n);
    size_t pad = (size_t)std::llround(sr * 0.100);  // 100 ms

    // ---- split ----
    SplitCtx sc{5000.0, 6};
    auto hi = fft_mag_filter(x.data(), n, sr, pad, hp_mag, &sc);
    std::vector<double> lo(n);
    for (size_t i = 0; i < n; i++) lo[i] = x[i] - hi[i];

    // ---- detector ----
    auto e_h = exp_env_pow(hi.data(), n, 3.0, sr);
    auto e_b = exp_env_pow(x.data(), n, 3.0, sr);
    const double thr = -5.0, slope = 2.0, cap = 10.0;
    std::vector<double> gr_t(n);
    for (size_t i = 0; i < n; i++) {
        double R = 10.0 * std::log10((e_h[i] + 1e-18) / (e_b[i] + 1e-18));
        gr_t[i] = std::min(std::max(slope * (R - thr), 0.0), cap);
    }

    // ---- attack/release smoothing on a 0.5 ms grid + linear interpolation ----
    size_t bs = std::max<size_t>(1, (size_t)std::llround(sr * 0.0005));
    size_t nb = (n + bs - 1) / bs;
    std::vector<double> gb(nb);
    double a_att = 1.0 - std::exp(-0.5 / 0.5);    // grid step 0.5 ms over attack tau 0.5 ms
    double a_rel = 1.0 - std::exp(-0.5 / 50.0);   // release tau 50 ms
    double cur = 0.0;
    for (size_t b = 0; b < nb; b++) {
        size_t idx = std::min(b * bs + bs / 2, n - 1);
        double t = gr_t[idx];
        cur += (t > cur ? a_att : a_rel) * (t - cur);
        gb[b] = cur;
    }
    std::vector<double> y(n);
    for (size_t i = 0; i < n; i++) {
        double gr;
        if (nb == 1) gr = gb[0];
        else {
            double pos = ((double)i - (double)(bs / 2)) / (double)bs;
            if (pos <= 0) gr = gb[0];
            else if (pos >= (double)(nb - 1)) gr = gb[nb - 1];
            else {
                size_t b = (size_t)pos;
                double fr = pos - (double)b;
                gr = gb[b] * (1 - fr) + gb[b + 1] * fr;
            }
        }
        double g = std::pow(10.0, -gr / 20.0);
        y[i] = lo[i] + g * hi[i];
    }

    // ---- air shelf ----
    ShelfCtx sh{11000.0, std::min(16000.0, sr / 2.0), 3.0};
    auto z = fft_mag_filter(y.data(), n, sr, pad, shelf_mag, &sh);
    for (size_t i = 0; i < n; i++) out[i] = (float)z[i];
}

// ---- final soft clip (output-chain amendment, user decision 2026-11-08) ----
// Chain: synthesis -> deess2air -> silence trim -> soft clip -> write.
// tanh soft clip, y = tanh(d*x)/d, driven per sentence so the sentence peak
// lands just past the knee (d * peak = 1.2), then the result is rescaled to
// the sentence's original RMS: only the peak shape is compressed, loudness
// is unchanged. Per sentence, no state across sentences. Provenance: the
// samples-fun angry take1 experiment (peak 0.81741 -> d = 1.468, RMS
// -17.15 dB before and after, crest factor 15.40 -> 13.11 dB).
struct LevelStats {
    double peak_in = 0, peak_out = 0;    // linear, full-scale units
    double rms_in = 0, rms_out = 0;      // linear
    double drive = 0;                    // d actually used (0 = not applied)
};
inline LevelStats softclip(float* x, size_t n) {
    LevelStats s;
    if (n == 0) return s;
    double pk = 0, acc = 0;
    for (size_t i = 0; i < n; i++) {
        double a = std::fabs((double)x[i]);
        if (a > pk) pk = a;
        acc += (double)x[i] * x[i];
    }
    s.peak_in = pk;
    s.rms_in = std::sqrt(acc / (double)n);
    if (pk <= 0.0 || s.rms_in <= 0.0) { s.peak_out = pk; s.rms_out = s.rms_in; return s; }
    const double d = 1.2 / pk;
    s.drive = d;
    double acc2 = 0;
    for (size_t i = 0; i < n; i++) {
        double y = std::tanh(d * (double)x[i]) / d;
        x[i] = (float)y;
        acc2 += y * y;
    }
    double rms2 = std::sqrt(acc2 / (double)n);
    double g = s.rms_in / rms2;  // restore the sentence's original RMS
    double po = 0;
    for (size_t i = 0; i < n; i++) {
        x[i] = (float)(x[i] * g);
        double a = std::fabs((double)x[i]);
        if (a > po) po = a;
    }
    s.peak_out = po;
    s.rms_out = s.rms_in;  // exact by construction (rescale above)
    return s;
}

}  // namespace deess
