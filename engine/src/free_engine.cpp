// ORT-free static-plan engine: plan compiler + executor + kernels.
// Design notes live in ORT-FREE.md. This file includes packed_core.h
// (split out of packed.h: the thread pool, the fused-dequant kernels
// packed::gemm_* and the BlobStore -- none of them reference ORT, and
// binaries built on this engine link no onnxruntime library; verify
// with ldd). packed.h re-includes packed_core.h and adds the ORT glue.
#include "free_engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>

#include "cpu.h"
#include "packed_core.h"
#include "vits_kernels.h"  // phase 2: packed conv/gather kernels (ORT-free header)

namespace freeeng {
namespace {

// ------------------------------------------------------------------ RT --
// Internal resolved tensor: explicit strides (elements), always set.
struct RT {
    int dtype = 0;
    uint8_t* p = nullptr;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    int64_t numel() const {
        int64_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }
    bool contig() const {
        int64_t exp = 1;
        for (int i = (int)shape.size() - 1; i >= 0; i--) {
            if (shape[i] == 0) return true;
            if (shape[i] > 1) {
                if (strides[i] != exp) return false;
                exp *= shape[i];
            }
        }
        return true;
    }
};

std::vector<int64_t> rowmajor(const std::vector<int64_t>& shape) {
    std::vector<int64_t> s(shape.size(), 1);
    int64_t acc = 1;
    for (int i = (int)shape.size() - 1; i >= 0; i--) {
        s[i] = acc;
        acc *= shape[i];
    }
    return s;
}

[[noreturn]] void fail(const std::string& m) { throw std::runtime_error("free_engine: " + m); }

// Read/write one element as double (slow generic path only).
double read_d(const RT& t, int64_t off) {
    switch (t.dtype) {
        case 1: { float v; memcpy(&v, t.p + off * 4, 4); return v; }
        case 11: { double v; memcpy(&v, t.p + off * 8, 8); return v; }
        case 6: { int32_t v; memcpy(&v, t.p + off * 4, 4); return v; }
        case 7: { int64_t v; memcpy(&v, t.p + off * 8, 8); return (double)v; }
        case 3: { int8_t v; memcpy(&v, t.p + off, 1); return v; }
        default: return t.p[off];
    }
}
void write_d(const RT& t, int64_t off, double x) {
    switch (t.dtype) {
        case 1: { float v = (float)x; memcpy(t.p + off * 4, &v, 4); break; }
        case 11: { double v = x; memcpy(t.p + off * 8, &v, 8); break; }
        case 6: { int32_t v = (int32_t)x; memcpy(t.p + off * 4, &v, 4); break; }
        case 7: { int64_t v = (int64_t)x; memcpy(t.p + off * 8, &v, 8); break; }
        case 3: { int8_t v = (int8_t)x; memcpy(t.p + off, &v, 1); break; }
        default: t.p[off] = (uint8_t)(x != 0); break;
    }
}

// Odometer over `shape`, invoking fn(flat_index) while per-operand
// offsets are maintained by the caller through coord tracking. Here we
// expose coords; callers compute offsets from strides.
template <typename F>
void odometer(const std::vector<int64_t>& shape, F&& fn) {
    int64_t n = 1;
    for (auto d : shape) n *= d;
    if (n == 0) return;
    std::vector<int64_t> coord(shape.size(), 0);
    for (int64_t i = 0; i < n; i++) {
        fn(coord);
        for (int d = (int)shape.size() - 1; d >= 0; d--) {
            if (++coord[d] < shape[d]) break;
            coord[d] = 0;
        }
    }
}

int64_t off_of(const RT& t, const std::vector<int64_t>& coord, const std::vector<int64_t>& align) {
    // align[d] = dim of t corresponding to output dim d, or -1 (broadcast
    // from a leading position / scalar).
    int64_t off = 0;
    for (size_t d = 0; d < coord.size(); d++) {
        int td = align[d];
        if (td < 0) continue;
        int64_t c = coord[d];
        if (t.shape[td] == 1) c = 0;
        off += c * t.strides[td];
    }
    return off;
}

// Align an operand's dims to the output rank (right-aligned broadcast).
std::vector<int64_t> align_of(const RT& t, size_t out_rank) {
    std::vector<int64_t> a(out_rank, -1);
    int64_t shift = (int64_t)out_rank - (int64_t)t.shape.size();
    for (size_t i = 0; i < t.shape.size(); i++) a[shift + i] = (int64_t)i;
    return a;
}

void copy_tensor(const RT& dst, const RT& src);  // fwd

// Materialize src into fresh contiguous storage (scratch or dst buffer).
RT materialize(const RT& src, uint8_t* buf) {
    RT d;
    d.dtype = src.dtype;
    d.p = buf;
    d.shape = src.shape;
    d.strides = rowmajor(src.shape);
    copy_tensor(d, src);
    return d;
}

void copy_tensor(const RT& dst, const RT& src) {
    if (dst.numel() == 0) return;
    size_t es = onnxm::elem_size(dst.dtype);
    if (dst.contig() && src.contig() && dst.shape == src.shape) {
        memcpy(dst.p, src.p, (size_t)dst.numel() * es);
        return;
    }
    // Strided walk with incremental offsets (dst rank == src rank here
    // by construction of callers; src dims may broadcast from size 1).
    if (dst.shape.size() <= 8 && src.shape.size() == dst.shape.size()) {
        int r = (int)dst.shape.size();
        int64_t ss[8], ds[8];
        for (int d = 0; d < r; d++) {
            ss[d] = src.shape[d] == 1 ? 0 : src.strides[d];
            ds[d] = dst.strides[d];
        }
        int64_t coord[8] = {0};
        int64_t so = 0, dof = 0;
        int64_t n = dst.numel();
        for (int64_t i = 0; i < n; i++) {
            memcpy(dst.p + dof * (int64_t)es, src.p + so * (int64_t)es, es);
            for (int d = r - 1; d >= 0; d--) {
                so += ss[d]; dof += ds[d];
                if (++coord[d] < dst.shape[d]) break;
                coord[d] = 0;
                so -= ss[d] * dst.shape[d]; dof -= ds[d] * dst.shape[d];
            }
        }
        return;
    }
    auto sa = align_of(src, dst.shape.size());
    auto da = align_of(dst, dst.shape.size());
    odometer(dst.shape, [&](const std::vector<int64_t>& c) {
        int64_t so = off_of(src, c, sa);
        int64_t dof = off_of(dst, c, da);
        memcpy(dst.p + dof * (int64_t)es, src.p + so * (int64_t)es, es);
    });
}

// Logical-order copy (reshape semantics): element i of dst in row-major
// order takes element i of src in its own row-major logical order.
void flatten_copy(const RT& dst, const RT& src) {
    size_t es = onnxm::elem_size(dst.dtype);
    if (src.contig() && dst.contig()) {
        memcpy(dst.p, src.p, (size_t)dst.numel() * es);
        return;
    }
    int64_t idx = 0;
    odometer(src.shape, [&](const std::vector<int64_t>& c) {
        int64_t so = 0;
        for (size_t d = 0; d < c.size(); d++) so += c[d] * src.strides[d];
        memcpy(dst.p + idx * (int64_t)es, src.p + so * (int64_t)es, es);
        idx++;
    });
}

// Scratch bump allocator for per-step temporaries (contiguity fixes).
// Chunked: blocks never move once allocated, because a step may hold
// several materialized tensors at once (a growing vector would leave
// the earlier pointers dangling -- that was a real segfault source).
struct Scratch {
    std::vector<std::vector<uint8_t>> blocks;
    size_t bi = 0, top = 0;
    struct Mark { size_t bi, top; };
    Mark mark() const { return {bi, top}; }
    void reset(Mark m) { bi = m.bi; top = m.top; }
    uint8_t* alloc(size_t bytes) {
        if (blocks.empty()) blocks.emplace_back(1 << 20);
        size_t at = (top + 63) & ~(size_t)63;
        if (at + bytes > blocks[bi].size()) {
            size_t want = std::max<size_t>(bytes + 4096, 1 << 20);
            if (bi + 1 < blocks.size() && blocks[bi + 1].size() >= want) {
                bi++;
            } else if (bi + 1 < blocks.size()) {
                blocks[bi + 1].resize(want);
                bi++;
            } else {
                blocks.emplace_back(want);
                bi = blocks.size() - 1;
            }
            at = 0;
        }
        top = at + bytes;
        return blocks[bi].data() + at;
    }
    RT contig_of(const RT& t) {
        if (t.contig()) return t;
        return materialize(t, alloc((size_t)t.numel() * onnxm::elem_size(t.dtype)));
    }
};

// ------------------------------------------------------------- kernels --

void k_matmul(const RT& A0, const RT& B0, RT& C, Scratch& sc) {
    // FE_MM_DEBUG=1: dump shapes/strides of the first 400 calls.
    if (std::getenv("FE_MM_DEBUG")) {
        static int mm_dbg = 0;
        if (mm_dbg++ < 400) {
            auto fmt = [](const RT& t) {
                std::string s = "[";
                for (size_t i = 0; i < t.shape.size(); i++) { if (i) s += ","; s += std::to_string(t.shape[i]); }
                s += "]<";
                for (size_t i = 0; i < t.strides.size(); i++) { if (i) s += ","; s += std::to_string(t.strides[i]); }
                return s + ">";
            };
            std::fprintf(stderr, "[mm] A%s B%s C%s\n", fmt(A0).c_str(), fmt(B0).c_str(), fmt(C).c_str());
        }
    }
    // Fast path: batched GEMV against a transposed view of a contiguous
    // matrix (attention scores Q.K^T, one row per head). B0 (..., K, N)
    // with last-two strides (1, K) means the underlying (..., N, K) rows
    // are contiguous: y[b][j] = sum_k a[b][k] * B0.p[b][j*K + k]. The
    // general path below computes the same product as a stride-K SAXPY
    // sweep, re-reading the whole K cache once per k (stride-K gathers);
    // the dot form reads it exactly once, sequentially. Accumulation is
    // double, in k order, as in the original single-batch fast path.
    if (A0.dtype == 1 && B0.dtype == 1 && C.contig()) {
        int ra0 = (int)A0.shape.size(), rb0 = (int)B0.shape.size();
        if (ra0 >= 2 && rb0 >= 2) {
            int64_t M = A0.shape[ra0 - 2], K = A0.shape[ra0 - 1], N = B0.shape[rb0 - 1];
            if (M == 1 && A0.strides[ra0 - 1] == 1 && B0.strides[rb0 - 2] == 1 && B0.strides[rb0 - 1] == K) {
                int batch = (int)(C.numel() / N);
                auto bst = [](const RT& t, const std::vector<int64_t>& cshape) {
                    std::vector<int64_t> st;
                    int rank = (int)t.shape.size();
                    int b = (int)cshape.size() - 2;
                    for (int i = 0; i < b; i++) {
                        int ti = i - (b - (rank - 2));
                        if (ti < 0 || (t.shape[ti] == 1 && cshape[i] != 1)) st.push_back(0);
                        else st.push_back(t.strides[ti]);
                    }
                    return st;
                };
                std::vector<int64_t> as = bst(A0, C.shape), bs = bst(B0, C.shape);
                std::vector<int64_t> bdims(C.shape.begin(), C.shape.end() - 2);
                std::vector<int64_t> aoff(batch), boff(batch);
                for (int b = 0; b < batch; b++) {
                    int64_t rem = b, ao = 0, bo = 0;
                    for (int i = (int)bdims.size() - 1; i >= 0; i--) {
                        int64_t c = bdims[i] ? rem % bdims[i] : 0;
                        rem /= bdims[i] ? bdims[i] : 1;
                        ao += c * as[i];
                        bo += c * bs[i];
                    }
                    aoff[b] = ao;
                    boff[b] = bo;
                }
                const float* ap0 = (const float*)A0.p;
                const float* bp0 = (const float*)B0.p;
                float* cp0 = (float*)C.p;
                int64_t work = (int64_t)batch * N;
                auto body = [&](int idx, int nt) {
                    packed::chunk(idx, nt, work, [&](int64_t w0, int64_t w1) {
                        for (int64_t w = w0; w < w1; w++) {
                            int b = (int)(w / N);
                            int64_t j = w % N;
                            const float* a = ap0 + aoff[b];
                            const float* row = bp0 + boff[b] + j * K;
                            double s = 0;
                            for (int64_t k = 0; k < K; k++) s += (double)a[k] * row[k];
                            cp0[w] = (float)s;
                        }
                    });
                };
                // Small GEMVs serially: a pool rendezvous costs more
                // than the whole product at this size.
                if (work * K >= (1 << 20)) packed::kernel_pool().run(body);
                else body(0, 1);
                return;
            }
        }
    }
    // No contiguity forcing: every access below goes through strides,
    // and materializing a transposed view (attention V) costs more
    // than the strided reads it saves.
    RT A = A0, B = B0;
    // Shapes: A [..., M, K], B [..., K, N] with broadcast batch dims.
    int ra = (int)A.shape.size(), rb = (int)B.shape.size();
    int64_t M = A.shape[ra - 2], K = A.shape[ra - 1], N = B.shape[rb - 1];
    int batch = (int)C.numel() / (int)(M * N);
    // Batch strides (elements) after broadcast expansion.
    auto batch_strides = [](const RT& t, const std::vector<int64_t>& cshape, int mat_dims) {
        std::vector<int64_t> st;
        int rank = (int)t.shape.size();
        int b = (int)cshape.size() - mat_dims;
        for (int i = 0; i < b; i++) {
            int ti = i - (b - (rank - mat_dims));
            if (ti < 0 || t.shape[ti] == 1 && cshape[i] != 1) st.push_back(0);
            else st.push_back(t.strides[ti]);
        }
        return st;
    };
    std::vector<int64_t> as = batch_strides(A, C.shape, 2), bs = batch_strides(B, C.shape, 2);
    std::vector<int64_t> bdims(C.shape.begin(), C.shape.end() - 2);
    const float* ap0 = (const float*)A.p;
    const float* bp0 = (const float*)B.p;
    float* cp0 = (float*)C.p;
    int64_t arow = A.strides[ra - 2], acol = A.strides[ra - 1];
    int64_t brow = B.strides[rb - 2], bcol = B.strides[rb - 1];
    // Per-batch operand offsets.
    std::vector<int64_t> aoff(batch), boff(batch);
    for (int b = 0; b < batch; b++) {
        int64_t rem = b, ao = 0, bo = 0;
        for (int i = (int)bdims.size() - 1; i >= 0; i--) {
            int64_t c = bdims[i] ? rem % bdims[i] : 0;
            rem /= bdims[i] ? bdims[i] : 1;
            ao += c * as[i];
            bo += c * bs[i];
        }
        aoff[b] = ao;
        boff[b] = bo;
    }
    int64_t rows = (int64_t)batch * M;
    if (rows >= 4) {
        // One pool run over all (batch, row) pairs: per-batch pool
        // synchronization would dominate multi-head attention. Rows are
        // processed in tiles of 16 with k as the outer loop, so each B
        // row segment is read once per tile instead of once per row
        // (prefill M~200 against a 16 MB B otherwise re-reads B ~200x).
        const int64_t RT = 16;
        int64_t ntiles = (rows + RT - 1) / RT;
        auto tile_body = [&](int idx, int nt) {
            packed::chunk(idx, nt, ntiles, [&](int64_t t0, int64_t t1) {
                for (int64_t t = t0; t < t1; t++) {
                    int64_t r0 = t * RT, r1 = std::min(rows, r0 + RT);
                    for (int64_t r = r0; r < r1; r++) {
                        float* crow = cp0 + r * N;
                        for (int64_t j = 0; j < N; j++) crow[j] = 0.0f;
                    }
                    for (int64_t k = 0; k < K; k++) {
                        for (int64_t r = r0; r < r1; r++) {
                            int b = (int)(r / M);
                            int64_t m = r % M;
                            const float* a = ap0 + aoff[b] + m * arow;
                            const float* bb = bp0 + boff[b];
                            float av = a[k * acol];
                            const float* browp = bb + k * brow;
                            float* crow = cp0 + r * N;
                            if (bcol == 1) {
                                for (int64_t j = 0; j < N; j++) crow[j] += av * browp[j];
                            } else {
                                for (int64_t j = 0; j < N; j++) crow[j] += av * browp[j * bcol];
                            }
                        }
                    }
                }
            });
        };
        // Decode attention products are ~1e5 MAC: serial wins over a
        // pool rendezvous. Prefill stays pooled.
        if ((int64_t)batch * M * N * K >= (1 << 20)) packed::kernel_pool().run(tile_body);
        else tile_body(0, 1);
        return;
    }
    for (int b = 0; b < batch; b++) {
        const float* a = ap0 + aoff[b];
        const float* bb = bp0 + boff[b];
        float* c = cp0 + (int64_t)b * M * N;
        if (M == 1) {
            // GEMV: y[j] = sum_k a[k] * B[k][j]; parallel over j blocks.
            auto gemv_body = [&](int idx, int nt) {
                packed::chunk(idx, nt, N, [&](int64_t j0, int64_t j1) {
                    for (int64_t j = j0; j < j1; j++) c[j] = 0.0f;
                    for (int64_t k = 0; k < K; k++) {
                        float av = a[k * acol];
                        const float* browp = bb + k * brow + j0 * bcol;
                        if (bcol == 1) {
                            for (int64_t j = j0; j < j1; j++) c[j] += av * browp[j - j0];
                        } else {
                            for (int64_t j = j0; j < j1; j++) c[j] += av * browp[(j - j0) * bcol];
                        }
                    }
                });
            };
            if (M * N * K >= (1 << 20)) packed::kernel_pool().run(gemv_body);
            else gemv_body(0, 1);
        } else {
            auto gemm_body = [&](int idx, int nt) {
                packed::chunk(idx, nt, M, [&](int64_t m0, int64_t m1) {
                    for (int64_t m = m0; m < m1; m++) {
                        float* crow = c + m * N;
                        for (int64_t j = 0; j < N; j++) crow[j] = 0.0f;
                        for (int64_t k = 0; k < K; k++) {
                            float av = a[m * arow + k * acol];
                            const float* browp = bb + k * brow;
                            if (bcol == 1) {
                                for (int64_t j = 0; j < N; j++) crow[j] += av * browp[j];
                            } else {
                                for (int64_t j = 0; j < N; j++) crow[j] += av * browp[j * bcol];
                            }
                        }
                    }
                });
            };
            if (M * N * K >= (1 << 20)) packed::kernel_pool().run(gemm_body);
            else gemm_body(0, 1);
        }
    }
}

void k_layernorm(const RT& X0, const RT* W, const RT* B, RT& Y, int64_t axis, float eps, Scratch& sc) {
    RT X = sc.contig_of(X0);
    int rank = (int)X.shape.size();
    if (axis < 0) axis += rank;
    int64_t D = 1;
    for (int i = (int)axis; i < rank; i++) D *= X.shape[i];
    int64_t rows = X.numel() / D;
    const float* x = (const float*)X.p;
    float* y = (float*)Y.p;
    const float* w = W ? (const float*)W->p : nullptr;
    const float* b = B ? (const float*)B->p : nullptr;
    // W/B may be strided views; normalize through contig copies if needed.
    RT wc, bc;
    if (W && !W->contig()) wc = sc.contig_of(*W), w = (const float*)wc.p;
    if (B && !B->contig()) bc = sc.contig_of(*B), b = (const float*)bc.p;
    auto body = [&](int idx, int nt) {
        packed::chunk(idx, nt, rows, [&](int64_t r0, int64_t r1) {
            for (int64_t r = r0; r < r1; r++) {
                const float* xr = x + r * D;
                float* yr = y + r * D;
                double mean = 0;
                for (int64_t i = 0; i < D; i++) mean += xr[i];
                mean /= D;
                double var = 0;
                for (int64_t i = 0; i < D; i++) { double d = xr[i] - mean; var += d * d; }
                var /= D;
                double inv = 1.0 / std::sqrt(var + eps);
                for (int64_t i = 0; i < D; i++) {
                    float v = (float)((xr[i] - mean) * inv);
                    if (w) v *= w[i];
                    if (b) v += b[i];
                    yr[i] = v;
                }
            }
        });
    };
    // Small work serially: the pool barrier costs more than the math.
    if (X.numel() < 8192) body(0, 1);
    else packed::kernel_pool().run(body);
}

void k_softmax(const RT& X0, RT& Y, int64_t axis, Scratch& sc) {
    RT X = sc.contig_of(X0);
    int rank = (int)X.shape.size();
    if (axis < 0) axis += rank;
    int64_t D = X.shape[axis];
    int64_t outer = 1, inner = 1;
    for (int i = 0; i < axis; i++) outer *= X.shape[i];
    for (size_t i = (size_t)axis + 1; i < X.shape.size(); i++) inner *= X.shape[i];
    const float* x = (const float*)X.p;
    float* y = (float*)Y.p;
    int64_t lines = outer * inner;
    auto body = [&](int idx, int nt) {
        packed::chunk(idx, nt, lines, [&](int64_t l0, int64_t l1) {
            for (int64_t l = l0; l < l1; l++) {
                int64_t o = l / inner, in = l % inner;
                const float* xp = x + o * D * inner + in;
                float* yp = y + o * D * inner + in;
                float mx = -INFINITY;
                for (int64_t i = 0; i < D; i++) mx = std::max(mx, xp[i * inner]);
                double sum = 0;
                for (int64_t i = 0; i < D; i++) { yp[i * inner] = std::exp(xp[i * inner] - mx); sum += yp[i * inner]; }
                float inv = (float)(1.0 / sum);
                for (int64_t i = 0; i < D; i++) yp[i * inner] *= inv;
            }
        });
    };
    if (X.numel() < 8192) body(0, 1);
    else packed::kernel_pool().run(body);
}


// Elementwise binary op codes, shared by compiler + kernels.
enum class BinOp { ADD, SUB, MUL, DIV, POW, MAXIMUM, LESS, GREATER, EQUAL, OR, AND };
enum class UnOp { NEG, SQRT, SIN, COS, EXP, TANH, SIGMOID, RELU, ABS, LEAKY };

double binop_apply(BinOp op, double a, double b) {
    switch (op) {
        case BinOp::ADD: return a + b;
        case BinOp::SUB: return a - b;
        case BinOp::MUL: return a * b;
        case BinOp::DIV: return a / b;
        case BinOp::POW: return std::pow(a, b);
        case BinOp::MAXIMUM: return std::max(a, b);
        case BinOp::LESS: return a < b ? 1 : 0;
        case BinOp::GREATER: return a > b ? 1 : 0;
        case BinOp::EQUAL: return a == b ? 1 : 0;
        case BinOp::OR: return (a != 0 || b != 0) ? 1 : 0;
        case BinOp::AND: return (a != 0 && b != 0) ? 1 : 0;
    }
    return 0;
}
double unop_apply(UnOp op, double a, double alpha = 0.0) {
    switch (op) {
        case UnOp::NEG: return -a;
        case UnOp::SQRT: return std::sqrt(a);
        case UnOp::SIN: return std::sin(a);
        case UnOp::COS: return std::cos(a);
        case UnOp::EXP: return std::exp(a);
        case UnOp::TANH: return std::tanh(a);
        case UnOp::SIGMOID: return 1.0 / (1.0 + std::exp(-a));
        case UnOp::RELU: return a > 0 ? a : 0;
        case UnOp::ABS: return std::fabs(a);
        case UnOp::LEAKY: return a > 0 ? a : alpha * a;
    }
    return 0;
}

float binop_apply_f(BinOp op, float a, float b);  // fwd, defined by k_binop
float unop_apply_f(UnOp op, float a, double alpha);  // fwd

// f32 flat fast path for binary ops when both operands are contiguous
// and either same-shape or scalar.
bool binop_f32_fast(BinOp op, const RT& A, const RT& B, RT& Y) {
    if (A.dtype != 1 || B.dtype != 1 || Y.dtype != 1) return false;
    if (!A.contig() || !B.contig() || !Y.contig()) return false;
    const float* a = (const float*)A.p;
    const float* b = (const float*)B.p;
    float* y = (float*)Y.p;
    int64_t n = Y.numel();
    bool as = A.numel() == n, bs = B.numel() == n;
    if (!as && A.numel() != 1) return false;
    if (!bs && B.numel() != 1) return false;
    if (n >= (1 << 17)) {
        packed::kernel_pool().run([&](int idx, int nt) {
            packed::chunk(idx, nt, n, [&](int64_t i0, int64_t i1) {
                for (int64_t i = i0; i < i1; i++)
                    y[i] = binop_apply_f(op, a[as ? i : 0], b[bs ? i : 0]);
            });
        });
    } else {
        for (int64_t i = 0; i < n; i++) y[i] = binop_apply_f(op, a[as ? i : 0], b[bs ? i : 0]);
    }
    return true;
}

// Aligned strides of an operand against the output shape; broadcast
// dims get stride 0. False when rank exceeds the walker limit.
bool aligned_strides(const RT& t, const std::vector<int64_t>& oshape, int64_t* out) {
    int r = (int)oshape.size();
    if (r > 8) return false;
    int shift = r - (int)t.shape.size();
    for (int d = 0; d < r; d++) {
        int td = d - shift;
        out[d] = td < 0 ? 0 : (t.shape[td] == 1 ? 0 : t.strides[td]);
    }
    return true;
}

// Pool-parallel strided binary walk (f32): each thread decomposes its
// start index once, then walks incrementally.
void binop_walk_par(BinOp op, const RT& A, const RT& B, RT& Y, const int64_t* as, const int64_t* bs, const int64_t* ys) {
    int r = (int)Y.shape.size();
    int64_t suf[8];
    suf[r - 1] = 1;
    for (int d = r - 2; d >= 0; d--) suf[d] = suf[d + 1] * Y.shape[d + 1];
    int64_t n = Y.numel();
    const float* a = (const float*)A.p;
    const float* b = (const float*)B.p;
    float* y = (float*)Y.p;
    packed::kernel_pool().run([&](int idx, int nt) {
        packed::chunk(idx, nt, n, [&](int64_t i0, int64_t i1) {
            int64_t coord[8] = {0};
            int64_t ao = 0, bo = 0, yo = 0;
            int64_t rem = i0;
            for (int d = 0; d < r; d++) {
                coord[d] = rem / suf[d];
                rem -= coord[d] * suf[d];
                ao += coord[d] * as[d]; bo += coord[d] * bs[d]; yo += coord[d] * ys[d];
            }
            for (int64_t i = i0; i < i1; i++) {
                y[yo] = binop_apply_f(op, a[ao], b[bo]);
                for (int d = r - 1; d >= 0; d--) {
                    ao += as[d]; bo += bs[d]; yo += ys[d];
                    if (++coord[d] < Y.shape[d]) break;
                    coord[d] = 0;
                    ao -= as[d] * Y.shape[d]; bo -= bs[d] * Y.shape[d]; yo -= ys[d] * Y.shape[d];
                }
            }
        });
    });
}

// Float form of binop_apply for the ops whose float evaluation is the
// correctly-rounded one (ORT evaluates these in float too). POW and
// anything unlisted fall back to the double form.
float binop_apply_f(BinOp op, float a, float b) {
    switch (op) {
        case BinOp::ADD: return a + b;
        case BinOp::SUB: return a - b;
        case BinOp::MUL: return a * b;
        case BinOp::DIV: return a / b;
        case BinOp::MAXIMUM: return std::max(a, b);
        case BinOp::LESS: return a < b ? 1.0f : 0.0f;
        case BinOp::GREATER: return a > b ? 1.0f : 0.0f;
        case BinOp::EQUAL: return a == b ? 1.0f : 0.0f;
        case BinOp::OR: return (a != 0 || b != 0) ? 1.0f : 0.0f;
        case BinOp::AND: return (a != 0 && b != 0) ? 1.0f : 0.0f;
        default: return (float)binop_apply(op, a, b);
    }
}
float unop_apply_f(UnOp op, float a, double alpha) {
    switch (op) {
        case UnOp::NEG: return -a;
        case UnOp::SQRT: return std::sqrt(a);
        case UnOp::SIN: return std::sin(a);
        case UnOp::COS: return std::cos(a);
        case UnOp::EXP: return std::exp(a);
        case UnOp::TANH: return std::tanh(a);
        case UnOp::SIGMOID: return 1.0f / (1.0f + std::exp(-a));
        case UnOp::RELU: return a > 0 ? a : 0.0f;
        case UnOp::ABS: return std::fabs(a);
        case UnOp::LEAKY: return a > 0 ? a : (float)(alpha * a);
    }
    return 0;
}

// Row-organized elementwise walk (f32): when the output's last dim is
// contiguous and each operand's last-dim stride is 0 (broadcast) or 1,
// each output row is one flat loop the compiler can vectorize, instead
// of a per-element odometer walk. Outer offsets are decomposed once
// per row. Returns false when the layout does not qualify.
bool binop_rows(BinOp op, const RT& A, const RT& B, RT& Y, const int64_t* as, const int64_t* bs, const int64_t* ys) {
    int r = (int)Y.shape.size();
    if (r < 1 || ys[r - 1] != 1) return false;
    if ((as[r - 1] != 0 && as[r - 1] != 1) || (bs[r - 1] != 0 && bs[r - 1] != 1)) return false;
    int64_t D = Y.shape[r - 1];
    int64_t rows = Y.numel() / D;
    int64_t suf[8];
    if (r >= 2) {
        suf[r - 2] = 1;
        for (int d = r - 3; d >= 0; d--) suf[d] = suf[d + 1] * Y.shape[d + 1];
    }
    const float* a = (const float*)A.p;
    const float* b = (const float*)B.p;
    float* y = (float*)Y.p;
    int64_t asl = as[r - 1], bsl = bs[r - 1];
    auto body = [&](int idx, int nt) {
        packed::chunk(idx, nt, rows, [&](int64_t r0, int64_t r1) {
            for (int64_t rr = r0; rr < r1; rr++) {
                int64_t ao = 0, bo = 0, yo = 0, rem = rr;
                for (int d = 0; d < r - 1; d++) {
                    int64_t c = rem / suf[d];
                    rem -= c * suf[d];
                    ao += c * as[d]; bo += c * bs[d]; yo += c * ys[d];
                }
                const float* ap = a + ao;
                const float* bp = b + bo;
                float* yp = y + yo;
                if (asl == 1 && bsl == 1) {
                    for (int64_t i = 0; i < D; i++) yp[i] = binop_apply_f(op, ap[i], bp[i]);
                } else if (asl == 1) {
                    float bv = bp[0];
                    for (int64_t i = 0; i < D; i++) yp[i] = binop_apply_f(op, ap[i], bv);
                } else if (bsl == 1) {
                    float av = ap[0];
                    for (int64_t i = 0; i < D; i++) yp[i] = binop_apply_f(op, av, bp[i]);
                } else {
                    float v = binop_apply_f(op, ap[0], bp[0]);
                    for (int64_t i = 0; i < D; i++) yp[i] = v;
                }
            }
        });
    };
    if (Y.numel() >= (1 << 17)) packed::kernel_pool().run(body);
    else body(0, 1);
    return true;
}
bool unop_rows(UnOp op, const RT& A, RT& Y, double alpha, const int64_t* as, const int64_t* ys) {
    int r = (int)Y.shape.size();
    if (r < 1 || ys[r - 1] != 1) return false;
    if (as[r - 1] != 0 && as[r - 1] != 1) return false;
    int64_t D = Y.shape[r - 1];
    int64_t rows = Y.numel() / D;
    int64_t suf[8];
    if (r >= 2) {
        suf[r - 2] = 1;
        for (int d = r - 3; d >= 0; d--) suf[d] = suf[d + 1] * Y.shape[d + 1];
    }
    const float* a = (const float*)A.p;
    float* y = (float*)Y.p;
    int64_t asl = as[r - 1];
    auto body = [&](int idx, int nt) {
        packed::chunk(idx, nt, rows, [&](int64_t r0, int64_t r1) {
            for (int64_t rr = r0; rr < r1; rr++) {
                int64_t ao = 0, yo = 0, rem = rr;
                for (int d = 0; d < r - 1; d++) {
                    int64_t c = rem / suf[d];
                    rem -= c * suf[d];
                    ao += c * as[d]; yo += c * ys[d];
                }
                const float* ap = a + ao;
                float* yp = y + yo;
                if (asl == 1) {
                    for (int64_t i = 0; i < D; i++) yp[i] = unop_apply_f(op, ap[i], alpha);
                } else {
                    float v = unop_apply_f(op, ap[0], alpha);
                    for (int64_t i = 0; i < D; i++) yp[i] = v;
                }
            }
        });
    };
    if (Y.numel() >= (1 << 17)) packed::kernel_pool().run(body);
    else body(0, 1);
    return true;
}

void k_binop(BinOp op, const RT& A, const RT& B, RT& Y) {
    if (binop_f32_fast(op, A, B, Y)) return;
    if (A.dtype == 1 && B.dtype == 1 && Y.dtype == 1) {
        int64_t as[8], bs[8], ys[8];
        if (aligned_strides(A, Y.shape, as) && aligned_strides(B, Y.shape, bs) && aligned_strides(Y, Y.shape, ys)) {
            if (binop_rows(op, A, B, Y, as, bs, ys)) return;
            if (Y.numel() >= 8192) { binop_walk_par(op, A, B, Y, as, bs, ys); return; }
            int r = (int)Y.shape.size();
            int64_t coord[8] = {0};
            int64_t ao = 0, bo = 0, yo = 0;
            const float* a = (const float*)A.p;
            const float* b = (const float*)B.p;
            float* y = (float*)Y.p;
            int64_t n = Y.numel();
            for (int64_t i = 0; i < n; i++) {
                y[yo] = (float)binop_apply(op, a[ao], b[bo]);
                for (int d = r - 1; d >= 0; d--) {
                    ao += as[d]; bo += bs[d]; yo += ys[d];
                    if (++coord[d] < Y.shape[d]) break;
                    coord[d] = 0;
                    ao -= as[d] * Y.shape[d]; bo -= bs[d] * Y.shape[d]; yo -= ys[d] * Y.shape[d];
                }
            }
            return;
        }
    }
    auto aa = align_of(A, Y.shape.size());
    auto ab = align_of(B, Y.shape.size());
    auto ay = align_of(Y, Y.shape.size());
    odometer(Y.shape, [&](const std::vector<int64_t>& c) {
        double va = read_d(A, off_of(A, c, aa));
        double vb = read_d(B, off_of(B, c, ab));
        write_d(Y, off_of(Y, c, ay), binop_apply(op, va, vb));
    });
}

void k_unop(UnOp op, const RT& A, RT& Y, double alpha = 0.0) {
    if (A.dtype == 1 && A.contig() && Y.contig() && A.numel() == Y.numel()) {
        const float* a = (const float*)A.p;
        float* y = (float*)Y.p;
        int64_t n = Y.numel();
        if (n >= (1 << 17)) {
            packed::kernel_pool().run([&](int idx, int nt) {
                packed::chunk(idx, nt, n, [&](int64_t i0, int64_t i1) {
                    for (int64_t i = i0; i < i1; i++) y[i] = unop_apply_f(op, a[i], alpha);
                });
            });
        } else {
            for (int64_t i = 0; i < n; i++) y[i] = unop_apply_f(op, a[i], alpha);
        }
        return;
    }
    if (A.dtype == 1 && Y.dtype == 1) {
        int64_t as[8], ys[8];
        if (aligned_strides(A, Y.shape, as) && aligned_strides(Y, Y.shape, ys)) {
            if (unop_rows(op, A, Y, alpha, as, ys)) return;
            int r = (int)Y.shape.size();
            int64_t coord[8] = {0};
            int64_t ao = 0, yo = 0;
            const float* a = (const float*)A.p;
            float* y = (float*)Y.p;
            int64_t n = Y.numel();
            for (int64_t i = 0; i < n; i++) {
                y[yo] = (float)unop_apply(op, a[ao], alpha);
                for (int d = r - 1; d >= 0; d--) {
                    ao += as[d]; yo += ys[d];
                    if (++coord[d] < Y.shape[d]) break;
                    coord[d] = 0;
                    ao -= as[d] * Y.shape[d]; yo -= ys[d] * Y.shape[d];
                }
            }
            return;
        }
    }
    auto aa = align_of(A, Y.shape.size());
    auto ay = align_of(Y, Y.shape.size());
    odometer(Y.shape, [&](const std::vector<int64_t>& c) {
        write_d(Y, off_of(Y, c, ay), unop_apply(op, read_d(A, off_of(A, c, aa)), alpha));
    });
}

template <typename InT, typename OutT>
void cast_loop(const RT& A, RT& Y) {
    auto aa = align_of(A, Y.shape.size());
    auto ay = align_of(Y, Y.shape.size());
    odometer(Y.shape, [&](const std::vector<int64_t>& c) {
        InT v;
        memcpy(&v, A.p + off_of(A, c, aa) * (int64_t)sizeof(InT), sizeof(InT));
        OutT o = (OutT)v;
        memcpy(Y.p + off_of(Y, c, ay) * (int64_t)sizeof(OutT), &o, sizeof(OutT));
    });
}
void k_cast(const RT& A, RT& Y) {
    if (A.dtype == Y.dtype) { copy_tensor(Y, A); return; }
    auto run = [&](auto in_tag, auto out_tag) {
        cast_loop<decltype(in_tag), decltype(out_tag)>(A, Y);
    };
    // Dispatch through read_d/write_d for the long tail of type pairs;
    // sizes here are small (metadata casts, position tensors).
    (void)run;
    auto aa = align_of(A, Y.shape.size());
    auto ay = align_of(Y, Y.shape.size());
    odometer(Y.shape, [&](const std::vector<int64_t>& c) {
        write_d(Y, off_of(Y, c, ay), read_d(A, off_of(A, c, aa)));
    });
}

void k_where(const RT& C, const RT& A, const RT& B, RT& Y) {
    auto ac = align_of(C, Y.shape.size());
    auto aa = align_of(A, Y.shape.size());
    auto ab = align_of(B, Y.shape.size());
    auto ay = align_of(Y, Y.shape.size());
    odometer(Y.shape, [&](const std::vector<int64_t>& cd) {
        bool sel = read_d(C, off_of(C, cd, ac)) != 0;
        write_d(Y, off_of(Y, cd, ay), sel ? read_d(A, off_of(A, cd, aa)) : read_d(B, off_of(B, cd, ab)));
    });
}

void k_reducesum(const RT& X, RT& Y, const std::vector<int64_t>& axes, bool keepdims) {
    int rank = (int)X.shape.size();
    std::vector<char> red(rank, 0);
    for (auto a : axes) red[a] = 1;
    // Map each X dim to its Y dim (keepdims: same index; else: index
    // among the non-reduced dims), or -1 when reduced away.
    std::vector<int64_t> ydim(rank, -1);
    int64_t yn = 0;
    for (int d = 0; d < rank; d++) {
        if (keepdims) ydim[d] = d;
        else if (!red[d]) ydim[d] = yn++;
    }
    memset(Y.p, 0, (size_t)Y.numel() * onnxm::elem_size(Y.dtype));
    odometer(X.shape, [&](const std::vector<int64_t>& c) {
        int64_t xo = 0, yo = 0;
        for (int d = 0; d < rank; d++) {
            xo += c[d] * X.strides[d];
            if (!red[d]) yo += c[d] * Y.strides[ydim[d]];
            else if (keepdims) yo += 0;  // coord contributes 0 on size-1 dim
        }
        write_d(Y, yo, read_d(Y, yo) + read_d(X, xo));
    });
}

void k_argmax(const RT& X, RT& Y, int64_t axis, bool keepdims, bool last) {
    (void)keepdims;
    int rank = (int)X.shape.size();
    if (axis < 0) axis += rank;
    int64_t D = X.shape[axis];
    int64_t outer = 1, inner = 1;
    for (int i = 0; i < axis; i++) outer *= X.shape[i];
    for (int i = axis + 1; i < rank; i++) inner *= X.shape[i];
    int64_t s_axis = X.strides[axis];
    int64_t s_outer = axis > 0 ? X.strides[axis - 1] : X.numel();
    (void)s_outer;
    // Lines are addressed via (outer, inner) decomposition over strides.
    std::vector<int64_t> oc_shape = X.shape;
    oc_shape[axis] = 1;
    int64_t lines = outer * inner;
    for (int64_t l = 0; l < lines; l++) {
        int64_t o = l / inner, in = l % inner;
        // base offset: walk outer dims and inner dims through strides
        int64_t base = 0;
        {
            int64_t rem = o;
            for (int d = (int)axis - 1; d >= 0; d--) {
                int64_t c = X.shape[d] ? rem % X.shape[d] : 0;
                rem /= X.shape[d] ? X.shape[d] : 1;
                base += c * X.strides[d];
            }
            rem = in;
            for (int d = rank - 1; d > axis; d--) {
                int64_t c = X.shape[d] ? rem % X.shape[d] : 0;
                rem /= X.shape[d] ? X.shape[d] : 1;
                base += c * X.strides[d];
            }
        }
        double best = -INFINITY;
        int64_t bi = 0;
        for (int64_t i = 0; i < D; i++) {
            double v = read_d(X, base + i * s_axis);
            if (v > best || (last && v == best)) { best = v; bi = i; }
        }
        int64_t yo = 0;
        {
            int64_t rem = o;
            std::vector<int64_t> yshape = Y.shape;
            // Y rank == X rank (keepdims) or rank-1 less; map via oc walk:
            std::vector<int64_t> coord(rank, 0);
            for (int d = (int)axis - 1; d >= 0; d--) {
                coord[d] = X.shape[d] ? (rem % X.shape[d]) : 0;
                rem /= X.shape[d] ? X.shape[d] : 1;
            }
            rem = in;
            for (int d = rank - 1; d > axis; d--) {
                coord[d] = X.shape[d] ? (rem % X.shape[d]) : 0;
                rem /= X.shape[d] ? X.shape[d] : 1;
            }
            coord[axis] = 0;
            // Convert coord in X-rank space to Y offset.
            if ((int)Y.shape.size() == rank) {
                for (int d = 0; d < rank; d++) yo += coord[d] * Y.strides[d];
            } else {
                for (int d = 0; d < rank; d++) {
                    if (d == (int)axis) continue;
                    int yd = d < (int)axis ? d : d - 1;
                    yo += coord[d] * Y.strides[yd];
                }
            }
            (void)yshape;
        }
        int64_t v = bi;
        memcpy(Y.p + yo * 8, &v, 8);
    }
}

void k_topk(const RT& X, RT& Yv, RT& Yi, int64_t axis, int64_t k, bool largest, bool sorted) {
    int rank = (int)X.shape.size();
    if (axis < 0) axis += rank;
    int64_t D = X.shape[axis];
    int64_t outer = 1, inner = 1;
    for (int i = 0; i < axis; i++) outer *= X.shape[i];
    for (int i = axis + 1; i < rank; i++) inner *= X.shape[i];
    int64_t lines = outer * inner;
    std::vector<std::pair<double, int64_t>> buf(D);
    for (int64_t l = 0; l < lines; l++) {
        int64_t o = l / inner, in = l % inner;
        int64_t base = 0;
        {
            int64_t rem = o;
            for (int d = (int)axis - 1; d >= 0; d--) {
                int64_t c = X.shape[d] ? rem % X.shape[d] : 0;
                rem /= X.shape[d] ? X.shape[d] : 1;
                base += c * X.strides[d];
            }
            rem = in;
            for (int d = rank - 1; d > axis; d--) {
                int64_t c = X.shape[d] ? rem % X.shape[d] : 0;
                rem /= X.shape[d] ? X.shape[d] : 1;
                base += c * X.strides[d];
            }
        }
        for (int64_t i = 0; i < D; i++) buf[i] = {read_d(X, base + i * X.strides[axis]), i};
        std::partial_sort(buf.begin(), buf.begin() + k, buf.end(),
                          [&](const std::pair<double, int64_t>& a, const std::pair<double, int64_t>& b) {
                              return largest ? a.first > b.first : a.first < b.first;
                          });
        if (!sorted) std::sort(buf.begin(), buf.begin() + k, [](const std::pair<double, int64_t>& a, const std::pair<double, int64_t>& b) { return a.second < b.second; });
        // Output line offset: same (o, in) decomposition on Y shapes.
        auto line_off = [&](const RT& T) {
            int64_t off = 0;
            int64_t rem = o;
            for (int d = (int)axis - 1; d >= 0; d--) {
                int64_t c = X.shape[d] ? rem % X.shape[d] : 0;
                rem /= X.shape[d] ? X.shape[d] : 1;
                off += c * T.strides[d];
            }
            rem = in;
            for (int d = rank - 1; d > axis; d--) {
                int64_t c = X.shape[d] ? rem % X.shape[d] : 0;
                rem /= X.shape[d] ? X.shape[d] : 1;
                off += c * T.strides[d];
            }
            return off;
        };
        int64_t vo = line_off(Yv), io = line_off(Yi);
        for (int64_t i = 0; i < k; i++) {
            write_d(Yv, vo + i * Yv.strides[axis], buf[i].first);
            int64_t id = buf[i].second;
            memcpy(Yi.p + (io + i * Yi.strides[axis]) * 8, &id, 8);
        }
    }
}

void k_cumsum(const RT& X, RT& Y, int64_t axis, bool exclusive, bool reverse) {
    int rank = (int)X.shape.size();
    if (axis < 0) axis += rank;
    int64_t D = X.shape[axis];
    int64_t outer = 1, inner = 1;
    for (int i = 0; i < axis; i++) outer *= X.shape[i];
    for (int i = axis + 1; i < rank; i++) inner *= X.shape[i];
    int64_t lines = outer * inner;
    for (int64_t l = 0; l < lines; l++) {
        int64_t o = l / inner, in = l % inner;
        int64_t base = 0;
        {
            int64_t rem = o;
            for (int d = (int)axis - 1; d >= 0; d--) {
                int64_t c = X.shape[d] ? rem % X.shape[d] : 0;
                rem /= X.shape[d] ? X.shape[d] : 1;
                base += c * X.strides[d];
            }
            rem = in;
            for (int d = rank - 1; d > axis; d--) {
                int64_t c = X.shape[d] ? rem % X.shape[d] : 0;
                rem /= X.shape[d] ? X.shape[d] : 1;
                base += c * X.strides[d];
            }
        }
        double acc = 0;
        for (int64_t ii = 0; ii < D; ii++) {
            int64_t i = reverse ? D - 1 - ii : ii;
            double v = read_d(X, base + i * X.strides[axis]);
            if (exclusive) {
                write_d(Y, base + i * Y.strides[axis], acc);
                acc += v;
            } else {
                acc += v;
                write_d(Y, base + i * Y.strides[axis], acc);
            }
        }
    }
}

void k_gather(const RT& D, const RT& I, RT& Y, int64_t axis) {
    int rank = (int)D.shape.size();
    if (axis < 0) axis += rank;
    size_t es = onnxm::elem_size(D.dtype);
    int64_t outer = 1;
    for (int i = 0; i < axis; i++) outer *= D.shape[i];
    int64_t inner = 1;
    for (int i = axis + 1; i < rank; i++) inner *= D.shape[i];
    int64_t icount = I.numel();
    // Fast path: data contiguous, output contiguous, inner block memcpy.
    if (D.contig() && Y.contig()) {
        int64_t d_axis_stride = inner;
        for (int64_t o = 0; o < outer; o++) {
            for (int64_t ii = 0; ii < icount; ii++) {
                int64_t idx = (int64_t)read_d(I, ii);
                if (idx < 0) idx += D.shape[axis];
                const uint8_t* s = D.p + ((o * D.shape[axis] + idx) * inner) * (int64_t)es;
                uint8_t* d = Y.p + ((o * icount + ii) * inner) * (int64_t)es;
                memcpy(d, s, (size_t)inner * es);
            }
        }
        return;
    }
    auto ya = align_of(Y, Y.shape.size());
    odometer(Y.shape, [&](const std::vector<int64_t>& c) {
        // Y coords: [data outer dims..., indices dims..., data inner dims...]
        std::vector<int64_t> dc(D.shape.size(), 0), ic(I.shape.size(), 0);
        for (int i = 0; i < axis; i++) dc[i] = c[i];
        for (size_t i = 0; i < I.shape.size(); i++) ic[i] = c[axis + i];
        for (int i = axis + 1; i < rank; i++) dc[i] = c[axis + (int64_t)I.shape.size() + (i - axis - 1)];
        int64_t io = 0;
        for (size_t i = 0; i < I.shape.size(); i++) io += ic[i] * I.strides[i];
        int64_t idx = (int64_t)read_d(I, io);
        if (idx < 0) idx += D.shape[axis];
        dc[axis] = idx;
        int64_t so = 0;
        for (int i = 0; i < rank; i++) so += dc[i] * D.strides[i];
        int64_t dof = off_of(Y, c, ya);
        memcpy(Y.p + dof * (int64_t)es, D.p + so * (int64_t)es, es);
    });
}

void k_gather_elements(const RT& D, const RT& I, RT& Y, int64_t axis) {
    int rank = (int)D.shape.size();
    if (axis < 0) axis += rank;
    size_t es = onnxm::elem_size(D.dtype);
    auto ia = align_of(I, Y.shape.size());
    auto ya = align_of(Y, Y.shape.size());
    odometer(Y.shape, [&](const std::vector<int64_t>& c) {
        int64_t idx = (int64_t)read_d(I, off_of(I, c, ia));
        if (idx < 0) idx += D.shape[axis];
        int64_t so = 0;
        for (int i = 0; i < rank; i++) so += (i == (int)axis ? idx : c[i]) * D.strides[i];
        memcpy(Y.p + off_of(Y, c, ya) * (int64_t)es, D.p + so * (int64_t)es, es);
    });
}

void k_scatter_elements(const RT& D, const RT& I, const RT& U, RT& Y, int64_t axis, int reduction) {
    copy_tensor(Y, D);
    int rank = (int)D.shape.size();
    if (axis < 0) axis += rank;
    auto ia = align_of(I, I.shape.size());
    auto ua = align_of(U, I.shape.size());
    odometer(I.shape, [&](const std::vector<int64_t>& c) {
        int64_t idx = (int64_t)read_d(I, off_of(I, c, ia));
        if (idx < 0) idx += D.shape[axis];
        int64_t yo = 0;
        for (int i = 0; i < rank; i++) yo += (i == (int)axis ? idx : c[i]) * Y.strides[i];
        double v = read_d(U, off_of(U, c, ua));
        if (reduction == 1) write_d(Y, yo, read_d(Y, yo) + v);        // add
        else if (reduction == 2) write_d(Y, yo, read_d(Y, yo) * v);   // mul
        else write_d(Y, yo, v);                                       // none
    });
}

void k_concat(const std::vector<const RT*>& ins, RT& Y, int64_t axis, Scratch& sc) {
    int rank = (int)Y.shape.size();
    if (axis < 0) axis += rank;
    int64_t off = 0;
    for (auto* ip : ins) {
        RT sub;
        sub.dtype = Y.dtype;
        sub.p = Y.p;
        sub.shape = ip->shape;
        sub.strides = Y.strides;
        // Offset along axis in elements of Y's layout.
        sub.p = Y.p + off * Y.strides[axis] * (int64_t)onnxm::elem_size(Y.dtype);
        RT in = sc.contig_of(*ip);
        copy_tensor(sub, in);
        off += ip->shape[axis];
    }
}

void k_conv(const RT& X0, const RT& W0, const RT* B, RT& Y, const std::vector<int64_t>& strides_a,
            const std::vector<int64_t>& pads, const std::vector<int64_t>& dil, int64_t group, Scratch& sc) {
    if (group != 1) fail("Conv: group != 1 not implemented");
    RT X = sc.contig_of(X0), W = sc.contig_of(W0);
    RT Bc;
    const float* bias = nullptr;
    if (B) { Bc = sc.contig_of(*B); bias = (const float*)Bc.p; }
    int sp = (int)X.shape.size() - 2;
    if (sp < 1 || sp > 2) fail("Conv: spatial rank not 1..2");
    int64_t N = X.shape[0], C = X.shape[1], O = W.shape[0];
    const float* x = (const float*)X.p;
    const float* w = (const float*)W.p;
    float* y = (float*)Y.p;
    if (sp == 1) {
        // Direct conv1d, SAXPY-over-l form (same loop shape as the
        // packed conv1d_fast): one output line (n, o) per task; for
        // each (c, k) the valid l range is clamped once and the inner
        // loop is a contiguous (stride 1) or regularly strided SAXPY.
        int64_t L = X.shape[2], K = W.shape[2], Lout = Y.shape[2];
        int64_t stride = strides_a[0], dilv = dil[0], pl = pads[0];
        packed::kernel_pool().run([&](int idx, int nt) {
            packed::chunk(idx, nt, N * O, [&](int64_t t0, int64_t t1) {
                for (int64_t t = t0; t < t1; t++) {
                    int64_t n = t / O, o = t % O;
                    float* yl = y + (size_t)(n * O + o) * Lout;
                    float bv = bias ? bias[o] : 0.0f;
                    for (int64_t l = 0; l < Lout; l++) yl[l] = bv;
                    for (int64_t c = 0; c < C; c++) {
                        const float* xr = x + (size_t)(n * C + c) * L;
                        const float* wr = w + (size_t)(o * C + c) * K;
                        for (int64_t k = 0; k < K; k++) {
                            float wv = wr[k];
                            if (wv == 0.0f) continue;
                            int64_t base = k * dilv - pl;
                            int64_t l_lo = 0, l_hi = Lout;
                            if (base < 0) l_lo = (-base + stride - 1) / stride;
                            if (base > L - 1) continue;
                            l_hi = std::min(Lout, (L - 1 - base) / stride + 1);
                            if (l_lo >= l_hi) continue;
                            if (stride == 1) {
                                const float* xs = xr + l_lo + base;
                                for (int64_t l = l_lo; l < l_hi; l++) yl[l] += wv * xs[l - l_lo];
                            } else {
                                for (int64_t l = l_lo; l < l_hi; l++) yl[l] += wv * xr[l * stride + base];
                            }
                        }
                    }
                }
            });
        });
        return;
    }
    // sp == 2: direct dot-product form, one output line (n, o) per task.
    int64_t H = X.shape[2], Wd = X.shape[3], KH = W.shape[2], KW = W.shape[3];
    int64_t Hout = Y.shape[2], Wout = Y.shape[3];
    packed::kernel_pool().run([&](int idx, int nt) {
        packed::chunk(idx, nt, N * O, [&](int64_t t0, int64_t t1) {
            for (int64_t t = t0; t < t1; t++) {
                int64_t n = t / O, o = t % O;
                float* yl = y + (size_t)(n * O + o) * Hout * Wout;
                for (int64_t h = 0; h < Hout; h++) {
                    for (int64_t wd = 0; wd < Wout; wd++) {
                        double acc = bias ? bias[o] : 0.0;
                        for (int64_t c = 0; c < C; c++) {
                            const float* xr = x + (size_t)(n * C + c) * H * Wd;
                            const float* wr = w + (size_t)(o * C + c) * KH * KW;
                            for (int64_t kh = 0; kh < KH; kh++) {
                                int64_t ph = h * strides_a[0] + kh * dil[0] - pads[0];
                                if (ph < 0 || ph >= H) continue;
                                for (int64_t kw = 0; kw < KW; kw++) {
                                    int64_t pw = wd * strides_a[1] + kw * dil[1] - pads[1];
                                    if (pw < 0 || pw >= Wd) continue;
                                    acc += (double)xr[ph * Wd + pw] * wr[kh * KW + kw];
                                }
                            }
                        }
                        yl[h * Wout + wd] = (float)acc;
                    }
                }
            }
        });
    });
}

// fp32 ConvTranspose (phase 2; the baked vits graph's 5 nodes are all
// sp=1, group=1). Gather form: each output element accumulates over
// (ci, k) with li = (lo + pad_begin - k*dil) / stride where divisible.
// Output lines (n, o) are independent, so they fan out over the pool.
void k_convtranspose(const RT& X0, const RT& W0, const RT* B, RT& Y,
                     const std::vector<int64_t>& strides_a, const std::vector<int64_t>& pads_b,
                     const std::vector<int64_t>& dil, Scratch& sc) {
    RT X = sc.contig_of(X0), W = sc.contig_of(W0);
    int sp = (int)X.shape.size() - 2;
    if (sp < 1 || sp > 2) fail("ConvTranspose: spatial rank not 1..2");
    if (X.dtype != 1 || W.dtype != 1 || Y.dtype != 1) fail("ConvTranspose: f32 only");
    int64_t N = X.shape[0], Ci = X.shape[1], O = W.shape[1];
    int64_t xsp = 1, ksp = 1, osp = 1;
    for (int i = 0; i < sp; i++) {
        xsp *= X.shape[2 + i];
        ksp *= W.shape[2 + i];
        osp *= Y.shape[2 + i];
    }
    const float* x = (const float*)X.p;
    const float* w = (const float*)W.p;
    float* y = (float*)Y.p;
    if (sp == 1 && dil[0] == 1) {
        // Phase decomposition (same structure as the packed
        // conv_transpose1d_fast): output positions lo = m*stride + p
        // draw on taps kk with (p + pad_b - kk) % stride == 0; per
        // phase the result is a plain stride-1 conv1d with a gathered
        // phase kernel, written back interleaved. The gather form
        // below costs an integer division per (output, tap) candidate
        // and is ~100x slower on the vits decoder upsamples.
        RT Bc;
        const float* bias = nullptr;
        if (B) { Bc = sc.contig_of(*B); bias = (const float*)Bc.p; }
        int64_t L = X.shape[2], K = W.shape[2], Tout = Y.shape[2];
        int64_t stride = strides_a[0], pl = pads_b[0];
        for (int64_t p = 0; p < stride; p++) {
            if (p >= Tout) break;
            int64_t kks[64];
            int64_t Kp = 0;
            bool overflow = false;
            for (int64_t kk = K - 1; kk >= 0; kk--) {
                int64_t rem = (p + pl - kk) % stride;
                if (rem < 0) rem += stride;
                if (rem == 0) {
                    if (Kp >= 64) { overflow = true; break; }
                    kks[Kp++] = kk;
                }
            }
            if (overflow) break;  // falls through to the gather form, which rewrites all of Y
            if (Kp == 0) {
                // No tap lands on this phase: those outputs are bias only.
                int64_t Lm0 = (Tout - 1 - p) / stride + 1;
                for (int64_t n = 0; n < N; n++)
                    for (int64_t o = 0; o < O; o++) {
                        float* dst = y + ((size_t)n * O + o) * Tout + p;
                        float bv = bias ? bias[o] : 0.0f;
                        for (int64_t m = 0; m < Lm0; m++) dst[(size_t)m * stride] = bv;
                    }
                continue;
            }
            int64_t shift0 = (p + pl - kks[0]) / stride;
            int64_t Lm = (Tout - 1 - p) / stride + 1;
            // prow[(o*Ci + ci)*Kp + j] = W[ci][o][kks[j]]
            std::vector<float> prow((size_t)O * Ci * Kp);
            for (int64_t o = 0; o < O; o++)
                for (int64_t ci = 0; ci < Ci; ci++) {
                    const float* wrow = w + ((size_t)ci * O + o) * K;
                    float* dst = prow.data() + ((size_t)o * Ci + ci) * Kp;
                    for (int64_t j = 0; j < Kp; j++) dst[j] = wrow[kks[j]];
                }
            std::vector<float> tmp((size_t)N * O * Lm);
            // Phase conv: out[m] = bias + sum_{ci,j} X[ci][m + shift0 + j] * prow[o][ci][j]
            // (kks descending makes the implied input shift increase by
            // exactly 1 per tap: a stride-1 conv with pad_l = -shift0,
            // the same form conv_transpose1d_fast uses).
            packed::kernel_pool().run([&](int idx, int nt) {
                packed::chunk(idx, nt, N * O, [&](int64_t t0, int64_t t1) {
                    for (int64_t t = t0; t < t1; t++) {
                        int64_t n = t / O, o = t % O;
                        float* yl = tmp.data() + (size_t)t * Lm;
                        float bv = bias ? bias[o] : 0.0f;
                        for (int64_t m = 0; m < Lm; m++) yl[m] = bv;
                        for (int64_t ci = 0; ci < Ci; ci++) {
                            const float* xr = x + ((size_t)n * Ci + ci) * L;
                            const float* wr = prow.data() + ((size_t)o * Ci + ci) * Kp;
                            for (int64_t j = 0; j < Kp; j++) {
                                float wv = wr[j];
                                if (wv == 0.0f) continue;
                                int64_t base = shift0 + j;
                                int64_t m_lo = base >= 0 ? 0 : -base;
                                int64_t m_hi = std::min(Lm, L - base);
                                if (m_lo >= m_hi) continue;
                                const float* xs = xr + m_lo + base;
                                for (int64_t m = m_lo; m < m_hi; m++) yl[m] += wv * xs[m - m_lo];
                            }
                        }
                    }
                });
            });
            for (int64_t n = 0; n < N; n++)
                for (int64_t o = 0; o < O; o++) {
                    const float* src = tmp.data() + ((size_t)n * O + o) * Lm;
                    float* dst = y + ((size_t)n * O + o) * Tout + p;
                    for (int64_t m = 0; m < Lm; m++) dst[(size_t)m * stride] = src[m];
                }
            if (p == stride - 1) return;
            continue;
        }
        // overflow (Kp > 64) falls through to the gather form.
    }
    packed::kernel_pool().run([&](int idx, int nt) {
        packed::chunk(idx, nt, N * O, [&](int64_t l0, int64_t l1) {
            std::vector<int64_t> oc(sp), kc(sp), li(sp);
            for (int64_t ln = l0; ln < l1; ln++) {
                int64_t n = ln / O, o = ln % O;
                float bv = B ? ((const float*)B->p)[o] : 0.0f;
                float* yrow = y + (size_t)ln * osp;
                std::fill(oc.begin(), oc.end(), 0);
                for (int64_t oi = 0; oi < osp; oi++) {
                    double acc = bv;
                    for (int64_t ci = 0; ci < Ci; ci++) {
                        const float* xrow = x + ((size_t)n * Ci + ci) * xsp;
                        const float* wrow = w + ((size_t)ci * O + o) * ksp;
                        std::fill(kc.begin(), kc.end(), 0);
                        for (int64_t ki = 0; ki < ksp; ki++) {
                            int64_t xoff = 0;
                            bool ok = true;
                            for (int d = 0; d < sp; d++) {
                                int64_t num = oc[d] + pads_b[d] - kc[d] * dil[d];
                                if (num < 0 || num % strides_a[d] != 0) { ok = false; break; }
                                li[d] = num / strides_a[d];
                                if (li[d] >= X.shape[2 + d]) { ok = false; break; }
                                xoff = xoff * X.shape[2 + d] + li[d];
                            }
                            if (ok) acc += (double)xrow[xoff] * (double)wrow[ki];
                            for (int d = sp - 1; d >= 0; d--) {
                                if (++kc[d] < W.shape[2 + d]) break;
                                kc[d] = 0;
                            }
                        }
                    }
                    yrow[oi] = (float)acc;
                    for (int d = sp - 1; d >= 0; d--) {
                        if (++oc[d] < Y.shape[2 + d]) break;
                        oc[d] = 0;
                    }
                }
            }
        });
    });
}

// Pad (phase 2): mode 0 constant (fill), 1 reflect (mirror, edge not
// repeated), 2 edge (clamp). pads has 2*rank entries [begin..., end...].
// Fast path (f32, both contiguous): rows along the last dim. Outer
// coords are mapped once per row; the last dim is segment copies
// (memcpy for the interior, short reversed runs for reflect). The
// generic path below walks every element through the odometer and was
// the vits glue hotspot (96 calls on multi-M-element tensors).
bool k_pad_rows(const RT& X, RT& Y, const std::vector<int64_t>& pads, int mode, double fill) {
    if (X.dtype != 1 || Y.dtype != 1 || !X.contig() || !Y.contig()) return false;
    int r = (int)Y.shape.size();
    if (r < 1 || (int)X.shape.size() != r) return false;
    int64_t Wout = Y.shape[r - 1], Win = X.shape[r - 1];
    int64_t pb = pads[r - 1];
    int64_t rows = Y.numel() / Wout;
    std::vector<int64_t> oshape(Y.shape.begin(), Y.shape.end() - 1);
    const float* x = (const float*)X.p;
    float* y = (float*)Y.p;
    float fv = (float)fill;
    auto mapidx = [&](int64_t v, int64_t n) -> int64_t {
        if (mode == 2) return std::max<int64_t>(0, std::min(v, n - 1));
        if (mode == 1) {
            if (n == 1) return 0;
            while (v < 0 || v >= n) {
                if (v < 0) v = -v;
                if (v >= n) v = 2 * n - 2 - v;
            }
            return v;
        }
        return v;  // mode 0: caller range-checks
    };
    auto body = [&](int idx, int nt) {
        packed::chunk(idx, nt, rows, [&](int64_t r0, int64_t r1) {
            std::vector<int64_t> oc(oshape.size(), 0);
            // Seek oc to r0.
            {
                int64_t rem = r0;
                for (int d = (int)oshape.size() - 1; d >= 0; d--) {
                    oc[d] = oshape[d] ? rem % oshape[d] : 0;
                    rem /= oshape[d] ? oshape[d] : 1;
                }
            }
            for (int64_t rr = r0; rr < r1; rr++) {
                // Map outer coords to a source row (or none for mode 0).
                int64_t srow = 0;
                bool outside = false;
                for (int d = 0; d < (int)oshape.size(); d++) {
                    int64_t n = X.shape[d];
                    int64_t v = oc[d] - pads[d];
                    if (mode == 0 && (v < 0 || v >= n)) { outside = true; break; }
                    srow = srow * n + mapidx(v, n);
                }
                float* yp = y + rr * Wout;
                if (outside) {
                    for (int64_t j = 0; j < Wout; j++) yp[j] = fv;
                } else {
                    const float* xp = x + srow * Win;
                    int64_t lo = std::max<int64_t>(0, pb);           // first j with src >= 0
                    int64_t hi = std::min<int64_t>(Wout, pb + Win);  // first j with src >= Win
                    if (mode == 0) {
                        for (int64_t j = 0; j < lo; j++) yp[j] = fv;
                        if (hi > lo) memcpy(yp + lo, xp + (lo - pb), (size_t)(hi - lo) * 4);
                        for (int64_t j = hi; j < Wout; j++) yp[j] = fv;
                    } else if (mode == 2) {
                        for (int64_t j = 0; j < lo; j++) yp[j] = xp[0];
                        if (hi > lo) memcpy(yp + lo, xp + (lo - pb), (size_t)(hi - lo) * 4);
                        for (int64_t j = hi; j < Wout; j++) yp[j] = xp[Win - 1];
                    } else {
                        for (int64_t j = 0; j < lo; j++) yp[j] = xp[mapidx(j - pb, Win)];
                        if (hi > lo) memcpy(yp + lo, xp + (lo - pb), (size_t)(hi - lo) * 4);
                        for (int64_t j = hi; j < Wout; j++) yp[j] = xp[mapidx(j - pb, Win)];
                    }
                }
                for (int d = (int)oshape.size() - 1; d >= 0; d--) {
                    if (++oc[d] < oshape[d]) break;
                    oc[d] = 0;
                }
            }
        });
    };
    if (Y.numel() >= 8192) packed::kernel_pool().run(body);
    else body(0, 1);
    return true;
}

void k_pad(const RT& X, RT& Y, const std::vector<int64_t>& pads, int mode, double fill) {
    if (k_pad_rows(X, Y, pads, mode, fill)) return;
    int r = (int)Y.shape.size();
    if (mode == 0) {
        // Fill everything, then copy the interior overlap.
        odometer(Y.shape, [&](const std::vector<int64_t>& c) {
            int64_t yo = 0;
            for (int d = 0; d < r; d++) yo += c[d] * Y.strides[d];
            write_d(Y, yo, fill);
        });
    }
    auto xa = align_of(X, Y.shape.size());
    odometer(Y.shape, [&](const std::vector<int64_t>& c) {
        std::vector<int64_t> xc = c;
        for (int d = 0; d < r; d++) {
            int td = xa[d];
            if (td < 0) continue;
            int64_t n = X.shape[td];
            int64_t v = c[d] - pads[td];
            if (mode == 0) {
                if (v < 0 || v >= n) return;  // outside: keep fill
            } else if (mode == 2) {
                v = std::max<int64_t>(0, std::min(v, n - 1));
            } else {  // reflect
                if (n == 1) v = 0;
                else {
                    while (v < 0 || v >= n) {
                        if (v < 0) v = -v;
                        if (v >= n) v = 2 * n - 2 - v;
                    }
                }
            }
            xc[d] = v;
        }
        int64_t yo = 0;
        for (int d = 0; d < r; d++) yo += c[d] * Y.strides[d];
        write_d(Y, yo, read_d(X, off_of(X, xc, xa)));
    });
}

// Expand (phase 2): broadcast copy to the (compile-time computed)
// output shape; off_of already treats size-1 source dims as broadcast.
void k_expand(const RT& X, RT& Y) {
    auto xa = align_of(X, Y.shape.size());
    auto ya = align_of(Y, Y.shape.size());
    odometer(Y.shape, [&](const std::vector<int64_t>& c) {
        write_d(Y, off_of(Y, c, ya), read_d(X, off_of(X, c, xa)));
    });
}


// -------------------------------------------------------------- Expr -----
// Shape dimensions and symbolic int elements are expression trees over
// runtime dim symbols. Everything evaluates numerically at bind time;
// compile time only needs construction, constant folding, and equality.
struct Expr {
    enum K { C, S, ADD, SUB, MUL, DIV, SLEN, MAXK, MINK } k = C;
    int64_t c = 0;
    int sym = -1;
    std::shared_ptr<Expr> a, b;
    int64_t step = 1;  // SLEN: slice length from (a=start, b=end, step, d handled via a/b only)
};
using EP = std::shared_ptr<Expr>;
EP ec(int64_t v) { auto e = std::make_shared<Expr>(); e->k = Expr::C; e->c = v; return e; }
EP esym(int s) { auto e = std::make_shared<Expr>(); e->k = Expr::S; e->sym = s; return e; }
bool econst(const EP& e, int64_t& v) {
    if (!e) { v = 0; return true; }
    if (e->k == Expr::C) { v = e->c; return true; }
    return false;
}
int64_t eeval(const EP& e, const std::vector<int64_t>& sv) {
    switch (e->k) {
        case Expr::C: return e->c;
        case Expr::S: return sv[e->sym];
        case Expr::ADD: return eeval(e->a, sv) + eeval(e->b, sv);
        case Expr::SUB: return eeval(e->a, sv) - eeval(e->b, sv);
        case Expr::MUL: return eeval(e->a, sv) * eeval(e->b, sv);
        case Expr::DIV: {
            int64_t d = eeval(e->b, sv);
            if (d == 0) fail("expr div by zero");
            int64_t num = eeval(e->a, sv);
            if (num == INT64_MIN && d == -1) return INT64_MAX;  // saturate, never trap
            return num / d;
        }
        case Expr::SLEN: return e->c;  // resolved to a constant at bind-precompute; see compiler note
        case Expr::MAXK: return std::max(eeval(e->a, sv), eeval(e->b, sv));
        case Expr::MINK: return std::min(eeval(e->a, sv), eeval(e->b, sv));
    }
    return 0;
}
bool eseq(const EP& x, const EP& y) {
    if (x == y) return true;
    if (!x || !y || x->k != y->k) return false;
    if (x->k == Expr::C) return x->c == y->c;
    if (x->k == Expr::S) return x->sym == y->sym;
    return eseq(x->a, y->a) && eseq(x->b, y->b);
}
EP ebin(Expr::K k, const EP& a, const EP& b) {
    int64_t ca, cb;
    bool ka = econst(a, ca), kb = econst(b, cb);
    if (ka && kb) {
        // Fold through __int128 with saturation: slice-bound sentinel
        // arithmetic (e.g. end = -2^63+1 with a negative step) must not
        // trap on INT64_MIN / -1 or wrap (phase 2).
        switch (k) {
            case Expr::ADD: case Expr::SUB: case Expr::MUL: {
                __int128 r = k == Expr::ADD ? (__int128)ca + cb
                             : k == Expr::SUB ? (__int128)ca - cb
                                              : (__int128)ca * cb;
                if (r > INT64_MAX) r = INT64_MAX;
                if (r < INT64_MIN) r = INT64_MIN;
                return ec((int64_t)r);
            }
            case Expr::DIV:
                if (cb == 0) return ec(0);
                if (ca == INT64_MIN && cb == -1) return ec(INT64_MAX);
                return ec(ca / cb);
            case Expr::MAXK: return ec(std::max(ca, cb));
            case Expr::MINK: return ec(std::min(ca, cb));
            default: break;
        }
    }
    if ((k == Expr::MAXK || k == Expr::MINK) && eseq(a, b)) return a;
    if (k == Expr::ADD && ka && ca == 0) return b;
    if (k == Expr::ADD && kb && cb == 0) return a;
    if (k == Expr::SUB && kb && cb == 0) return a;
    if (k == Expr::MUL && ka && ca == 1) return b;
    if (k == Expr::MUL && kb && cb == 1) return a;
    if (k == Expr::MUL && ((ka && ca == 0) || (kb && cb == 0))) return ec(0);
    if (k == Expr::DIV && kb && cb == 1) return a;
    auto e = std::make_shared<Expr>();
    e->k = k;
    e->a = a;
    e->b = b;
    return e;
}
EP eadd(const EP& a, const EP& b) { return ebin(Expr::ADD, a, b); }
EP esub(const EP& a, const EP& b) { return ebin(Expr::SUB, a, b); }
EP emul(const EP& a, const EP& b) { return ebin(Expr::MUL, a, b); }
EP ediv(const EP& a, const EP& b) { return ebin(Expr::DIV, a, b); }
EP enumel(const std::vector<EP>& shape) {
    EP n = ec(1);
    for (auto& d : shape) n = emul(n, d);
    return n;
}

// ------------------------------------------------------- values/steps --
enum class SK { MATMUL, GEMM, PACKED, CONV, LN, SOFTMAX, REDUCESUM, ARGMAX, TOPK, CUMSUM,
                GATHER, GATHER_EL, SCATTER_EL, CONCAT, CONSTSHAPE, RANDNORM,
                BINOP, UNOP, CAST, WHERE, COPY,
                // Phase 2 (vits): packed conv family, fp32 ConvTranspose,
                // Pad, Expand. PCONV also carries PackedConvTranspose
                // (pc_transpose), PGATHER is PackedGather.
                PCONV, PGATHER, CONVT, PAD, EXPAND };
struct Step {
    SK k = SK::COPY;
    std::vector<int> in, out;
    BinOp bop = BinOp::ADD;
    UnOp uop = UnOp::NEG;
    int64_t axis = 0;
    bool keepdims = true, exclusive = false, reverse = false, largest = true, sorted = true;
    int64_t topk = 0;
    int reduction = 0;
    float alpha = 1, beta = 1, eps = 0;
    int transA = 0, transB = 0;
    std::vector<int64_t> axes;
    std::vector<int64_t> c_strides{1}, c_pads{0}, c_dil{1};
    std::vector<int64_t> c_pads_end{0};  // ConvTranspose: end pads (begin in c_pads)
    int64_t c_group = 1;
    bool flat_copy = false;  // COPY step: logical-order (reshape) copy
    int pm_mode = 0;
    int64_t pm_rows = 0, pm_cols = 0;
    bool pm_zero = false, pm_bias = false;
    std::string pm_base;
    // PackedConv / PackedConvTranspose geometry (attr-carried).
    int64_t pc_k = 1, pc_stride = 1, pc_pad_l = 0, pc_pad_r = 0, pc_dil = 1, pc_out_pad = 0;
    bool pc_transpose = false;
    int64_t ct_out_pad = 0;   // fp32 ConvTranspose output_padding
    int pad_mode = 0;         // PAD: 0 constant, 1 reflect, 2 edge
    bool mean = false;        // REDUCESUM: ReduceMean (scale by 1/count)
    onnxm::Tensor fill;
    int out_dtype = 0;
};

struct Value {
    enum Kind { CONSTV, SYMVAL, RT } kind = RT;
    onnxm::Tensor ct;                    // CONSTV
    int sdt = 7;                         // SYMVAL materialization dtype
    std::vector<int64_t> sshape;         // SYMVAL
    std::vector<EP> selems;              // SYMVAL
    int dtype = 0;                       // RT
    std::vector<EP> rshape;              // RT
    int view_kind = 0;                   // RT: 0 materialized/input, 1 transpose, 2 reshape,
                                         // 3 unsqueeze, 4 squeeze, 5 slice, 6 identity
    int base = -1;
    std::vector<int64_t> vperm, vaxes;
    int meta0 = -1, meta1 = -1, meta2 = -1, meta3 = -1;  // slice: starts/ends/axes/steps value ids; reshape: meta0 = shape value id
    int def_step = -1, last_use = -1;
    bool is_input = false, is_output = false;
};

struct Impl {
    std::string label;
    std::vector<Value> vals;
    std::vector<Step> steps;
    std::map<std::string, int> by_name;
    std::vector<std::string> in_names, out_names;
    std::vector<int> in_vals, out_vals;
    int nsyms = 0;
    std::vector<std::pair<int, int>> sym_origin;  // sym -> (input value id, dim)
    // Per-run state (single-threaded use, like a session).
    std::vector<uint8_t> arena, out_arena;
    std::vector<std::vector<uint8_t>> view_blocks;  // reshape materializations (never moved)
    std::vector<std::vector<uint8_t>> sym_bufs;
    uint64_t rng_ctr = 0;
    PackedLookup lookup;
};

// ------------------------------------------------------- shape helpers -
std::vector<EP> shape_exprs(Impl& im, const Value& v) {
    if (v.kind == Value::CONSTV) {
        std::vector<EP> r;
        for (auto d : v.ct.dims) r.push_back(ec(d));
        return r;
    }
    if (v.kind == Value::SYMVAL) {
        std::vector<EP> r;
        for (auto d : v.sshape) r.push_back(ec(d));
        return r;
    }
    return v.rshape;
}
int64_t numel_of(const std::vector<int64_t>& s) {
    int64_t n = 1;
    for (auto d : s) n *= d;
    return n;
}
// Read a CONST int tensor's elements.
std::vector<int64_t> const_ints(const onnxm::Tensor& t) {
    std::vector<int64_t> r;
    int64_t n = t.numel();
    for (int64_t i = 0; i < n; i++) {
        if (t.data_type == 7) { int64_t v; memcpy(&v, t.data.data() + i * 8, 8); r.push_back(v); }
        else if (t.data_type == 6) { int32_t v; memcpy(&v, t.data.data() + i * 4, 4); r.push_back(v); }
        else fail("const_ints: dtype not int");
    }
    return r;
}
// If a value is CONST or a SYMVAL of all-constant exprs, get its ints.
bool value_const_ints(Impl& im, int vid, std::vector<int64_t>& out) {
    Value& v = im.vals[vid];
    if (v.kind == Value::CONSTV) {
        if (v.ct.data_type != 7 && v.ct.data_type != 6) return false;
        out = const_ints(v.ct);
        return true;
    }
    if (v.kind == Value::SYMVAL) {
        out.clear();
        for (auto& e : v.selems) {
            int64_t c;
            if (!econst(e, c)) return false;
            out.push_back(c);
        }
        return true;
    }
    return false;
}
// Symbolic element list of a value used as int metadata (may be symbolic).
bool value_sym_elems(Impl& im, int vid, std::vector<EP>& out) {
    Value& v = im.vals[vid];
    if (v.kind == Value::SYMVAL) { out = v.selems; return true; }
    if (v.kind == Value::CONSTV) {
        if (v.ct.data_type != 7 && v.ct.data_type != 6) return false;
        out.clear();
        for (auto i : const_ints(v.ct)) out.push_back(ec(i));
        return true;
    }
    return false;
}

std::vector<EP> broadcast_shape(const std::vector<EP>& A, const std::vector<EP>& B) {
    size_t r = std::max(A.size(), B.size());
    std::vector<EP> out(r);
    for (size_t i = 0; i < r; i++) {
        EP a = i < r - A.size() ? ec(1) : A[i - (r - A.size())];
        EP b = i < r - B.size() ? ec(1) : B[i - (r - B.size())];
        int64_t ca, cb;
        bool ka = econst(a, ca), kb = econst(b, cb);
        if (eseq(a, b)) out[i] = a;
        else if (ka && ca == 1) out[i] = b;
        else if (kb && cb == 1) out[i] = a;
        else if (ka && kb) {
            if (ca != cb) fail("broadcast: incompatible const dims");
            out[i] = a;
        } else if (!ka) out[i] = a;
        else out[i] = b;
    }
    return out;
}

int64_t norm_axis(int64_t a, size_t rank) {
    if (a < 0) a += (int64_t)rank;
    return a;
}

// ONNX Slice element count + first selected index for one axis, with
// the spec's clamping (negative bounds count from the end, then clip:
// step > 0 clips both to [0, D]; step < 0 clips start to [0, D-1] and
// end to [-1, D-1]). Shared by the compile-time shape computation, the
// const-data slice path, the symbolic slice, and bind-time view
// resolution so the four can never drift. Sentinel bounds (INT64_MIN
// class) are safe: they stay far inside int64 after += D.
int64_t slice_count(int64_t D, int64_t s, int64_t e, int64_t step, int64_t& first) {
    if (D <= 0 || step == 0) { first = 0; return 0; }
    if (step > 0) {
        if (s < 0) s += D;
        if (e < 0) e += D;
        s = std::max<int64_t>(0, std::min(s, D));
        e = std::max<int64_t>(0, std::min(e, D));
        first = s;
        return e > s ? (e - s + step - 1) / step : 0;
    }
    if (s < 0) s += D;
    if (e < 0) e += D;
    s = std::max<int64_t>(0, std::min(s, D - 1));
    e = std::max<int64_t>(-1, std::min(e, D - 1));
    first = s;
    int64_t m = -step;
    return s > e ? (s - e - 1) / m + 1 : 0;
}

// ------------------------------------------- symbolic value operations -
// Elementwise combine of two SYMVAL/CONST int/float element lists.
EP sym_bin(Expr::K k, const EP& a, const EP& b) { return ebin(k, a, b); }

// Try to compute an elementwise binary op symbolically. Operands are
// value ids known to be CONSTV or SYMVAL. Returns new value id or -1.
int sym_binop(Impl& im, const std::string& op, int va, int vb) {
    if (op == "Pow") {
        // Symbolic Pow only as repeated multiplication by a small
        // non-negative constant exponent (phase 2: vits attention
        // reshapes use Pow(dim, 2) in their target chains).
        std::vector<int64_t> ex;
        if (!value_const_ints(im, vb, ex) || ex.size() != 1 || ex[0] < 0 || ex[0] > 8) return -1;
        std::vector<EP> ea;
        if (!value_sym_elems(im, va, ea)) return -1;
        Value& A = im.vals[va];
        Value v;
        v.kind = Value::SYMVAL;
        v.sshape = A.kind == Value::SYMVAL ? A.sshape : A.ct.dims;
        for (auto& e : ea) {
            EP r = ec(1);
            for (int64_t i = 0; i < ex[0]; i++) r = ebin(Expr::MUL, r, e);
            v.selems.push_back(r);
        }
        im.vals.push_back(std::move(v));
        return (int)im.vals.size() - 1;
    }
    std::vector<EP> ea, eb;
    if (!value_sym_elems(im, va, ea) || !value_sym_elems(im, vb, eb)) return -1;
    std::vector<int64_t> sa, sb;
    {
        Value& A = im.vals[va];
        Value& B = im.vals[vb];
        sa = A.kind == Value::SYMVAL ? A.sshape : A.ct.dims;
        sb = B.kind == Value::SYMVAL ? B.sshape : B.ct.dims;
    }
    // broadcast shapes (concrete small shapes)
    size_t r = std::max(sa.size(), sb.size());
    std::vector<int64_t> so(r);
    for (size_t i = 0; i < r; i++) {
        int64_t da = i < r - sa.size() ? 1 : sa[i - (r - sa.size())];
        int64_t db = i < r - sb.size() ? 1 : sb[i - (r - sb.size())];
        if (da == db) so[i] = da;
        else if (da == 1) so[i] = db;
        else if (db == 1) so[i] = da;
        else return -1;
    }
    auto at = [](const std::vector<EP>& e, const std::vector<int64_t>& s, const std::vector<int64_t>& coord) {
        int64_t off = 0, stride = 1;
        for (int d = (int)s.size() - 1; d >= 0; d--) {
            int64_t c = s[d] == 1 ? 0 : coord[d + (coord.size() - s.size())];
            off += c * stride;
            stride *= s[d];
        }
        return e[off];
    };
    Expr::K k = op == "Add" ? Expr::ADD : op == "Sub" ? Expr::SUB : op == "Mul" ? Expr::MUL
                : op == "Max" ? Expr::MAXK : op == "Min" ? Expr::MINK : Expr::DIV;
    Value v;
    v.kind = Value::SYMVAL;
    v.sshape = so;
    int64_t n = numel_of(so);
    v.selems.resize(n);
    std::vector<int64_t> coord(r, 0);
    for (int64_t i = 0; i < n; i++) {
        v.selems[i] = sym_bin(k, at(ea, sa, coord), at(eb, sb, coord));
        for (int d = (int)r - 1; d >= 0; d--) {
            if (++coord[d] < so[d]) break;
            coord[d] = 0;
        }
    }
    im.vals.push_back(std::move(v));
    return (int)im.vals.size() - 1;
}

// If all elements are constant, collapse a SYMVAL into a CONST int tensor.
int maybe_const_collapse(Impl& im, int vid) {
    Value& v = im.vals[vid];
    if (v.kind != Value::SYMVAL || v.sdt != 7) return vid;
    std::vector<int64_t> ints;
    for (auto& e : v.selems) {
        int64_t c;
        if (!econst(e, c)) return vid;
        ints.push_back(c);
    }
    Value nv;
    nv.kind = Value::CONSTV;
    nv.ct.data_type = 7;
    nv.ct.dims = v.sshape;
    nv.ct.data.resize(ints.size() * 8);
    memcpy(nv.ct.data.data(), ints.data(), ints.size() * 8);
    im.vals.push_back(std::move(nv));
    return (int)im.vals.size() - 1;
}


// ------------------------------------------------------ step executor --
// Shared by the runtime and by compile-time constant folding.
void exec_step(Impl& im, const Step& st, const std::vector<RT>& ins, std::vector<RT>& outs, Scratch& sc);

RT tensor_of_const(const onnxm::Tensor& t) {
    RT r;
    r.dtype = t.data_type;
    r.p = const_cast<uint8_t*>(t.data.data());
    r.shape = t.dims;
    r.strides = rowmajor(t.dims);
    return r;
}

// ---------------------------------------------------------- compiler ---
struct Compiler {
    Impl& im;
    const onnxm::Graph& g;
    std::string label;
    explicit Compiler(Impl& i, const onnxm::Graph& gr, std::string l) : im(i), g(gr), label(std::move(l)) { im.label = label; }

    int vid(const std::string& name) const {
        auto it = im.by_name.find(name);
        if (it == im.by_name.end()) fail("unknown value: " + name + " in " + label);
        return it->second;
    }
    void reg(const std::string& name, int v) {
        if (!name.empty()) im.by_name[name] = v;
    }
    int new_rt(int dtype, std::vector<EP> shape) {
        Value v;
        v.kind = Value::RT;
        v.dtype = dtype;
        v.rshape = std::move(shape);
        im.vals.push_back(std::move(v));
        return (int)im.vals.size() - 1;
    }
    int new_const(onnxm::Tensor t) {
        Value v;
        v.kind = Value::CONSTV;
        v.ct = std::move(t);
        im.vals.push_back(std::move(v));
        return (int)im.vals.size() - 1;
    }
    void add_step(Step st, const std::vector<int>& outs) {
        int idx = (int)im.steps.size();
        for (int o : outs) im.vals[o].def_step = idx;
        st.out = outs;
        im.steps.push_back(std::move(st));
    }
    bool all_const(const std::vector<int>& ins) const {
        for (int v : ins)
            if (v >= 0 && im.vals[v].kind != Value::CONSTV) return false;
        return true;
    }
    bool all_sym_or_const(const std::vector<int>& ins) const {
        for (int v : ins)
            if (v >= 0 && im.vals[v].kind == Value::RT) return false;
        return true;
    }
};

// Execute one node immediately over constant inputs; returns CONST value
// ids. Used for folding. Supports the compute step set plus layout ops
// via ad-hoc copies. Throws on unsupported ops (caller falls back).
std::vector<int> fold_node(Compiler& C, const onnxm::Node& n, const std::vector<int>& ins, Step st,
                           const std::vector<std::pair<int, std::vector<EP>>>& out_specs,
                           const std::vector<int64_t>& symvals_dummy) {
    (void)symvals_dummy;
    std::vector<RT> inT, outT;
    for (int v : ins) inT.push_back(tensor_of_const(C.im.vals[v].ct));
    std::vector<onnxm::Tensor> bufs(out_specs.size());
    for (size_t i = 0; i < out_specs.size(); i++) {
        bufs[i].data_type = out_specs[i].first;
        int64_t nel = 1;
        for (auto& e : out_specs[i].second) {
            int64_t c;
            if (!econst(e, c)) fail("fold: symbolic output shape");
            bufs[i].dims.push_back(c);
            nel *= c;
        }
        bufs[i].data.resize((size_t)nel * onnxm::elem_size(bufs[i].data_type));
        RT r;
        r.dtype = bufs[i].data_type;
        r.p = bufs[i].data.data();
        r.shape = bufs[i].dims;
        r.strides = rowmajor(bufs[i].dims);
        outT.push_back(r);
    }
    Scratch sc;
    exec_step(C.im, st, inT, outT, sc);
    std::vector<int> r;
    for (auto& b : bufs) r.push_back(C.new_const(std::move(b)));
    return r;
}

// Symbolic structural ops on SYMVAL element lists (small int tensors).
int sym_structural(Compiler& C, const onnxm::Node& n, const std::vector<int>& ins) {
    Impl& im = C.im;
    const std::string& op = n.op;
    Value& A = im.vals[ins[0]];
    if (std::getenv("FE_DEBUG") && n.name.find("attn_layers.0") != std::string::npos)
        std::fprintf(stderr, "[sym] %s %s in0kind=%d\n", op.c_str(), n.name.c_str(), (int)A.kind);
    if (op == "Concat") {
        // Concat is handled before the SYMVAL-first-input check below:
        // its inputs may lead with a CONSTV (phase 2: vits pads chains
        // concat unsqueezed constants ahead of the symbolic elements).
        auto mkc = [&](std::vector<int64_t> shape, std::vector<EP> elems, int sdt) {
            Value v;
            v.kind = Value::SYMVAL;
            v.sshape = std::move(shape);
            v.selems = std::move(elems);
            v.sdt = sdt;
            im.vals.push_back(std::move(v));
            return (int)im.vals.size() - 1;
        };
        int64_t axis = n.attr_i("axis", 0);
        std::vector<int64_t> nshape;
        std::vector<EP> elems;
        std::vector<std::vector<EP>> per_in;
        for (int v : ins) {
            Value& B = im.vals[v];
            std::vector<EP> be;
            if (!value_sym_elems(im, v, be)) return -1;
            std::vector<int64_t> bshape = B.kind == Value::SYMVAL ? B.sshape : B.ct.dims;
            if (nshape.empty()) nshape = bshape;
            else {
                if (bshape.size() != nshape.size()) return -1;
                int64_t ax = norm_axis(axis, (int64_t)nshape.size());
                nshape[ax] += bshape[ax];
            }
            per_in.push_back(std::move(be));
        }
        // Only axis-concat of equal-rank small tensors; do it by blocks
        // when axis == 0 (the observed case: shape vectors), else bail.
        int64_t ax = norm_axis(axis, (int64_t)nshape.size());
        if (ax != 0) return -1;
        for (auto& be : per_in) elems.insert(elems.end(), be.begin(), be.end());
        int rsdt = A.kind == Value::SYMVAL ? A.sdt : A.ct.data_type;
        return mkc(nshape, elems, rsdt);
    }
    if (A.kind != Value::SYMVAL) return -1;
    auto mk = [&](std::vector<int64_t> shape, std::vector<EP> elems, int sdt) {
        Value v;
        v.kind = Value::SYMVAL;
        v.sshape = std::move(shape);
        v.selems = std::move(elems);
        v.sdt = sdt;
        im.vals.push_back(std::move(v));
        return (int)im.vals.size() - 1;
    };
    if (op == "Unsqueeze" || op == "Squeeze") {
        std::vector<int64_t> axes;
        if (ins.size() > 1 && ins[1] >= 0) {
            if (!value_const_ints(im, ins[1], axes)) return -1;
        } else if (auto* at = n.attr("axes")) axes = at->ints;
        int64_t rank = (int64_t)A.sshape.size();
        std::vector<int64_t> nshape;
        std::vector<char> drop(rank, 0), add;
        if (op == "Unsqueeze") {
            int64_t nr = rank + (int64_t)axes.size();
            std::vector<char> ins_at(nr, 0);
            for (auto a : axes) ins_at[norm_axis(a, nr)] = 1;
            size_t src = 0;
            for (int64_t i = 0; i < nr; i++) nshape.push_back(ins_at[i] ? 1 : A.sshape[src++]);
            return mk(nshape, A.selems, A.sdt);
        }
        if (axes.empty()) {
            for (int64_t i = 0; i < rank; i++)
                if (A.sshape[i] == 1) drop[i] = 1;
        } else
            for (auto a : axes) drop[norm_axis(a, rank)] = 1;
        for (int64_t i = 0; i < rank; i++)
            if (!drop[i]) nshape.push_back(A.sshape[i]);
        // Elements: squeezing only removes size-1 dims; but our SYMVAL
        // shapes with dropped dims must have those dims == 1 (checked at
        // materialization time in the runtime path; here trust + verify).
        for (int64_t i = 0; i < rank; i++)
            if (drop[i] && A.sshape[i] != 1) return -1;
        return mk(nshape, A.selems, A.sdt);
    }
    if (op == "Reshape") {
        std::vector<int64_t> target;
        if (!value_const_ints(im, ins[1], target)) return -1;
        int64_t total = numel_of(A.sshape), prod = 1;
        int infer = -1;
        for (size_t i = 0; i < target.size(); i++) {
            if (target[i] == -1) infer = (int)i;
            else if (target[i] == 0) target[i] = A.sshape[i];
            else prod *= target[i];
        }
        if (infer >= 0) target[infer] = total / prod;
        return mk(target, A.selems, A.sdt);
    }
    if (op == "Slice") {
        std::vector<int64_t> starts, ends, axes, steps;
        if (!value_const_ints(im, ins[1], starts) || !value_const_ints(im, ins[2], ends)) return -1;
        if (ins.size() > 3 && ins[3] >= 0 && !value_const_ints(im, ins[3], axes)) return -1;
        if (ins.size() > 4 && ins[4] >= 0 && !value_const_ints(im, ins[4], steps)) return -1;
        int64_t rank = (int64_t)A.sshape.size();
        if (axes.empty()) { axes.resize(rank); for (int64_t i = 0; i < rank; i++) axes[i] = i; }
        if (steps.empty()) steps.assign(axes.size(), 1);
        std::vector<int64_t> nshape = A.sshape;
        std::vector<int64_t> off(rank, 0), stp(rank, 1);
        for (size_t i = 0; i < axes.size(); i++) {
            int64_t ax = norm_axis(axes[i], rank);
            int64_t first;
            nshape[ax] = slice_count(A.sshape[ax], starts[i], ends[i], steps[i], first);
            off[ax] = first;
            stp[ax] = steps[i];
        }
        int64_t n = numel_of(nshape);
        std::vector<EP> elems(n);
        std::vector<int64_t> coord(nshape.size(), 0);
        auto astr = rowmajor(A.sshape);
        for (int64_t i = 0; i < n; i++) {
            int64_t so = 0;
            for (size_t d = 0; d < coord.size(); d++) so += (off[d] + coord[d] * stp[d]) * astr[d];
            elems[i] = A.selems[so];
            for (int d = (int)coord.size() - 1; d >= 0; d--) {
                if (++coord[d] < nshape[d]) break;
                coord[d] = 0;
            }
        }
        return mk(nshape, elems, A.sdt);
    }
    if (op == "Gather") {
        std::vector<int64_t> idx;
        if (!value_const_ints(im, ins[1], idx)) return -1;
        int64_t axis = norm_axis(n.attr_i("axis", 0), (int64_t)A.sshape.size());
        // Output shape = A.shape[:axis] + indices.shape + A.shape[axis+1:]
        // (a 0-dim scalar index REMOVES the axis -- the vits shape chains
        // rely on this; the phase-1 code assumed 1-D indices throughout).
        Value& B = im.vals[ins[1]];
        std::vector<int64_t> ishape = B.kind == Value::SYMVAL ? B.sshape : B.ct.dims;
        int64_t outer = 1, inner = 1;
        for (int64_t i = 0; i < axis; i++) outer *= A.sshape[i];
        for (size_t i = (size_t)axis + 1; i < A.sshape.size(); i++) inner *= A.sshape[i];
        std::vector<int64_t> nshape;
        for (int64_t i = 0; i < axis; i++) nshape.push_back(A.sshape[i]);
        for (auto d : ishape) nshape.push_back(d);
        for (size_t i = (size_t)axis + 1; i < A.sshape.size(); i++) nshape.push_back(A.sshape[i]);
        std::vector<EP> elems((size_t)outer * idx.size() * inner);
        for (int64_t o = 0; o < outer; o++)
            for (size_t ii = 0; ii < idx.size(); ii++) {
                int64_t s = idx[ii] < 0 ? idx[ii] + A.sshape[axis] : idx[ii];
                for (int64_t k = 0; k < inner; k++)
                    elems[((o * (int64_t)idx.size()) + (int64_t)ii) * inner + k] =
                        A.selems[((o * A.sshape[axis]) + s) * inner + k];
            }
        return mk(nshape, elems, A.sdt);
    }
    if (op == "Cast") {
        int64_t to = n.attr_i("to", 0);
        return mk(A.sshape, A.selems, (int)to);
    }
    if (op == "Transpose") {
        std::vector<int64_t> perm;
        if (auto* at = n.attr("perm")) perm = at->ints;
        int64_t rank = (int64_t)A.sshape.size();
        if (perm.empty()) { perm.resize(rank); for (int64_t i = 0; i < rank; i++) perm[i] = rank - 1 - i; }
        std::vector<int64_t> nshape(rank);
        for (int64_t i = 0; i < rank; i++) nshape[i] = A.sshape[perm[i]];
        int64_t n = numel_of(nshape);
        std::vector<EP> elems(n);
        std::vector<int64_t> coord(nshape.size(), 0);
        auto astr = rowmajor(A.sshape);
        for (int64_t i = 0; i < n; i++) {
            int64_t so = 0;
            for (int64_t d = 0; d < rank; d++) so += coord[d] * astr[perm[d]];
            elems[i] = A.selems[so];
            for (int d = (int)coord.size() - 1; d >= 0; d--) {
                if (++coord[d] < nshape[d]) break;
                coord[d] = 0;
            }
        }
        return mk(nshape, elems, A.sdt);
    }
    // Concat is handled at the top of this function (its first input
    // may be a CONSTV).
    return -1;
}


// --------------------------------------------------------- lowering ----
int dtype_of(Impl& im, int vid) {
    Value& v = im.vals[vid];
    if (v.kind == Value::CONSTV) return v.ct.data_type;
    if (v.kind == Value::SYMVAL) return v.sdt;
    return v.dtype;
}
int promoted_dtype(int a, int b) {
    if (a == b) return a;
    if (a == 11 || b == 11) return 11;
    if (a == 1 || b == 1) return 1;
    if (a == 7 || b == 7) return 7;
    if (a == 6 || b == 6) return 6;
    return a;
}

std::vector<EP> matmul_shape(const std::vector<EP>& A, const std::vector<EP>& B) {
    std::vector<EP> a = A, b = B;
    bool a1 = a.size() == 1, b1 = b.size() == 1;
    if (a1) a = {ec(1), a[0]};
    if (b1) b = {b[0], ec(1)};
    size_t r = std::max(a.size(), b.size());
    std::vector<EP> out;
    for (size_t i = 0; i + 2 < r; i++) {
        EP da = i < r - a.size() ? ec(1) : a[i - (r - a.size())];
        EP db = i < r - b.size() ? ec(1) : b[i - (r - b.size())];
        out.push_back(eseq(da, db) || [&] { int64_t c; return econst(da, c) && c == 1; }() ? db : da);
    }
    out.push_back(a[a.size() - 2]);
    out.push_back(b[b.size() - 1]);
    if (a1) out.erase(out.begin() + (int)out.size() - 2);
    if (b1) out.pop_back();
    return out;
}

// Generic permute copy for constant folding of Transpose.
onnxm::Tensor permute_copy(const onnxm::Tensor& t, const std::vector<int64_t>& perm) {
    onnxm::Tensor o;
    o.data_type = t.data_type;
    o.name = t.name;
    for (auto p : perm) o.dims.push_back(t.dims[p]);
    o.data.resize(t.data.size());
    size_t es = onnxm::elem_size(t.data_type);
    std::vector<int64_t> coord(o.dims.size(), 0);
    auto istr = rowmajor(t.dims);
    int64_t n = numel_of(o.dims);
    for (int64_t i = 0; i < n; i++) {
        int64_t so = 0;
        for (size_t d = 0; d < coord.size(); d++) so += coord[d] * istr[perm[d]];
        int64_t dof = 0;
        auto ostr = rowmajor(o.dims);
        for (size_t d = 0; d < coord.size(); d++) dof += coord[d] * ostr[d];
        memcpy(o.data.data() + dof * (int64_t)es, t.data.data() + so * (int64_t)es, es);
        for (int d = (int)coord.size() - 1; d >= 0; d--) {
            if (++coord[d] < o.dims[d]) break;
            coord[d] = 0;
        }
    }
    return o;
}

void lower_node(Compiler& C, const onnxm::Node& n) {
    Impl& im = C.im;
    const std::string& op = n.op;
    if (std::getenv("FE_DEBUG")) std::fprintf(stderr, "[lower] %s %s\n", op.c_str(), n.name.c_str());
    std::vector<int> ins;
    if (op == "PackedMatMul") {
        // Packed slots (__pk/__sc/__ze) are not plan values; only the
        // data input and the optional bias resolve through by_name.
        bool hz = n.attr_i("has_zero", 0) != 0;
        bool hb = n.attr_i("has_bias", 0) != 0;
        ins.push_back(C.vid(n.inputs[0]));
        ins.push_back(-2);
        ins.push_back(-2);
        if (hz) ins.push_back(-2);
        if (hb) ins.push_back(C.vid(n.inputs[3 + (hz ? 1 : 0)]));
    } else if (op == "PackedConv" || op == "PackedConvTranspose") {
        // Fixed 5 slots [X, base__pk, base__sc, base__ze-or-"", bias-or-""];
        // only X and the bias (slot 4) are plan values.
        bool hb = n.attr_i("has_bias", 0) != 0;
        ins.push_back(C.vid(n.inputs[0]));
        ins.push_back(-2);
        ins.push_back(-2);
        ins.push_back(-2);
        ins.push_back(hb ? C.vid(n.inputs[4]) : -1);
    } else if (op == "PackedGather") {
        // Slots [base__pk, base__sc, base__ze-or-"", indices]; only the
        // indices (slot 3) are a plan value.
        ins.push_back(C.vid(n.inputs[3]));
    } else {
        for (auto& name : n.inputs) ins.push_back(name.empty() ? -1 : C.vid(name));
    }
    auto reg_out = [&](int idx, int v) { C.reg(n.outputs[idx], v); };
    auto fail_op = [&](const std::string& m) { fail("op " + op + " (" + n.name + "): " + m); };

    // ---- Constant / Shape / Identity ----
    if (op == "Constant") {
        auto* t = n.attr("value");
        if (!t || !t->t) fail_op("no value attr");
        reg_out(0, C.new_const(*t->t));
        return;
    }
    if (op == "Shape") {
        Value v;
        v.kind = Value::SYMVAL;
        v.sdt = 7;
        auto sh = shape_exprs(im, im.vals[ins[0]]);
        v.sshape = {(int64_t)sh.size()};
        v.selems = sh;
        im.vals.push_back(std::move(v));
        reg_out(0, maybe_const_collapse(im, (int)im.vals.size() - 1));
        return;
    }
    if (op == "Identity") {
        C.reg(n.outputs[0], ins[0]);
        return;
    }

    // ---- PackedMatMul (potato domain) ----
    if (op == "PackedMatMul") {
        Step st;
        st.k = SK::PACKED;
        st.pm_mode = (int)n.attr_i("mode", 0);
        st.pm_rows = n.attr_i("rows", 0);
        st.pm_cols = n.attr_i("cols", 0);
        st.pm_zero = n.attr_i("has_zero", 0) != 0;
        st.pm_bias = n.attr_i("has_bias", 0) != 0;
        // Input names: [x, base__pk, base__sc, (base__ze), (bias)].
        const std::string& pkn = n.inputs[1];
        size_t suf = pkn.rfind("__pk");
        st.pm_base = suf == std::string::npos ? pkn : pkn.substr(0, suf);
        st.in.push_back(ins[0]);
        size_t bi = 3 + (st.pm_zero ? 1 : 0);
        if (st.pm_bias) st.in.push_back(ins[bi]);
        auto xs = shape_exprs(im, im.vals[ins[0]]);
        xs.back() = ec((st.pm_mode == 0 || st.pm_mode == 2) ? st.pm_cols : st.pm_rows);
        int o = C.new_rt(1, xs);
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }

    // ---- PackedConv / PackedConvTranspose / PackedGather (potato) ----
    // Glue only: the step resolves a PackedBinding through the same
    // lookup as PackedMatMul and calls the vits_kernels.cpp kernels --
    // the exact functions the ORT custom ops call (see vc_compute_impl
    // in vits_packed.h). Output-shape formulas mirror that glue.
    if (op == "PackedConv" || op == "PackedConvTranspose") {
        Step st;
        st.k = SK::PCONV;
        st.pc_transpose = op == "PackedConvTranspose";
        st.pm_mode = (int)n.attr_i("mode", 0);
        st.pm_rows = n.attr_i("w_rows", 0);
        st.pm_cols = n.attr_i("w_cols", 0);
        st.pm_zero = n.attr_i("has_zero", 0) != 0;
        st.pm_bias = n.attr_i("has_bias", 0) != 0;
        st.pc_k = n.attr_i("k", 1);
        st.pc_stride = n.attr_i("stride", 1);
        st.pc_pad_l = n.attr_i("pad_l", 0);
        st.pc_pad_r = n.attr_i("pad_r", 0);
        st.pc_dil = n.attr_i("dilation", 1);
        st.pc_out_pad = n.attr_i("out_pad", 0);
        const std::string& pkn = n.inputs[1];
        size_t suf = pkn.rfind("__pk");
        st.pm_base = suf == std::string::npos ? pkn : pkn.substr(0, suf);
        st.in.push_back(ins[0]);
        if (st.pm_bias) st.in.push_back(ins[4]);
        auto xs = shape_exprs(im, im.vals[ins[0]]);
        if (xs.size() != 3) fail_op("PackedConv: X must be (N, C, L)");
        EP och, olen;
        if (!st.pc_transpose) {
            och = ec(st.pm_rows);
            EP num = eadd(xs[2], ec(st.pc_pad_l + st.pc_pad_r - 1));
            num = esub(num, emul(ec(st.pc_dil), ec(st.pc_k - 1)));
            olen = eadd(ediv(num, ec(st.pc_stride)), ec(1));
        } else {
            och = ec(st.pm_cols / st.pc_k);
            EP t = emul(esub(xs[2], ec(1)), ec(st.pc_stride));
            t = esub(t, ec(st.pc_pad_l + st.pc_pad_r));
            t = eadd(t, emul(ec(st.pc_dil), ec(st.pc_k - 1)));
            olen = eadd(t, ec(st.pc_out_pad + 1));
        }
        int o = C.new_rt(1, {xs[0], och, olen});
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }
    if (op == "PackedGather") {
        Step st;
        st.k = SK::PGATHER;
        st.pm_mode = (int)n.attr_i("mode", 0);
        st.pm_rows = n.attr_i("w_rows", 0);
        st.pm_cols = n.attr_i("w_cols", 0);
        st.pm_zero = n.attr_i("has_zero", 0) != 0;
        const std::string& pkn = n.inputs[0];
        size_t suf = pkn.rfind("__pk");
        st.pm_base = suf == std::string::npos ? pkn : pkn.substr(0, suf);
        st.in.push_back(ins[0]);  // indices
        auto os = shape_exprs(im, im.vals[ins[0]]);
        os.push_back(ec(st.pm_cols));
        int o = C.new_rt(1, os);
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }

    // ---- elementwise binary ----
    static const std::map<std::string, BinOp> binops = {
        {"Add", BinOp::ADD}, {"Sub", BinOp::SUB}, {"Mul", BinOp::MUL}, {"Div", BinOp::DIV},
        {"Pow", BinOp::POW}, {"Max", BinOp::MAXIMUM}, {"Less", BinOp::LESS},
        {"Greater", BinOp::GREATER}, {"Equal", BinOp::EQUAL}, {"Or", BinOp::OR}, {"And", BinOp::AND}};
    if (binops.count(op)) {
        BinOp bop = binops.at(op);
        bool cmp = op == "Less" || op == "Greater" || op == "Equal" || op == "Or" || op == "And";
        if (!cmp && (bop == BinOp::ADD || bop == BinOp::SUB || bop == BinOp::MUL || bop == BinOp::DIV ||
                     bop == BinOp::MAXIMUM || bop == BinOp::POW) &&
            C.all_sym_or_const(ins)) {
            int r = sym_binop(im, op, ins[0], ins[1]);
            if (r >= 0) {
                reg_out(0, maybe_const_collapse(im, r));
                return;
            }
        }
        auto os = broadcast_shape(shape_exprs(im, im.vals[ins[0]]), shape_exprs(im, im.vals[ins[1]]));
        int odt = cmp ? 9 : promoted_dtype(dtype_of(im, ins[0]), dtype_of(im, ins[1]));
        Step st;
        st.k = SK::BINOP;
        st.bop = bop;
        st.in = ins;
        if (C.all_const(ins)) {
            auto outs = fold_node(C, n, ins, st, {{odt, os}}, {});
            reg_out(0, outs[0]);
            return;
        }
        int o = C.new_rt(odt, os);
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }

    // ---- elementwise unary ----
    static const std::map<std::string, UnOp> unops = {
        {"Neg", UnOp::NEG}, {"Sqrt", UnOp::SQRT}, {"Sin", UnOp::SIN}, {"Cos", UnOp::COS},
        {"Exp", UnOp::EXP}, {"Tanh", UnOp::TANH}, {"Sigmoid", UnOp::SIGMOID}, {"Relu", UnOp::RELU},
        {"Abs", UnOp::ABS}, {"LeakyRelu", UnOp::LEAKY}};
    if (unops.count(op)) {
        if (op == "Neg" && im.vals[ins[0]].kind == Value::SYMVAL) {
            Value v = im.vals[ins[0]];
            for (auto& e : v.selems) e = esub(ec(0), e);
            im.vals.push_back(std::move(v));
            reg_out(0, maybe_const_collapse(im, (int)im.vals.size() - 1));
            return;
        }
        Step st;
        st.k = SK::UNOP;
        st.uop = unops.at(op);
        if (op == "LeakyRelu") st.alpha = n.attr_f("alpha", 0.01f);
        st.in = ins;
        auto os = shape_exprs(im, im.vals[ins[0]]);
        int odt = dtype_of(im, ins[0]);
        if (C.all_const(ins)) {
            auto outs = fold_node(C, n, ins, st, {{odt, os}}, {});
            reg_out(0, outs[0]);
            return;
        }
        int o = C.new_rt(odt, os);
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }
    if (op == "Not") {
        // Not(x) == Equal(x, false)
        onnxm::Tensor f;
        f.data_type = 9;
        f.dims = {1};
        f.data = {0};
        int fc = C.new_const(std::move(f));
        Step st;
        st.k = SK::BINOP;
        st.bop = BinOp::EQUAL;
        st.in = {ins[0], fc};
        auto os = shape_exprs(im, im.vals[ins[0]]);
        if (C.all_const(ins)) {
            auto outs = fold_node(C, n, st.in, st, {{9, os}}, {});
            reg_out(0, outs[0]);
            return;
        }
        int o = C.new_rt(9, os);
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }

    // ---- Cast ----
    if (op == "Cast") {
        int64_t to = n.attr_i("to", 0);
        if (im.vals[ins[0]].kind == Value::SYMVAL) {
            int r = sym_structural(C, n, ins);
            if (r >= 0) {
                reg_out(0, r);
                return;
            }
        }
        Step st;
        st.k = SK::CAST;
        st.in = ins;
        auto os = shape_exprs(im, im.vals[ins[0]]);
        if (C.all_const(ins)) {
            auto outs = fold_node(C, n, ins, st, {{(int)to, os}}, {});
            reg_out(0, outs[0]);
            return;
        }
        int o = C.new_rt((int)to, os);
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }

    // ---- MatMul / Gemm ----
    if (op == "MatMul" || op == "Gemm") {
        Step st;
        st.in = ins;
        std::vector<EP> os;
        int odt = dtype_of(im, ins[0]);
        if (op == "MatMul") {
            st.k = SK::MATMUL;
            st.in = {ins[0], ins[1]};
            os = matmul_shape(shape_exprs(im, im.vals[ins[0]]), shape_exprs(im, im.vals[ins[1]]));
        } else {
            st.k = SK::GEMM;
            st.alpha = n.attr_f("alpha", 1.0f);
            st.beta = n.attr_f("beta", 1.0f);
            st.transA = (int)n.attr_i("transA", 0);
            st.transB = (int)n.attr_i("transB", 0);
            auto as = shape_exprs(im, im.vals[ins[0]]);
            auto bs = shape_exprs(im, im.vals[ins[1]]);
            if (as.size() != 2 || bs.size() != 2) fail_op("Gemm rank != 2");
            os = {st.transA ? as[1] : as[0], st.transB ? bs[0] : bs[1]};
            st.in = {ins[0], ins[1]};
            if (ins.size() > 2 && ins[2] >= 0) st.in.push_back(ins[2]);
        }
        if (C.all_const(st.in)) {
            auto outs = fold_node(C, n, st.in, st, {{odt, os}}, {});
            reg_out(0, outs[0]);
            return;
        }
        int o = C.new_rt(odt, os);
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }

    // ---- Conv ----
    if (op == "Conv") {
        Step st;
        st.k = SK::CONV;
        st.in = ins;
        auto* at = n.attr("strides");
        if (at) st.c_strides = at->ints;
        at = n.attr("dilations");
        if (at) st.c_dil = at->ints;
        at = n.attr("pads");
        if (at) {
            // ONNX pads: [begin..., end...]; kernel takes begin per dim.
            size_t sp = at->ints.size() / 2;
            st.c_pads.assign(at->ints.begin(), at->ints.begin() + sp);
        }
        st.c_group = n.attr_i("group", 1);
        auto xs = shape_exprs(im, im.vals[ins[0]]);
        auto ws = shape_exprs(im, im.vals[ins[1]]);
        int sp = (int)xs.size() - 2;
        if ((int)st.c_strides.size() != sp) st.c_strides.assign(sp, 1);
        if ((int)st.c_dil.size() != sp) st.c_dil.assign(sp, 1);
        if ((int)st.c_pads.size() != sp) st.c_pads.assign(sp, 0);
        std::vector<EP> os = {xs[0], ws[0]};
        for (int i = 0; i < sp; i++) {
            // out = floor((in + 2*pad - dil*(k-1) - 1) / stride) + 1
            EP num = eadd(xs[2 + i], ec(2 * st.c_pads[i] - 1));
            num = esub(num, emul(ec(st.c_dil[i]), esub(ws[2 + i], ec(1))));
            os.push_back(eadd(ediv(num, ec(st.c_strides[i])), ec(1)));
        }
        int odt = dtype_of(im, ins[0]);
        if (C.all_const(st.in)) {
            auto outs = fold_node(C, n, st.in, st, {{odt, os}}, {});
            reg_out(0, outs[0]);
            return;
        }
        int o = C.new_rt(odt, os);
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }

    // ---- ConvTranspose (fp32; phase 2 for the baked vits graph) ----
    if (op == "ConvTranspose") {
        Step st;
        st.k = SK::CONVT;
        st.in = ins;
        st.c_group = n.attr_i("group", 1);
        if (st.c_group != 1) fail_op("ConvTranspose: group != 1 not implemented");
        auto* at = n.attr("strides");
        if (at) st.c_strides = at->ints;
        at = n.attr("dilations");
        if (at) st.c_dil = at->ints;
        at = n.attr("pads");
        if (at) {
            size_t spn = at->ints.size() / 2;
            st.c_pads.assign(at->ints.begin(), at->ints.begin() + spn);
            st.c_pads_end.assign(at->ints.begin() + spn, at->ints.end());
        }
        at = n.attr("output_padding");
        if (at && !at->ints.empty()) st.ct_out_pad = at->ints[0];
        auto xs = shape_exprs(im, im.vals[ins[0]]);
        auto ws = shape_exprs(im, im.vals[ins[1]]);
        int sp = (int)xs.size() - 2;
        if (sp < 1) fail_op("ConvTranspose: rank < 3");
        if ((int)st.c_strides.size() != sp) st.c_strides.assign(sp, 1);
        if ((int)st.c_dil.size() != sp) st.c_dil.assign(sp, 1);
        if ((int)st.c_pads.size() != sp) st.c_pads.assign(sp, 0);
        if ((int)st.c_pads_end.size() != sp) st.c_pads_end.assign(sp, 0);
        // out = (in-1)*stride - pad_b - pad_e + dil*(k-1) + out_pad + 1;
        // weight shape is [Ci, O, K...], so out channels = ws[1].
        std::vector<EP> os = {xs[0], ws[1]};
        for (int i = 0; i < sp; i++) {
            EP t = emul(esub(xs[2 + i], ec(1)), ec(st.c_strides[i]));
            t = eadd(t, emul(ec(st.c_dil[i]), esub(ws[2 + i], ec(1))));
            t = eadd(t, ec(st.ct_out_pad + 1 - st.c_pads[i] - st.c_pads_end[i]));
            os.push_back(t);
        }
        int odt = dtype_of(im, ins[0]);
        if (C.all_const(st.in)) {
            auto outs = fold_node(C, n, st.in, st, {{odt, os}}, {});
            reg_out(0, outs[0]);
            return;
        }
        int o = C.new_rt(odt, os);
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }

    // ---- LayerNormalization / Softmax ----
    if (op == "LayerNormalization") {
        Step st;
        st.k = SK::LN;
        st.in = ins;
        st.axis = n.attr_i("axis", -1);
        st.eps = n.attr_f("epsilon", 1e-5f);
        auto os = shape_exprs(im, im.vals[ins[0]]);
        int odt = dtype_of(im, ins[0]);
        if (C.all_const(st.in)) {
            auto outs = fold_node(C, n, st.in, st, {{odt, os}}, {});
            reg_out(0, outs[0]);
            return;
        }
        int o = C.new_rt(odt, os);
        C.add_step(st, {o});
        reg_out(0, o);
        for (size_t i = 1; i < n.outputs.size(); i++) {
            // Mean / InvStd outputs: not consumed in these graphs; give
            // them a degenerate value so name lookup succeeds.
            std::vector<EP> ds = os;
            int64_t ax = norm_axis(st.axis, ds.size());
            for (size_t d = (size_t)ax; d < ds.size(); d++) ds[d] = ec(1);
            reg_out((int)i, C.new_rt(odt, ds));
            im.vals.back().def_step = -2;  // mark: no producer (never materialized unless used)
        }
        return;
    }
    if (op == "Softmax") {
        Step st;
        st.k = SK::SOFTMAX;
        st.in = ins;
        st.axis = n.attr_i("axis", -1);
        auto os = shape_exprs(im, im.vals[ins[0]]);
        int odt = dtype_of(im, ins[0]);
        if (C.all_const(ins)) {
            auto outs = fold_node(C, n, ins, st, {{odt, os}}, {});
            reg_out(0, outs[0]);
            return;
        }
        int o = C.new_rt(odt, os);
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }

    // ---- ReduceSum / ReduceMean / ArgMax / TopK / CumSum ----
    // ReduceMean (phase 2) shares the ReduceSum path with a mean flag;
    // its noop_with_empty_axes form is an identity copy.
    if (op == "ReduceMean" && n.attr_i("noop_with_empty_axes", 0) != 0) {
        std::vector<int64_t> axes;
        bool have_axes = false;
        if (ins.size() > 1 && ins[1] >= 0) have_axes = value_const_ints(im, ins[1], axes);
        else if (auto* at = n.attr("axes")) { axes = at->ints; have_axes = true; }
        if (have_axes && axes.empty()) {
            Step st;
            st.k = SK::COPY;
            st.in = {ins[0]};
            auto os = shape_exprs(im, im.vals[ins[0]]);
            int odt = dtype_of(im, ins[0]);
            if (C.all_const(st.in)) {
                auto outs = fold_node(C, n, st.in, st, {{odt, os}}, {});
                reg_out(0, outs[0]);
                return;
            }
            int o = C.new_rt(odt, os);
            C.add_step(st, {o});
            reg_out(0, o);
            return;
        }
    }
    if (op == "ReduceSum" || op == "ReduceMean") {
        Step st;
        st.k = SK::REDUCESUM;
        st.mean = op == "ReduceMean";
        st.in = {ins[0]};
        st.keepdims = n.attr_i("keepdims", 1) != 0;
        std::vector<int64_t> axes;
        bool have_axes = false;
        if (ins.size() > 1 && ins[1] >= 0) {
            have_axes = value_const_ints(im, ins[1], axes);
            if (!have_axes) fail_op("Reduce axes not const");
        } else if (auto* at = n.attr("axes")) {
            axes = at->ints;
            have_axes = true;
        }
        auto xs = shape_exprs(im, im.vals[ins[0]]);
        if (!have_axes || axes.empty()) {
            axes.clear();
            for (size_t i = 0; i < xs.size(); i++) axes.push_back((int64_t)i);
        }
        for (auto& a : axes) a = norm_axis(a, xs.size());
        st.axes = axes;
        std::vector<EP> os;
        std::vector<char> red(xs.size(), 0);
        for (auto a : axes) red[a] = 1;
        for (size_t i = 0; i < xs.size(); i++) {
            if (red[i]) {
                if (st.keepdims) os.push_back(ec(1));
            } else
                os.push_back(xs[i]);
        }
        int odt = dtype_of(im, ins[0]);
        if (C.all_const(st.in)) {
            auto outs = fold_node(C, n, st.in, st, {{odt, os}}, {});
            reg_out(0, outs[0]);
            return;
        }
        int o = C.new_rt(odt, os);
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }
    if (op == "ArgMax") {
        Step st;
        st.k = SK::ARGMAX;
        st.in = {ins[0]};
        st.axis = n.attr_i("axis", 0);
        st.keepdims = n.attr_i("keepdims", 1) != 0;
        st.exclusive = n.attr_i("select_last_index", 0) != 0;  // field reuse
        auto xs = shape_exprs(im, im.vals[ins[0]]);
        int64_t ax = norm_axis(st.axis, xs.size());
        std::vector<EP> os;
        for (size_t i = 0; i < xs.size(); i++) {
            if ((int64_t)i == ax) {
                if (st.keepdims) os.push_back(ec(1));
            } else
                os.push_back(xs[i]);
        }
        if (C.all_const(st.in)) {
            auto outs = fold_node(C, n, st.in, st, {{7, os}}, {});
            reg_out(0, outs[0]);
            return;
        }
        int o = C.new_rt(7, os);
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }
    if (op == "TopK") {
        std::vector<int64_t> kv;
        if (!value_const_ints(im, ins[1], kv) || kv.empty()) fail_op("TopK k not const");
        Step st;
        st.k = SK::TOPK;
        st.in = {ins[0]};
        st.topk = kv[0];
        st.axis = n.attr_i("axis", -1);
        st.largest = n.attr_i("largest", 1) != 0;
        st.sorted = n.attr_i("sorted", 1) != 0;
        auto xs = shape_exprs(im, im.vals[ins[0]]);
        int64_t ax = norm_axis(st.axis, xs.size());
        std::vector<EP> os = xs;
        os[ax] = ec(st.topk);
        int odt = dtype_of(im, ins[0]);
        int o1 = C.new_rt(odt, os);
        int o2 = C.new_rt(7, os);
        C.add_step(st, {o1, o2});
        reg_out(0, o1);
        reg_out(1, o2);
        return;
    }
    if (op == "CumSum") {
        std::vector<int64_t> ax;
        if (!value_const_ints(im, ins[1], ax) || ax.empty()) fail_op("CumSum axis not const");
        Step st;
        st.k = SK::CUMSUM;
        st.in = {ins[0]};
        st.axis = ax[0];
        st.exclusive = n.attr_i("exclusive", 0) != 0;
        st.reverse = n.attr_i("reverse", 0) != 0;
        auto os = shape_exprs(im, im.vals[ins[0]]);
        int odt = dtype_of(im, ins[0]);
        if (C.all_const(st.in)) {
            auto outs = fold_node(C, n, st.in, st, {{odt, os}}, {});
            reg_out(0, outs[0]);
            return;
        }
        int o = C.new_rt(odt, os);
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }

    // ---- Gather family ----
    if (op == "Gather") {
        if (im.vals[ins[0]].kind == Value::SYMVAL) {
            int r = sym_structural(C, n, ins);
            if (r >= 0) {
                reg_out(0, r);
                return;
            }
        }
        Step st;
        st.k = SK::GATHER;
        st.in = ins;
        st.axis = n.attr_i("axis", 0);
        auto ds = shape_exprs(im, im.vals[ins[0]]);
        auto is = shape_exprs(im, im.vals[ins[1]]);
        int64_t ax = norm_axis(st.axis, ds.size());
        std::vector<EP> os;
        for (int64_t i = 0; i < ax; i++) os.push_back(ds[i]);
        for (auto& e : is) os.push_back(e);
        for (size_t i = (size_t)ax + 1; i < ds.size(); i++) os.push_back(ds[i]);
        int odt = dtype_of(im, ins[0]);
        if (C.all_const(st.in)) {
            auto outs = fold_node(C, n, st.in, st, {{odt, os}}, {});
            reg_out(0, outs[0]);
            return;
        }
        int o = C.new_rt(odt, os);
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }
    if (op == "GatherElements") {
        Step st;
        st.k = SK::GATHER_EL;
        st.in = ins;
        st.axis = n.attr_i("axis", 0);
        auto os = shape_exprs(im, im.vals[ins[1]]);
        int odt = dtype_of(im, ins[0]);
        if (C.all_const(st.in)) {
            auto outs = fold_node(C, n, st.in, st, {{odt, os}}, {});
            reg_out(0, outs[0]);
            return;
        }
        int o = C.new_rt(odt, os);
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }
    if (op == "ScatterElements") {
        Step st;
        st.k = SK::SCATTER_EL;
        st.in = ins;
        st.axis = n.attr_i("axis", 0);
        std::string red = n.attr("reduction") ? n.attr("reduction")->s : "none";
        st.reduction = red == "add" ? 1 : red == "mul" ? 2 : 0;
        auto os = shape_exprs(im, im.vals[ins[0]]);
        int odt = dtype_of(im, ins[0]);
        if (C.all_const(st.in)) {
            auto outs = fold_node(C, n, st.in, st, {{odt, os}}, {});
            reg_out(0, outs[0]);
            return;
        }
        int o = C.new_rt(odt, os);
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }

    // ---- Concat ----
    if (op == "Concat") {
        if (C.all_sym_or_const(ins)) {
            int r = sym_structural(C, n, ins);
            if (std::getenv("FE_DEBUG") && n.name.find("attn_layers.0") != std::string::npos)
                std::fprintf(stderr, "[concat] %s sym=%d kinds=%d,%d,%d,%d\n", n.name.c_str(), r,
                             (int)im.vals[ins[0]].kind, (int)im.vals[ins[1]].kind,
                             ins.size() > 2 ? (int)im.vals[ins[2]].kind : -9,
                             ins.size() > 3 ? (int)im.vals[ins[3]].kind : -9);
            if (r >= 0) {
                reg_out(0, r);
                return;
            }
        }
        Step st;
        st.k = SK::CONCAT;
        st.in = ins;
        st.axis = n.attr_i("axis", 0);
        auto os = shape_exprs(im, im.vals[ins[0]]);
        int64_t ax = norm_axis(st.axis, os.size());
        EP sum = ec(0);
        for (int v : ins) sum = eadd(sum, shape_exprs(im, im.vals[v])[ax]);
        os[ax] = sum;
        int odt = dtype_of(im, ins[0]);
        if (C.all_const(st.in)) {
            auto outs = fold_node(C, n, st.in, st, {{odt, os}}, {});
            reg_out(0, outs[0]);
            return;
        }
        int o = C.new_rt(odt, os);
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }

    // ---- ConstantOfShape / RandomNormalLike ----
    if (op == "ConstantOfShape") {
        std::vector<EP> elems;
        if (!value_sym_elems(im, ins[0], elems)) fail_op("shape input not symbolic/const");
        Step st;
        st.k = SK::CONSTSHAPE;
        auto* at = n.attr("value");
        if (at && at->t) st.fill = *at->t;
        else {
            st.fill.data_type = 1;
            st.fill.dims = {1};
            st.fill.data.assign(4, 0);
        }
        // Constant spec -> constant tensor: fold it (phase 2; the vits
        // attention pads chains route through a ConstantOfShape whose
        // spec is a fixed [0], and the RT step would poison the
        // symbolic Concat that follows). Symbolic specs keep the step.
        {
            std::vector<int64_t> dims;
            bool allc = true;
            for (auto& e : elems) {
                int64_t c;
                if (!econst(e, c) || c < 0) { allc = false; break; }
                dims.push_back(c);
            }
            int64_t nel = 1;
            for (auto d : dims) nel *= d;
            if (allc && nel <= (1 << 20)) {
                onnxm::Tensor t;
                t.data_type = st.fill.data_type;
                t.dims = dims;
                size_t es = onnxm::elem_size(t.data_type);
                t.data.resize((size_t)nel * es);
                size_t fbytes = st.fill.data.size();
                if (fbytes != es) fail_op("ConstantOfShape fill size mismatch");
                for (int64_t i = 0; i < nel; i++) memcpy(t.data.data() + (size_t)i * es, st.fill.data.data(), es);
                reg_out(0, C.new_const(std::move(t)));
                return;
            }
        }
        int odt = st.fill.data_type;
        int o = C.new_rt(odt, elems);
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }
    if (op == "RandomNormalLike") {
        Step st;
        st.k = SK::RANDNORM;
        st.out_dtype = (int)n.attr_i("dtype", dtype_of(im, ins[0]));
        auto os = shape_exprs(im, im.vals[ins[0]]);
        int o = C.new_rt(st.out_dtype, os);
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }

    // ---- Where ----
    if (op == "Where") {
        auto os = broadcast_shape(shape_exprs(im, im.vals[ins[1]]), shape_exprs(im, im.vals[ins[2]]));
        os = broadcast_shape(shape_exprs(im, im.vals[ins[0]]), os);
        Step st;
        st.k = SK::WHERE;
        st.in = ins;
        int odt = dtype_of(im, ins[1]);
        if (C.all_const(ins)) {
            auto outs = fold_node(C, n, ins, st, {{odt, os}}, {});
            reg_out(0, outs[0]);
            return;
        }
        int o = C.new_rt(odt, os);
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }

    // ---- layout ops (views) ----
    if (op == "Transpose") {
        std::vector<int64_t> perm;
        if (auto* at = n.attr("perm")) perm = at->ints;
        if (im.vals[ins[0]].kind == Value::SYMVAL) {
            int r = sym_structural(C, n, ins);
            if (r >= 0) { reg_out(0, r); return; }
        }
        if (im.vals[ins[0]].kind == Value::CONSTV) {
            auto is = shape_exprs(im, im.vals[ins[0]]);
            if (perm.empty()) { perm.resize(is.size()); for (size_t i = 0; i < is.size(); i++) perm[i] = (int64_t)is.size() - 1 - i; }
            reg_out(0, C.new_const(permute_copy(im.vals[ins[0]].ct, perm)));
            return;
        }
        auto is = shape_exprs(im, im.vals[ins[0]]);
        if (perm.empty()) { perm.resize(is.size()); for (size_t i = 0; i < is.size(); i++) perm[i] = (int64_t)is.size() - 1 - i; }
        std::vector<EP> os;
        for (auto p : perm) os.push_back(is[p]);
        Value v;
        v.kind = Value::RT;
        v.dtype = dtype_of(im, ins[0]);
        v.rshape = os;
        v.view_kind = 1;
        v.base = ins[0];
        v.vperm = perm;
        im.vals.push_back(std::move(v));
        reg_out(0, (int)im.vals.size() - 1);
        return;
    }
    if (op == "Reshape") {
        if (im.vals[ins[0]].kind == Value::SYMVAL) {
            int r = sym_structural(C, n, ins);
            if (r >= 0) { reg_out(0, r); return; }
        }
        std::vector<EP> target;
        if (!value_sym_elems(im, ins[1], target)) fail_op("Reshape target not symbolic/const");
        auto is = shape_exprs(im, im.vals[ins[0]]);
        std::vector<EP> os;
        int infer = -1;
        for (size_t i = 0; i < target.size(); i++) {
            int64_t c;
            if (econst(target[i], c) && c == -1) {
                infer = (int)i;
                os.push_back(ec(-1));  // placeholder, replaced below
            } else if (econst(target[i], c) && c == 0) {
                os.push_back(is[i]);
            } else
                os.push_back(target[i]);
        }
        if (infer >= 0) {
            EP prod = ec(1);
            for (size_t i = 0; i < os.size(); i++)
                if ((int)i != infer) prod = emul(prod, os[i]);
            os[infer] = ediv(enumel(is), prod);
        }
        if (im.vals[ins[0]].kind == Value::CONSTV) {
            std::vector<int64_t> dims;
            for (auto& e : os) {
                int64_t c;
                if (!econst(e, c)) fail_op("Reshape of const with symbolic shape");
                dims.push_back(c);
            }
            onnxm::Tensor t = im.vals[ins[0]].ct;
            t.dims = dims;
            reg_out(0, C.new_const(std::move(t)));
            return;
        }
        // A reshape is a pure view only when its base is a materialized
        // (row-major contiguous) tensor. On a view or an input the data
        // would have to be copied, and copying cannot happen at bind
        // time (the producer step has not run yet) -- emit a COPY step
        // with reshape (logical-order) semantics instead.
        Value& bv = im.vals[ins[0]];
        bool view_ok = bv.kind == Value::RT && bv.view_kind == 0 && !bv.is_input;
        if (!view_ok) {
            Step st;
            st.k = SK::COPY;
            st.flat_copy = true;
            st.in = {ins[0]};
            int o = C.new_rt(dtype_of(im, ins[0]), os);
            C.add_step(st, {o});
            reg_out(0, o);
            return;
        }
        Value v;
        v.kind = Value::RT;
        v.dtype = dtype_of(im, ins[0]);
        v.rshape = os;
        v.view_kind = 2;
        v.base = ins[0];
        v.meta0 = ins[1];
        im.vals.push_back(std::move(v));
        reg_out(0, (int)im.vals.size() - 1);
        return;
    }
    if (op == "Unsqueeze" || op == "Squeeze") {
        std::vector<int64_t> axes;
        bool have_axes = false;
        if (ins.size() > 1 && ins[1] >= 0) have_axes = value_const_ints(im, ins[1], axes);
        else if (auto* at = n.attr("axes")) { axes = at->ints; have_axes = true; }
        if (im.vals[ins[0]].kind == Value::SYMVAL) {
            int r = sym_structural(C, n, ins);
            if (r >= 0) { reg_out(0, r); return; }
        }
        auto is = shape_exprs(im, im.vals[ins[0]]);
        std::vector<EP> os;
        std::vector<int64_t> naxes;
        if (op == "Unsqueeze") {
            if (!have_axes) fail_op("Unsqueeze axes unknown");
            int64_t nr = (int64_t)is.size() + (int64_t)axes.size();
            std::vector<char> at(nr, 0);
            for (auto a : axes) {
                int64_t na = norm_axis(a, nr);
                at[na] = 1;
                naxes.push_back(na);
            }
            size_t src = 0;
            for (int64_t i = 0; i < nr; i++) os.push_back(at[i] ? ec(1) : is[src++]);
        } else {
            std::vector<char> drop(is.size(), 0);
            if (have_axes) {
                for (auto a : axes) {
                    int64_t na = norm_axis(a, (int64_t)is.size());
                    drop[na] = 1;
                    naxes.push_back(na);
                }
            } else {
                for (size_t i = 0; i < is.size(); i++) {
                    int64_t c;
                    if (econst(is[i], c) && c == 1) { drop[i] = 1; naxes.push_back((int64_t)i); }
                }
            }
            for (size_t i = 0; i < is.size(); i++)
                if (!drop[i]) os.push_back(is[i]);
        }
        if (im.vals[ins[0]].kind == Value::CONSTV) {
            onnxm::Tensor t = im.vals[ins[0]].ct;
            t.dims.clear();
            for (auto& e : os) {
                int64_t c;
                if (!econst(e, c)) fail_op("layout of const with symbolic shape");
                t.dims.push_back(c);
            }
            reg_out(0, C.new_const(std::move(t)));
            return;
        }
        Value v;
        v.kind = Value::RT;
        v.dtype = dtype_of(im, ins[0]);
        v.rshape = os;
        v.view_kind = op == "Unsqueeze" ? 3 : 4;
        v.base = ins[0];
        v.vaxes = naxes;
        im.vals.push_back(std::move(v));
        reg_out(0, (int)im.vals.size() - 1);
        return;
    }
    if (op == "Slice") {
        if (im.vals[ins[0]].kind == Value::SYMVAL) {
            int r = sym_structural(C, n, ins);
            if (r >= 0) { reg_out(0, r); return; }
        }
        // Resolve axes at compile time (must be const); starts/ends/steps
        // may stay symbolic for the view, but shape exprs need them.
        std::vector<int64_t> axes;
        if (ins.size() > 3 && ins[3] >= 0) {
            if (!value_const_ints(im, ins[3], axes)) fail_op("Slice axes not const");
        }
        auto is = shape_exprs(im, im.vals[ins[0]]);
        int64_t rank = (int64_t)is.size();
        if (axes.empty()) { axes.resize(rank); for (int64_t i = 0; i < rank; i++) axes[i] = i; }
        for (auto& a : axes) a = norm_axis(a, rank);
        std::vector<EP> sev, eev, pev;
        if (!value_sym_elems(im, ins[1], sev) || !value_sym_elems(im, ins[2], eev)) fail_op("Slice bounds not symbolic/const");
        if (ins.size() > 4 && ins[4] >= 0) {
            if (!value_sym_elems(im, ins[4], pev)) fail_op("Slice steps not symbolic/const");
        } else
            pev.assign(axes.size(), ec(1));
        std::vector<EP> os = is;
        for (size_t i = 0; i < axes.size(); i++) {
            int64_t ax = axes[i];
            EP start = sev[i], end = eev[i];
            int64_t cs, ce, cp;
            bool ks = econst(start, cs), ke = econst(end, ce);
            int64_t step = econst(pev[i], cp) ? cp : 1;
            if (step == 0) fail_op("Slice step 0");
            int64_t dc;
            if (ks && ke && econst(is[ax], dc)) {
                int64_t first;
                os[ax] = ec(slice_count(dc, cs, ce, step, first));
                continue;
            }
            // Symbolic dim: build the length from clamped bound exprs.
            // Bounds beyond +-2^29 are sentinels (real dims in these
            // graphs never reach 2^29; the phase-1 code used the same
            // threshold for large positive ends) and act as infinities;
            // ordinary negative bounds count from the end (D + v), the
            // same normalization slice_count applies numerically.
            const int64_t INF = (int64_t)1 << 29;
            EP sE = start, eE = end;
            if (ks) {
                if (cs <= -INF) sE = step > 0 ? ec(0) : esub(is[ax], ec(1));
                else if (cs >= INF) sE = step > 0 ? is[ax] : esub(is[ax], ec(1));
                else if (cs < 0) sE = eadd(is[ax], ec(cs));
                else sE = ec(cs);
            }
            if (ke) {
                if (ce <= -INF) eE = step > 0 ? ec(0) : ec(-1);
                else if (ce >= INF) eE = step > 0 ? is[ax] : esub(is[ax], ec(1));
                else if (ce < 0) eE = eadd(is[ax], ec(ce));
                else eE = ec(ce);
            }
            if (step > 0) os[ax] = ediv(eadd(esub(eE, sE), ec(step - 1)), ec(step));
            else os[ax] = ediv(eadd(esub(sE, eE), ec(-step - 1)), ec(-step));
        }
        if (im.vals[ins[0]].kind == Value::CONSTV) {
            // Materialize via a view resolution on the const buffer.
            RT base = tensor_of_const(im.vals[ins[0]].ct);
            std::vector<int64_t> s_ints, e_ints, p_ints(axes.size(), 1);
            for (auto& e : sev) { int64_t c; if (!econst(e, c)) fail_op("const slice symbolic start"); s_ints.push_back(c); }
            for (auto& e : eev) { int64_t c; if (!econst(e, c)) fail_op("const slice symbolic end"); e_ints.push_back(c); }
            for (size_t i = 0; i < pev.size(); i++) { int64_t c; if (!econst(pev[i], c)) fail_op("const slice symbolic step"); p_ints[i] = c; }
            RT view = base;
            int64_t off = 0;
            for (size_t i = 0; i < axes.size(); i++) {
                int64_t ax = axes[i];
                int64_t first;
                view.shape[ax] = slice_count(base.shape[ax], s_ints[i], e_ints[i], p_ints[i], first);
                off += first * base.strides[ax];
                view.strides[ax] *= p_ints[i];
            }
            view.p = base.p + off * (int64_t)onnxm::elem_size(base.dtype);
            onnxm::Tensor t;
            t.data_type = base.dtype;
            t.dims = view.shape;
            t.data.resize((size_t)view.numel() * onnxm::elem_size(base.dtype));
            RT dst;
            dst.dtype = base.dtype;
            dst.p = t.data.data();
            dst.shape = t.dims;
            dst.strides = rowmajor(t.dims);
            copy_tensor(dst, view);
            reg_out(0, C.new_const(std::move(t)));
            return;
        }
        Value v;
        v.kind = Value::RT;
        v.dtype = dtype_of(im, ins[0]);
        v.rshape = os;
        v.view_kind = 5;
        v.base = ins[0];
        v.meta0 = ins[1];
        v.meta1 = ins[2];
        v.meta2 = ins.size() > 3 ? ins[3] : -1;
        v.meta3 = ins.size() > 4 ? ins[4] : -1;
        im.vals.push_back(std::move(v));
        reg_out(0, (int)im.vals.size() - 1);
        return;
    }

    // ---- Pad (phase 2) ----
    if (op == "Pad") {
        std::string mode = "constant";
        if (auto* at = n.attr("mode")) mode = at->s;
        Step st;
        st.k = SK::PAD;
        if (mode == "constant") st.pad_mode = 0;
        else if (mode == "reflect") st.pad_mode = 1;
        else if (mode == "edge") st.pad_mode = 2;
        else fail_op("Pad mode not implemented: " + mode);
        int pads_vid = -1;
        if (ins.size() > 1 && ins[1] >= 0) pads_vid = ins[1];
        else if (auto* at = n.attr("pads")) {
            // Pre-opset-11 form: pads as an attribute.
            onnxm::Tensor t;
            t.data_type = 7;
            t.dims = {(int64_t)at->ints.size()};
            t.data.resize(at->ints.size() * 8);
            memcpy(t.data.data(), at->ints.data(), at->ints.size() * 8);
            pads_vid = C.new_const(std::move(t));
        }
        if (pads_vid < 0) fail_op("Pad: no pads input or attr");
        std::vector<EP> pads_e;
        if (!value_sym_elems(im, pads_vid, pads_e)) fail_op("Pad pads not symbolic/const");
        auto xs = shape_exprs(im, im.vals[ins[0]]);
        int64_t r = (int64_t)xs.size();
        if ((int64_t)pads_e.size() != 2 * r) fail_op("Pad: pads length != 2*rank");
        std::vector<EP> os = xs;
        for (int64_t d = 0; d < r; d++) os[d] = eadd(eadd(xs[d], pads_e[d]), pads_e[r + d]);
        st.in = {ins[0], pads_vid};
        if (ins.size() > 2 && ins[2] >= 0) st.in.push_back(ins[2]);
        int odt = dtype_of(im, ins[0]);
        if (C.all_const(st.in)) {
            auto outs = fold_node(C, n, st.in, st, {{odt, os}}, {});
            reg_out(0, outs[0]);
            return;
        }
        int o = C.new_rt(odt, os);
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }

    // ---- Split (phase 2): outputs are views into the input ----
    if (op == "Split") {
        std::vector<int64_t> sizes;
        if (ins.size() > 1 && ins[1] >= 0) {
            if (!value_const_ints(im, ins[1], sizes)) fail_op("Split sizes not const");
        } else if (auto* at = n.attr("split")) sizes = at->ints;
        auto xs = shape_exprs(im, im.vals[ins[0]]);
        int64_t ax = norm_axis(n.attr_i("axis", 0), xs.size());
        if (sizes.empty()) {
            // Equal split: input dim must be a compile-time constant.
            int64_t dim;
            if (!econst(xs[ax], dim)) fail_op("Split: equal split of symbolic dim");
            int64_t cnt = (int64_t)n.outputs.size();
            if (cnt == 0 || dim % cnt != 0) fail_op("Split: dim not divisible");
            sizes.assign(cnt, dim / cnt);
        }
        if (sizes.size() != n.outputs.size()) fail_op("Split: sizes/outputs mismatch");
        int odt = dtype_of(im, ins[0]);
        if (im.vals[ins[0]].kind == Value::CONSTV) {
            RT base = tensor_of_const(im.vals[ins[0]].ct);
            size_t es = onnxm::elem_size(base.dtype);
            int64_t off = 0;
            for (size_t i = 0; i < sizes.size(); i++) {
                RT view = base;
                view.shape[ax] = sizes[i];
                view.p = base.p + (size_t)off * base.strides[ax] * es;
                onnxm::Tensor t;
                t.data_type = base.dtype;
                t.dims = view.shape;
                t.data.resize((size_t)view.numel() * es);
                RT dst;
                dst.dtype = base.dtype;
                dst.p = t.data.data();
                dst.shape = t.dims;
                dst.strides = rowmajor(t.dims);
                copy_tensor(dst, view);
                reg_out((int)i, C.new_const(std::move(t)));
                off += sizes[i];
            }
            return;
        }
        int64_t off = 0;
        for (size_t i = 0; i < sizes.size(); i++) {
            std::vector<EP> os = xs;
            os[ax] = ec(sizes[i]);
            Value v;
            v.kind = Value::RT;
            v.dtype = odt;
            v.rshape = os;
            v.view_kind = 7;  // split view
            v.base = ins[0];
            v.meta0 = (int)ax;
            v.meta1 = (int)off;
            im.vals.push_back(std::move(v));
            reg_out((int)i, (int)im.vals.size() - 1);
            off += sizes[i];
        }
        return;
    }

    // ---- Expand (phase 2): broadcast copy to the target shape ----
    if (op == "Expand") {
        std::vector<EP> target;
        if (!value_sym_elems(im, ins[1], target)) fail_op("Expand target not symbolic/const");
        auto os = broadcast_shape(shape_exprs(im, im.vals[ins[0]]), target);
        Step st;
        st.k = SK::EXPAND;
        st.in = {ins[0]};
        int odt = dtype_of(im, ins[0]);
        if (C.all_const(st.in)) {
            auto outs = fold_node(C, n, st.in, st, {{odt, os}}, {});
            reg_out(0, outs[0]);
            return;
        }
        int o = C.new_rt(odt, os);
        C.add_step(st, {o});
        reg_out(0, o);
        return;
    }

    fail_op("unsupported op in free engine (phase-1 GPT set)");
}



// ------------------------------------------------------ step execute ---
void exec_step(Impl& im, const Step& st, const std::vector<RT>& ins, std::vector<RT>& outs, Scratch& sc) {
    switch (st.k) {
        case SK::MATMUL: k_matmul(ins[0], ins[1], outs[0], sc); break;
        case SK::GEMM: {
            RT A = ins[0], B = ins[1];
            if (st.transA) {
                std::swap(A.shape[A.shape.size() - 2], A.shape[A.shape.size() - 1]);
                std::swap(A.strides[A.strides.size() - 2], A.strides[A.strides.size() - 1]);
            }
            if (st.transB) {
                std::swap(B.shape[B.shape.size() - 2], B.shape[B.shape.size() - 1]);
                std::swap(B.strides[B.strides.size() - 2], B.strides[B.strides.size() - 1]);
            }
            k_matmul(A, B, outs[0], sc);
            bool have_c = ins.size() > 2;
            if (st.alpha != 1.0f || have_c) {
                const RT* Cp = have_c ? &ins[2] : nullptr;
                auto ca = Cp ? align_of(*Cp, outs[0].shape.size()) : std::vector<int64_t>{};
                auto ya = align_of(outs[0], outs[0].shape.size());
                odometer(outs[0].shape, [&](const std::vector<int64_t>& c) {
                    int64_t yo = off_of(outs[0], c, ya);
                    double v = read_d(outs[0], yo) * st.alpha;
                    if (Cp) v += st.beta * read_d(*Cp, off_of(*Cp, c, ca));
                    write_d(outs[0], yo, v);
                });
            }
            break;
        }
        case SK::PACKED: {
            RT X = sc.contig_of(ins[0]);
            int64_t D = (st.pm_mode == 0 || st.pm_mode == 2) ? st.pm_cols : st.pm_rows;
            int64_t K = X.shape.back();
            int64_t M = X.numel() / K;
            if (!im.lookup) fail("PackedMatMul: no packed lookup set");
            PackedBinding pb;
            if (!im.lookup(st.pm_base, pb)) fail("PackedMatMul: unknown weight " + st.pm_base);
            const float* bias = nullptr;
            RT bc;
            if (st.pm_bias) {
                bc = sc.contig_of(ins[1]);
                bias = (const float*)bc.p;
            }
            float* Y = (float*)outs[0].p;
            switch (st.pm_mode) {
                case 0: packed::gemm_a_w4((const float*)X.p, M, K, D, pb.pk, pb.sc, pb.ze, bias, Y); break;
                case 1: packed::gemm_b_w4((const float*)X.p, M, K, D, pb.pk, pb.sc, pb.ze, bias, Y); break;
                case 2: packed::gemm_a_w8((const float*)X.p, M, K, D, pb.pk, pb.sc, bias, Y); break;
                case 3: packed::gemm_b_w8((const float*)X.p, M, K, D, pb.pk, pb.sc, bias, Y); break;
                default: fail("PackedMatMul: bad mode");
            }
            break;
        }
        case SK::CONV: k_conv(ins[0], ins[1], ins.size() > 2 ? &ins[2] : nullptr, outs[0], st.c_strides, st.c_pads, st.c_dil, st.c_group, sc); break;
        case SK::CONVT: k_convtranspose(ins[0], ins[1], ins.size() > 2 ? &ins[2] : nullptr, outs[0], st.c_strides, st.c_pads, st.c_dil, sc); break;
        case SK::PCONV: {
            RT X = sc.contig_of(ins[0]);
            if (X.shape.size() != 3) fail("PackedConv: X must be (N, C, L)");
            if (X.dtype != 1 || outs[0].dtype != 1) fail("PackedConv: f32 only");
            if (!im.lookup) fail("PackedConv: no packed lookup set");
            PackedBinding pb;
            if (!im.lookup(st.pm_base, pb)) fail("PackedConv: unknown weight " + st.pm_base);
            const float* bias = nullptr;
            RT bc;
            if (st.pm_bias) {
                bc = sc.contig_of(ins[1]);
                bias = (const float*)bc.p;
            }
            int64_t N = X.shape[0], Ci = X.shape[1], L = X.shape[2];
            const uint8_t* ze = st.pm_zero ? pb.ze : nullptr;
            float* Y = (float*)outs[0].p;
            if (!st.pc_transpose) {
                if (Ci * st.pc_k != st.pm_cols) fail("PackedConv: Ci*K != w_cols");
                packed::conv1d_fast((const float*)X.p, N, Ci, L, pb.pk, pb.sc, ze, st.pm_mode,
                                    st.pm_rows, st.pc_k, st.pc_stride, st.pc_pad_l, st.pc_dil,
                                    outs[0].shape[2], bias, Y);
            } else {
                if (Ci != st.pm_rows) fail("PackedConvTranspose: Ci != w_rows");
                int64_t O = st.pm_cols / st.pc_k;
                packed::conv_transpose1d_fast((const float*)X.p, N, Ci, L, pb.pk, pb.sc, ze,
                                              st.pm_mode, O, st.pc_k, st.pc_stride, st.pc_pad_l,
                                              st.pc_dil, outs[0].shape[2], bias, Y);
            }
            break;
        }
        case SK::PGATHER: {
            RT idx = sc.contig_of(ins[0]);
            if (idx.dtype != 7) fail("PackedGather: indices must be int64");
            if (!im.lookup) fail("PackedGather: no packed lookup set");
            PackedBinding pb;
            if (!im.lookup(st.pm_base, pb)) fail("PackedGather: unknown weight " + st.pm_base);
            packed::gather_packed(pb.pk, pb.sc, st.pm_zero ? pb.ze : nullptr, st.pm_mode,
                                  st.pm_rows, st.pm_cols, (const int64_t*)idx.p, idx.numel(),
                                  (float*)outs[0].p);
            break;
        }
        case SK::PAD: {
            // Exact integer read of the pads tensor (small values, but
            // keep them out of the double path on principle).
            const RT& pt = ins[1];
            std::vector<int64_t> pads;
            for (int64_t i = 0; i < pt.numel(); i++) {
                int64_t v = 0;
                if (pt.dtype == 7) memcpy(&v, pt.p + i * 8, 8);
                else if (pt.dtype == 6) { int32_t x; memcpy(&x, pt.p + i * 4, 4); v = x; }
                else v = (int64_t)read_d(pt, i);
                pads.push_back(v);
            }
            double fillv = ins.size() > 2 ? read_d(ins[2], 0) : 0.0;
            k_pad(ins[0], outs[0], pads, st.pad_mode, fillv);
            break;
        }
        case SK::EXPAND: k_expand(ins[0], outs[0]); break;
        case SK::LN: k_layernorm(ins[0], ins.size() > 1 ? &ins[1] : nullptr, ins.size() > 2 ? &ins[2] : nullptr, outs[0], st.axis, st.eps, sc); break;
        case SK::SOFTMAX: k_softmax(ins[0], outs[0], st.axis, sc); break;
        case SK::REDUCESUM: {
            k_reducesum(ins[0], outs[0], st.axes, st.keepdims);
            if (st.mean) {
                double f = (double)outs[0].numel() / (double)ins[0].numel();
                RT& Y = outs[0];
                if (Y.dtype == 1 && Y.contig()) {
                    float* p = (float*)Y.p;
                    int64_t n = Y.numel();
                    for (int64_t i = 0; i < n; i++) p[i] = (float)(p[i] * f);
                } else {
                    odometer(Y.shape, [&](const std::vector<int64_t>& c) {
                        int64_t yo = 0;
                        for (size_t d = 0; d < c.size(); d++) yo += c[d] * Y.strides[d];
                        write_d(Y, yo, read_d(Y, yo) * f);
                    });
                }
            }
            break;
        }
        case SK::ARGMAX: k_argmax(ins[0], outs[0], st.axis, st.keepdims, st.exclusive); break;
        case SK::TOPK: k_topk(ins[0], outs[0], outs[1], st.axis, st.topk, st.largest, st.sorted); break;
        case SK::CUMSUM: k_cumsum(ins[0], outs[0], st.axis, st.exclusive, st.reverse); break;
        case SK::GATHER: k_gather(ins[0], ins[1], outs[0], st.axis); break;
        case SK::GATHER_EL: k_gather_elements(ins[0], ins[1], outs[0], st.axis); break;
        case SK::SCATTER_EL: k_scatter_elements(ins[0], ins[1], ins[2], outs[0], st.axis, st.reduction); break;
        case SK::CONCAT: {
            std::vector<const RT*> ptrs;
            for (auto& t : ins) ptrs.push_back(&t);
            k_concat(ptrs, outs[0], st.axis, sc);
            break;
        }
        case SK::CONSTSHAPE: {
            double v = read_d(tensor_of_const(st.fill), 0);
            RT& Y = outs[0];
            if (v == 0) {
                memset(Y.p, 0, (size_t)Y.numel() * onnxm::elem_size(Y.dtype));
            } else {
                for (int64_t i = 0; i < Y.numel(); i++) write_d(Y, i, v);
            }
            break;
        }
        case SK::RANDNORM: {
            RT& Y = outs[0];
            std::mt19937_64 rng(0x9E3779B97F4A7C15ULL + (++im.rng_ctr) * 0x2545F4914F6CDD1DULL);
            std::normal_distribution<double> nd(0.0, 1.0);
            for (int64_t i = 0; i < Y.numel(); i++) write_d(Y, i, nd(rng));
            break;
        }
        case SK::BINOP: k_binop(st.bop, ins[0], ins[1], outs[0]); break;
        case SK::UNOP: k_unop(st.uop, ins[0], outs[0], st.alpha); break;
        case SK::CAST: k_cast(ins[0], outs[0]); break;
        case SK::WHERE: k_where(ins[0], ins[1], ins[2], outs[0]); break;
        case SK::COPY:
            if (st.flat_copy) flatten_copy(outs[0], ins[0]);
            else copy_tensor(outs[0], ins[0]);
            break;
    }
}

}  // namespace

// ---------------------------------------------------------- compile ----
std::shared_ptr<Plan> Plan::compile(const onnxm::Graph& g, std::string label) {
    auto plan = std::make_shared<Plan>();
    auto im = std::make_shared<Impl>();
    plan->impl_ = im;
    Compiler C(*im, g, label);
    // Initializers first (they may also be listed as inputs; parse_graph
    // already filtered them out of g.inputs).
    for (auto& t : g.initializers) {
        Value v;
        v.kind = Value::CONSTV;
        v.ct = t;
        im->vals.push_back(std::move(v));
        C.reg(t.name, (int)im->vals.size() - 1);
    }
    // Runtime inputs; packed weight slots (__pk/__sc/__ze) are not
    // values: PackedMatMul steps bind them through the lookup.
    for (auto& vi : g.inputs) {
        if (vi.name.size() >= 4) {
            std::string suf = vi.name.substr(vi.name.size() - 4);
            if (suf == "__pk" || suf == "__sc" || suf == "__ze") continue;
        }
        Value v;
        v.kind = Value::RT;
        v.dtype = vi.elem_type;
        v.is_input = true;
        v.def_step = -1;
        for (size_t d = 0; d < vi.dims.size(); d++) {
            if (vi.dims[d] >= 0) v.rshape.push_back(ec(vi.dims[d]));
            else {
                int s = im->nsyms++;
                im->sym_origin.push_back({(int)im->vals.size(), (int)d});
                v.rshape.push_back(esym(s));
            }
        }
        im->vals.push_back(std::move(v));
        int id = (int)im->vals.size() - 1;
        C.reg(vi.name, id);
        im->in_names.push_back(vi.name);
        im->in_vals.push_back(id);
    }
    for (auto& n : g.nodes) {
        lower_node(C, n);
        if (std::getenv("FE_DEBUG") && !n.outputs.empty() &&
            n.name.find("attn_layers.0") != std::string::npos) {
            auto it = im->by_name.find(n.outputs[0]);
            if (it != im->by_name.end())
                std::fprintf(stderr, "[kind] %s %s -> kind=%d\n", n.op.c_str(), n.name.c_str(),
                             (int)im->vals[it->second].kind);
        }
    }
    for (auto& vo : g.outputs) {
        int id = C.vid(vo.name);
        im->vals[id].is_output = true;
        im->out_names.push_back(vo.name);
        im->out_vals.push_back(id);
    }
    // Liveness: last use over steps, then propagate through views
    // (reverse creation order: views are created after their bases, and
    // a view's last use must extend its base's).
    int nsteps = (int)im->steps.size();
    for (int s = 0; s < nsteps; s++)
        for (int v : im->steps[s].in) im->vals[v].last_use = std::max(im->vals[v].last_use, s);
    for (auto& v : im->vals)
        if (v.is_output) v.last_use = nsteps + 1;
    for (int i = (int)im->vals.size() - 1; i >= 0; i--) {
        Value& v = im->vals[i];
        if (v.kind == Value::RT && v.view_kind != 0 && v.base >= 0)
            im->vals[v.base].last_use = std::max(im->vals[v.base].last_use, v.last_use);
    }
    plan->in_names_ = im->in_names;
    plan->out_names_ = im->out_names;
    plan->n_steps_ = im->steps.size();
    return plan;
}

// -------------------------------------------------------------- bind ---
// Read integer metadata (slice bounds, reshape target) from a resolved
// value at bind time. Exact integer reads: routing through double (as
// read_d does) corrupts sentinel values like INT64_MAX slice ends.
std::vector<int64_t> read_meta_ints(const RT& t) {
    std::vector<int64_t> r;
    odometer(t.shape, [&](const std::vector<int64_t>& c) {
        int64_t off = 0;
        for (size_t d = 0; d < c.size(); d++) off += c[d] * t.strides[d];
        int64_t v = 0;
        switch (t.dtype) {
            case 7: memcpy(&v, t.p + off * 8, 8); break;
            case 6: { int32_t x; memcpy(&x, t.p + off * 4, 4); v = x; break; }
            case 1: { float x; memcpy(&x, t.p + off * 4, 4); v = (int64_t)x; break; }
            default: v = (int64_t)read_d(t, off); break;
        }
        r.push_back(v);
    });
    return r;
}

// FE_PROFILE accumulators (see Plan::run). Kind order matches enum SK.
// Totals are kept per plan label so a decode step (stage plans) can be
// read separately from prefill (first) and the encoder.
struct ProfAcc { double us[26] = {}; long n[26] = {}; };
static std::map<std::string, ProfAcc> g_prof;
static const char* g_prof_names[26] = {"MATMUL", "GEMM", "PACKED", "CONV", "LN", "SOFTMAX",
                                       "REDUCESUM", "ARGMAX", "TOPK", "CUMSUM", "GATHER",
                                       "GATHER_EL", "SCATTER_EL", "CONCAT", "CONSTSHAPE",
                                       "RANDNORM", "BINOP", "UNOP", "CAST", "WHERE", "COPY",
                                       "PCONV", "PGATHER", "CONVT", "PAD", "EXPAND"};
static void prof_print() {
    for (auto& kv : g_prof) {
        std::fprintf(stderr, "[FE_PROFILE] per-kind totals, plan=%s:\n", kv.first.c_str());
        for (int i = 0; i < 26; i++)
            if (kv.second.n[i]) std::fprintf(stderr, "  %-10s %8ld calls %12.1f ms\n", g_prof_names[i], kv.second.n[i], kv.second.us[i] / 1000.0);
    }
}
static void prof_add(const std::string& label, int kind, double us) {
    static bool reg = (std::atexit(prof_print), true);
    (void)reg;
    if (kind >= 0 && kind < 26) { ProfAcc& a = g_prof[label]; a.us[kind] += us; a.n[kind]++; }
}

void Plan::release_arena() {
    Impl& im = *std::static_pointer_cast<Impl>(impl_);
    std::vector<uint8_t>().swap(im.arena);
    std::vector<uint8_t>().swap(im.out_arena);
    arena_bytes_ = 0;
}

std::map<std::string, Tensor> Plan::run(const std::map<std::string, Tensor>& inputs) {
    Impl& im = *std::static_pointer_cast<Impl>(impl_);
    im.lookup = lookup_;
    int nv = (int)im.vals.size();
    // Symbol values from the caller's input shapes.
    std::vector<int64_t> sv(im.nsyms, 0);
    std::vector<RT> res(nv);
    std::vector<char> done(nv, 0);
    // Inputs.
    for (size_t i = 0; i < im.in_vals.size(); i++) {
        int vid = im.in_vals[i];
        auto it = inputs.find(im.in_names[i]);
        if (it == inputs.end()) {
            if (im.vals[vid].last_use >= 0 || im.vals[vid].is_output)
                fail("missing input: " + im.in_names[i]);
            res[vid] = RT{};
            done[vid] = 1;
            continue;
        }
        RT r;
        r.dtype = it->second.dtype;
        r.p = const_cast<uint8_t*>(it->second.data);
        r.shape = it->second.shape;
        r.strides = it->second.strides.empty() ? rowmajor(r.shape) : it->second.strides;
        res[vid] = r;
        done[vid] = 1;
    }
    for (int s = 0; s < im.nsyms; s++) {
        auto [vid, dim] = im.sym_origin[s];
        sv[s] = dim < (int)res[vid].shape.size() ? res[vid].shape[dim] : 0;
    }
    // Consts and symbolic values.
    im.sym_bufs.assign(nv, {});
    for (int i = 0; i < nv; i++) {
        Value& v = im.vals[i];
        if (v.kind == Value::CONSTV) {
            res[i] = tensor_of_const(v.ct);
            done[i] = 1;
        } else if (v.kind == Value::SYMVAL) {
            auto& buf = im.sym_bufs[i];
            buf.resize((size_t)numel_of(v.sshape) * onnxm::elem_size(v.sdt));
            RT r;
            r.dtype = v.sdt;
            r.p = buf.data();
            r.shape = v.sshape;
            r.strides = rowmajor(v.sshape);
            for (size_t e = 0; e < v.selems.size(); e++) write_d(r, (int64_t)e, (double)eeval(v.selems[e], sv));
            res[i] = r;
            done[i] = 1;
        }
    }
    // Concrete shapes for RT values (from exprs).
    std::vector<std::vector<int64_t>> cshape(nv);
    for (int i = 0; i < nv; i++) {
        Value& v = im.vals[i];
        if (v.kind != Value::RT) continue;
        for (auto& e : v.rshape) cshape[i].push_back(eeval(e, sv));
    }
    // Arena assignment for materialized RT values (not views / inputs).
    struct Alloc { int vid; int64_t size; int def, use; bool out; };
    std::vector<Alloc> allocs;
    for (int i = 0; i < nv; i++) {
        Value& v = im.vals[i];
        if (v.kind != Value::RT || v.view_kind != 0 || v.is_input) continue;
        if (v.def_step == -2) continue;  // degenerate LN side output
        int64_t sz = 1;
        for (auto d : cshape[i]) sz *= d;
        sz *= (int64_t)onnxm::elem_size(v.dtype);
        allocs.push_back({i, sz, std::max(v.def_step, 0), v.last_use, v.is_output});
    }
    auto plan_arena = [](std::vector<Alloc>& as, std::vector<uint8_t>& buf,
                         std::map<int, size_t>& offs) {
        std::stable_sort(as.begin(), as.end(), [](const Alloc& a, const Alloc& b) { return a.def < b.def; });
        struct Block { size_t off, size; int use; };
        std::vector<Block> blocks;
        size_t total = 0;
        for (auto& a : as) {
            size_t need = (size_t)((a.size + 63) & ~63LL);
            if (need == 0) need = 64;
            int found = -1;
            for (size_t bi = 0; bi < blocks.size(); bi++) {
                if (blocks[bi].size >= need && blocks[bi].use < a.def) {
                    found = (int)bi;
                    break;
                }
            }
            if (found >= 0) {
                offs[a.vid] = blocks[found].off;
                int old_use = blocks[found].use;
                blocks[found].use = a.use;
                if (blocks[found].size > need + 4096) {
                    // Split large leftovers back into the free pool; the
                    // leftover keeps the old block's (earlier) last use.
                    blocks.push_back({blocks[found].off + need, blocks[found].size - need, old_use});
                    blocks[found].size = need;
                }
            } else {
                offs[a.vid] = total;
                blocks.push_back({total, need, a.use});
                total += need;
            }
        }
        if (buf.size() < total) buf.resize(total);
        return total;
    };
    std::vector<Alloc> work, outa;
    for (auto& a : allocs) (a.out ? outa : work).push_back(a);
    std::map<int, size_t> off_w, off_o;
    size_t arena_total = plan_arena(work, im.arena, off_w);
    size_t out_total = plan_arena(outa, im.out_arena, off_o);
    arena_bytes_ = arena_total + out_total;
    for (auto& a : allocs) {
        RT r;
        r.dtype = im.vals[a.vid].dtype;
        r.shape = cshape[a.vid];
        r.strides = rowmajor(r.shape);
        r.p = (a.out ? im.out_arena.data() + off_o[a.vid] : im.arena.data() + off_w[a.vid]);
        res[a.vid] = r;
        done[a.vid] = 1;
    }
    // Degenerate unproduced values (LN side outputs): zero-size views.
    for (int i = 0; i < nv; i++) {
        Value& v = im.vals[i];
        if (v.kind == Value::RT && v.def_step == -2 && !done[i]) {
            RT r;
            r.dtype = v.dtype;
            r.p = im.arena.data();
            r.shape = cshape[i];
            r.strides = rowmajor(r.shape);
            res[i] = r;
            done[i] = 1;
        }
    }
    // Views, in creation order (bases always precede their views).
    im.view_blocks.clear();
    for (int i = 0; i < nv; i++) {
        Value& v = im.vals[i];
        if (v.kind != Value::RT || v.view_kind == 0 || done[i]) continue;
        const RT& B = res[v.base];
        RT r;
        r.dtype = v.dtype;
        if (v.view_kind == 1) {  // transpose
            r.p = B.p;
            for (auto p : v.vperm) {
                r.shape.push_back(B.shape[p]);
                r.strides.push_back(B.strides[p]);
            }
        } else if (v.view_kind == 2) {  // reshape
            RT src = B;
            if (!src.contig()) {
                size_t bytes = (size_t)src.numel() * onnxm::elem_size(src.dtype);
                im.view_blocks.emplace_back(bytes);
                src = materialize(B, im.view_blocks.back().data());
            }
            r.p = src.p;
            r.shape = cshape[i];
            r.strides = rowmajor(r.shape);
        } else if (v.view_kind == 3) {  // unsqueeze
            r.p = B.p;
            std::vector<char> at(B.shape.size() + v.vaxes.size(), 0);
            for (auto a : v.vaxes) at[a] = 1;
            size_t src = 0;
            for (size_t d = 0; d < at.size(); d++) {
                if (at[d]) {
                    r.shape.push_back(1);
                    r.strides.push_back(d + 1 < at.size() ? 1 : 1);
                } else {
                    r.shape.push_back(B.shape[src]);
                    r.strides.push_back(B.strides[src]);
                    src++;
                }
            }
            // Stride of a size-1 dim is irrelevant; keep it simple.
        } else if (v.view_kind == 4) {  // squeeze
            r.p = B.p;
            std::vector<char> drop(B.shape.size(), 0);
            for (auto a : v.vaxes) drop[a] = 1;
            for (size_t d = 0; d < B.shape.size(); d++)
                if (!drop[d]) {
                    r.shape.push_back(B.shape[d]);
                    r.strides.push_back(B.strides[d]);
                }
        } else if (v.view_kind == 5) {  // slice
            std::vector<int64_t> starts = read_meta_ints(res[v.meta0]);
            std::vector<int64_t> ends = read_meta_ints(res[v.meta1]);
            std::vector<int64_t> axes;
            if (v.meta2 >= 0) axes = read_meta_ints(res[v.meta2]);
            if (axes.empty()) {
                axes.resize(B.shape.size());
                for (size_t a = 0; a < axes.size(); a++) axes[a] = (int64_t)a;
            }
            std::vector<int64_t> steps(axes.size(), 1);
            if (v.meta3 >= 0) steps = read_meta_ints(res[v.meta3]);
            r = B;
            int64_t off = 0;
            for (size_t ai = 0; ai < axes.size(); ai++) {
                int64_t ax = norm_axis(axes[ai], B.shape.size());
                int64_t first;
                r.shape[ax] = slice_count(B.shape[ax], starts[ai], ends[ai], steps[ai], first);
                off += first * B.strides[ax];
                r.strides[ax] = B.strides[ax] * steps[ai];
            }
            r.p = B.p + off * (int64_t)onnxm::elem_size(B.dtype);
        } else if (v.view_kind == 7) {  // split: contiguous-axis window of the base
            int64_t ax = norm_axis(v.meta0, B.shape.size());
            r = B;
            r.shape[ax] = cshape[i][ax];
            r.p = B.p + (int64_t)v.meta1 * B.strides[ax] * (int64_t)onnxm::elem_size(B.dtype);
        }
        // Validate compile-time shape exprs against the resolved view.
        if (r.shape != cshape[i]) fail("view shape mismatch at bind (slice/reshape exprs)");
        res[i] = r;
        done[i] = 1;
    }
    // Execute steps. FE_PROFILE=1 accumulates wall time per step kind
    // across all plans in the process and prints a table at exit.
    static const bool profile = std::getenv("FE_PROFILE") != nullptr;
    Scratch sc;
    std::vector<RT> sin, sout;
    for (auto& st : im.steps) {
        sin.clear();
        sout.clear();
        for (int v : st.in) sin.push_back(res[v]);
        for (int v : st.out) sout.push_back(res[v]);
        Scratch::Mark mk = sc.mark();
        if (profile) {
            auto p0 = std::chrono::steady_clock::now();
            exec_step(im, st, sin, sout, sc);
            double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - p0).count();
            prof_add(im.label, (int)st.k, us);
        } else {
            exec_step(im, st, sin, sout, sc);
        }
        sc.reset(mk);
    }
    // Outputs.
    std::map<std::string, Tensor> out;
    for (size_t i = 0; i < im.out_vals.size(); i++) {
        const RT& r = res[im.out_vals[i]];
        Tensor t;
        t.dtype = r.dtype;
        t.data = r.p;
        t.shape = r.shape;
        if (!r.contig()) t.strides = r.strides;
        out[im.out_names[i]] = t;
    }
    return out;
}

}  // namespace freeeng
