// vits_check_free: numeric validation for the ORT-free engine's vits
// path (ORT-FREE phase 2), replicating vits_check.cpp's protocol:
// same feed (work/memopt/vits_feed.npz), same noiseless probe graphs,
// same per-layer comparison and the same stage-1 denoise口径 against
// work/memopt/vits_fz.npy. No onnxruntime session exists in this
// process (ldd shows no libonnxruntime).
//
// Comparisons reported:
//   * free packed  vs free baked E   (per-layer, all probe outputs)
//   * free packed  vs ORT packed     (when --ort-dump <prefix> given:
//                                     vits_check --dump files)
//   * free baked E vs ORT baked E    (same)
//   * final audio of both free paths vs the fp32 reference; the E-pack
//     baseline under this口径 is corr 0.9599 / relRMS 0.2933 and the
//     packed path must not fall below it.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "cpu.h"
#include "free_engine.h"
#include "free_pool.h"
#include "json_min.h"
#include "npz.h"
#include "packed_core.h"

static std::string read_text(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open: " + path);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
static std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open: " + path);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
static long rss_kb() {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line))
        if (line.rfind("VmRSS:", 0) == 0) return std::atol(line.c_str() + 6);
    return 0;
}

// Minimal .npy reader (float32, C order) for the fp32 reference audio.
static std::vector<float> read_npy_f32(const std::string& path) {
    auto bytes = read_file(path);
    if (bytes.size() < 10 || memcmp(bytes.data(), "\x93NUMPY", 6) != 0)
        throw std::runtime_error("bad npy " + path);
    uint16_t hlen = (uint16_t)bytes[8] | ((uint16_t)bytes[9] << 8);
    size_t off = 10 + hlen;
    size_t nbytes = bytes.size() - off;
    std::vector<float> out(nbytes / 4);
    memcpy(out.data(), bytes.data() + off, nbytes - nbytes % 4);
    return out;
}

// Dump reader for vits_check --dump files (u32 name_len, name, u64
// numel, f32 data per record).
static std::map<std::string, std::vector<float>> read_dump(const std::string& path) {
    auto bytes = read_file(path);
    std::map<std::string, std::vector<float>> m;
    size_t pos = 0;
    while (pos < bytes.size()) {
        uint32_t nl;
        memcpy(&nl, bytes.data() + pos, 4);
        pos += 4;
        std::string name((const char*)bytes.data() + pos, nl);
        pos += nl;
        uint64_t n;
        memcpy(&n, bytes.data() + pos, 8);
        pos += 8;
        std::vector<float> v(n);
        memcpy(v.data(), bytes.data() + pos, (size_t)n * 4);
        pos += (size_t)n * 4;
        m[name] = std::move(v);
    }
    return m;
}

struct Stats {
    double maxerr = 0, corr = 1, relrms = 0;
};

static Stats compare(const float* a, const float* b, size_t n) {
    Stats s;
    double sa = 0, sb = 0, sab = 0, saa = 0, sbb = 0, num = 0, den = 0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)a[i] - b[i];
        s.maxerr = std::max(s.maxerr, std::abs(d));
        num += d * d;
        den += (double)b[i] * b[i];
        sa += a[i]; sb += b[i]; sab += (double)a[i] * b[i]; saa += (double)a[i] * a[i]; sbb += (double)b[i] * b[i];
    }
    s.relrms = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
    double cov = sab - sa * sb / (double)n;
    double va = saa - sa * sa / (double)n, vb = sbb - sb * sb / (double)n;
    s.corr = (va > 0 && vb > 0) ? cov / std::sqrt(va * vb) : (s.maxerr == 0 ? 1.0 : 0.0);
    return s;
}

static freeeng::Tensor npy_tensor(const NpyTensor& t) {
    freeeng::Tensor ft;
    ft.data = t.data.data();
    ft.shape = t.shape;
    if (t.descr == "<f4") ft.dtype = 1;
    else if (t.descr == "<i8") ft.dtype = 7;
    else if (t.descr == "<i4") ft.dtype = 6;
    else throw std::runtime_error("npz: unexpected descr " + t.descr);
    return ft;
}

using Clk = std::chrono::steady_clock;
static double ms_since(Clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clk::now() - t0).count();
}

int main(int argc, char** argv) {
    std::string root = ".";  // override with --root
    int threads = 4;
    std::string ort_dump;
    bool skip_baked = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--root") root = argv[++i];
        else if (a == "--threads") threads = std::atoi(argv[++i]);
        else if (a == "--ort-dump") ort_dump = argv[++i];
        else if (a == "--skip-baked") skip_baked = true;
        else { std::cerr << "unknown arg " << a << "\n"; return 2; }
    }
    cpu::log_features("vits_check_free");
    try {
        packed::set_kernel_threads(threads);

        // ---- feed (same tensors as vits_check)
        Npz npz;
        npz.load(root + "/work/memopt/vits_feed.npz");
        std::map<std::string, freeeng::Tensor> feed;
        for (const char* nm : {"text_seq", "pred_semantic", "ge", "ge_advanced"})
            feed.emplace(nm, npy_tensor(npz.entries.at(nm)));
        auto fz = read_npy_f32(root + "/work/memopt/vits_fz.npy");

        auto run_plan = [&](const std::string& graph_path, freeeng::Plan* plan,
                            std::map<std::string, std::vector<float>>& out_map,
                            double& run_ms) {
            auto t0 = Clk::now();
            auto outs = plan->run(feed);
            run_ms = ms_since(t0);
            for (auto& kv : outs) {
                if (kv.second.dtype != 1)
                    throw std::runtime_error("probe output not f32: " + kv.first);
                const float* p = (const float*)kv.second.data;
                // Outputs may be strided views; copy in logical order.
                if (kv.second.contiguous()) {
                    out_map[kv.first].assign(p, p + kv.second.numel());
                } else {
                    // Walk the tensor through its strides.
                    std::vector<int64_t> coord(kv.second.shape.size(), 0);
                    std::vector<float> v;
                    v.reserve((size_t)kv.second.numel());
                    for (int64_t i = 0; i < kv.second.numel(); i++) {
                        int64_t off = 0;
                        for (size_t d = 0; d < coord.size(); d++) off += coord[d] * kv.second.strides[d];
                        v.push_back(p[off]);
                        for (int d = (int)coord.size() - 1; d >= 0; d--) {
                            if (++coord[d] < kv.second.shape[d]) break;
                            coord[d] = 0;
                        }
                    }
                    out_map[kv.first] = std::move(v);
                }
            }
        };

        // ---- packed probes on the free engine
        auto blob_store = std::make_shared<packed::BlobStore>();
        FreePool pool_v(blob_store);
        Json man_h = JsonParser(read_text(root + "/wo-packs/manifest_vits_w4hybrid.json")).parse();
        Json side = JsonParser(read_text(root + "/host/packed_graphs/vits_wnfold.onnx.json")).parse();
        for (auto& pi : side.find("packed_inputs")->arr) pool_v.filter[pi.find("tensor")->as_str()] = true;
        pool_v.load(root, man_h, "w4hybrid", "sovits_vits");
        std::cerr << "[vits_check_free] vits pool: entries " << pool_v.entries << " store unique "
                  << blob_store->unique_bytes() << " B logical " << pool_v.logical << " B; rss "
                  << rss_kb() / 1024 << " MB\n";

        std::map<std::string, std::vector<float>> packed_out, baked_out;
        double t_compile_p, t_compile_b = 0, run_p, run_b = 0;
        size_t arena_p, arena_b = 0, steps_p, steps_b = 0;
        {
            auto t0 = Clk::now();
            onnxm::Graph g = onnxm::load_model(root + "/work/memopt/vits_packed_noiseless_probes.onnx");
            auto plan = freeeng::Plan::compile(g, "vits-packed");
            t_compile_p = ms_since(t0);
            plan->set_packed_lookup(pool_v.lookup());
            run_plan("", plan.get(), packed_out, run_p);
            arena_p = plan->arena_bytes();
            steps_p = plan->step_count();
            std::cerr << "[vits_check_free] packed: compile " << (long long)t_compile_p << " ms, "
                      << steps_p << " steps, arena " << arena_p / 1048576.0 << " MB, run "
                      << (long long)run_p << " ms, " << packed_out.size() << " outputs; rss "
                      << rss_kb() / 1024 << " MB\n";
        }

        // ---- baked E probes on the free engine
        if (!skip_baked) {
            auto t0 = Clk::now();
            onnxm::Graph g = onnxm::load_model(root + "/work/memopt/vits_bakedE_noiseless_probes.onnx");
            auto plan = freeeng::Plan::compile(g, "vits-baked");
            t_compile_b = ms_since(t0);
            run_plan("", plan.get(), baked_out, run_b);
            arena_b = plan->arena_bytes();
            steps_b = plan->step_count();
            std::cerr << "[vits_check_free] baked: compile " << (long long)t_compile_b << " ms, "
                      << steps_b << " steps, arena " << arena_b / 1048576.0 << " MB, run "
                      << (long long)run_b << " ms, " << baked_out.size() << " outputs; rss "
                      << rss_kb() / 1024 << " MB\n";
        }

        auto layer_report = [&](const std::map<std::string, std::vector<float>>& A,
                                const std::map<std::string, std::vector<float>>& B,
                                const char* an, const char* bn) {
            struct Row { std::string name; Stats st; size_t n; };
            std::vector<Row> rows;
            for (auto& kv : B) {
                auto it = A.find(kv.first);
                if (it == A.end() || it->second.size() != kv.second.size()) continue;
                rows.push_back({kv.first, compare(it->second.data(), kv.second.data(), kv.second.size()),
                                kv.second.size()});
            }
            double worst_corr = 1, worst_err = 0;
            for (auto& r : rows) {
                worst_corr = std::min(worst_corr, r.st.corr);
                worst_err = std::max(worst_err, r.st.maxerr);
            }
            std::printf("{\"layers_%s_vs_%s\": {\"compared\": %zu, \"worst_corr\": %.6f, \"worst_maxerr\": %.6g}}\n",
                        an, bn, rows.size(), worst_corr, worst_err);
            std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.st.corr < b.st.corr; });
            std::cerr << "[layers " << an << " vs " << bn << "] worst 5 by correlation:\n";
            for (size_t i = 0; i < std::min<size_t>(rows.size(), 5); i++)
                std::cerr << "  " << rows[i].name << " n=" << rows[i].n << " maxerr=" << rows[i].st.maxerr
                          << " corr=" << rows[i].st.corr << " relrms=" << rows[i].st.relrms << "\n";
            return worst_corr;
        };

        if (!skip_baked) layer_report(packed_out, baked_out, "free_packed", "free_baked");
        if (!ort_dump.empty()) {
            auto ort_packed = read_dump(ort_dump + "_packed.bin");
            layer_report(packed_out, ort_packed, "free_packed", "ort_packed");
            if (!skip_baked) {
                auto ort_baked = read_dump(ort_dump + "_baked.bin");
                layer_report(baked_out, ort_baked, "free_baked", "ort_baked");
            }
        }

        // ---- final audio vs fp32 reference (stage-1 denoise口径)
        Stats sp = compare(packed_out["audio"].data(), fz.data(), fz.size());
        if (skip_baked) {
            std::printf("{\"free_packed_vs_fp32\": {\"corr\": %.6f, \"relrms\": %.6f, \"maxerr\": %.6g}, "
                        "\"compile_ms\": %.0f, \"rss_mb\": %ld}\n",
                        sp.corr, sp.relrms, sp.maxerr, t_compile_p, rss_kb() / 1024);
            return 0;
        }
        Stats sb = compare(baked_out["audio"].data(), fz.data(), fz.size());
        Stats pp = compare(packed_out["audio"].data(), baked_out["audio"].data(), fz.size());
        std::printf("{\"free_baked_vs_fp32\": {\"corr\": %.6f, \"relrms\": %.6f, \"maxerr\": %.6g}, "
                    "\"free_packed_vs_fp32\": {\"corr\": %.6f, \"relrms\": %.6f, \"maxerr\": %.6g}, "
                    "\"free_packed_vs_free_baked\": {\"corr\": %.6f, \"relrms\": %.6f, \"maxerr\": %.6g}, "
                    "\"compile_ms\": {\"packed\": %.0f, \"baked\": %.0f}, \"rss_mb\": %ld}\n",
                    sb.corr, sb.relrms, sb.maxerr, sp.corr, sp.relrms, sp.maxerr, pp.corr, pp.relrms,
                    pp.maxerr, t_compile_p, t_compile_b, rss_kb() / 1024);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "vits_check_free error: " << e.what() << "\n";
        return 1;
    }
}
