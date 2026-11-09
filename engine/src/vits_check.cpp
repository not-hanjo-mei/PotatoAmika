// vits_check: numeric validation for the packed-residency vits path
// (Stage 3 S10 step 2).
//
// Runs the deterministic stage-1 eval feed (work/memopt/vits_feed.npz:
// neutral1 text, fp32-GPT tokens, prompt ge/ge_advanced) through two
// noiseless probe graphs built by make_packed_vits.py --probes:
//   * baked E graph  (weights restored to fp32 at load; the shipping
//     behaviour before this step)
//   * packed graph   (potato::PackedConv / PackedConvTranspose /
//     PackedGather over the E pack bytes)
// and compares EVERY rewritten node's output between them (max abs err +
// correlation per tensor), plus the final audio of both against the fp32
// reference under the stage-1 denoise口径 (work/memopt/vits_fz.npy,
// produced from work/vits_noiseless.onnx). The E-pack baked baseline
// under this口径 is corr 0.9599 / relRMS 0.2933; the packed path must not
// fall below it.

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
#include "json_min.h"
#include "npz.h"
#include "packed.h"
#include "vits_packed.h"

static std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
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
    if (bytes.size() < 10 || bytes.compare(0, 6, "\x93NUMPY") != 0)
        throw std::runtime_error("bad npy " + path);
    uint16_t hlen = (uint8_t)bytes[8] | ((uint16_t)(uint8_t)bytes[9] << 8);
    size_t off = 10 + hlen;
    if (bytes.find("'descr': '<f4'") > hlen + 10 && bytes.find("<f4") == std::string::npos)
        throw std::runtime_error("npy not <f4: " + path);
    size_t nbytes = bytes.size() - off;
    std::vector<float> out(nbytes / 4);
    memcpy(out.data(), bytes.data() + off, nbytes - nbytes % 4);
    return out;
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

// Dump format shared with vits_check_free (--dump <prefix> writes
// <prefix>_baked.bin / <prefix>_packed.bin): per tensor a record of
// u32 name_len, name bytes, u64 numel, f32 data (shapes are not kept
// by vits_check's in-memory maps; element counts identify tensors).
static void dump_map(const std::map<std::string, std::vector<float>>& m, const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write dump " + path);
    for (auto& kv : m) {
        uint32_t nl = (uint32_t)kv.first.size();
        f.write((const char*)&nl, 4);
        f.write(kv.first.data(), nl);
        uint64_t n = kv.second.size();
        f.write((const char*)&n, 8);
        f.write((const char*)kv.second.data(), (std::streamsize)(n * 4));
    }
}

int main(int argc, char** argv) {
    std::string root = ".";
    int threads = 4;
    int ort_threads = -1;
    bool skip_baked = false;
    std::string dump_prefix;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--root") root = argv[++i];
        else if (a == "--threads") threads = std::atoi(argv[++i]);
        else if (a == "--ort-threads") ort_threads = std::atoi(argv[++i]);
        else if (a == "--skip-baked") skip_baked = true;
        else if (a == "--dump") dump_prefix = argv[++i];
        else { std::cerr << "unknown arg " << a << "\n"; return 2; }
    }
    if (ort_threads < 0) ort_threads = threads;
    // The packed session defaults to ONE ORT intra-op thread: its heavy
    // ops are the potato kernels, which run on the kernel pool; ORT's own
    // spinning intra-op threads otherwise oversubscribe the cores and
    // roughly double the packed run time (measured S10).
    cpu::log_features("vits_check");
    std::cerr << "[vits_check] kernel tier: "
              << (packed::active_tier() == packed::KernelTier::Scalar ? "scalar"
                  : packed::active_tier() == packed::KernelTier::Avx2 ? "avx2"
                  : packed::active_tier() == packed::KernelTier::Neon ? "neon" : "sdot")
              << " (implementations registered this build: scalar)\n";
    try {
        packed::set_kernel_threads(threads);
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "vits-check");
        Ort::SessionOptions so;
        so.SetIntraOpNumThreads(ort_threads);
        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
        auto domain = packed::make_domain_full();
        so.Add(domain);
        const char* prof_prefix = std::getenv("VITS_PROFILE");
        if (prof_prefix) so.EnableProfiling(prof_prefix);

        // ---- feed
        Npz npz;
        npz.load(root + "/work/memopt/vits_feed.npz");
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto make_feed = [&]() {
            std::map<std::string, Ort::Value> feed;
            for (const char* nm : {"text_seq", "pred_semantic", "ge", "ge_advanced"}) {
                const NpyTensor& t = npz.entries.at(nm);
                if (t.descr == "<i8")
                    feed.emplace(nm, Ort::Value::CreateTensor<int64_t>(mem, (int64_t*)t.data.data(),
                                                                       t.numel(),
                                                                       t.shape.data(), t.shape.size()));
                else
                    feed.emplace(nm, Ort::Value::CreateTensor<float>(mem, (float*)t.data.data(),
                                                                     t.numel(),
                                                                     t.shape.data(), t.shape.size()));
            }
            return feed;
        };
        auto fz = read_npy_f32(root + "/work/memopt/vits_fz.npy");

        // ---- baked E session (plain)
        std::map<std::string, std::vector<float>> baked_out;
        if (!skip_baked) {
        Ort::Session baked(env, (root + "/work/memopt/vits_bakedE_noiseless_probes.onnx").c_str(), so);
        Ort::AllocatorWithDefaultOptions alloc;
        auto feed_b = make_feed();
        auto out_names_b = baked.GetOutputNames();
        auto in_name_store = baked.GetInputNames();
        std::vector<const char*> in_names, out_names;
        std::vector<Ort::Value> in_vals;
        for (auto& nm : in_name_store) {
            auto it = feed_b.find(nm);
            if (it == feed_b.end()) continue;  // initializer listed as input; ORT fills it
            in_names.push_back(it->first.c_str());
            in_vals.push_back(std::move(it->second));
        }
        for (auto& n : out_names_b) out_names.push_back(n.c_str());
        auto t0 = std::chrono::steady_clock::now();
        auto outs_b = baked.Run(Ort::RunOptions{nullptr}, in_names.data(), in_vals.data(),
                                in_vals.size(), out_names.data(), out_names.size());
        auto t1 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < outs_b.size(); i++) {
            auto info = outs_b[i].GetTensorTypeAndShapeInfo();
            size_t n = info.GetElementCount();
            const float* p = outs_b[i].GetTensorData<float>();
            baked_out[out_names[i]].assign(p, p + n);
        }
        std::cerr << "[vits_check] baked E run " << std::chrono::duration<double, std::milli>(t1 - t0).count()
                  << " ms, " << baked_out.size() << " outputs; rss " << rss_kb() / 1024 << " MB\n";
        }  // if (!skip_baked)

        // ---- packed session (pool + named session)
        auto blob_store = std::make_shared<packed::BlobStore>();
        packed::WeightPool pool_v(blob_store);
        Json man_h = JsonParser(read_file(root + "/wo-packs/manifest_vits_w4hybrid.json")).parse();
        Json side = JsonParser(read_file(root + "/host/packed_graphs/vits_wnfold.onnx.json")).parse();
        std::map<std::string, bool> filter;
        for (auto& pi : side.find("packed_inputs")->arr) filter[pi.find("tensor")->as_str()] = true;
        pool_v.set_filter(filter);
        pool_v.load(root, man_h, "w4hybrid", {"sovits_vits"});
        std::cerr << "[vits_check] vits pool: entries " << pool_v.entries() << " store unique "
                  << blob_store->unique_bytes() << " B logical " << pool_v.logical_bytes()
                  << " B; rss " << rss_kb() / 1024 << " MB\n";
        Ort::SessionOptions so_p;
        so_p.SetIntraOpNumThreads(1);
        so_p.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
        so_p.Add(domain);
        if (prof_prefix) so_p.EnableProfiling(prof_prefix);
        packed::NamedSession packed_s;
        auto pbytes = read_file(root + "/work/memopt/vits_packed_noiseless_probes.onnx");
        packed_s.init(env, pbytes.data(), pbytes.size(), so_p);
        auto feed_p = make_feed();
        auto t0 = std::chrono::steady_clock::now();
        auto pouts = packed_s.run(feed_p, &pool_v);
        auto t1 = std::chrono::steady_clock::now();
        std::map<std::string, std::vector<float>> packed_out;
        for (auto& kv : pouts) {
            auto info = kv.second.GetTensorTypeAndShapeInfo();
            size_t n = info.GetElementCount();
            const float* p = kv.second.GetTensorData<float>();
            packed_out[kv.first].assign(p, p + n);
        }
        std::cerr << "[vits_check] packed run " << std::chrono::duration<double, std::milli>(t1 - t0).count()
                  << " ms, " << packed_out.size() << " outputs; rss " << rss_kb() / 1024 << " MB\n";
        if (!dump_prefix.empty()) {
            if (!baked_out.empty()) dump_map(baked_out, dump_prefix + "_baked.bin");
            dump_map(packed_out, dump_prefix + "_packed.bin");
            std::cerr << "[vits_check] dumps written to " << dump_prefix << "_*.bin\n";
        }
        auto& vs = packed::vc_stats();
        std::cerr << "[vc_stats] conv calls " << vs.calls << " dequant " << vs.dequant_us / 1000
                  << " ms, compute " << vs.compute_us / 1000 << " ms, max call " << vs.max_call_us / 1000
                  << " ms, total " << vs.total_mac / 1000000 << " MMAC, hist(ms) <1:" << vs.hist[0]
                  << " 1-5:" << vs.hist[1] << " 5-20:" << vs.hist[2] << " 20-50:" << vs.hist[3]
                  << " 50-200:" << vs.hist[4] << " >200:" << vs.hist[5] << "\n";

        // ---- per-layer comparison over common outputs
        struct Row { std::string name; Stats st; size_t n; };
        std::vector<Row> rows;
        for (auto& kv : baked_out) {
            auto it = packed_out.find(kv.first);
            if (it == packed_out.end() || it->second.size() != kv.second.size()) continue;
            rows.push_back({kv.first, compare(it->second.data(), kv.second.data(), kv.second.size()),
                            kv.second.size()});
        }
        std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.st.maxerr > b.st.maxerr; });
        double worst_corr = 1;
        for (auto& r : rows) worst_corr = std::min(worst_corr, r.st.corr);
        std::printf("{\"layers_compared\": %zu, \"worst_corr\": %.6f, \"worst_maxerr\": %.6g}\n",
                    rows.size(), worst_corr, rows.empty() ? 0.0 : rows[0].st.maxerr);
        std::cerr << "[layers] worst 10 by max abs err (packed vs baked E):\n";
        for (size_t i = 0; i < std::min<size_t>(rows.size(), 10); i++)
            std::cerr << "  " << rows[i].name << " n=" << rows[i].n << " maxerr=" << rows[i].st.maxerr
                      << " corr=" << rows[i].st.corr << " relrms=" << rows[i].st.relrms << "\n";
        std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.st.corr < b.st.corr; });
        std::cerr << "[layers] worst 5 by correlation:\n";
        for (size_t i = 0; i < std::min<size_t>(rows.size(), 5); i++)
            std::cerr << "  " << rows[i].name << " n=" << rows[i].n << " maxerr=" << rows[i].st.maxerr
                      << " corr=" << rows[i].st.corr << " relrms=" << rows[i].st.relrms << "\n";

        // ---- final audio vs fp32 reference (stage-1 denoise口径)
        Stats sp = compare(packed_out["audio"].data(), fz.data(), fz.size());
        if (skip_baked) {
            std::printf("{\"packed_vs_fp32\": {\"corr\": %.6f, \"relrms\": %.6f, \"maxerr\": %.6g}, "
                        "\"rss_mb\": %ld}\n", sp.corr, sp.relrms, sp.maxerr, rss_kb() / 1024);
            return 0;
        }
        Stats sb = compare(baked_out["audio"].data(), fz.data(), fz.size());
        Stats pp = compare(packed_out["audio"].data(), baked_out["audio"].data(), fz.size());
        std::printf("{\"baked_vs_fp32\": {\"corr\": %.6f, \"relrms\": %.6f, \"maxerr\": %.6g}, "
                    "\"packed_vs_fp32\": {\"corr\": %.6f, \"relrms\": %.6f, \"maxerr\": %.6g}, "
                    "\"packed_vs_baked\": {\"corr\": %.6f, \"relrms\": %.6f, \"maxerr\": %.6g}, "
                    "\"rss_mb\": %ld}\n",
                    sb.corr, sb.relrms, sb.maxerr, sp.corr, sp.relrms, sp.maxerr, pp.corr, pp.relrms,
                    pp.maxerr, rss_kb() / 1024);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "vits_check error: " << e.what() << "\n";
        return 1;
    }
}
