// PotatoAmika TTS host (Stage 2): GPT-SoVITS synthesis with a host-controlled
// decode loop. Loads weight-only packs (W4 or W8 GPT, W8 vits), dequantizes
// them in-process and injects the fp32 tensors into the fp32 graph templates
// from Stage 1. Reference conditioning (incl. prompt-encoder outputs) comes
// from offline .rpk packs; prompt_encoder never runs at runtime.
// Interface: resident process, stdin one JSON request per line, one wav file
// per request into --outdir, one JSON result line on stdout per request.
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>

#include "cpu.h"
#include "deess.h"
#include "free_engine.h"
#include "free_pool.h"
#include "g2p.h"
#include "json_min.h"
#include "npz.h"
#include "packed.h"
#include "patch.h"
#include "rpk.h"
#include "vits_packed.h"

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}
static long rss_kb() {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) if (line.rfind("VmRSS:", 0) == 0) return std::stol(line.substr(6));
    return -1;
}
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
static void write_file(const std::string& path, const uint8_t* data, size_t n) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write: " + path);
    f.write((const char*)data, n);
}

// ---------------- model loading ----------------
struct SessionPack {
    std::unique_ptr<Ort::Session> session;
    std::vector<std::string> in_names, out_names;
    std::vector<const char*> in_ptrs, out_ptrs;
};

static std::string load_and_patch(const std::string& template_path, const std::string& pack_path,
                                  const Json& manifest_sec, const char* scheme,
                                  std::string& report) {
    auto modelb = read_file(template_path);
    std::string model(modelb.begin(), modelb.end());
    std::map<std::string, RestoredTensor> repl;
    if (!pack_path.empty()) {
        Npz npz;
        npz.load(pack_path);
        std::vector<std::pair<std::string, std::string>> name_scheme;
        std::map<std::string, std::vector<int64_t>> shapes;
        const Json* entries = manifest_sec.find("manifest");
        if (!entries || entries->type != Json::Arr) throw std::runtime_error("manifest: no entries list");
        for (auto& e : entries->arr) {
            std::string name = e.find("name")->as_str();
            std::string sch = e.find("scheme")->as_str();
            std::vector<int64_t> shp;
            for (auto& d : e.find("shape")->arr) shp.push_back((int64_t)d.as_num());
            name_scheme.push_back({name, sch});
            shapes[name] = shp;
        }
        repl = restore_pack(npz, name_scheme, shapes);
    }
    int replaced = 0;
    size_t bytes = 0;
    auto patched = pb::patch_model(model, repl, replaced, bytes);
    std::ostringstream rs;
    rs << template_path.substr(template_path.find_last_of('/') + 1) << ": template " << model.size()
       << " B, pack " << (pack_path.empty() ? std::string("(none)") : pack_path.substr(pack_path.find_last_of('/') + 1))
       << " scheme " << scheme << ", " << repl.size() << " pack entries, " << replaced
       << " initializers replaced (" << bytes << " B fp32 injected)";
    report = rs.str();
    return patched;
}

static SessionPack make_session(Ort::Env& env, const std::string& model, int threads,
                                bool no_arena = false, const std::string& opt_level = "all") {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(threads);
    if (no_arena) opts.DisableCpuMemArena();
    opts.SetGraphOptimizationLevel(opt_level == "basic" ? GraphOptimizationLevel::ORT_ENABLE_BASIC
                                                        : GraphOptimizationLevel::ORT_ENABLE_ALL);
    SessionPack sp;
    sp.session = std::make_unique<Ort::Session>(env, model.data(), model.size(), opts);
    Ort::AllocatorWithDefaultOptions alloc;
    for (size_t i = 0; i < sp.session->GetInputCount(); i++) sp.in_names.push_back(sp.session->GetInputNameAllocated(i, alloc).get());
    for (size_t i = 0; i < sp.session->GetOutputCount(); i++) sp.out_names.push_back(sp.session->GetOutputNameAllocated(i, alloc).get());
    for (auto& n : sp.in_names) sp.in_ptrs.push_back(n.c_str());
    for (auto& n : sp.out_names) sp.out_ptrs.push_back(n.c_str());
    return sp;
}

static Ort::Value f32_tensor(Ort::AllocatorWithDefaultOptions& alloc, const float* data, std::vector<int64_t> shape) {
    size_t n = 1;
    for (auto d : shape) n *= (size_t)d;
    auto v = Ort::Value::CreateTensor<float>(alloc, shape.data(), shape.size());
    memcpy(v.GetTensorMutableData<float>(), data, n * 4);
    return v;
}
static Ort::Value i64_tensor(Ort::AllocatorWithDefaultOptions& alloc, const int64_t* data, std::vector<int64_t> shape) {
    size_t n = 1;
    for (auto d : shape) n *= (size_t)d;
    auto v = Ort::Value::CreateTensor<int64_t>(alloc, shape.data(), shape.size());
    memcpy(v.GetTensorMutableData<int64_t>(), data, n * 8);
    return v;
}

// ---------------- ORT-free engine state (--engine free) ----------------
// The free engine runs the same packed-residency configuration as the
// ORT --packed path (W4 GPT pools + E vits pool over one BlobStore),
// with plans compiled from the same graph files. --vits-baked in free
// mode runs the baked fp32 vits graph on the free engine instead: the
// template's initializers are replaced in place by the E pack's
// restored fp32 tensors (the same values load_and_patch injects for
// ORT), so the plan sees exactly the ORT baked model's weights.
struct FreeHost {
    std::shared_ptr<packed::BlobStore> store;
    std::unique_ptr<FreePool> pool_f, pool_s, pool_v;
    std::shared_ptr<freeeng::Plan> enc, first, stage_a, stage_b, vits;
    bool vits_baked = false;
};

static void patch_graph_inplace(onnxm::Graph& g, const std::string& root, const std::string& pack_path,
                                const Json& manifest_sec) {
    Npz npz;
    npz.load(pack_path);
    std::vector<std::pair<std::string, std::string>> name_scheme;
    std::map<std::string, std::vector<int64_t>> shapes;
    const Json* entries = manifest_sec.find("manifest");
    if (!entries || entries->type != Json::Arr) throw std::runtime_error("manifest: no entries list");
    for (auto& e : entries->arr) {
        std::string name = e.find("name")->as_str();
        name_scheme.push_back({name, e.find("scheme")->as_str()});
        std::vector<int64_t> shp;
        for (auto& d : e.find("shape")->arr) shp.push_back((int64_t)d.as_num());
        shapes[name] = shp;
    }
    auto repl = restore_pack(npz, name_scheme, shapes);
    int replaced = 0;
    for (auto& t : g.initializers) {
        auto it = repl.find(t.name);
        if (it == repl.end()) continue;
        t.data_type = 1;
        t.dims = it->second.shape;
        t.data.assign((const uint8_t*)it->second.data.data(),
                      (const uint8_t*)it->second.data.data() + it->second.data.size() * 4);
        replaced++;
    }
    if (replaced != (int)repl.size())
        throw std::runtime_error("patch_graph_inplace: replaced " + std::to_string(replaced) + " of " +
                                 std::to_string(repl.size()) + " pack entries");
}

static freeeng::Tensor ft_from_rpk(const RpkTensor& t) {
    freeeng::Tensor ft;
    ft.dtype = t.dtype == 1 ? 7 : 1;
    ft.data = t.data.data();
    ft.shape = t.shape;
    return ft;
}
static std::vector<float> ft_to_vec(const freeeng::Tensor& t) {
    std::vector<float> v;
    v.reserve((size_t)t.numel());
    const float* p = (const float*)t.data;
    if (t.contiguous()) {
        v.assign(p, p + t.numel());
        return v;
    }
    std::vector<int64_t> coord(t.shape.size(), 0);
    for (int64_t i = 0; i < t.numel(); i++) {
        int64_t off = 0;
        for (size_t d = 0; d < coord.size(); d++) off += coord[d] * t.strides[d];
        v.push_back(p[off]);
        for (int d = (int)coord.size() - 1; d >= 0; d--) {
            if (++coord[d] < t.shape[d]) break;
            coord[d] = 0;
        }
    }
    return v;
}

// ---------------- sampling ----------------
static int64_t sample_token(const float* logits, int n, int top_k, double top_p, double temperature,
                            std::mt19937_64& rng, bool greedy) {
    if (greedy || temperature <= 0.0) {
        return std::max_element(logits, logits + n) - logits;
    }
    std::vector<std::pair<float, int>> cand;
    cand.reserve(n);
    float mx = *std::max_element(logits, logits + n);
    for (int i = 0; i < n; i++) cand.push_back({(logits[i] - mx) / (float)temperature, i});
    if (top_k > 0 && top_k < n) {
        std::nth_element(cand.begin(), cand.begin() + top_k, cand.end(),
                         [](const auto& a, const auto& b) { return a.first > b.first; });
        cand.resize(top_k);
    }
    double sum = 0;
    std::vector<std::pair<double, int>> sp;
    for (auto& c : cand) { double p = std::exp((double)c.first); sp.push_back({p, c.second}); sum += p; }
    for (auto& pr : sp) pr.first /= sum;
    std::sort(sp.begin(), sp.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    std::vector<std::pair<double, int>> kept;
    double cum = 0;
    for (auto& pr : sp) {
        kept.push_back(pr);
        cum += pr.first;
        if (cum >= top_p) break;
    }
    double ksum = 0;
    for (auto& pr : kept) ksum += pr.first;
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    double u = uni(rng) * ksum, acc = 0;
    for (auto& pr : kept) { acc += pr.first; if (u <= acc) return pr.second; }
    return kept.back().second;
}

// ---------------- wav ----------------
static void write_wav(const std::string& path, const float* audio, size_t n, int sr) {
    std::ofstream f(path, std::ios::binary);
    auto w32 = [&](uint32_t v) { f.write((char*)&v, 4); };
    auto w16 = [&](uint16_t v) { f.write((char*)&v, 2); };
    f.write("RIFF", 4); w32(36 + (uint32_t)n * 2); f.write("WAVE", 4);
    f.write("fmt ", 4); w32(16); w16(1); w16(1); w32(sr); w32(sr * 2); w16(2); w16(16);
    f.write("data", 4); w32((uint32_t)n * 2);
    for (size_t i = 0; i < n; i++) {
        float x = std::max(-1.0f, std::min(1.0f, audio[i]));
        int16_t v = (int16_t)std::lrint(x * 32767.0f);
        f.write((char*)&v, 2);
    }
}

// ---- minimal wav IO for the offline --deess-file verification mode ----
static bool read_wav_any(const std::string& path, std::vector<float>& out, int& sr) {
    auto buf = read_file(path);
    if (buf.size() < 44 || memcmp(buf.data(), "RIFF", 4) != 0) return false;
    uint16_t fmt = 0, chans = 0, bits = 0;
    size_t pos = 12;
    const uint8_t* data = nullptr;
    size_t data_len = 0;
    while (pos + 8 <= buf.size()) {
        uint32_t csz = rd32(&buf[pos + 4]);
        if (memcmp(&buf[pos], "fmt ", 4) == 0) {
            fmt = rd16(&buf[pos + 8]); chans = rd16(&buf[pos + 10]);
            sr = (int)rd32(&buf[pos + 12]); bits = rd16(&buf[pos + 22]);
        } else if (memcmp(&buf[pos], "data", 4) == 0) {
            data = &buf[pos + 8]; data_len = csz;
        }
        pos += 8 + csz + (csz & 1);
    }
    if (!data || chans == 0) return false;
    if (fmt == 1 && bits == 16) {
        size_t frames = data_len / 2 / chans;
        out.resize(frames);
        for (size_t i = 0; i < frames; i++) out[i] = (int16_t)rd16(data + (i * chans) * 2) / 32768.0f;
    } else if (fmt == 3 && bits == 32) {
        size_t frames = data_len / 4 / chans;
        out.resize(frames);
        for (size_t i = 0; i < frames; i++) { float v; memcpy(&v, data + (i * chans) * 4, 4); out[i] = v; }
    } else return false;
    return true;
}
static void write_wav_f32(const std::string& path, const float* audio, size_t n, int sr) {
    std::ofstream f(path, std::ios::binary);
    auto w32 = [&](uint32_t v) { f.write((char*)&v, 4); };
    auto w16 = [&](uint16_t v) { f.write((char*)&v, 2); };
    f.write("RIFF", 4); w32(36 + (uint32_t)n * 4); f.write("WAVE", 4);
    f.write("fmt ", 4); w32(16); w16(3); w16(1); w32(sr); w32(sr * 4); w16(4); w16(32);
    f.write("data", 4); w32((uint32_t)n * 4);
    f.write((const char*)audio, n * 4);
}

// Trim leading/trailing near-silence, each side capped (shell behavior: up to 1.3 s).
static std::pair<size_t, size_t> trim_range(const float* a, size_t n, double max_trim_s, int sr) {
    float peak = 0;
    for (size_t i = 0; i < n; i++) peak = std::max(peak, std::fabs(a[i]));
    float thr = peak * 0.0032f;  // about -50 dB of peak
    size_t cap = (size_t)(max_trim_s * sr);
    size_t lead = 0;
    while (lead < n && std::fabs(a[lead]) <= thr && lead < cap) lead++;
    if (lead >= n) return {0, n};
    size_t trail = 0;
    while (trail < n - lead && std::fabs(a[n - 1 - trail]) <= thr && trail < cap) trail++;
    return {lead, n - trail};
}

int main(int argc, char** argv) {
    std::string root = ".";  // override with --root
    std::string outdir, emit_dir;
    int gpt_bits = 4, threads = 4;
    // --ort-threads: ORT intra-op threads for the PACKED sessions only
    // (baked sessions always use --threads). The packed graphs' heavy ops
    // run on the kernel pool (--threads); ORT's own spinning intra-op
    // threads oversubscribe the cores and roughly double packed vits
    // time (measured S10), so the packed default is 1. Numerics are
    // unaffected (thread count changes no result for these ops).
    int ort_threads = 1;
    // Stage 3: --packed runs the two GPT decoder sessions from packed graphs
    // (weights stay packed in per-graph pools, fused-dequant kernels; W4 only).
    // --no-arena / --opt-level exist to attribute RSS in the stage-3 ledger;
    // defaults reproduce the stage-2 session configuration exactly.
    bool use_packed = false, no_arena = false, vits_baked = false;
    std::string opt_level = "all";
    // --engine free: the ORT-free engine (ORT-FREE.md phase 2). GPT runs
    // packed residency (W4) and vits packed residency by default, or the
    // baked fp32 vits graph with --vits-baked -- all on free_engine
    // plans. The ORT path stays the default and is unchanged.
    std::string engine = "ort";
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--root") root = next();
        else if (a == "--engine") engine = next();
        else if (a == "--outdir") outdir = next();
        else if (a == "--gpt-bits") gpt_bits = std::stoi(next());
        else if (a == "--threads") threads = std::stoi(next());
        else if (a == "--packed") use_packed = true;
        else if (a == "--ort-threads") ort_threads = std::stoi(next());
        // --vits-baked: with --packed, keep vits on the baked path (stage-3
        // behaviour) while GPT stays packed. Attribution/debugging only;
        // the default with --packed is vits packed residency (S10).
        else if (a == "--vits-baked") vits_baked = true;
        else if (a == "--no-arena") no_arena = true;
        else if (a == "--opt-level") opt_level = next();
        else if (a == "--emit-models") emit_dir = next();
        else if (a == "--g2p-stdin") { /* handled below */ }
        else if (a == "--deess-file") { i += 2; /* handled below */ }
        else { std::cerr << "unknown arg: " << a << "\n"; return 2; }
    }
    // Offline verification mode: --deess-file in.wav out.wav applies the
    // deess2air output stage to a wav file (no models loaded).
    for (int i = 1; i + 2 < argc; i++) {
        if (std::string(argv[i]) == "--deess-file") {
            std::vector<float> sig;
            int fsr = 0;
            if (!read_wav_any(argv[i + 1], sig, fsr)) { std::cerr << "deess-file: cannot read wav\n"; return 2; }
            std::vector<float> out(sig.size());
            deess::deess2air(sig.data(), sig.size(), fsr, out.data());
            write_wav_f32(argv[i + 2], out.data(), out.size(), fsr);
            std::cerr << "deess-file: " << sig.size() << " samples @ " << fsr << " Hz processed\n";
            return 0;
        }
    }
    bool g2p_stdin = false;
    for (int i = 1; i < argc; i++) if (std::string(argv[i]) == "--g2p-stdin") g2p_stdin = true;
    if (g2p_stdin) {
        // Validation helper: one input line of text -> one JSON array of phone ids.
        G2P g2p;
        g2p.load(root + "/host/data");
        std::string l;
        while (std::getline(std::cin, l)) {
            if (l.empty()) continue;
            int fb = 0;
            auto ids = g2p.text_to_ids("." + l, fb);
            std::cout << "{\"ids\": [";
            for (size_t i = 0; i < ids.size(); i++) { if (i) std::cout << ","; std::cout << ids[i]; }
            std::cout << "], \"fallback\": " << fb << "}" << std::endl;
        }
        return 0;
    }
    if (outdir.empty() && emit_dir.empty()) { std::cerr << "--outdir required\n"; return 2; }
    if (gpt_bits != 4 && gpt_bits != 8) { std::cerr << "--gpt-bits must be 4 or 8\n"; return 2; }
    if (use_packed && gpt_bits != 4) { std::cerr << "--packed supports --gpt-bits 4 only (packed graphs are W4; W8 stays on the baked path)\n"; return 2; }
    if (opt_level != "all" && opt_level != "basic") { std::cerr << "--opt-level must be all or basic\n"; return 2; }
    if (engine != "ort" && engine != "free") { std::cerr << "--engine must be ort|free\n"; return 2; }
    bool engine_free = engine == "free";
    if (engine_free && gpt_bits != 4) { std::cerr << "--engine free supports --gpt-bits 4 only (packed graphs are W4)\n"; return 2; }
    if (engine_free && !emit_dir.empty()) { std::cerr << "--emit-models is ORT-path only\n"; return 2; }
    std::string scheme = gpt_bits == 4 ? "w4" : "w8";

    try {
        auto t_boot = Clock::now();
        Json man = JsonParser(read_text(root + "/wo-packs/manifest.json")).parse();

        Json man_hybrid = JsonParser(read_text(root + "/wo-packs/manifest_vits_w4hybrid.json")).parse();
        // vits always loads the variant-E hybrid pack (enc_p + quantizer codebook
        // in W8, the rest W4; see VITS-W4.md). GPT packs follow --gpt-bits.
        struct GSpec { const char* tpl; const char* sec; const char* packfmt; const char* schemekey; const Json* manifest; };
        std::vector<GSpec> specs = {
            {"t2s_encoder_fp32.onnx", "gpt_encoder", "gpt_encoder_w%d.npz", nullptr, &man},
            {"t2s_first_stage_decoder_fp32.onnx", "gpt_first", "gpt_first_w%d.npz", nullptr, &man},
            {"t2s_stage_decoder_fp32.onnx", "gpt_stage", "gpt_stage_w%d.npz", nullptr, &man},
            {"vits_wnfold.onnx", "sovits_vits", "sovits_vits_w4hybrid.npz", "w4hybrid", &man_hybrid},
        };
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "potato-tts-host");
        std::vector<SessionPack> sessions;
        // Packed GPT state (stage 3): one descriptor pool per graph, all
        // backed by ONE content-addressed BlobStore (see packed.h), plus
        // name-bound sessions for specs 1/2.
        auto blob_store = std::make_shared<packed::BlobStore>();
        packed::WeightPool pool_f(blob_store), pool_s(blob_store), pool_v(blob_store);
        packed::NamedSession gpt_first_p, gpt_stage_p, vits_p;
        bool vits_packed = false;
        Ort::CustomOpDomain domain(nullptr);
        bool pool_ready = false;
        cpu::log_features("host");
        std::cerr << "[load] gpt_bits=" << gpt_bits << " threads=" << threads
                  << " packed=" << (use_packed ? 1 : 0) << " arena=" << (no_arena ? "off" : "on")
                  << " opt=" << opt_level << " engine=" << engine << "\n";
        // ---- free-engine load (--engine free) ----
        FreeHost fh;
        if (engine_free) {
            packed::set_kernel_threads(threads);
            fh.store = std::make_shared<packed::BlobStore>();
            fh.pool_f = std::make_unique<FreePool>(fh.store);
            fh.pool_s = std::make_unique<FreePool>(fh.store);
            fh.pool_v = std::make_unique<FreePool>(fh.store);
            std::map<std::string, bool> filter;
            for (auto f : {"t2s_first_stage_decoder_fp32.onnx", "t2s_stage_decoder_fp32.onnx"}) {
                Json side = JsonParser(read_text(root + "/host/packed_graphs/" + f + ".json")).parse();
                for (auto& pi : side.find("packed_inputs")->arr) filter[pi.find("tensor")->as_str()] = true;
            }
            fh.pool_f->filter = filter;
            fh.pool_f->load(root, man, "w4", "gpt_first");
            fh.pool_s->filter = filter;
            fh.pool_s->load(root, man, "w4", "gpt_stage");
            {
                Json side = JsonParser(read_text(root + "/host/packed_graphs/vits_wnfold.onnx.json")).parse();
                for (auto& pi : side.find("packed_inputs")->arr) fh.pool_v->filter[pi.find("tensor")->as_str()] = true;
            }
            fh.pool_v->load(root, man_hybrid, "w4hybrid", "sovits_vits");
            std::cerr << "[pool] free: first entries " << fh.pool_f->entries << "; stage entries "
                      << fh.pool_s->entries << "; vits entries " << fh.pool_v->entries
                      << "; store unique " << fh.store->unique_bytes() << " B; rss " << rss_kb() / 1024
                      << " MB\n";
            auto compile = [&](const std::string& path, const std::string& label) {
                auto t0 = Clock::now();
                onnxm::Graph g = onnxm::load_model(path);
                auto p = freeeng::Plan::compile(g, label);
                std::cerr << "[load] free plan " << label << " compiled in " << (int)ms_since(t0)
                          << " ms, " << p->step_count() << " steps; rss " << rss_kb() / 1024 << " MB\n";
                return p;
            };
            fh.enc = compile(root + "/work/wo_graphs/w4/t2s_encoder_fp32.onnx", "gpt_encoder");
            fh.first = compile(root + "/host/packed_graphs/t2s_first_stage_decoder_fp32.onnx", "gpt_first");
            fh.first->set_packed_lookup(fh.pool_f->lookup());
            fh.stage_a = compile(root + "/host/packed_graphs/t2s_stage_decoder_fp32.onnx", "gpt_stage_a");
            fh.stage_a->set_packed_lookup(fh.pool_s->lookup());
            fh.stage_b = compile(root + "/host/packed_graphs/t2s_stage_decoder_fp32.onnx", "gpt_stage_b");
            fh.stage_b->set_packed_lookup(fh.pool_s->lookup());
            fh.vits_baked = vits_baked;
            if (vits_baked) {
                auto t0 = Clock::now();
                onnxm::Graph g = onnxm::load_model(root + "/work/folded/vits_wnfold.onnx");
                const Json* secnode = man_hybrid.find("sovits_vits")->find("w4hybrid");
                patch_graph_inplace(g, root, root + "/wo-packs/sovits_vits_w4hybrid.npz", *secnode);
                fh.vits = freeeng::Plan::compile(g, "vits_baked");
                std::cerr << "[load] free plan vits_baked compiled in " << (int)ms_since(t0) << " ms, "
                          << fh.vits->step_count() << " steps; rss " << rss_kb() / 1024 << " MB\n";
            } else {
                fh.vits = compile(root + "/host/packed_graphs/vits_wnfold.onnx", "vits");
                fh.vits->set_packed_lookup(fh.pool_v->lookup());
            }
        }
        int spec_idx = 0;
        for (auto& gs : specs) {
            if (engine_free) break;
            int idx = spec_idx++;
            if (use_packed && (idx == 1 || idx == 2)) {
                if (!pool_ready) {
                    packed::set_kernel_threads(threads);
                    std::map<std::string, bool> filter;
                    for (auto f : {"t2s_first_stage_decoder_fp32.onnx", "t2s_stage_decoder_fp32.onnx"}) {
                        Json side = JsonParser(read_text(root + "/host/packed_graphs/" + f + ".json")).parse();
                        for (auto& pi : side.find("packed_inputs")->arr) filter[pi.find("tensor")->as_str()] = true;
                    }
                    pool_f.set_filter(filter);
                    pool_f.load(root, man, "w4", {"gpt_first"});
                    pool_s.set_filter(filter);
                    pool_s.load(root, man, "w4", {"gpt_stage"});
                    domain = packed::make_domain_full();
                    pool_ready = true;
                    std::cerr << "[pool] first: entries " << pool_f.entries()
                              << "; stage: entries " << pool_s.entries()
                              << "; store unique " << blob_store->unique_bytes()
                              << " B; logical " << pool_f.logical_bytes() + pool_s.logical_bytes()
                              << " B; rss " << rss_kb() / 1024 << " MB\n";
                }
                Ort::SessionOptions opts;
                opts.SetIntraOpNumThreads(ort_threads);
                if (no_arena) opts.DisableCpuMemArena();
                opts.SetGraphOptimizationLevel(opt_level == "basic" ? GraphOptimizationLevel::ORT_ENABLE_BASIC
                                                                    : GraphOptimizationLevel::ORT_ENABLE_ALL);
                opts.Add(domain);
                auto bytes = read_file(root + "/host/packed_graphs/" + gs.tpl);
                auto t0 = Clock::now();
                (idx == 1 ? gpt_first_p : gpt_stage_p).init(env, bytes.data(), bytes.size(), opts);
                std::cerr << "[load] packed session " << gs.tpl << " ready in " << (int)ms_since(t0)
                          << " ms; rss " << rss_kb() / 1024 << " MB\n";
                continue;
            }
            if (use_packed && idx == 3 && !vits_baked) {
                // vits packed residency (S10 step 2): E-pack tensors stay
                // packed in pool_v (shared store); the graph runs
                // potato::PackedConv / PackedConvTranspose / PackedGather.
                if (!pool_ready) { std::cerr << "internal: domain not ready for vits\n"; return 1; }
                Json side = JsonParser(read_text(root + "/host/packed_graphs/vits_wnfold.onnx.json")).parse();
                std::map<std::string, bool> filter;
                for (auto& pi : side.find("packed_inputs")->arr) filter[pi.find("tensor")->as_str()] = true;
                pool_v.set_filter(filter);
                pool_v.load(root, man_hybrid, "w4hybrid", {"sovits_vits"});
                Ort::SessionOptions opts;
                opts.SetIntraOpNumThreads(ort_threads);
                if (no_arena) opts.DisableCpuMemArena();
                opts.SetGraphOptimizationLevel(opt_level == "basic" ? GraphOptimizationLevel::ORT_ENABLE_BASIC
                                                                    : GraphOptimizationLevel::ORT_ENABLE_ALL);
                opts.Add(domain);
                auto bytes = read_file(root + "/host/packed_graphs/vits_wnfold.onnx");
                auto t0 = Clock::now();
                vits_p.init(env, bytes.data(), bytes.size(), opts);
                vits_packed = true;
                std::cerr << "[load] packed session vits_wnfold.onnx ready in " << (int)ms_since(t0)
                          << " ms; pool_v entries " << pool_v.entries() << "; store unique "
                          << blob_store->unique_bytes() << " B; rss " << rss_kb() / 1024 << " MB\n";
                continue;
            }
            char packname[64];
            if (strstr(gs.packfmt, "%d")) snprintf(packname, 64, gs.packfmt, gpt_bits);
            else snprintf(packname, 64, "%s", gs.packfmt);
            std::string rep;
            const Json* secnode = gs.manifest->find(gs.sec)->find(gs.schemekey ? gs.schemekey : scheme.c_str());
            auto patched = load_and_patch(root + "/work/folded/" + gs.tpl, root + "/wo-packs/" + packname,
                                          *secnode, gs.schemekey ? gs.schemekey : scheme.c_str(), rep);
            std::cerr << "[load] " << rep << "\n";
            if (!emit_dir.empty()) {
                write_file(emit_dir + "/patched_" + gs.tpl, (const uint8_t*)patched.data(), patched.size());
                continue;
            }
            auto t0 = Clock::now();
            sessions.push_back(make_session(env, patched, threads, no_arena, opt_level));
            std::cerr << "[load] session ready in " << (int)ms_since(t0) << " ms; rss " << rss_kb() / 1024 << " MB\n";
        }
        // In packed mode with baked vits, sessions holds [encoder, vits];
        // with vits packed residency it holds [encoder] only; baked mode
        // holds all 4. vits_sess is only dereferenced when !vits_packed
        // (and never in free mode, which has no ORT sessions at all).
        SessionPack* vits_sess_p = nullptr;
        if (!engine_free) vits_sess_p = &(vits_packed ? sessions[0] : (use_packed ? sessions[1] : sessions[3]));
        if (!emit_dir.empty()) { std::cerr << "[emit] patched models written to " << emit_dir << "\n"; return 0; }

        G2P g2p;
        g2p.load(root + "/host/data");
        Json index_doc = JsonParser(read_text(root + "/host/refpacks/index.json")).parse();
        std::cerr << "[load] g2p + refpack index ready; boot " << (int)ms_since(t_boot) << " ms; rss " << rss_kb() / 1024 << " MB\n";

        Ort::AllocatorWithDefaultOptions alloc;
        std::map<std::string, std::map<std::string, RpkTensor>> refcache;
        auto get_ref = [&](const std::string& tag) -> const std::map<std::string, RpkTensor>& {
            auto it = refcache.find(tag);
            if (it != refcache.end()) return it->second;
            const Json* node = index_doc.find(tag);
            if (!node) throw std::runtime_error("tag not in index: " + tag);
            auto rp = load_rpk(root + "/host/refpacks/" + node->find("file")->as_str());
            refcache[tag] = std::move(rp);
            return refcache[tag];
        };

        const int SR = 32000, EOS_ID = 1024, VOCAB = 1025, MAX_STEPS = 500;
        std::string line;
        long req_no = 0;
        while (std::getline(std::cin, line)) {
            if (line.empty()) continue;
            req_no++;
            auto t_req = Clock::now();
            try {
                Json r = JsonParser(line).parse();
                auto getstr = [&](const char* k, const std::string& d) { const Json* j = r.find(k); return j ? j->as_str(d) : d; };
                auto getnum = [&](const char* k, double d) { const Json* j = r.find(k); return j ? j->as_num(d) : d; };
                std::string text = getstr("text", "");
                std::string tag = getstr("tag", "neutral1");
                double speed = getnum("speed", 1.0);
                (void)speed;
                int top_k = (int)getnum("top_k", 20);
                double top_p = getnum("top_p", 0.9);
                double temperature = getnum("temperature", 0.85);
                long long seed = (long long)getnum("seed", 42);
                const Json* gj = r.find("greedy");
                bool greedy = gj && gj->type == Json::Bool && gj->b;
                std::vector<std::string> notes;
                if (r.find("speed")) notes.push_back("speed accepted but not applied: no graph input carries it");
                std::string tag_used = tag;
                if (!index_doc.find(tag)) { tag_used = "neutral1"; notes.push_back("tag '" + tag + "' missing, fell back to neutral1"); }
                const auto& ref = get_ref(tag_used);

                // ---- G2P (shell behavior: prepend ".") ----
                auto t0 = Clock::now();
                int fb = 0;
                auto text_ids = g2p.text_to_ids("." + text, fb);
                double t_g2p = ms_since(t0);
                int64_t L = text_ids.size();
                std::vector<float> text_bert((size_t)L * 1024, 0.0f);
                const auto& ref_seq = ref.at("ref_seq");
                const auto& ref_bert = ref.at("ref_bert");
                const auto& ssl = ref.at("ssl_content");

                // ---- GPT encoder ----
                t0 = Clock::now();
                std::vector<Ort::Value> enc_out;
                std::map<std::string, freeeng::Tensor> fenc_out;
                int64_t plen = 0;
                if (engine_free) {
                    std::map<std::string, freeeng::Tensor> ein;
                    ein.emplace("ref_seq", ft_from_rpk(ref_seq));
                    freeeng::Tensor tt; tt.dtype = 7; tt.data = (const uint8_t*)text_ids.data(); tt.shape = {1, L};
                    ein.emplace("text_seq", tt);
                    ein.emplace("ref_bert", ft_from_rpk(ref_bert));
                    freeeng::Tensor tb; tb.dtype = 1; tb.data = (const uint8_t*)text_bert.data(); tb.shape = {L, 1024};
                    ein.emplace("text_bert", tb);
                    ein.emplace("ssl_content", ft_from_rpk(ssl));
                    fenc_out = fh.enc->run(ein);
                    plen = fenc_out.at("prompts").shape[1];
                } else {
                    std::vector<Ort::Value> enc_in;
                    enc_in.push_back(i64_tensor(alloc, ref_seq.i64(), ref_seq.shape));
                    enc_in.push_back(i64_tensor(alloc, text_ids.data(), {1, L}));
                    enc_in.push_back(f32_tensor(alloc, ref_bert.f32(), ref_bert.shape));
                    enc_in.push_back(f32_tensor(alloc, text_bert.data(), {L, 1024}));
                    enc_in.push_back(f32_tensor(alloc, ssl.f32(), ssl.shape));
                    enc_out = sessions[0].session->Run(Ort::RunOptions{nullptr}, sessions[0].in_ptrs.data(), enc_in.data(), enc_in.size(), sessions[0].out_ptrs.data(), sessions[0].out_names.size());
                    plen = enc_out[1].GetTensorTypeAndShapeInfo().GetShape()[1];
                }
                double t_enc = ms_since(t0);

                // ---- first stage + decode loop ----
                std::vector<int64_t> gen;
                int steps = 0;
                double t_first = 0, t_dec = 0;
                if (engine_free) {
                    // Name-bound protocol on free plans (same as
                    // tf_check_free): stage state ping-pongs between
                    // two Plan instances so the previous step's outputs
                    // feed back as inputs without copies.
                    t0 = Clock::now();
                    std::map<std::string, freeeng::Tensor> fin;
                    fin.emplace("x", fenc_out.at("x"));
                    fin.emplace("prompts", fenc_out.at("prompts"));
                    auto fstate = fh.first->run(fin);
                    t_first = ms_since(t0);
                    fh.enc->release_arena();  // x/prompts were consumed by the first run
                    {
                        auto& y0 = fstate.at("y");
                        const int64_t* yd = (const int64_t*)y0.data;
                        gen.push_back(yd[y0.shape[1] - 1]);
                    }
                    t0 = Clock::now();
                    std::mt19937_64 rng((uint64_t)seed);
                    bool use_a = true;
                    while (gen.back() != EOS_ID && steps < MAX_STEPS) {
                        std::map<std::string, freeeng::Tensor> sin;
                        sin.emplace("iy", fstate.at("y"));
                        sin.emplace("iy_emb", fstate.at("y_emb"));
                        for (auto& kv : fstate) {
                            if (kv.first.rfind("present_k_", 0) == 0) sin.emplace("past_k_" + kv.first.substr(10), kv.second);
                            else if (kv.first.rfind("present_v_", 0) == 0) sin.emplace("past_v_" + kv.first.substr(10), kv.second);
                        }
                        fstate = (use_a ? fh.stage_a : fh.stage_b)->run(sin);
                        use_a = !use_a;
                        steps++;
                        auto& lg = fstate.at("logits_penalized");
                        const float* logits = (const float*)lg.data;
                        size_t lnum = (size_t)lg.numel();
                        const float* last_row = logits + (lnum - VOCAB);
                        int64_t nxt = sample_token(last_row, VOCAB, top_k, top_p, temperature, rng, greedy);
                        auto& yv = fstate.at("y");
                        int64_t* yd = (int64_t*)yv.data;
                        yd[yv.shape[1] - 1] = nxt;
                        gen.push_back(nxt);
                    }
                    t_dec = ms_since(t0);
                    fh.first->release_arena();  // stage plans own the live state now
                } else if (use_packed) {
                    // Name-bound protocol (same as tf_check): state tensors are
                    // carried by graph input/output name, weights come from
                    // each graph's own pool as zero-copy views on every Run.
                    t0 = Clock::now();
                    std::map<std::string, Ort::Value> pstate;
                    {
                        std::map<std::string, Ort::Value> fin;
                        std::vector<std::string> want;
                        for (auto& n : gpt_first_p.in_names) if (!pool_f.has(n)) want.push_back(n);
                        if (want.size() != 2) throw std::runtime_error("packed first stage: unexpected state inputs");
                        fin.emplace(want[0], std::move(enc_out[0]));
                        fin.emplace(want[1], std::move(enc_out[1]));
                        pstate = gpt_first_p.run(fin, &pool_f);
                    }
                    t_first = ms_since(t0);
                    {
                        auto& y0 = pstate.at(gpt_first_p.out_names[0]);
                        auto* yd = y0.GetTensorMutableData<int64_t>();
                        auto ys = y0.GetTensorTypeAndShapeInfo().GetShape();
                        gen.push_back(yd[ys[1] - 1]);
                    }
                    t0 = Clock::now();
                    std::mt19937_64 rng((uint64_t)seed);
                    const std::string& yname = gpt_stage_p.out_names[0];
                    while (gen.back() != EOS_ID && steps < MAX_STEPS) {
                        // Stage graph input names differ from the state/output
                        // names: iy <- y, iy_emb <- y_emb, past_* <- present_*
                        // (same rename tf_check uses; baked mode hides it by
                        // feeding positionally).
                        std::map<std::string, Ort::Value> sin;
                        sin["iy"] = std::move(pstate.at("y"));
                        sin["iy_emb"] = std::move(pstate.at("y_emb"));
                        for (auto& [n, v] : pstate) {
                            if (n.rfind("present_k_", 0) == 0) sin["past_k_" + n.substr(10)] = std::move(v);
                            else if (n.rfind("present_v_", 0) == 0) sin["past_v_" + n.substr(10)] = std::move(v);
                        }
                        pstate.clear();
                        pstate = gpt_stage_p.run(sin, &pool_s);
                        steps++;
                        auto& lg = pstate.at("logits_penalized");
                        const float* logits = lg.GetTensorData<float>();
                        size_t lnum = lg.GetTensorTypeAndShapeInfo().GetElementCount();
                        const float* last_row = logits + (lnum - VOCAB);
                        int64_t nxt = sample_token(last_row, VOCAB, top_k, top_p, temperature, rng, greedy);
                        auto& yv = pstate.at(yname);
                        auto* yd = yv.GetTensorMutableData<int64_t>();
                        auto ys = yv.GetTensorTypeAndShapeInfo().GetShape();
                        yd[ys[1] - 1] = nxt;
                        gen.push_back(nxt);
                    }
                    t_dec = ms_since(t0);
                } else {
                    t0 = Clock::now();
                    std::vector<Ort::Value> first_in;
                    first_in.push_back(std::move(enc_out[0]));
                    first_in.push_back(std::move(enc_out[1]));
                    auto cur = sessions[1].session->Run(Ort::RunOptions{nullptr}, sessions[1].in_ptrs.data(), first_in.data(), first_in.size(), sessions[1].out_ptrs.data(), sessions[1].out_names.size());
                    t_first = ms_since(t0);
                    // cur: [y, y_emb, kv x48]
                    {
                        auto* yd = cur[0].GetTensorMutableData<int64_t>();
                        auto ys = cur[0].GetTensorTypeAndShapeInfo().GetShape();
                        gen.push_back(yd[ys[1] - 1]);
                    }
                    t0 = Clock::now();
                    std::mt19937_64 rng((uint64_t)seed);
                    while (gen.back() != EOS_ID && steps < MAX_STEPS) {
                        std::vector<Ort::Value> st_in;
                        st_in.reserve(cur.size());
                        for (auto& v : cur) st_in.push_back(std::move(v));
                        auto out = sessions[2].session->Run(Ort::RunOptions{nullptr}, sessions[2].in_ptrs.data(), st_in.data(), st_in.size(), sessions[2].out_ptrs.data(), sessions[2].out_names.size());
                        steps++;
                        // out: [y, y_emb, stop, kv x48, logits]
                        // logits_penalized is a flat [1025] vector (last position only).
                        const float* logits = out.back().GetTensorData<float>();
                        size_t lnum = out.back().GetTensorTypeAndShapeInfo().GetElementCount();
                        const float* last_row = logits + (lnum - VOCAB);
                        int64_t nxt = sample_token(last_row, VOCAB, top_k, top_p, temperature, rng, greedy);
                        auto* yd = out[0].GetTensorMutableData<int64_t>();
                        auto ys = out[0].GetTensorTypeAndShapeInfo().GetShape();
                        yd[ys[1] - 1] = nxt;
                        gen.push_back(nxt);
                        cur.clear();
                        cur.push_back(std::move(out[0]));
                        cur.push_back(std::move(out[1]));
                        for (size_t k = 3; k + 1 < out.size(); k++) cur.push_back(std::move(out[k]));  // skip stop (idx2) and logits (last)
                    }
                    t_dec = ms_since(t0);
                }

                // pred_semantic = tokens before first EOS
                std::vector<int64_t> pred;
                for (auto tk : gen) { if (tk >= EOS_ID) break; pred.push_back(tk); }
                if (pred.empty()) throw std::runtime_error("empty semantic sequence");

                // ---- vits ----
                t0 = Clock::now();
                std::vector<Ort::Value> vits_in;
                vits_in.push_back(i64_tensor(alloc, text_ids.data(), {1, L}));
                vits_in.push_back(i64_tensor(alloc, pred.data(), {1, 1, (int64_t)pred.size()}));
                vits_in.push_back(f32_tensor(alloc, ref.at("ge").f32(), ref.at("ge").shape));
                vits_in.push_back(f32_tensor(alloc, ref.at("ge_advanced").f32(), ref.at("ge_advanced").shape));
                std::vector<Ort::Value> vout;
                Ort::Value vout_packed(nullptr);
                std::vector<float> audio_free;
                const float* audio = nullptr;
                size_t n_audio = 0;
                if (engine_free) {
                    std::map<std::string, freeeng::Tensor> vin;
                    freeeng::Tensor vt; vt.dtype = 7; vt.data = (const uint8_t*)text_ids.data(); vt.shape = {1, L};
                    vin.emplace("text_seq", vt);
                    freeeng::Tensor vp; vp.dtype = 7; vp.data = (const uint8_t*)pred.data(); vp.shape = {1, 1, (int64_t)pred.size()};
                    vin.emplace("pred_semantic", vp);
                    vin.emplace("ge", ft_from_rpk(ref.at("ge")));
                    vin.emplace("ge_advanced", ft_from_rpk(ref.at("ge_advanced")));
                    auto vout_f = fh.vits->run(vin);
                    audio_free = ft_to_vec(vout_f.at("audio"));
                    audio = audio_free.data();
                    n_audio = audio_free.size();
                    fh.vits->release_arena();  // audio is copied out; drop the vits peak
                } else if (vits_packed) {
                    std::map<std::string, Ort::Value> vin;
                    vin.emplace("text_seq", std::move(vits_in[0]));
                    vin.emplace("pred_semantic", std::move(vits_in[1]));
                    vin.emplace("ge", std::move(vits_in[2]));
                    vin.emplace("ge_advanced", std::move(vits_in[3]));
                    auto pout = vits_p.run(vin, &pool_v);
                    auto it = pout.find("audio");
                    if (it == pout.end()) throw std::runtime_error("packed vits: no audio output");
                    vout_packed = std::move(it->second);
                    audio = vout_packed.GetTensorData<float>();
                    n_audio = vout_packed.GetTensorTypeAndShapeInfo().GetElementCount();
                } else {
                    vout = vits_sess_p->session->Run(Ort::RunOptions{nullptr}, vits_sess_p->in_ptrs.data(), vits_in.data(), vits_in.size(), vits_sess_p->out_ptrs.data(), 1);
                    audio = vout[0].GetTensorData<float>();
                    n_audio = vout[0].GetTensorTypeAndShapeInfo().GetElementCount();
                }
                double t_vits = ms_since(t0);

                // Output stage (DEESS.md v2): deess2air before silence trimming.
                // Per-request field "deess" (default true) toggles it.
                const Json* dj = r.find("deess");
                bool use_deess = !(dj && dj->type == Json::Bool && !dj->b);
                std::vector<float> processed;
                const float* audio_out = audio;
                double t_deess = 0;
                if (use_deess) {
                    processed.assign(audio, audio + n_audio);
                    auto td = Clock::now();
                    deess::deess2air(processed.data(), n_audio, SR, processed.data());
                    t_deess = ms_since(td);
                    audio_out = processed.data();
                }

                auto [lo, hi] = trim_range(audio_out, n_audio, 1.3, SR);

                // Final stage (chain amendment): tanh soft clip on the trimmed
                // segment, RMS-preserving; per sentence, no carried state.
                // Request field "softclip" (default true) toggles it for A/B.
                const Json* sj = r.find("softclip");
                bool use_softclip = !(sj && sj->type == Json::Bool && !sj->b);
                std::vector<float> seg(audio_out + lo, audio_out + hi);
                deess::LevelStats lv{};
                if (use_softclip) {
                    lv = deess::softclip(seg.data(), seg.size());
                } else if (!seg.empty()) {
                    double pk = 0, acc = 0;
                    for (float v : seg) { double a = std::fabs((double)v); if (a > pk) pk = a; acc += (double)v * v; }
                    lv.peak_in = lv.peak_out = pk;
                    lv.rms_in = lv.rms_out = std::sqrt(acc / (double)seg.size());
                }
                auto crest_db = [](double pk, double rms) { return rms > 0 ? 20.0 * std::log10(pk / rms) : 0.0; };
                auto rms_db = [](double rms) { return rms > 0 ? 20.0 * std::log10(rms) : -120.0; };

                char fname[64];
                snprintf(fname, 64, "req%03ld.wav", req_no);
                std::string fpath = outdir + "/" + fname;
                write_wav(fpath, seg.data(), seg.size(), SR);

                std::ostringstream os;
                os << "{\"file\": \"" << fpath << "\", \"duration_s\": " << (hi - lo) / (double)SR
                   << ", \"elapsed_ms\": " << (long long)ms_since(t_req)
                   << ", \"steps\": " << steps << ", \"tokens\": [";
                for (size_t i = 0; i < gen.size(); i++) { if (i) os << ","; os << gen[i]; }
                os << "], \"tag_used\": \"" << tag_used << "\", \"g2p_fallback\": " << fb
                   << ", \"timing_ms\": {\"g2p\": " << t_g2p << ", \"encoder\": " << t_enc
                   << ", \"first\": " << t_first << ", \"decode\": " << t_dec
                   << ", \"decode_per_step\": " << (steps ? t_dec / steps : 0) << ", \"vits\": " << t_vits
                   << ", \"deess\": " << t_deess
                   << ", \"total\": " << ms_since(t_req) << "}, \"deess_applied\": " << (use_deess ? "true" : "false")
                   << ", \"softclip_applied\": " << (use_softclip ? "true" : "false")
                   << ", \"levels\": {\"peak_in\": " << lv.peak_in << ", \"peak_out\": " << lv.peak_out
                   << ", \"rms_in_db\": " << rms_db(lv.rms_in) << ", \"rms_out_db\": " << rms_db(lv.rms_out)
                   << ", \"crest_in_db\": " << crest_db(lv.peak_in, lv.rms_in)
                   << ", \"crest_out_db\": " << crest_db(lv.peak_out, lv.rms_out)
                   << ", \"drive\": " << lv.drive << "}"
                   << ", \"rss_mb\": " << rss_kb() / 1024.0
                   << ", \"notes\": [";
                for (size_t i = 0; i < notes.size(); i++) { if (i) os << ", "; os << "\"" << json_escape(notes[i]) << "\""; }
                os << "]}";
                std::cout << os.str() << std::endl;
            } catch (const std::exception& e) {
                std::cout << "{\"error\": \"" << json_escape(e.what()) << "\"}" << std::endl;
            }
        }
        if (engine_free) {
            auto mb = [](size_t b) { return b / 1048576.0; };
            std::cerr << "[free] final arenas MB: enc " << mb(fh.enc->arena_bytes()) << ", first "
                      << mb(fh.first->arena_bytes()) << ", stage_a " << mb(fh.stage_a->arena_bytes())
                      << ", stage_b " << mb(fh.stage_b->arena_bytes()) << ", vits "
                      << mb(fh.vits->arena_bytes()) << "; rss " << rss_kb() / 1024 << " MB\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "fatal: " << e.what() << "\n";
        return 1;
    }
}
