// Teacher-forcing validation (Stage 3), replicating work/greedy.py's
// protocol exactly:
//   state = (y full sequence, y_emb, KV) from encoder + first-stage runs;
//   y[0, plen:] is seeded with the fp32 greedy trajectory's first token, then
//   each stage call's last-position logits argmax is compared against the
//   fp32 trajectory's next token and y[0, -1] is overwritten with it.
// Modes:
//   baked  -- stage-1 baked W4 graphs in work/wo_graphs/w4 (positional I/O,
//             the same files greedy.py used; baseline 299/361 = 82.83%).
//   packed -- first/stage graphs from host/packed_graphs with the shared
//             packed weight pool + potato::PackedMatMul (name-bound I/O).
// --dump writes every step's logits row (f32, 1025 floats/step, in
// ref/sentence/step order) for cross-build comparison; --ref reads one back
// and reports max |diff| and argmax agreement against it.
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>

#include "json_min.h"
#include "npz.h"
#include "packed.h"

static std::string read_text(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open: " + path);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
static long rss_kb() {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) if (line.rfind("VmRSS:", 0) == 0) return std::stol(line.substr(6));
    return -1;
}

struct PlainSession {
    std::unique_ptr<Ort::Session> s;
    std::vector<std::string> in_names, out_names;
    std::vector<const char*> in_ptrs, out_ptrs;
    void init(Ort::Env& env, const std::string& path, int threads) {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(threads);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        s = std::make_unique<Ort::Session>(env, path.c_str(), opts);
        Ort::AllocatorWithDefaultOptions alloc;
        for (size_t i = 0; i < s->GetInputCount(); i++) in_names.push_back(s->GetInputNameAllocated(i, alloc).get());
        for (size_t i = 0; i < s->GetOutputCount(); i++) out_names.push_back(s->GetOutputNameAllocated(i, alloc).get());
        for (auto& n : in_names) in_ptrs.push_back(n.c_str());
        for (auto& n : out_names) out_ptrs.push_back(n.c_str());
    }
    std::vector<Ort::Value> run(std::vector<Ort::Value>& in) {
        return s->Run(Ort::RunOptions{nullptr}, in_ptrs.data(), in.data(), in.size(), out_ptrs.data(), out_names.size());
    }
};

static Ort::Value tensor_from_npy(Ort::AllocatorWithDefaultOptions& alloc, const NpyTensor& t) {
    if (t.descr == "<f4") {
        auto v = Ort::Value::CreateTensor<float>(alloc, const_cast<int64_t*>(t.shape.data()), t.shape.size());
        memcpy(v.GetTensorMutableData<float>(), t.data.data(), t.data.size());
        return v;
    }
    if (t.descr == "<i8") {
        auto v = Ort::Value::CreateTensor<int64_t>(alloc, const_cast<int64_t*>(t.shape.data()), t.shape.size());
        memcpy(v.GetTensorMutableData<int64_t>(), t.data.data(), t.data.size());
        return v;
    }
    throw std::runtime_error("npz: unexpected descr " + t.descr);
}

int main(int argc, char** argv) {
    std::string root = ".";  // override with --root
    std::string mode = "baked", dump_path, ref_path;
    int threads = 4, max_rows = 6;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--root") root = next();
        else if (a == "--mode") mode = next();
        else if (a == "--threads") threads = std::stoi(next());
        else if (a == "--dump") dump_path = next();
        else if (a == "--ref") ref_path = next();
        else if (a == "--rows") max_rows = std::stoi(next());
        else { std::cerr << "unknown arg: " << a << "\n"; return 2; }
    }
    if (mode != "baked" && mode != "packed") { std::cerr << "--mode must be baked|packed\n"; return 2; }

    try {
        Npz z;
        z.load(root + "/work/inputs.npz");
        Json tm = JsonParser(read_text(root + "/work/token_metrics.json")).parse();
        const Json* fp32_gen = tm.find("fp32_gen");
        const std::vector<std::string> refs = {"neutral1", "caring1"};

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "tf-check");
        Ort::AllocatorWithDefaultOptions alloc;

        PlainSession enc;
        enc.init(env, root + "/work/wo_graphs/w4/t2s_encoder_fp32.onnx", threads);

        // Decoder sessions: two backends behind one step interface.
        PlainSession first_b, stage_b;
        packed::NamedSession first_p, stage_p;
        auto blob_store = std::make_shared<packed::BlobStore>();
        packed::WeightPool pool_f(blob_store), pool_s(blob_store);  // per-graph descriptors, shared content store
        Ort::CustomOpDomain domain(nullptr);
        if (mode == "baked") {
            first_b.init(env, root + "/work/wo_graphs/w4/t2s_first_stage_decoder_fp32.onnx", threads);
            stage_b.init(env, root + "/work/wo_graphs/w4/t2s_stage_decoder_fp32.onnx", threads);
        } else {
            packed::set_kernel_threads(threads);
            Json man = JsonParser(read_text(root + "/wo-packs/manifest.json")).parse();
            std::map<std::string, bool> filter;
            for (auto f : {"t2s_first_stage_decoder_fp32.onnx", "t2s_stage_decoder_fp32.onnx"}) {
                Json side = JsonParser(read_text(root + "/host/packed_graphs/" + f + ".json")).parse();
                for (auto& pi : side.find("packed_inputs")->arr) filter[pi.find("tensor")->as_str()] = true;
            }
            pool_f.set_filter(filter);
            pool_f.load(root, man, "w4", {"gpt_first"});
            pool_s.set_filter(filter);
            pool_s.load(root, man, "w4", {"gpt_stage"});
            std::cerr << "[pool] first: entries " << pool_f.entries()
                      << "; stage: entries " << pool_s.entries()
                      << "; store unique " << blob_store->unique_bytes()
                      << " B; logical " << pool_f.logical_bytes() + pool_s.logical_bytes() << " B\n";
            domain = packed::make_domain();
            auto mk = [&](packed::NamedSession& ns, const std::string& file) {
                Ort::SessionOptions opts;
                opts.SetIntraOpNumThreads(threads);
                opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
                opts.Add(domain);
                std::ifstream f(root + "/host/packed_graphs/" + file, std::ios::binary);
                std::string bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
                ns.init(env, bytes.data(), bytes.size(), opts);
            };
            mk(first_p, "t2s_first_stage_decoder_fp32.onnx");
            mk(stage_p, "t2s_stage_decoder_fp32.onnx");
        }
        std::cerr << "[load] sessions ready; rss " << rss_kb() / 1024 << " MB\n";

        std::vector<float> all_logits;
        long agree = 0, total = 0;
        std::vector<std::pair<int, int>> rows;
        using Clk = std::chrono::steady_clock;

        int row_idx = 0;
        for (auto& ref : refs) {
            for (int s = 0; s < 3; s++) {
                if (row_idx++ >= max_rows) break;
                std::string key = ref + "_" + std::to_string(s);
                const Json* traj_j = fp32_gen->find(key);
                std::vector<int64_t> traj;
                for (auto& v : traj_j->arr) traj.push_back((int64_t)v.as_num());

                // ---- initial state: encoder + first stage ----
                auto t_row0 = Clk::now();
                Ort::Value y, y_emb;
                std::vector<Ort::Value> kv;                    // baked: positional
                std::map<std::string, Ort::Value> state;       // packed: by name
                int64_t plen = 0;
                {
                    std::vector<Ort::Value> ein;
                    ein.push_back(tensor_from_npy(alloc, z.entries.at(ref + "_ref_seq")));
                    ein.push_back(tensor_from_npy(alloc, z.entries.at("s" + std::to_string(s) + "_text_seq")));
                    ein.push_back(tensor_from_npy(alloc, z.entries.at(ref + "_ref_bert")));
                    ein.push_back(tensor_from_npy(alloc, z.entries.at("s" + std::to_string(s) + "_text_bert")));
                    ein.push_back(tensor_from_npy(alloc, z.entries.at(ref + "_ssl_content")));
                    auto eout = enc.run(ein);
                    plen = eout[1].GetTensorTypeAndShapeInfo().GetShape()[1];
                    if (mode == "baked") {
                        std::vector<Ort::Value> fin;
                        fin.push_back(std::move(eout[0]));
                        fin.push_back(std::move(eout[1]));
                        auto fout = first_b.run(fin);
                        y = std::move(fout[0]);
                        y_emb = std::move(fout[1]);
                        for (size_t i = 2; i < fout.size(); i++) kv.push_back(std::move(fout[i]));
                    } else {
                        std::map<std::string, Ort::Value> fin;
                        fin["x"] = std::move(eout[0]);
                        fin["prompts"] = std::move(eout[1]);
                        auto fout = first_p.run(fin, &pool_f);
                        state = std::move(fout);  // y, y_emb, present_* by name
                    }
                }
                auto ydata = [&]() -> int64_t* {
                    if (mode == "baked") return y.GetTensorMutableData<int64_t>();
                    return state.at("y").GetTensorMutableData<int64_t>();
                };
                auto yshape = [&]() -> std::vector<int64_t> {
                    if (mode == "baked") return y.GetTensorTypeAndShapeInfo().GetShape();
                    return state.at("y").GetTensorTypeAndShapeInfo().GetShape();
                };
                // seed y[0, plen:] = traj[0]
                {
                    auto sh = yshape();
                    int64_t* d = ydata();
                    std::cerr << "[first-token] " << key << " model=" << d[sh[1] - 1]
                              << " teacher=" << traj[0] << "\n";
                    for (int64_t i = plen; i < sh[1]; i++) d[i] = traj[0];
                }
                auto t_init = Clk::now();
                int row_agree = 0;
                for (size_t t = 0; t + 1 < traj.size(); t++) {
                    if (mode == "baked") {
                        std::vector<Ort::Value> sin;
                        sin.push_back(std::move(y));
                        sin.push_back(std::move(y_emb));
                        for (auto& v : kv) sin.push_back(std::move(v));
                        kv.clear();
                        auto sout = stage_b.run(sin);
                        y = std::move(sout[0]);
                        y_emb = std::move(sout[1]);
                        for (size_t i = 3; i + 1 < sout.size(); i++) kv.push_back(std::move(sout[i]));
                        // logits_penalized: last row of the last output (greedy.py
                        // reshapes to (-1, 1025) and takes [-1]).
                        auto& lb = sout.back();
                        size_t ln = lb.GetTensorTypeAndShapeInfo().GetElementCount();
                        const float* last = lb.GetTensorData<float>() + (ln - 1025);
                        all_logits.insert(all_logits.end(), last, last + 1025);
                    } else {
                        std::map<std::string, Ort::Value> sin;
                        sin["iy"] = std::move(state.at("y"));
                        sin["iy_emb"] = std::move(state.at("y_emb"));
                        for (auto& [n, v] : state) {
                            if (n.rfind("present_k_", 0) == 0) sin["past_k_" + n.substr(10)] = std::move(v);
                            else if (n.rfind("present_v_", 0) == 0) sin["past_v_" + n.substr(10)] = std::move(v);
                        }
                        state.clear();
                        state = stage_p.run(sin, &pool_s);
                        auto& lb = state.at("logits_penalized");
                        size_t ln = lb.GetTensorTypeAndShapeInfo().GetElementCount();
                        const float* last = lb.GetTensorData<float>() + (ln - 1025);
                        all_logits.insert(all_logits.end(), last, last + 1025);
                    }
                    const float* last = &all_logits[all_logits.size() - 1025];
                    int64_t pred = std::max_element(last, last + 1025) - last;
                    if (pred == traj[t + 1]) { agree++; row_agree++; }
                    total++;
                    ydata()[yshape()[1] - 1] = traj[t + 1];
                }
                rows.push_back({row_agree, (int)traj.size() - 1});
                auto ms = [](Clk::time_point a, Clk::time_point b) {
                    return std::chrono::duration<double, std::milli>(b - a).count();
                };
                std::cerr << "[tf] " << key << " agree " << row_agree << "/" << (traj.size() - 1)
                          << "; init " << (long)ms(t_row0, t_init) << " ms, steps "
                          << (long)ms(t_init, Clk::now()) << " ms; rss " << rss_kb() / 1024 << " MB\n";
            }
        }

        double maxdiff = -1;
        long ref_argmax_agree = -1;
        if (!ref_path.empty()) {
            std::ifstream f(ref_path, std::ios::binary);
            std::vector<char> raw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            std::vector<float> ref(raw.size() / 4);
            memcpy(ref.data(), raw.data(), ref.size() * 4);
            if (ref.size() != all_logits.size()) throw std::runtime_error("ref dump size mismatch");
            maxdiff = 0;
            ref_argmax_agree = 0;
            for (size_t step = 0; step * 1025 < ref.size(); step++) {
                const float* a = &ref[step * 1025];
                const float* b = &all_logits[step * 1025];
                for (int i = 0; i < 1025; i++) maxdiff = std::max(maxdiff, (double)std::fabs(a[i] - b[i]));
                if ((std::max_element(a, a + 1025) - a) == (std::max_element(b, b + 1025) - b)) ref_argmax_agree++;
            }
        }
        if (!dump_path.empty()) {
            std::ofstream f(dump_path, std::ios::binary);
            f.write((const char*)all_logits.data(), all_logits.size() * 4);
        }
        std::cout << "{\"mode\": \"" << mode << "\", \"tf_agree\": " << agree << ", \"tf_total\": " << total
                  << ", \"tf_rate\": " << (total ? (double)agree / total : 0)
                  << ", \"steps\": " << all_logits.size() / 1025;
        if (maxdiff >= 0) std::cout << ", \"ref_max_logit_diff\": " << maxdiff << ", \"ref_argmax_agree\": " << ref_argmax_agree;
        std::cout << ", \"rss_mb\": " << rss_kb() / 1024.0 << "}" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "fatal: " << e.what() << "\n";
        return 1;
    }
}
