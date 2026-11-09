// Teacher-forcing validation for the ORT-free engine (free_engine),
// replicating tf_check.cpp's protocol exactly (see that file's header):
// same inputs, same seeding, same overwrite rule, same dump/ref format.
// The GPT encoder, first-stage and stage graphs all run on free_engine
// Plans; no onnxruntime session exists in this process (ldd shows no
// libonnxruntime). Modes mirror tf_check: baked = wo_graphs fp32 graphs
// compiled by the same engine (linear steps bind fp32 constant tensors),
// packed = packed_graphs + the shared BlobStore weight pools.
//
// Stage state ping-pongs between two Plan instances so the previous
// step's outputs can be fed back as inputs without copies (a Plan's
// output storage stays valid until that Plan runs again, and by then its
// outputs have been consumed by the other Plan's run).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>

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
static long rss_kb() {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) if (line.rfind("VmRSS:", 0) == 0) return std::stol(line.substr(6));
    return -1;
}

// FreePool lives in free_pool.h (shared with the host + vits_check_free).

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

int main(int argc, char** argv) {
    std::string root = ".";  // override with --root
    std::string mode = "packed", dump_path, ref_path;
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

        packed::set_kernel_threads(threads);
        using Clk = std::chrono::steady_clock;

        auto compile = [&](const std::string& path, const std::string& label) {
            auto t0 = Clk::now();
            onnxm::Graph g = onnxm::load_model(path);
            auto plan = freeeng::Plan::compile(g, label);
            auto ms = std::chrono::duration<double, std::milli>(Clk::now() - t0).count();
            std::cerr << "[compile] " << label << ": " << plan->step_count() << " steps in " << (long)ms << " ms\n";
            return plan;
        };

        auto enc = compile(root + "/work/wo_graphs/w4/t2s_encoder_fp32.onnx", "encoder");
        std::shared_ptr<freeeng::Plan> first, stageA, stageB;
        auto blob_store = std::make_shared<packed::BlobStore>();
        FreePool pool_f(blob_store), pool_s(blob_store);
        if (mode == "baked") {
            first = compile(root + "/work/wo_graphs/w4/t2s_first_stage_decoder_fp32.onnx", "first");
            stageA = compile(root + "/work/wo_graphs/w4/t2s_stage_decoder_fp32.onnx", "stageA");
            stageB = compile(root + "/work/wo_graphs/w4/t2s_stage_decoder_fp32.onnx", "stageB");
        } else {
            Json man = JsonParser(read_text(root + "/wo-packs/manifest.json")).parse();
            std::map<std::string, bool> filter;
            for (auto f : {"t2s_first_stage_decoder_fp32.onnx", "t2s_stage_decoder_fp32.onnx"}) {
                Json side = JsonParser(read_text(root + "/host/packed_graphs/" + f + ".json")).parse();
                for (auto& pi : side.find("packed_inputs")->arr) filter[pi.find("tensor")->as_str()] = true;
            }
            pool_f.filter = filter;
            pool_f.load(root, man, "w4", "gpt_first");
            pool_s.filter = filter;
            pool_s.load(root, man, "w4", "gpt_stage");
            std::cerr << "[pool] first: entries " << pool_f.entries << "; stage: entries " << pool_s.entries
                      << "; store unique " << blob_store->unique_bytes()
                      << " B; logical " << pool_f.logical + pool_s.logical << " B\n";
            first = compile(root + "/host/packed_graphs/t2s_first_stage_decoder_fp32.onnx", "first");
            stageA = compile(root + "/host/packed_graphs/t2s_stage_decoder_fp32.onnx", "stageA");
            stageB = compile(root + "/host/packed_graphs/t2s_stage_decoder_fp32.onnx", "stageB");
            first->set_packed_lookup(pool_f.lookup());
            stageA->set_packed_lookup(pool_s.lookup());
            stageB->set_packed_lookup(pool_s.lookup());
        }
        std::cerr << "[load] plans ready; rss " << rss_kb() / 1024 << " MB\n";

        std::vector<float> all_logits;
        long agree = 0, total = 0;
        std::vector<std::pair<int, int>> rows;

        int row_idx = 0;
        for (auto& ref : refs) {
            for (int s = 0; s < 3; s++) {
                if (row_idx++ >= max_rows) break;
                std::string key = ref + "_" + std::to_string(s);
                const Json* traj_j = fp32_gen->find(key);
                std::vector<int64_t> traj;
                for (auto& v : traj_j->arr) traj.push_back((int64_t)v.as_num());

                auto t_row0 = Clk::now();
                std::map<std::string, freeeng::Tensor> state;
                int64_t plen = 0;
                {
                    std::map<std::string, freeeng::Tensor> ein;
                    ein["ref_seq"] = npy_tensor(z.entries.at(ref + "_ref_seq"));
                    ein["text_seq"] = npy_tensor(z.entries.at("s" + std::to_string(s) + "_text_seq"));
                    ein["ref_bert"] = npy_tensor(z.entries.at(ref + "_ref_bert"));
                    ein["text_bert"] = npy_tensor(z.entries.at("s" + std::to_string(s) + "_text_bert"));
                    ein["ssl_content"] = npy_tensor(z.entries.at(ref + "_ssl_content"));
                    auto eout = enc->run(ein);
                    plen = eout.at("prompts").shape[1];
                    std::map<std::string, freeeng::Tensor> fin;
                    fin["x"] = eout.at("x");
                    fin["prompts"] = eout.at("prompts");
                    state = first->run(fin);
                }
                auto ydata = [&]() -> int64_t* {
                    return (int64_t*)const_cast<uint8_t*>(state.at("y").data);
                };
                {
                    auto sh = state.at("y").shape;
                    int64_t* d = ydata();
                    std::cerr << "[first-token] " << key << " model=" << d[sh[1] - 1]
                              << " teacher=" << traj[0] << "\n";
                    for (int64_t i = plen; i < sh[1]; i++) d[i] = traj[0];
                }
                auto t_init = Clk::now();
                int row_agree = 0;
                bool useA = true;
                for (size_t t = 0; t + 1 < traj.size(); t++) {
                    auto& plan = useA ? stageA : stageB;
                    useA = !useA;
                    std::map<std::string, freeeng::Tensor> sin;
                    for (auto& name : plan->input_names()) {
                        if (name == "iy") sin[name] = state.at("y");
                        else if (name == "iy_emb") sin[name] = state.at("y_emb");
                        else if (name.rfind("past_", 0) == 0) sin[name] = state.at("present_" + name.substr(5));
                        else throw std::runtime_error("unexpected stage input: " + name);
                    }
                    state = plan->run(sin);
                    auto& lb = state.at("logits_penalized");
                    size_t ln = (size_t)lb.numel();
                    const float* last = (const float*)lb.data + (ln - 1025);
                    all_logits.insert(all_logits.end(), last, last + 1025);
                    int64_t pred = std::max_element(last, last + 1025) - last;
                    if (pred == traj[t + 1]) { agree++; row_agree++; }
                    total++;
                    auto ysh = state.at("y").shape;
                    ydata()[ysh[1] - 1] = traj[t + 1];
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
            std::vector<float> refv(raw.size() / 4);
            memcpy(refv.data(), raw.data(), refv.size() * 4);
            if (refv.size() != all_logits.size()) throw std::runtime_error("ref dump size mismatch");
            maxdiff = 0;
            ref_argmax_agree = 0;
            for (size_t step = 0; step * 1025 < refv.size(); step++) {
                const float* a = &refv[step * 1025];
                const float* b = &all_logits[step * 1025];
                for (int i = 0; i < 1025; i++) maxdiff = std::max(maxdiff, (double)std::fabs(a[i] - b[i]));
                if ((std::max_element(a, a + 1025) - a) == (std::max_element(b, b + 1025) - b)) ref_argmax_agree++;
            }
        }
        if (!dump_path.empty()) {
            std::ofstream f(dump_path, std::ios::binary);
            f.write((const char*)all_logits.data(), all_logits.size() * 4);
        }
        std::cout << "{\"mode\": \"free-" << mode << "\", \"tf_agree\": " << agree << ", \"tf_total\": " << total
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
