// Packed-resident decoder weights + fused dequant GEMV/GEMM (Stage 3).
// The ORT-free core (thread pool, gemm kernels, BlobStore) lives in
// packed_core.h; this header adds the ORT custom-op glue,
// WeightPool and NamedSession on top of it.
#pragma once
#include <onnxruntime_cxx_api.h>

#include "packed_core.h"

namespace packed {

// ---------------- custom op (classic OrtCustomOp) ----------------
struct PMKernel {
    int64_t mode = 0, rows = 0, cols = 0, has_zero = 0, has_bias = 0;
};

inline const OrtApi* g_api = nullptr;

inline void* ORT_API_CALL pm_create(const OrtCustomOp* /*op*/, const OrtApi* api, const OrtKernelInfo* info) {
    auto* k = new PMKernel();
    api->KernelInfoGetAttribute_int64(info, "mode", &k->mode);
    api->KernelInfoGetAttribute_int64(info, "rows", &k->rows);
    api->KernelInfoGetAttribute_int64(info, "cols", &k->cols);
    api->KernelInfoGetAttribute_int64(info, "has_zero", &k->has_zero);
    api->KernelInfoGetAttribute_int64(info, "has_bias", &k->has_bias);
    return k;
}
inline const char* ORT_API_CALL pm_name(const OrtCustomOp*) { return "PackedMatMul"; }
inline const char* ORT_API_CALL pm_ep(const OrtCustomOp*) { return nullptr; }
inline ONNXTensorElementDataType ORT_API_CALL pm_in_type(const OrtCustomOp*, size_t i) {
    // 0: X float, 1: pk uint8, 2: scale float, 3: zero uint8, 4: bias float
    return (i == 1 || i == 3) ? ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}
inline size_t ORT_API_CALL pm_in_count(const OrtCustomOp*) { return 5; }
inline ONNXTensorElementDataType ORT_API_CALL pm_out_type(const OrtCustomOp*, size_t) {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}
inline size_t ORT_API_CALL pm_out_count(const OrtCustomOp*) { return 1; }
inline OrtCustomOpInputOutputCharacteristic ORT_API_CALL pm_in_char(const OrtCustomOp*, size_t i) {
    return i >= 3 ? INPUT_OUTPUT_OPTIONAL : INPUT_OUTPUT_REQUIRED;
}
inline OrtCustomOpInputOutputCharacteristic ORT_API_CALL pm_out_char(const OrtCustomOp*, size_t) {
    return INPUT_OUTPUT_REQUIRED;
}

inline void ORT_API_CALL pm_compute(void* op_kernel, OrtKernelContext* ctx) {
    auto* k = (PMKernel*)op_kernel;
    const OrtApi* api = g_api;
    Ort::KernelContext kc(ctx);
    auto xinfo = kc.GetInput(0).GetTensorTypeAndShapeInfo();
    auto xshape = xinfo.GetShape();
    int64_t K = xshape.back();
    int64_t M = 1;
    for (size_t i = 0; i + 1 < xshape.size(); i++) M *= xshape[i];
    int64_t D = (k->mode == 0 || k->mode == 2) ? k->cols : k->rows;
    int64_t Kexp = (k->mode == 0 || k->mode == 2) ? k->rows : k->cols;
    if (K != Kexp) throw std::runtime_error("PackedMatMul: input last dim != weight rows/cols");
    const float* X = kc.GetInput(0).GetTensorData<float>();
    const uint8_t* pk = kc.GetInput(1).GetTensorData<uint8_t>();
    const float* sc = kc.GetInput(2).GetTensorData<float>();
    const uint8_t* ze = nullptr;
    const float* bias = nullptr;
    size_t in_idx = 3;
    if (k->has_zero) ze = kc.GetInput(in_idx++).GetTensorData<uint8_t>();
    if (k->has_bias) bias = kc.GetInput(in_idx++).GetTensorData<float>();
    auto oshape = xshape;
    oshape.back() = D;
    auto out = kc.GetOutput(0, oshape.data(), oshape.size());
    float* Y = out.GetTensorMutableData<float>();
    (void)api;
    switch (k->mode) {
        case 0: gemm_a_w4(X, M, K, D, pk, sc, ze, bias, Y); break;
        case 1: gemm_b_w4(X, M, K, D, pk, sc, ze, bias, Y); break;
        case 2: gemm_a_w8(X, M, K, D, pk, sc, bias, Y); break;
        case 3: gemm_b_w8(X, M, K, D, pk, sc, bias, Y); break;
        default: throw std::runtime_error("PackedMatMul: bad mode");
    }
}
inline void ORT_API_CALL pm_destroy(void* k) { delete (PMKernel*)k; }

inline OrtMemType ORT_API_CALL pm_in_memtype(const OrtCustomOp*, size_t) { return OrtMemTypeDefault; }
inline int ORT_API_CALL pm_variadic_min(const OrtCustomOp*) { return 0; }
inline int ORT_API_CALL pm_variadic_homo(const OrtCustomOp*) { return 1; }
inline int ORT_API_CALL pm_start_ver(const OrtCustomOp*) { return 1; }
inline int ORT_API_CALL pm_end_ver(const OrtCustomOp*) { return 1 << 30; }
inline size_t ORT_API_CALL pm_no_pairs(int**, int**) { return 0; }
inline void ORT_API_CALL pm_release_pairs(int*, int*) {}

inline const OrtCustomOp& packed_op() {
    static OrtCustomOp op = [] {
        OrtCustomOp o{};
        o.version = ORT_API_VERSION;
        o.CreateKernel = pm_create;
        o.GetName = pm_name;
        o.GetExecutionProviderType = pm_ep;
        o.GetInputType = pm_in_type;
        o.GetInputTypeCount = pm_in_count;
        o.GetOutputType = pm_out_type;
        o.GetOutputTypeCount = pm_out_count;
        o.KernelCompute = pm_compute;
        o.KernelDestroy = pm_destroy;
        o.GetInputCharacteristic = pm_in_char;
        o.GetOutputCharacteristic = pm_out_char;
        // ORT 1.22 calls some of these unconditionally during session init;
        // leaving them null segfaults. Fill every slot with a safe default.
        o.GetInputMemoryType = pm_in_memtype;
        o.GetVariadicInputMinArity = pm_variadic_min;
        o.GetVariadicInputHomogeneity = pm_variadic_homo;
        o.GetVariadicOutputMinArity = pm_variadic_min;
        o.GetVariadicOutputHomogeneity = pm_variadic_homo;
        o.GetStartVersion = pm_start_ver;
        o.GetEndVersion = pm_end_ver;
        o.GetMayInplace = pm_no_pairs;
        o.ReleaseMayInplace = pm_release_pairs;
        o.GetAliasMap = pm_no_pairs;
        o.ReleaseAliasMap = pm_release_pairs;
        return o;
    }();
    return op;
}

// Register the domain on session options; keep the returned domain alive via
// the caller's Ort::CustomOpDomain object.
inline Ort::CustomOpDomain make_domain() {
    Ort::CustomOpDomain domain("potato");
    domain.Add(&packed_op());
    return domain;
}

// ---------------- packed weight pool (descriptors, one per graph) --------
// Loads one pack section, interns its payloads into the (possibly shared)
// BlobStore, and hands out zero-copy Ort::Value views keyed by graph
// input name ("<tensor>__pk" etc.). Callers keep one pool per graph
// because the NAME -> tensor mapping is per-graph (the same name may
// legitimately map to different bytes in another graph's pack); only the
// storage is shared, by content.
class WeightPool {
  public:
    explicit WeightPool(std::shared_ptr<BlobStore> store = nullptr)
        : store_(store ? std::move(store) : std::make_shared<BlobStore>()) {}
    const std::shared_ptr<BlobStore>& store() const { return store_; }
    // manifest: parsed wo-packs/manifest.json; scheme key e.g. "w4".
    // sections: manifest sections to load (one per pool in practice).
    void load(const std::string& root, const Json& manifest, const std::string& scheme,
              const std::vector<std::string>& sections = {"gpt_first", "gpt_stage"}) {
        size_t logical = 0;
        for (auto sec : sections) {
            const Json* secnode = manifest.find(sec.c_str())->find(scheme.c_str());
            std::string packfile = secnode->find("pack_file")->as_str();
            if (!packfile.empty() && packfile[0] != '/') packfile = root + "/wo-packs/" + packfile;
            Npz npz;
            npz.load(packfile);
            for (auto& e : secnode->find("manifest")->arr) {
                std::string name = e.find("name")->as_str();
                std::string sch = e.find("scheme")->as_str();
                if (!is_rewritten_scheme(sch)) continue;
                if (!wanted(name)) continue;
                // Deduped pack layout: both decoder sections may point at one
                // shared pack file; an entry whose payload is stored under a
                // different name there carries "pack_name" (see STAGE3.md S9).
                const Json* pn = e.find("pack_name");
                std::string lookup = pn ? pn->as_str() : name;
                const NpyTensor& pk = npz.entries.at(lookup);
                const NpyTensor& sc = npz.entries.at(lookup + "@scale");
                const NpyTensor* ze = nullptr;
                if (sch.rfind("w4", 0) == 0) ze = &npz.entries.at(lookup + "@zero");
                logical += pk.data.size() + sc.data.size() + (ze ? ze->data.size() : 0);
                BlobStore::Canon c = store_->intern(pk, sc, ze);
                interned_++;
                add_desc(name + "__pk", c.pk, pk.data.size(), pk.shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
                add_desc(name + "__sc", c.sc, sc.data.size(), sc.shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
                if (ze) add_desc(name + "__ze", c.ze, ze->data.size(), ze->shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
            }
        }
        logical_bytes_ = logical;
    }

    static bool is_rewritten_scheme(const std::string& sch) {
        // Mirrors make_packed_graphs.py's rewrite set: only tensors consumed
        // by a MatMul/Gemm are in the graphs as packed inputs; the pool is
        // queried by input name, so extra entries would simply never be
        // looked up -- but loading them wastes memory. The graphs' sidecar
        // JSONs are the authority; load() is only called with names filtered
        // by the caller through filter_names(). Default: accept w4/w8.
        return sch.rfind("w4", 0) == 0 || sch.rfind("w8", 0) == 0;
    }

    // Restrict which tensor names get loaded (from the graphs' sidecars).
    void set_filter(std::map<std::string, bool> names) { filter_ = std::move(names); }
    bool wanted(const std::string& name) const { return filter_.empty() || filter_.count(name); }

    bool has(const std::string& input_name) const { return descs_.count(input_name) != 0; }
    // Fresh zero-copy view over the pooled bytes (Ort::Value is move-only,
    // so callers create a view per Run; the data itself is never copied).
    Ort::Value view(const std::string& input_name) const {
        const TensorDesc& d = descs_.at(input_name);
        return Ort::Value::CreateTensor(meminfo_, const_cast<void*>(d.data), d.bytes,
                                        d.shape.data(), d.shape.size(), d.type);
    }
    // Unique bytes are a property of the shared store, not of this pool:
    // with several pools on one store this reports the store-wide total.
    size_t unique_bytes() const { return store_->unique_bytes(); }
    size_t logical_bytes() const { return logical_bytes_; }
    size_t entries() const { return interned_; }

  private:
    struct TensorDesc {
        const void* data = nullptr;
        size_t bytes = 0;
        std::vector<int64_t> shape;
        ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    };
    void add_desc(const std::string& name, const void* data, size_t bytes, const std::vector<int64_t>& shape,
                  ONNXTensorElementDataType type) {
        if (descs_.count(name)) return;
        descs_[name] = TensorDesc{data, bytes, shape, type};
    }

    std::shared_ptr<BlobStore> store_;
    std::map<std::string, TensorDesc> descs_;
    std::map<std::string, bool> filter_;
    size_t logical_bytes_ = 0;
    size_t interned_ = 0;
    Ort::MemoryInfo meminfo_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
};

// ---------------- name-bound session helper ----------------
struct NamedSession {
    std::unique_ptr<Ort::Session> session;
    std::vector<std::string> in_names, out_names;
    std::vector<const char*> in_ptrs, out_ptrs;

    void init(Ort::Env& env, const void* data, size_t size, Ort::SessionOptions& opts) {
        session = std::make_unique<Ort::Session>(env, data, size, opts);
        Ort::AllocatorWithDefaultOptions alloc;
        for (size_t i = 0; i < session->GetInputCount(); i++) in_names.push_back(session->GetInputNameAllocated(i, alloc).get());
        for (size_t i = 0; i < session->GetOutputCount(); i++) out_names.push_back(session->GetOutputNameAllocated(i, alloc).get());
        for (auto& n : in_names) in_ptrs.push_back(n.c_str());
        for (auto& n : out_names) out_ptrs.push_back(n.c_str());
    }
    // in: name -> owned state value (moved in). Any graph input not present
    // in `in` is taken as a fresh view from `pool` (packed weight inputs).
    // Returns name -> value for all outputs.
    std::map<std::string, Ort::Value> run(std::map<std::string, Ort::Value>& in, const WeightPool* pool = nullptr) {
        std::vector<Ort::Value> inv;
        inv.reserve(in_names.size());
        for (auto& n : in_names) {
            auto it = in.find(n);
            if (it != in.end()) {
                inv.push_back(std::move(it->second));
            } else if (pool && pool->has(n)) {
                inv.push_back(pool->view(n));
            } else {
                throw std::runtime_error("NamedSession: missing input " + n);
            }
        }
        in.clear();
        auto outs = session->Run(Ort::RunOptions{nullptr}, in_ptrs.data(), inv.data(), inv.size(),
                                 out_ptrs.data(), out_names.size());
        std::map<std::string, Ort::Value> out;
        for (size_t i = 0; i < outs.size(); i++) out.emplace(out_names[i], std::move(outs[i]));
        return out;
    }
};

}  // namespace packed
