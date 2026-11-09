// ORT-free static-plan inference engine (see ORT-FREE.md for the design).
//
// A graph is compiled once into a Plan: a fixed sequence of kernel steps
// over values whose shapes are affine expressions of a few runtime dim
// symbols (for the GPT stage graph: just the sequence length). Each call
// binds concrete input tensors, resolves shapes numerically, assigns a
// static arena by value lifetime, and runs the steps with no per-op
// dispatch beyond a switch. Layout ops are views (offset + strides), not
// copies. PackedMatMul steps call the existing packed::gemm_* kernels
// through a caller-supplied weight lookup, so this engine consumes the
// same weight packs and BlobStore as the ORT host. The baked (fp32)
// form of a graph compiles through the identical path: its linear steps
// bind fp32 constant tensors instead of packed ones.
#pragma once
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "onnx_model.h"

namespace freeeng {

// Tensor view exchanged with the caller. Strides are in elements; an
// empty strides vector means contiguous row-major.
struct Tensor {
    int dtype = 0;  // ONNX elem type (1 f32, 2 u8, 6 i32, 7 i64, 9 bool)
    const uint8_t* data = nullptr;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    int64_t numel() const {
        int64_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }
    bool contiguous() const { return strides.empty(); }
};

// Resolved packed weight for one PackedMatMul node base name.
struct PackedBinding {
    const uint8_t* pk = nullptr;
    const float* sc = nullptr;
    const uint8_t* ze = nullptr;  // null for W8 modes
};
// Given the graph tensor base name (PackedMatMul input names carry
// __pk/__sc/__ze suffixes; the base is what remains after stripping),
// fill the binding. Return false if the name is unknown.
using PackedLookup = std::function<bool(const std::string& base, PackedBinding& out)>;

class Plan {
  public:
    // Compile from a parsed graph. Throws std::runtime_error on any op
    // or pattern outside the implemented set (the census set).
    static std::shared_ptr<Plan> compile(const onnxm::Graph& g, std::string label);
    void set_packed_lookup(PackedLookup fn) { lookup_ = std::move(fn); }

    struct Instance;
    // Bind inputs (name -> tensor) and run. Outputs reference instance
    // storage and stay valid until the next run on the same Plan (the
    // caller copies what must persist, exactly like the tf_check state
    // hand-off). Throws on missing inputs or shape errors.
    std::map<std::string, Tensor> run(const std::map<std::string, Tensor>& inputs);
    // Drop the run arenas (work + outputs). The next run re-plans and
    // regrows them. Callers must have copied out anything they keep:
    // outputs point into these arenas. Used by the host for plans whose
    // peak is large but whose lifetime is one phase of a request (GPT
    // encoder/first, vits), so steady RSS tracks concurrently-live
    // memory instead of the sum of per-phase peaks.
    void release_arena();
    const std::vector<std::string>& input_names() const { return in_names_; }
    const std::vector<std::string>& output_names() const { return out_names_; }
    size_t arena_bytes() const { return arena_bytes_; }
    size_t step_count() const { return n_steps_; }

  private:
    PackedLookup lookup_;
    std::vector<std::string> in_names_, out_names_;
    size_t arena_bytes_ = 0;
    size_t n_steps_ = 0;
    std::shared_ptr<void> impl_;  // free_engine.cpp's internal Impl
};

}  // namespace freeeng
