// Packed residency for the vits (SoVITS) graph: potato::PackedConv,
// potato::PackedConvTranspose, potato::PackedGather.
//
// The E hybrid pack's quantized tensors stay packed in memory; each op
// dequantizes only the block of output channels it is about to compute
// with, into a thread-local scratch, and runs fp32 math on activations.
// This is the conv-side counterpart of PackedMatMul (which covers the
// GPT graphs' MatMul/Gemm only). Numerics: dequantization is the exact
// inverse of the pack (see restore_pack in npz.h / dequant_wo.py), so a
// packed run differs from the baked run only by floating-point
// accumulation order, never by weight values.
//
// Weight layout (identical to the GPT mode-0 pack): a tensor is stored
// flattened as (rows = shape[0], cols = numel / rows). For Conv weights
// (O, I, K) that means row o is the output channel and the columns run
// over (i, k) in C order; for ConvTranspose weights (I, O, K) row i is
// the input channel and columns run over (o, k); for Gather tables row r
// is embedding r. W4: group-64 asymmetric along the flattened columns,
// half-byte payloads (high nibble first). W8: per-row symmetric int8.
//
// Kernels live in vits_kernels.cpp. Dispatch is by RUNTIME CPU feature
// detection (cpu.h + IFUNC target clones on x86): AVX2 is the desktop
// baseline tier, the scalar build is the floor and is selected
// automatically when a feature is missing, and nothing -- pack format,
// manifests, graph layout -- depends on the host chip. See STAGE3.md
// S10 for the target-device table.
//
// The kernel-side declarations (dequant_row, VcStats, tiers, the three
// kernel entry points) live in vits_kernels.h, split out in ORT-free
// phase 2 so the free engine links the kernels without this ORT glue
// header. This header keeps the OrtCustomOp plumbing only.
#pragma once

#include "packed.h"
#include "vits_kernels.h"

namespace packed {

// ---------------- custom op plumbing (classic OrtCustomOp) ----------------
struct VCKernel {
    int64_t mode = 0, w_rows = 0, w_cols = 0, k = 1, stride = 1, pad_l = 0, pad_r = 0,
            dilation = 1, out_pad = 0, has_zero = 0, has_bias = 0;
    bool transpose = false;
};

template <bool TRANSPOSE>
void* ORT_API_CALL vc_create(const OrtCustomOp*, const OrtApi* api, const OrtKernelInfo* info) {
    auto* k = new VCKernel();
    k->transpose = TRANSPOSE;
    api->KernelInfoGetAttribute_int64(info, "mode", &k->mode);
    api->KernelInfoGetAttribute_int64(info, "w_rows", &k->w_rows);
    api->KernelInfoGetAttribute_int64(info, "w_cols", &k->w_cols);
    api->KernelInfoGetAttribute_int64(info, "k", &k->k);
    api->KernelInfoGetAttribute_int64(info, "stride", &k->stride);
    api->KernelInfoGetAttribute_int64(info, "pad_l", &k->pad_l);
    api->KernelInfoGetAttribute_int64(info, "pad_r", &k->pad_r);
    api->KernelInfoGetAttribute_int64(info, "dilation", &k->dilation);
    api->KernelInfoGetAttribute_int64(info, "has_zero", &k->has_zero);
    api->KernelInfoGetAttribute_int64(info, "has_bias", &k->has_bias);
    if (TRANSPOSE) api->KernelInfoGetAttribute_int64(info, "out_pad", &k->out_pad);
    return k;
}

inline const char* ORT_API_CALL vc_name(const OrtCustomOp*) { return "PackedConv"; }
inline const char* ORT_API_CALL vct_name(const OrtCustomOp*) { return "PackedConvTranspose"; }
inline ONNXTensorElementDataType ORT_API_CALL vc_in_type(const OrtCustomOp*, size_t i) {
    // 0: X float, 1: pk uint8, 2: scale float, 3: zero uint8, 4: bias float
    return (i == 1 || i == 3) ? ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}
inline size_t ORT_API_CALL vc_in_count(const OrtCustomOp*) { return 5; }

inline void vc_compute_impl(void* op_kernel, OrtKernelContext* ctx, bool transpose) {
    auto* k = (VCKernel*)op_kernel;
    Ort::KernelContext kc(ctx);
    auto xshape = kc.GetInput(0).GetTensorTypeAndShapeInfo().GetShape();
    if (xshape.size() != 3) throw std::runtime_error("PackedConv: X must be (N, C, L)");
    int64_t N = xshape[0], Ci = xshape[1], L = xshape[2];
    const float* X = kc.GetInput(0).GetTensorData<float>();
    const uint8_t* pk = kc.GetInput(1).GetTensorData<uint8_t>();
    const float* sc = kc.GetInput(2).GetTensorData<float>();
    // Fixed slots: 3 = zero (or "" placeholder), 4 = bias (or placeholder).
    const uint8_t* ze = k->has_zero ? kc.GetInput(3).GetTensorData<uint8_t>() : nullptr;
    const float* bias = k->has_bias ? kc.GetInput(4).GetTensorData<float>() : nullptr;
    if (!transpose) {
        if (Ci * k->k != k->w_cols) throw std::runtime_error("PackedConv: Ci*K != w_cols");
        int64_t Lout = (L + k->pad_l + k->pad_r - k->dilation * (k->k - 1) - 1) / k->stride + 1;
        std::vector<int64_t> oshape = {N, k->w_rows, Lout};
        float* Y = kc.GetOutput(0, oshape.data(), oshape.size()).GetTensorMutableData<float>();
        conv1d_fast(X, N, Ci, L, pk, sc, ze, (int)k->mode, k->w_rows, k->k, k->stride, k->pad_l,
                    k->dilation, Lout, bias, Y);
    } else {
        if (Ci != k->w_rows) throw std::runtime_error("PackedConvTranspose: Ci != w_rows");
        int64_t O = k->w_cols / k->k;
        int64_t Tout = (L - 1) * k->stride - k->pad_l - k->pad_r + k->dilation * (k->k - 1) +
                       k->out_pad + 1;
        std::vector<int64_t> oshape = {N, O, Tout};
        float* Y = kc.GetOutput(0, oshape.data(), oshape.size()).GetTensorMutableData<float>();
        conv_transpose1d_fast(X, N, Ci, L, pk, sc, ze, (int)k->mode, O, k->k, k->stride, k->pad_l,
                              k->dilation, Tout, bias, Y);
    }
}
inline void ORT_API_CALL vc_compute(void* op_kernel, OrtKernelContext* ctx) {
    vc_compute_impl(op_kernel, ctx, false);
}
inline void ORT_API_CALL vct_compute(void* op_kernel, OrtKernelContext* ctx) {
    vc_compute_impl(op_kernel, ctx, true);
}
inline void ORT_API_CALL vc_destroy(void* k) { delete (VCKernel*)k; }

struct VGKernel {
    int64_t mode = 0, w_rows = 0, w_cols = 0, has_zero = 0;
};
inline void* ORT_API_CALL vg_create(const OrtCustomOp*, const OrtApi* api, const OrtKernelInfo* info) {
    auto* k = new VGKernel();
    api->KernelInfoGetAttribute_int64(info, "mode", &k->mode);
    api->KernelInfoGetAttribute_int64(info, "w_rows", &k->w_rows);
    api->KernelInfoGetAttribute_int64(info, "w_cols", &k->w_cols);
    api->KernelInfoGetAttribute_int64(info, "has_zero", &k->has_zero);
    return k;
}
inline const char* ORT_API_CALL vg_name(const OrtCustomOp*) { return "PackedGather"; }
inline ONNXTensorElementDataType ORT_API_CALL vg_in_type(const OrtCustomOp*, size_t i) {
    // 0: pk uint8, 1: scale float, 2: zero uint8, 3: indices int64
    if (i == 3) return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    return (i == 0 || i == 2) ? ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}
inline size_t ORT_API_CALL vg_in_count(const OrtCustomOp*) { return 4; }
inline OrtCustomOpInputOutputCharacteristic ORT_API_CALL vg_in_char(const OrtCustomOp*, size_t i) {
    return i == 2 ? INPUT_OUTPUT_OPTIONAL : INPUT_OUTPUT_REQUIRED;
}
inline void ORT_API_CALL vg_compute(void* op_kernel, OrtKernelContext* ctx) {
    auto* k = (VGKernel*)op_kernel;
    Ort::KernelContext kc(ctx);
    const uint8_t* pk = kc.GetInput(0).GetTensorData<uint8_t>();
    const float* sc = kc.GetInput(1).GetTensorData<float>();
    // Fixed slots: 2 = zero (or "" placeholder), 3 = indices.
    const uint8_t* ze = k->has_zero ? kc.GetInput(2).GetTensorData<uint8_t>() : nullptr;
    auto idx_tensor = kc.GetInput(3);
    auto itype = idx_tensor.GetTensorTypeAndShapeInfo().GetElementType();
    if (itype != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
        throw std::runtime_error("PackedGather: indices must be int64");
    auto ishape = idx_tensor.GetTensorTypeAndShapeInfo().GetShape();
    const int64_t* idx = idx_tensor.GetTensorData<int64_t>();
    int64_t nidx = 1;
    for (auto d : ishape) nidx *= d;
    auto oshape = ishape;
    oshape.push_back(k->w_cols);
    float* Y = kc.GetOutput(0, oshape.data(), oshape.size()).GetTensorMutableData<float>();
    gather_packed(pk, sc, ze, (int)k->mode, k->w_rows, k->w_cols, idx, nidx, Y);
}
inline void ORT_API_CALL vg_destroy(void* k) { delete (VGKernel*)k; }

// Fill every OrtCustomOp slot (ORT calls some unconditionally; see packed.h).
template <typename CreateFn>
inline OrtCustomOp make_vc_op(CreateFn create, const char* (ORT_API_CALL* get_name)(const OrtCustomOp*),
                              ONNXTensorElementDataType (ORT_API_CALL* in_type)(const OrtCustomOp*, size_t),
                              size_t (ORT_API_CALL* in_count)(const OrtCustomOp*),
                              void (ORT_API_CALL* compute)(void*, OrtKernelContext*),
                              void (ORT_API_CALL* destroy)(void*),
                              OrtCustomOpInputOutputCharacteristic (ORT_API_CALL* in_char)(const OrtCustomOp*, size_t)) {
    OrtCustomOp o{};
    o.version = ORT_API_VERSION;
    o.CreateKernel = (void* (ORT_API_CALL*)(const OrtCustomOp*, const OrtApi*, const OrtKernelInfo*))create;
    o.GetName = get_name;
    o.GetExecutionProviderType = pm_ep;
    o.GetInputType = in_type;
    o.GetInputTypeCount = in_count;
    o.GetOutputType = pm_out_type;
    o.GetOutputTypeCount = pm_out_count;
    o.KernelCompute = compute;
    o.KernelDestroy = destroy;
    o.GetInputCharacteristic = in_char;
    o.GetOutputCharacteristic = pm_out_char;
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
}

inline const OrtCustomOp& packed_conv_op() {
    static OrtCustomOp op = make_vc_op(vc_create<false>, vc_name, vc_in_type, vc_in_count,
                                       vc_compute, vc_destroy, pm_in_char);
    return op;
}
inline const OrtCustomOp& packed_conv_transpose_op() {
    static OrtCustomOp op = make_vc_op(vc_create<true>, vct_name, vc_in_type, vc_in_count,
                                       vct_compute, vc_destroy, pm_in_char);
    return op;
}
inline const OrtCustomOp& packed_gather_op() {
    static OrtCustomOp op = make_vc_op(vg_create, vg_name, vg_in_type, vg_in_count,
                                       vg_compute, vg_destroy, vg_in_char);
    return op;
}

// Full domain: GPT PackedMatMul + the vits ops.
inline Ort::CustomOpDomain make_domain_full() {
    Ort::CustomOpDomain domain("potato");
    domain.Add(&packed_op());
    domain.Add(&packed_conv_op());
    domain.Add(&packed_conv_transpose_op());
    domain.Add(&packed_gather_op());
    return domain;
}

}  // namespace packed
