// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Opus-based MOE sorting torch-free binding.
// Self-contained: no CK header dependency.

#define MOE_SORTING_OPUS_IMPL
#include "moe_sorting_opus.h"

#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "aiter_tensor.h"
#include "mx_quant_utils.h"

#include <string>
#include <vector>

void moe_sorting_opus_fwd(aiter_tensor_t& topk_ids,
                          aiter_tensor_t& topk_weights,
                          aiter_tensor_t& sorted_token_ids,
                          aiter_tensor_t& sorted_weights,
                          aiter_tensor_t& sorted_expert_ids,
                          aiter_tensor_t& num_valid_ids,
                          aiter_tensor_t& moe_buf,
                          int num_experts,
                          int unit_size,
                          std::optional<aiter_tensor_t> local_expert_mask,
                          std::optional<aiter_tensor_t> num_local_tokens,
                          std::optional<aiter_tensor_t> workspace,
                          int dispatch_policy,
                          std::optional<aiter_tensor_t> local_topk_ids,
                          std::optional<aiter_tensor_t> m_indices,
                          std::optional<aiter_tensor_t> reverse_sorted)
{
    AITER_CHECK(topk_weights.dtype() == AITER_DTYPE_fp32,
                "topk_weights must be FP32 (float32)");

    auto dtype_str = AiterDtype_to_str(topk_ids.dtype());
    int num_tokens = topk_ids.size(0);
    int topk       = topk_ids.size(1);

    if(local_topk_ids.has_value())
    {
        auto& ids_out = local_topk_ids.value();
        AITER_CHECK(ids_out.dim() == 2 && ids_out.size(0) == topk_ids.size(0) &&
                        ids_out.size(1) == topk_ids.size(1),
                    "local_topk_ids must have the same [tokens, topk] shape as topk_ids");
        AITER_CHECK(ids_out.dtype() == topk_ids.dtype(),
                    "local_topk_ids dtype must match topk_ids");
        AITER_CHECK(ids_out.device_id == topk_ids.device_id,
                    "local_topk_ids must be on the same device as topk_ids");
        AITER_CHECK(ids_out.is_contiguous(), "local_topk_ids must be contiguous");
    }

    HipDeviceGuard device_guard(topk_ids.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    if(reverse_sorted.has_value())
    {
        auto& reverse = reverse_sorted.value();
        AITER_CHECK(reverse.dtype() == AITER_DTYPE_i32 && reverse.is_contiguous() &&
                        reverse.numel() >= static_cast<int64_t>(num_tokens) * topk,
                    "reverse_sorted must be a contiguous int32 buffer for tokens * topk routes");
    }

    void* ws_ptr = workspace.has_value() ? workspace.value().data_ptr() : nullptr;

    moe_sorting_opus(
        {
            dtype_str,
            "fp32",
            local_expert_mask.has_value(),
            true,
            dispatch_policy
        },
        {topk_ids.data_ptr(),
         topk_weights.data_ptr(),
         local_expert_mask.has_value() ? local_expert_mask.value().data_ptr() : nullptr,
         num_local_tokens.has_value() ? num_local_tokens.value().data_ptr() : nullptr,
         sorted_token_ids.data_ptr(),
         sorted_weights.data_ptr(),
         sorted_expert_ids.data_ptr(),
         num_valid_ids.data_ptr(),
         moe_buf.data_ptr(),
         ws_ptr,
         local_topk_ids.has_value() ? local_topk_ids.value().data_ptr() : nullptr,
         m_indices.has_value() ? m_indices.value().data_ptr() : nullptr,
         reverse_sorted.has_value() ? reverse_sorted.value().data_ptr() : nullptr,
         num_tokens,
         unit_size,
         num_experts,
         topk,
         static_cast<int>(moe_buf.size(-1)),
         static_cast<int>(moe_buf.element_size())},
        {stream});
}

namespace aiter::mxfp4_moe {

constexpr int kHiddenSize    = 4096;
constexpr int kBlockM        = 32;
constexpr int kThreads       = 1024;
constexpr int kSortQuantCtas = 16;
constexpr int kScaleCols     = kHiddenSize / 32;

// Routed-MoE shapes the sort kernel is instantiated for. NumExperts and TopK
// are template parameters (the expert histogram lives in LDS), so widening the
// support surface means adding a row here.
#define AITER_MXFP4_SORT_QUANT_ROUTES(_) \
    _(512, 10)                           \
    _(513, 11)

__device__ __forceinline__ int pack_token_topk_id(int token_id, int topk_id)
{
    return (token_id & 0x00ffffff) | ((topk_id & 0xff) << 24);
}

__device__ __forceinline__ int round_up_to_block(int value)
{
    return (value + kBlockM - 1) & ~(kBlockM - 1);
}

__device__ __forceinline__ int dpp_inclusive_scan_wave(int value)
{
    int tmp = __builtin_amdgcn_mov_dpp(value, 0x111, 0xF, 0xF, true);
    value += tmp;
    tmp = __builtin_amdgcn_mov_dpp(value, 0x112, 0xF, 0xF, true);
    value += tmp;
    tmp = __builtin_amdgcn_mov_dpp(value, 0x114, 0xF, 0xF, true);
    value += tmp;
    tmp = __builtin_amdgcn_mov_dpp(value, 0x118, 0xF, 0xF, true);
    value += tmp;
    tmp = __builtin_amdgcn_update_dpp(0, value, 0x142, 0xA, 0xF, true);
    value += tmp;
    tmp = __builtin_amdgcn_update_dpp(0, value, 0x143, 0xC, 0xF, true);
    return value + tmp;
}

template <int NumCtas>
__device__ __forceinline__ void zero_moe_output(void* output, int tokens)
{
    if(output == nullptr)
        return;

    constexpr long long row_bytes = static_cast<long long>(kHiddenSize) * 2;
    const long long total_vecs = (static_cast<long long>(tokens) * row_bytes) / sizeof(int4);
    const long long global_thread =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    constexpr long long stride = static_cast<long long>(NumCtas) * kThreads;
    auto* output_vec = reinterpret_cast<int4*>(output);
    const int4 zero = {0, 0, 0, 0};

    for(long long i = global_thread; i < total_vecs; i += stride)
        output_vec[i] = zero;
}

template <int NumExperts, int TopK>
__device__ __forceinline__ void build_sorted_metadata(
    const int32_t* topk_ids,
    const float* topk_weights,
    int32_t* sorted_token_ids,
    float* sorted_weights,
    int32_t* sorted_expert_ids,
    int32_t* num_valid_ids,
    int tokens)
{
    // Experts are scanned as complete 64-lane waves; a trailing partial wave
    // (E=385) is covered by the running prefix of the last full wave.
    constexpr int kFullExpertWaves = NumExperts / 64;
    __shared__ int expert_counts[NumExperts];
    __shared__ int expert_offsets[NumExperts + 1];
    __shared__ int expert_cursors[NumExperts];
    __shared__ int wave_totals[16];

    const int tid = threadIdx.x;
    for(int expert = tid; expert < NumExperts; expert += kThreads)
    {
        expert_counts[expert] = 0;
        expert_cursors[expert] = 0;
    }
    __syncthreads();

    const int total_pairs = tokens * TopK;
    for(int index = tid; index < total_pairs; index += kThreads)
    {
        const int expert = topk_ids[index];
        if(expert >= 0 && expert < NumExperts)
            atomicAdd(expert_counts + expert, 1);
    }
    __syncthreads();

    // At decode expert counts a small shared-memory prefix scan is cheaper
    // than the generic Opus one-shot bookkeeping.
    const int lane = tid & 63;
    const int wave = tid >> 6;
    int value = tid < NumExperts ? round_up_to_block(expert_counts[tid]) : 0;
    value = dpp_inclusive_scan_wave(value);
    if(lane == 63 && wave < kFullExpertWaves)
        wave_totals[wave] = value;
    __syncthreads();
    if(wave == 0)
    {
        int wave_value = lane < kFullExpertWaves ? wave_totals[lane] : 0;
        wave_value = dpp_inclusive_scan_wave(wave_value);
        if(lane < kFullExpertWaves)
            wave_totals[lane] = wave_value;
    }
    __syncthreads();

    const int wave_prefix = wave == 0 ? 0 : wave_totals[wave - 1];
    if(tid < NumExperts)
    {
        expert_offsets[tid + 1] = value + wave_prefix;
        expert_cursors[tid] = 0;
    }
    if(tid == 0)
        expert_offsets[0] = 0;
    __syncthreads();

    if(tid == 0)
    {
        num_valid_ids[0] = expert_offsets[NumExperts];
        num_valid_ids[1] = tokens;
    }
    __syncthreads();

    for(int index = tid; index < total_pairs; index += kThreads)
    {
        const int expert = topk_ids[index];
        if(expert < 0 || expert >= NumExperts)
            continue;

        const int token = index / TopK;
        const int topk = index - token * TopK;
        const int position = expert_offsets[expert] + atomicAdd(expert_cursors + expert, 1);
        sorted_token_ids[position] = pack_token_topk_id(token, topk);
        sorted_weights[position] = topk_weights[index];
    }
    __syncthreads();

    const int sentinel = pack_token_topk_id(tokens, 0);
    for(int expert = tid; expert < NumExperts; expert += kThreads)
    {
        const int count = expert_counts[expert];
        const int start = expert_offsets[expert];
        const int end = expert_offsets[expert + 1];

        for(int position = start + count; position < end; ++position)
        {
            sorted_token_ids[position] = sentinel;
            sorted_weights[position] = 0.0f;
        }
        for(int block = start / kBlockM; block < end / kBlockM; ++block)
            sorted_expert_ids[block] = expert;
    }
}

template <int QuantCtas>
__device__ __forceinline__ void quantize_activations_compact(
    const hip_bfloat16* hidden_states,
    uint8_t* activation_quant,
    uint8_t* activation_scale_token,
    int tokens)
{
    using bf16x2_t = __bf16 __attribute__((ext_vector_type(2)));
    constexpr int groups_per_token = kHiddenSize / 32;
    constexpr int groups_per_wave = 16;
    constexpr int groups_per_cta = (kThreads / 64) * groups_per_wave;

    const int tid = threadIdx.x;
    const int wave = tid >> 6;
    const int lane = tid & 63;
    const int group_in_wave = lane >> 2;
    const int lane_in_group = lane & 3;
    const int total_groups = tokens * groups_per_token;
    const int total_batches = (total_groups + groups_per_cta - 1) / groups_per_cta;
    const int quant_cta = blockIdx.x - 1;
    const int batches_per_cta = (total_batches + QuantCtas - 1) / QuantCtas;
    const int batch_begin = quant_cta * batches_per_cta;
    const int batch_end = min(batch_begin + batches_per_cta, total_batches);

    for(int batch = batch_begin; batch < batch_end; ++batch)
    {
        const int group = batch * groups_per_cta + wave * groups_per_wave + group_in_wave;
        if(group >= total_groups)
            continue;

        const int element_offset = group * 32 + lane_in_group * 8;
        uint32_t raw[4];
        *reinterpret_cast<int4*>(raw) =
            *reinterpret_cast<const int4*>(hidden_states + element_offset);

        uint16_t local_amax = 0;
#pragma unroll
        for(int i = 0; i < 4; ++i)
        {
            const uint16_t lo = static_cast<uint16_t>(raw[i] & 0xffffu) & 0x7fffu;
            const uint16_t hi = static_cast<uint16_t>(raw[i] >> 16) & 0x7fffu;
            local_amax = max(local_amax, max(lo, hi));
        }

        uint32_t max_bits = local_amax;
        max_bits = max(
            max_bits,
            static_cast<uint32_t>(
                __builtin_amdgcn_mov_dpp(static_cast<int>(max_bits), 0xB1, 0xF, 0xF, true)));
        max_bits = max(
            max_bits,
            static_cast<uint32_t>(
                __builtin_amdgcn_mov_dpp(static_cast<int>(max_bits), 0x4E, 0xF, 0xF, true)));
        const float amax =
            __uint_as_float(static_cast<uint32_t>(max_bits & 0xffffu) << 16);
        const float dequant_scale = aiter::fp4_f32_to_e8m0_scale(amax);
        const uint8_t scale =
            (static_cast<uint32_t>(__builtin_bit_cast(uint32_t, dequant_scale)) >> 23) & 0xff;

        uint32_t packed = 0;
#if defined(__gfx950__)
        packed = __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(
            packed, *reinterpret_cast<const bf16x2_t*>(&raw[0]), dequant_scale, 0);
        packed = __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(
            packed, *reinterpret_cast<const bf16x2_t*>(&raw[1]), dequant_scale, 1);
        packed = __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(
            packed, *reinterpret_cast<const bf16x2_t*>(&raw[2]), dequant_scale, 2);
        packed = __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(
            packed, *reinterpret_cast<const bf16x2_t*>(&raw[3]), dequant_scale, 3);
#else
        __builtin_trap();
#endif

        *reinterpret_cast<uint32_t*>(
            activation_quant + static_cast<size_t>(group) * 16 + lane_in_group * 4) = packed;
        if(lane_in_group == 0)
            activation_scale_token[group] = scale;
    }
}

template <int NumExperts, int TopK>
__global__ void __launch_bounds__(kThreads) moe_sort_quant_kernel(
    const hip_bfloat16* hidden_states,
    const int32_t* topk_ids,
    const float* topk_weights,
    int32_t* sorted_token_ids,
    float* sorted_weights,
    int32_t* sorted_expert_ids,
    int32_t* num_valid_ids,
    void* moe_buf,
    uint8_t* activation_quant,
    uint8_t* activation_scale_token,
    int tokens,
    const int32_t* num_local_tokens)
{
    // moe_buf is zeroed whole, so a row past the local-token count reads 0
    // rather than what the buffer held; only the local rows are sorted and
    // quantized, as moe_sorting_opus_fwd sorts them.
    zero_moe_output<kSortQuantCtas>(moe_buf, tokens);
    if(num_local_tokens != nullptr)
        tokens = min(tokens, *num_local_tokens);
    if(blockIdx.x == 0)
    {
        build_sorted_metadata<NumExperts, TopK>(
            topk_ids,
            topk_weights,
            sorted_token_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            tokens);
    }
    else
    {
        quantize_activations_compact<kSortQuantCtas - 1>(
            hidden_states, activation_quant, activation_scale_token, tokens);
    }
}

// The activation quantization above lowers to v_cvt_scalef32_pk_fp4_bf16, which
// is part of the fp4-cvt-scale instruction set introduced with gfx950 (CDNA4).
// Earlier CDNA parts have no FP4 convert instruction at all, so there is no
// gfx942 equivalent to substitute -- and the MXFP4 MoE GEMM that consumes this
// kernel's output is gfx950-only for the same reason, so a software-emulated
// fallback would produce bytes nothing downstream could use. Reject other
// architectures on the host instead, so an unsupported call reports an error
// rather than executing the __builtin_trap() the device path compiles to.
//
// Cached per device: this sits on the decode path and hipGetDeviceProperties is
// not cheap enough to call per launch.
inline bool is_fp4_cvt_capable(int device_id)
{
    static const std::vector<char> capable = [] {
        int count = 0;
        HIP_CALL(hipGetDeviceCount(&count));
        std::vector<char> caps(static_cast<size_t>(count), 0);
        for(int i = 0; i < count; ++i)
        {
            hipDeviceProp_t prop;
            HIP_CALL(hipGetDeviceProperties(&prop, i));
            caps[static_cast<size_t>(i)] =
                std::string(prop.gcnArchName).find("gfx950") != std::string::npos;
        }
        return caps;
    }();
    return device_id >= 0 && device_id < static_cast<int>(capable.size()) &&
           capable[static_cast<size_t>(device_id)] != 0;
}

} // namespace aiter::mxfp4_moe

void mxfp4_moe_sort_quant_fwd(aiter_tensor_t& hidden_states,
                              aiter_tensor_t& topk_ids,
                              aiter_tensor_t& topk_weights,
                              aiter_tensor_t& sorted_token_ids,
                              aiter_tensor_t& sorted_weights,
                              aiter_tensor_t& sorted_expert_ids,
                              aiter_tensor_t& num_valid_ids,
                              aiter_tensor_t& moe_buf,
                              aiter_tensor_t& activation_quant,
                              aiter_tensor_t& activation_scale_token,
                              int num_experts,
                              std::optional<aiter_tensor_t> num_local_tokens)
{
    using namespace aiter::mxfp4_moe;

    const int device_id = hidden_states.device_id;
    AITER_CHECK(is_fp4_cvt_capable(device_id),
                __func__,
                ": requires gfx950; the fp4 activation quantization uses the "
                "scalef32 fp4-cvt instructions, which no earlier architecture has");
    // An off-device tensor would otherwise be dereferenced by the kernel under
    // this call's HipDeviceGuard and fault, or silently read another device's
    // memory. moe_buf is exempt when empty: the zeroing pass skips it entirely.
    AITER_CHECK(topk_ids.device_id == device_id && topk_weights.device_id == device_id &&
                    sorted_token_ids.device_id == device_id &&
                    sorted_weights.device_id == device_id &&
                    sorted_expert_ids.device_id == device_id &&
                    num_valid_ids.device_id == device_id &&
                    (moe_buf.numel() == 0 || moe_buf.device_id == device_id) &&
                    activation_quant.device_id == device_id &&
                    activation_scale_token.device_id == device_id,
                __func__,
                ": all tensors must be on the same device");

    AITER_CHECK(hidden_states.dtype() == AITER_DTYPE_bf16,
                __func__,
                ": hidden_states must be BF16");
    AITER_CHECK(topk_ids.dtype() == AITER_DTYPE_i32 &&
                    topk_weights.dtype() == AITER_DTYPE_fp32,
                __func__,
                ": expected int32 topk_ids and fp32 topk_weights");
    AITER_CHECK(sorted_token_ids.dtype() == AITER_DTYPE_i32 &&
                    sorted_weights.dtype() == AITER_DTYPE_fp32 &&
                    sorted_expert_ids.dtype() == AITER_DTYPE_i32 &&
                    num_valid_ids.dtype() == AITER_DTYPE_i32,
                __func__,
                ": expected int32 sorted_token_ids, fp32 sorted_weights, int32 "
                "sorted_expert_ids, and int32 num_valid_ids");
    const int topk = topk_ids.dim() == 2 ? static_cast<int>(topk_ids.size(1)) : 0;

#define AITER_MXFP4_SORT_QUANT_MATCH(NUM_EXPERTS, TOPK) \
    || (num_experts == (NUM_EXPERTS) && topk == (TOPK))
    const bool route_supported =
        false AITER_MXFP4_SORT_QUANT_ROUTES(AITER_MXFP4_SORT_QUANT_MATCH);
#undef AITER_MXFP4_SORT_QUANT_MATCH

    AITER_CHECK(hidden_states.dim() == 2 && hidden_states.size(1) == kHiddenSize &&
                    topk_ids.dim() == 2 && topk_ids.size(0) == hidden_states.size(0) &&
                    route_supported,
                __func__,
                ": only H=4096 with (num_experts, topk) in "
                "{(512,10), (513,11)} is supported");
    AITER_CHECK(hidden_states.is_contiguous() && topk_ids.is_contiguous() &&
                    topk_weights.is_contiguous() && sorted_token_ids.is_contiguous() &&
                    sorted_weights.is_contiguous() && sorted_expert_ids.is_contiguous() &&
                    num_valid_ids.is_contiguous() && moe_buf.is_contiguous() &&
                    activation_quant.is_contiguous() &&
                    activation_scale_token.is_contiguous(),
                __func__,
                ": all tensors must be contiguous");
    AITER_CHECK(activation_quant.dtype() == AITER_DTYPE_fp4x2 ||
                    activation_quant.dtype() == AITER_DTYPE_u8,
                __func__,
                ": activation_quant must be packed FP4");
    AITER_CHECK(activation_scale_token.dtype() == AITER_DTYPE_fp8_e8m0 ||
                    activation_scale_token.dtype() == AITER_DTYPE_u8,
                __func__,
                ": activation_scale_token must hold e8m0 bytes");

    const int tokens = static_cast<int>(hidden_states.size(0));
    AITER_CHECK(tokens > 0 && tokens <= 128,
                __func__,
                ": only decode M in [1, 128] is supported");
    const int64_t route_count = topk_ids.numel();
    const int64_t active_experts =
        route_count < static_cast<int64_t>(num_experts) ? route_count : num_experts;
    const int64_t required_sorted =
        ((route_count + active_experts * (kBlockM - 1) + kBlockM - 1) /
         kBlockM) *
        kBlockM;
    AITER_CHECK(topk_weights.numel() == topk_ids.numel() &&
                    activation_quant.numel() == hidden_states.numel() / 2 &&
                    activation_scale_token.numel() ==
                        hidden_states.size(0) * kScaleCols &&
                    num_valid_ids.numel() >= 2,
                __func__,
                ": incompatible fused sort/quant tensor sizes");
    AITER_CHECK(moe_buf.numel() == 0 ||
                    ((moe_buf.dtype() == AITER_DTYPE_bf16 ||
                      moe_buf.dtype() == AITER_DTYPE_fp16) &&
                     moe_buf.numel() >=
                         static_cast<int64_t>(tokens) * kHiddenSize),
                __func__,
                ": moe_buf must be empty or contain at least M*4096 BF16/FP16 elements");

    AITER_CHECK(sorted_token_ids.numel() >= required_sorted &&
                    sorted_weights.numel() >= required_sorted &&
                    sorted_expert_ids.numel() >= required_sorted / kBlockM,
                __func__,
                ": output routing buffers are too small");

    HipDeviceGuard device_guard(hidden_states.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    const hip_bfloat16* hidden_ptr =
        static_cast<const hip_bfloat16*>(hidden_states.data_ptr());
    const int32_t* topk_ids_ptr = static_cast<const int32_t*>(topk_ids.data_ptr());
    const float* topk_weights_ptr = static_cast<const float*>(topk_weights.data_ptr());
    int32_t* sorted_ids_ptr = static_cast<int32_t*>(sorted_token_ids.data_ptr());
    float* sorted_weights_ptr = static_cast<float*>(sorted_weights.data_ptr());
    int32_t* sorted_experts_ptr = static_cast<int32_t*>(sorted_expert_ids.data_ptr());
    int32_t* num_valid_ptr = static_cast<int32_t*>(num_valid_ids.data_ptr());
    void* moe_buf_ptr = moe_buf.numel() == 0 ? nullptr : moe_buf.data_ptr();
    uint8_t* activation_quant_ptr = static_cast<uint8_t*>(activation_quant.data_ptr());
    uint8_t* activation_scale_token_ptr =
        static_cast<uint8_t*>(activation_scale_token.data_ptr());
    const int32_t* num_local_tokens_ptr =
        num_local_tokens.has_value()
            ? static_cast<const int32_t*>(num_local_tokens.value().data_ptr())
            : nullptr;
#define AITER_MXFP4_SORT_QUANT_LAUNCH(NUM_EXPERTS, TOPK)                            \
    if(num_experts == (NUM_EXPERTS) && topk == (TOPK))                              \
    {                                                                               \
        aiter::mxfp4_moe::moe_sort_quant_kernel<(NUM_EXPERTS), (TOPK)>              \
            <<<dim3(kSortQuantCtas), dim3(kThreads), 0, stream>>>(                  \
                hidden_ptr,                                                         \
                topk_ids_ptr,                                                       \
                topk_weights_ptr,                                                   \
                sorted_ids_ptr,                                                     \
                sorted_weights_ptr,                                                 \
                sorted_experts_ptr,                                                 \
                num_valid_ptr,                                                      \
                moe_buf_ptr,                                                        \
                activation_quant_ptr,                                               \
                activation_scale_token_ptr,                                         \
                tokens,                                                             \
                num_local_tokens_ptr);                                              \
    }
    AITER_MXFP4_SORT_QUANT_ROUTES(AITER_MXFP4_SORT_QUANT_LAUNCH)
#undef AITER_MXFP4_SORT_QUANT_LAUNCH

    const hipError_t launch_status = hipGetLastError();
    AITER_CHECK(
        launch_status == hipSuccess,
        __func__,
        ": fused sort+quant launch failed");
}

namespace aiter::mxfp4_moe {

// Opus multi-phase sort (P0_v2 + P23) with the MXFP4 activation prep folded in:
// P0_v2's grid gains quant CTAs (per-token MXFP4, the bytes
// dynamic_per_group_scaled_quant writes), and each P23 expert CTA writes the
// sorted, swizzled E8M0 scales of its own rows (the bytes mxfp4_moe_sort's
// coalesced path writes). The launch boundary orders quant before the scale
// sort, so no CTA waits on another.
constexpr int kSortQuantP0Ctas  = 256;
constexpr int kQuantUnroll      = 8;
constexpr int kScaleTileRows    = 32;
constexpr int kScaleColsMax     = 256;
constexpr int kScaleBatchTiles  = 4;

__device__ __forceinline__ void quantize_group_bf16x8(const uint32_t (&raw)[4],
                                                      uint32_t& packed,
                                                      uint8_t& scale)
{
    using bf16x2_t      = __bf16 __attribute__((ext_vector_type(2)));
    uint16_t local_amax = 0;
#pragma unroll
    for(int i = 0; i < 4; ++i)
    {
        const uint16_t lo = static_cast<uint16_t>(raw[i] & 0xffffu) & 0x7fffu;
        const uint16_t hi = static_cast<uint16_t>(raw[i] >> 16) & 0x7fffu;
        local_amax        = max(local_amax, max(lo, hi));
    }
    uint32_t max_bits = local_amax;
    max_bits          = max(max_bits,
                   static_cast<uint32_t>(__builtin_amdgcn_mov_dpp(
                       static_cast<int>(max_bits), 0xB1, 0xF, 0xF, true)));
    max_bits = max(max_bits,
                   static_cast<uint32_t>(__builtin_amdgcn_mov_dpp(
                       static_cast<int>(max_bits), 0x4E, 0xF, 0xF, true)));
    const float amax = __uint_as_float(static_cast<uint32_t>(max_bits & 0xffffu) << 16);
    const float dequant_scale = aiter::fp4_f32_to_e8m0_scale(amax);
    scale = (static_cast<uint32_t>(__builtin_bit_cast(uint32_t, dequant_scale)) >> 23) & 0xff;
    packed = 0;
#if defined(__gfx950__)
    packed = __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(
        packed, *reinterpret_cast<const bf16x2_t*>(&raw[0]), dequant_scale, 0);
    packed = __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(
        packed, *reinterpret_cast<const bf16x2_t*>(&raw[1]), dequant_scale, 1);
    packed = __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(
        packed, *reinterpret_cast<const bf16x2_t*>(&raw[2]), dequant_scale, 2);
    packed = __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(
        packed, *reinterpret_cast<const bf16x2_t*>(&raw[3]), dequant_scale, 3);
#else
    __builtin_trap();
#endif
}

struct SortQuantArgs
{
    const hip_bfloat16* hidden_states;
    uint8_t* activation_quant;       // [tokens, hidden / 2]
    uint8_t* activation_scale_token; // [tokens, hidden / 32]
    uint8_t* activation_scale;       // [pad32(sorted rows), hidden / 32], swizzled
    int hidden;
};

// Per-token MXFP4 quantization (the bytes dynamic_per_group_scaled_quant
// writes), strided over nq CTAs: 4 lanes x 8 bf16 per 32-group, 16 groups per
// wave per step, kUnroll loads in flight per lane.
template <int kBlockSize, int kUnroll = kQuantUnroll, bool kNtLoad = false>
__device__ __forceinline__ void
quantize_tokens_mxfp4(const SortQuantArgs& q, int tokens, int qcta, int nq)
{
    constexpr int groups_per_step = (kBlockSize / 64) * 16;
    constexpr int groups_per_iter = groups_per_step * kUnroll;
    const int wave          = threadIdx.x >> 6;
    const int lane          = threadIdx.x & 63;
    const int lane_in_group = lane & 3;
    const int total_groups  = tokens * (q.hidden / 32);
    for(int base = qcta * groups_per_iter; base < total_groups; base += nq * groups_per_iter)
    {
        uint32_t raw[kUnroll][4];
#pragma unroll
        for(int u = 0; u < kUnroll; ++u)
        {
            const int group = base + u * groups_per_step + wave * 16 + (lane >> 2);
            if(group < total_groups)
            {
                const hip_bfloat16* src =
                    q.hidden_states + static_cast<size_t>(group) * 32 + lane_in_group * 8;
                if constexpr(kNtLoad)
                {
                    using u32x4_t = uint32_t __attribute__((ext_vector_type(4)));
                    *reinterpret_cast<u32x4_t*>(raw[u]) =
                        __builtin_nontemporal_load(reinterpret_cast<const u32x4_t*>(src));
                }
                else
                    *reinterpret_cast<int4*>(raw[u]) = *reinterpret_cast<const int4*>(src);
            }
            else
                raw[u][0] = raw[u][1] = raw[u][2] = raw[u][3] = 0;
        }
#pragma unroll
        for(int u = 0; u < kUnroll; ++u)
        {
            const int group = base + u * groups_per_step + wave * 16 + (lane >> 2);
            uint32_t packed;
            uint8_t scale;
            quantize_group_bf16x8(raw[u], packed, scale);
            if(group < total_groups)
            {
                *reinterpret_cast<uint32_t*>(q.activation_quant + static_cast<size_t>(group) * 16 +
                                             lane_in_group * 4) = packed;
                if(lane_in_group == 0)
                    q.activation_scale_token[group] = scale;
            }
        }
    }
}

template <typename P0>
__global__ void __launch_bounds__(P0::kBlockSize)
    moe_sort_p0_quant_kernel(typename P0::Kargs kargs, SortQuantArgs q)
{
    const int num_experts = kargs.num_experts;
    if(static_cast<int>(blockIdx.x) < num_experts)
    {
        P0{}(kargs);
        return;
    }
    // P0 strides its reverse_sorted reset over the whole grid.
    impl::moe_sorting_reverse_sorted_set_invalid<P0::kBlockSize>(
        kargs.p_reverse_sorted, kargs.tokens, kargs.topk_mdiv.divisor);
    quantize_tokens_mxfp4<P0::kBlockSize>(
        q, kargs.tokens, blockIdx.x - num_experts, gridDim.x - num_experts);
}

template <typename P23>
__global__ void __launch_bounds__(P23::kBlockSize)
    moe_sort_p23_scale_kernel(typename P23::Kargs kargs, SortQuantArgs q)
{
    P23{}(kargs);
    // No sorted-scale output: the GEMM1 gathers the token-order scales itself.
    if(q.activation_scale == nullptr || static_cast<int>(blockIdx.x) >= kargs.num_experts)
        return;
    // This CTA wrote its expert's sorted rows; P23 keeps the padded expert
    // offsets in LDS after its block-level scan.
    __syncthreads();
    extern __shared__ char smem[];
    const int* expert_offsets =
        reinterpret_cast<const int*>(smem) + 4 + 2 * P23::kBlockSize / opus::get_warp_size();
    const int e_start = expert_offsets[blockIdx.x];
    const int e_end   = expert_offsets[blockIdx.x + 1];

    // Up to kScaleBatchTiles 32-row tiles per pass: every row's scales are
    // loaded at once, then each tile is written with coalesced dwords.
    constexpr int kBatchRows  = kScaleTileRows * kScaleBatchTiles;
    constexpr int kRowThreads = P23::kBlockSize / kBatchRows;
    constexpr int kLdsStride  = kScaleColsMax + 16;
    __shared__ uint8_t s_scale[kBatchRows * kLdsStride];
    const int tid             = threadIdx.x;
    const int tokens          = kargs.tokens;
    const int scale_cols      = q.hidden / 32; // multiple of 32 (host-checked)
    const int vecs_per_thread = scale_cols / 16 / kRowThreads;
    const int row_i           = tid / kRowThreads;
    const int col0            = (tid % kRowThreads) * vecs_per_thread * 16;
    const int tile_dw         = (scale_cols / 8) * 64;
    const int32_t* sorted_ids = reinterpret_cast<const int32_t*>(kargs.p_sorted_token_ids);
    for(int row0 = e_start; row0 < e_end; row0 += kBatchRows)
    {
        const int row    = row0 + row_i;
        const int token  = row < e_end ? sorted_ids[row] & 0xFFFFFF : tokens;
        const bool valid = token < tokens;
        __syncthreads();
        for(int j = 0; j < vecs_per_thread; ++j)
        {
            int4 v = {0, 0, 0, 0};
            if(valid)
                v = *reinterpret_cast<const int4*>(q.activation_scale_token +
                                                   static_cast<size_t>(token) * scale_cols +
                                                   col0 + j * 16);
            *reinterpret_cast<int4*>(s_scale + row_i * kLdsStride + col0 + j * 16) = v;
        }
        __syncthreads();
        const int tiles = min(kScaleBatchTiles, (e_end - row0) / kScaleTileRows);
        for(int d = tid; d < tiles * tile_dw; d += P23::kBlockSize)
        {
            const int tile    = d / tile_dw;
            const int dw      = d - tile * tile_dw;
            const int y_blk   = dw >> 6;
            const int t       = dw & 63;
            const int xl_lo   = tile * kScaleTileRows + (t & 15);
            const int c       = t >> 4;
            const int col_a   = y_blk * 8 + c;
            const int col_b   = col_a + 4;
            const uint32_t b0 = s_scale[xl_lo * kLdsStride + col_a];
            const uint32_t b1 = s_scale[(xl_lo + 16) * kLdsStride + col_a];
            const uint32_t b2 = s_scale[xl_lo * kLdsStride + col_b];
            const uint32_t b3 = s_scale[(xl_lo + 16) * kLdsStride + col_b];
            uint8_t* tile_out = q.activation_scale +
                                static_cast<int64_t>(row0 + tile * kScaleTileRows) * scale_cols;
            *reinterpret_cast<uint32_t*>(tile_out + y_blk * 256 + c * 64 + (t & 15) * 4) =
                b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
        }
    }
}


// One-launch replacement for P0_v2 + P23 when GEMM1 gathers token-order scales
// (no sorted-scale output). Each of the first sort_ctas CTAs histograms every
// route in LDS, scans the padded counts, and writes one slice of the routes plus
// the padding of one expert range; the other CTAs quantize the activations and
// zero moe_buf. Inside an expert, rows follow route order across slices and
// LDS-atomic order within a slice; every row is computed independently
// downstream, so the outputs are unchanged. MaxPer == 0 is the large-M form:
// only the CTA's own slice stays in registers and the other routes are counted
// in a streaming loop. Each sort CTA then takes about as long as the quant, so
// it uses few sort CTAs and one quant CTA on each of the remaining CUs (a CU
// shared with a sort CTA would make its quant share the straggler).
constexpr int kSort1MaxExperts = 1024;
constexpr int kSort1SliceMax    = 16;
constexpr int kSort1LoopSortCtas  = 24;
constexpr int kSort1LoopQuantCtas = 232;
constexpr int kSort1SortCtas    = 128;
constexpr int kSort1QuantCtas   = 256;
constexpr int kSort1QuantUnroll = 4;

struct Sort1Args
{
    const int32_t* topk_ids;
    const float* topk_weights;
    int32_t* sorted_token_ids;
    float* sorted_weights;
    int32_t* sorted_expert_ids;
    int32_t* num_valid_ids;
    int32_t* m_indices;
    int32_t* reverse_sorted;
    void* moe_buf;
    int moe_buf_cols;
    int moe_buf_elem_bytes;
    int tokens;
    int topk;
    int num_experts;
    int unit_size;
    int sort_ctas;
};

// Block barrier that orders LDS only: __syncthreads() would also drain every
// outstanding global load and store (vmcnt(0)) at each of the sort's barriers.
__device__ __forceinline__ void lds_barrier()
{
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// Counts routes [lo, hi) of a 4-aligned id array into counts, four ids per load.
__device__ __forceinline__ void
count_routes(const int32_t* ids, int* counts, int lo, int hi, int routes, int E, int lane)
{
    const int4* ids4 = reinterpret_cast<const int4*>(ids);
    for(int v = lo / 4 + static_cast<int>(threadIdx.x); v * 4 < hi; v += kThreads)
    {
        const int4 e4  = ids4[v];
        const int e[4] = {e4.x, e4.y, e4.z, e4.w};
#pragma unroll
        for(int j = 0; j < 4; ++j)
        {
            const int idx = v * 4 + j;
            if(idx >= lo && idx < hi)
                atomicAdd(counts + (idx < routes && e[j] >= 0 && e[j] < E ? e[j] : E + lane), 1);
        }
    }
}

template <int MaxPer>
__global__ void __launch_bounds__(kThreads) moe_sort1_quant_kernel(Sort1Args a, SortQuantArgs q)
{
    const int sort_ctas = a.sort_ctas;
    if(static_cast<int>(blockIdx.x) >= sort_ctas)
    {
        const int qcta = blockIdx.x - sort_ctas;
        const int nq   = gridDim.x - sort_ctas;
        if constexpr(MaxPer == 0)
            quantize_tokens_mxfp4<kThreads, 2, true>(q, a.tokens, qcta, nq);
        else
            quantize_tokens_mxfp4<kThreads, kSort1QuantUnroll>(q, a.tokens, qcta, nq);
        if(a.moe_buf != nullptr)
            impl::moe_buf_set_zero_kernel_2d<kThreads>(
                a.moe_buf, a.tokens, a.moe_buf_cols, a.moe_buf_elem_bytes, qcta, nq);
        return;
    }
    __shared__ int counts[kSort1MaxExperts + 64];
    __shared__ int offsets[kSort1MaxExperts];
    __shared__ int wave_totals[kThreads / 64];

    const int tid    = threadIdx.x;
    const int lane   = tid & 63;
    const int wave   = tid >> 6;
    const int E      = a.num_experts;
    const int unit   = a.unit_size;
    const int routes = a.tokens * a.topk;
    for(int e = tid; e < E + 64; e += kThreads)
        counts[e] = 0;
    lds_barrier();

    // This CTA writes routes [r_lo, r_hi). A route's rank inside its expert
    // counts that expert's routes before r_lo first, so all CTAs agree; invalid
    // routes count into per-lane spare slots past counts[E - 1].
    const int r_lo = static_cast<int>(blockIdx.x) * routes / sort_ctas;
    const int r_hi = (static_cast<int>(blockIdx.x) + 1) * routes / sort_ctas;
    constexpr bool kLoop = MaxPer == 0;
    constexpr int kRegs  = kLoop ? kSort1SliceMax : MaxPer;
    // Register slot i holds route route_of(i); only [r_lo, r_hi) is written.
    const int base_idx = kLoop ? r_lo + tid : tid;
    auto in_slice      = [&](int idx) { return idx >= r_lo && idx < r_hi; };
    auto valid_eid     = [&](int idx, int e) {
        return idx < routes && e >= 0 && e < E ? e : E + lane;
    };
    int eid[kRegs];
#pragma unroll
    for(int i = 0; i < kRegs; ++i)
    {
        const int idx = base_idx + i * kThreads;
        eid[i]        = a.topk_ids[idx < routes ? idx : 0];
    }
    if constexpr(kLoop)
        count_routes(a.topk_ids, counts, 0, r_lo, routes, E, lane);
#pragma unroll
    for(int i = 0; i < kRegs; ++i)
    {
        const int idx = base_idx + i * kThreads;
        eid[i]        = valid_eid(idx, eid[i]);
        if(!kLoop && idx < r_lo)
            atomicAdd(counts + eid[i], 1);
    }
    lds_barrier();
    int rank[kRegs];
    float wgt[kRegs];
#pragma unroll
    for(int i = 0; i < kRegs; ++i)
    {
        const int idx = base_idx + i * kThreads;
        if(in_slice(idx))
        {
            rank[i] = atomicAdd(counts + eid[i], 1);
            wgt[i]  = a.topk_weights[idx];
        }
    }
    lds_barrier();
    if constexpr(kLoop)
        count_routes(a.topk_ids, counts, r_hi, routes, routes, E, lane);
    else
    {
#pragma unroll
        for(int i = 0; i < kRegs; ++i)
            if(i * kThreads + tid >= r_hi)
                atomicAdd(counts + eid[i], 1);
    }
    lds_barrier();

    // Exclusive scan of the padded block counts, one expert per thread.
    const int blocks = tid < E ? (counts[tid] + unit - 1) / unit : 0;
    int value        = dpp_inclusive_scan_wave(blocks);
    if(lane == 63)
        wave_totals[wave] = value;
    lds_barrier();
    if(wave == 0)
    {
        int w = lane < kThreads / 64 ? wave_totals[lane] : 0;
        w     = dpp_inclusive_scan_wave(w);
        if(lane < kThreads / 64)
            wave_totals[lane] = w;
    }
    lds_barrier();
    value += wave == 0 ? 0 : wave_totals[wave - 1];
    if(tid < E)
        offsets[tid] = (value - blocks) * unit;
    if(blockIdx.x == 0 && tid == 0)
    {
        a.num_valid_ids[0] = wave_totals[kThreads / 64 - 1] * unit;
        a.num_valid_ids[1] = a.tokens;
    }
    lds_barrier();

#pragma unroll
    for(int i = 0; i < kRegs; ++i)
    {
        const int idx = base_idx + i * kThreads;
        if(in_slice(idx))
        {
            const int e = eid[i];
            if(e < E)
            {
                const int token         = idx / a.topk;
                const int pos           = offsets[e] + rank[i];
                a.sorted_token_ids[pos] = pack_token_topk_id(token, idx - token * a.topk);
                a.sorted_weights[pos]   = wgt[i];
                a.m_indices[pos]        = token;
                a.reverse_sorted[idx]   = pos;
            }
            else
                a.reverse_sorted[idx] = -1;
        }
    }

    // Padding rows and the block -> expert map of this CTA's expert range.
    const int e_lo     = static_cast<int>(blockIdx.x) * E / sort_ctas;
    const int e_hi     = (static_cast<int>(blockIdx.x) + 1) * E / sort_ctas;
    const int sentinel = pack_token_topk_id(a.tokens, a.topk);
    for(int e = e_lo + wave; e < e_hi; e += kThreads / 64)
    {
        const int count = counts[e];
        const int start = offsets[e];
        const int nb    = (count + unit - 1) / unit;
        for(int p = count + lane; p < nb * unit; p += 64)
        {
            a.sorted_token_ids[start + p] = sentinel;
            a.sorted_weights[start + p]   = 0.0f;
            a.m_indices[start + p]        = a.tokens;
        }
        for(int b = lane; b < nb; b += 64)
            a.sorted_expert_ids[start / unit + b] = e;
    }
}

} // namespace aiter::mxfp4_moe

void moe_sorting_opus_quant_fwd(aiter_tensor_t& topk_ids,
                                aiter_tensor_t& topk_weights,
                                aiter_tensor_t& sorted_token_ids,
                                aiter_tensor_t& sorted_weights,
                                aiter_tensor_t& sorted_expert_ids,
                                aiter_tensor_t& num_valid_ids,
                                aiter_tensor_t& moe_buf,
                                int num_experts,
                                int unit_size,
                                aiter_tensor_t& workspace,
                                aiter_tensor_t& m_indices,
                                aiter_tensor_t& reverse_sorted,
                                aiter_tensor_t& hidden_states,
                                aiter_tensor_t& activation_quant,
                                aiter_tensor_t& activation_scale_token,
                                aiter_tensor_t& activation_scale)
{
    using namespace aiter::mxfp4_moe;
    const int tokens = topk_ids.size(0);
    const int topk   = topk_ids.size(1);
    const int hidden = hidden_states.size(1);
    AITER_CHECK(is_fp4_cvt_capable(hidden_states.device_id), __func__, ": requires gfx950");
    AITER_CHECK(topk_ids.dtype() == AITER_DTYPE_i32 && topk_weights.dtype() == AITER_DTYPE_fp32 &&
                    hidden_states.dtype() == AITER_DTYPE_bf16 && hidden_states.is_contiguous() &&
                    hidden_states.size(0) == tokens,
                __func__,
                ": expected i32 topk_ids, fp32 topk_weights, contiguous bf16 hidden_states");
    AITER_CHECK(hidden % 1024 == 0 && hidden / 32 <= kScaleColsMax,
                __func__,
                ": hidden must be a multiple of 1024 and at most 8192");
    AITER_CHECK(unit_size % kScaleTileRows == 0, __func__, ": unit_size must be a multiple of 32");

    HipDeviceGuard device_guard(topk_ids.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    SortQuantArgs q{static_cast<const hip_bfloat16*>(hidden_states.data_ptr()),
                    static_cast<uint8_t*>(activation_quant.data_ptr()),
                    static_cast<uint8_t*>(activation_scale_token.data_ptr()),
                    activation_scale.numel() > 0
                        ? static_cast<uint8_t*>(activation_scale.data_ptr())
                        : nullptr,
                    hidden};

    // An empty workspace selects the one-launch sort (token-order scales only).
    if(workspace.numel() == 0)
    {
        const int routes = tokens * topk;
        AITER_CHECK(activation_scale.numel() == 0 && num_experts <= kSort1MaxExperts &&
                        routes % 4 == 0 && routes <= kSort1LoopSortCtas * kSort1SliceMax * kThreads,
                    __func__,
                    ": one-launch sort needs no sorted-scale output, <= 1024 experts and "
                    "routes % 4 == 0, <= 393216");
        Sort1Args s1{static_cast<const int32_t*>(topk_ids.data_ptr()),
                     static_cast<const float*>(topk_weights.data_ptr()),
                     static_cast<int32_t*>(sorted_token_ids.data_ptr()),
                     static_cast<float*>(sorted_weights.data_ptr()),
                     static_cast<int32_t*>(sorted_expert_ids.data_ptr()),
                     static_cast<int32_t*>(num_valid_ids.data_ptr()),
                     static_cast<int32_t*>(m_indices.data_ptr()),
                     static_cast<int32_t*>(reverse_sorted.data_ptr()),
                     moe_buf.numel() > 0 ? moe_buf.data_ptr() : nullptr,
                     static_cast<int>(moe_buf.size(-1)),
                     static_cast<int>(moe_buf.element_size()),
                     tokens,
                     topk,
                     num_experts,
                     unit_size,
                     routes <= 36 * kThreads ? kSort1SortCtas : kSort1LoopSortCtas};
        const dim3 grid(routes <= 36 * kThreads ? kSort1SortCtas + kSort1QuantCtas
                                                : kSort1LoopSortCtas + kSort1LoopQuantCtas);
        if(routes <= 8 * kThreads)
            hipLaunchKernelGGL(moe_sort1_quant_kernel<8>, grid, dim3(kThreads), 0, stream, s1, q);
        else if(routes <= 20 * kThreads)
            hipLaunchKernelGGL(moe_sort1_quant_kernel<20>, grid, dim3(kThreads), 0, stream, s1, q);
        else if(routes <= 36 * kThreads)
            hipLaunchKernelGGL(moe_sort1_quant_kernel<36>, grid, dim3(kThreads), 0, stream, s1, q);
        else
            hipLaunchKernelGGL(moe_sort1_quant_kernel<0>, grid, dim3(kThreads), 0, stream, s1, q);
        const hipError_t launch_status = hipGetLastError();
        AITER_CHECK(launch_status == hipSuccess, __func__, ": launch failed");
        return;
    }
    // Only the two-launch (P0_v2 + P23) uint8-mesh multi-phase sort is fused.
    AITER_CHECK(tokens >= 512 && topk < 255 && (tokens * topk) % 4 == 0 &&
                    (tokens < 2048 || tokens * topk <= 36864) &&
                    moe_sorting_opus_get_workspace_size(tokens, num_experts, topk, 0) > 0,
                __func__,
                ": shape is not on the P0_v2 + P23 uint8-mesh sort path");

    aiter::MoeSortingHostArgs a{topk_ids.data_ptr(),
                                topk_weights.data_ptr(),
                                nullptr,
                                nullptr,
                                sorted_token_ids.data_ptr(),
                                sorted_weights.data_ptr(),
                                sorted_expert_ids.data_ptr(),
                                num_valid_ids.data_ptr(),
                                moe_buf.data_ptr(),
                                workspace.data_ptr(),
                                nullptr,
                                m_indices.data_ptr(),
                                reverse_sorted.data_ptr(),
                                tokens,
                                unit_size,
                                num_experts,
                                topk,
                                static_cast<int>(moe_buf.size(-1)),
                                static_cast<int>(moe_buf.element_size())};

    using p0_problem  = aiter::MoeSortingProblemMp<opus::index_t, float, uint8_t, 4, false, false>;
    using p23_problem = aiter::MoeSortingProblemMp<opus::index_t, float, uint8_t, 16, false, false>;
    using p0_kernel   = aiter::MoeSortingMultiPhaseKernel_P0_v2<p0_problem>;
    using p23_kernel  = aiter::MoeSortingMultiPhaseKernel_P23<p23_problem>;

    hipLaunchKernelGGL(moe_sort_p0_quant_kernel<p0_kernel>,
                       dim3(num_experts + kSortQuantP0Ctas),
                       dim3(p0_kernel::kBlockSize),
                       0,
                       stream,
                       p0_kernel::MakeKargs(a),
                       q);
    const uint32_t lds = p23_kernel::GetSmemSize(a);
    hipLaunchKernelGGL(moe_sort_p23_scale_kernel<p23_kernel>,
                       p23_kernel::GridSize(a),
                       dim3(p23_kernel::kBlockSize),
                       lds,
                       stream,
                       p23_kernel::MakeKargs(a),
                       q);
    const hipError_t launch_status = hipGetLastError();
    AITER_CHECK(launch_status == hipSuccess, __func__, ": launch failed");
}
