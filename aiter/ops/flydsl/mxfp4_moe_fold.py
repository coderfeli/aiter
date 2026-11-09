# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Whole-graph MXFP4 (a4w4) decode MoE with the routing sort folded into stage 1,
selected by the tuned kernel name
``impl__flydsl_mxfp4_fold__[dpsk<N>__]<kernelName1>__<kernelName2>``:
two launches, the BM16 inline-quant GEMM1 ``<kernelName1>`` built with
``fold_tokens`` (kernels/mxfp4_gemm1.py) and the atomic v2 GEMM2 ``<kernelName2>``.
``dpsk<N>`` splits the K loop of the GEMM1 tiles past one per CU over N CTAs."""

import functools

import torch

from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe_registry import FusedMoeRequest
from aiter.ops.flydsl.kernels.mxfp4_gemm1 import compile_gemm1_a4w4_port
from aiter.ops.flydsl.kernels.mxfp4_gemm_common import kas_per_chunk_dw_for
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
from aiter.ops.flydsl.moe_kernels import runtime_swiglu_limit
from aiter.ops.flydsl.mxfp4_kname import _parse_mxfp4_g1_kname


@functools.cache
def _stage1(kname1, tokens, model_dim, inter_dim, experts, topk, swiglu_limit, dpsk):
    p = _parse_mxfp4_g1_kname(kname1)
    if not (
        p["BM"] == 16
        and p["inline_quant"]
        and p["a_dtype"] == "fp4"
        and p["out_dtype"] == "fp4"
        and p["act"] == "silu"
        and p["num_waves"] == 4
        and p["k_wave"] == 1
    ):
        raise NotImplementedError(f"mxfp4 fold MoE: unsupported stage 1 {kname1!r}")
    return compile_gemm1_a4w4_port(
        16,
        p["use_nt"],
        True,
        prefetch_hidden=p["prefetch_hidden"],
        D_HIDDEN=model_dim,
        D_INTER=inter_dim,
        NE=experts,
        BN=p["BN"],
        BK=p["BK"],
        xcd_swizzle=p["xcd_swizzle"],
        swiglu_limit=swiglu_limit,
        native_scale_layout=True,
        fold_tokens=tokens,
        topk=topk,
        fold_dpsk=dpsk,
    )


@functools.cache
def _dpsk_buffers(device, tiles, dpsk):
    # fp32 K-part tiles, and per-tile arrival counters that the last part resets.
    part = torch.empty(tiles * dpsk * 4096, dtype=torch.float32, device=device)
    return part, torch.zeros(tiles, dtype=torch.int32, device=device)


def run_mxfp4_moe_fold_impl(request: FusedMoeRequest, config: str) -> torch.Tensor:
    from aiter.fused_moe import _flydsl_v2_stage2_wrapper

    dpsk = 1
    if config.startswith("dpsk"):
        head, _, config = config.partition("__")
        dpsk = int(head[4:])
    kname1, sep, kname2 = config.partition("__")
    if not sep or "_atomic" not in kname2:
        raise ValueError(f"mxfp4 fold MoE: bad config {config!r}")
    x, w1, w2 = request.hidden_states, request.w1, request.w2
    ids, weights = request.topk_ids, request.topk_weight
    tokens, model_dim = x.shape
    experts = w1.shape[0]
    inter_dim = w1.shape[1] // 2
    topk = ids.shape[1]
    if not (
        request.quant_type == QuantType.per_1x32
        and request.activation == ActivationType.Silu
        and w1.dtype == dtypes.fp4x2
        and w2.dtype == dtypes.fp4x2
        and request.expert_mask is None
        and ids.dtype == torch.int32
        and weights.dtype == torch.float32
        and request.dtype in (None, x.dtype)
        and experts <= 257
        and tokens * topk <= 1024
    ):
        raise NotImplementedError(
            "mxfp4 fold MoE supports Silu a4w4 per_1x32, int32 ids, fp32 weights, "
            "no mask, <= 257 experts"
        )
    launch = _stage1(
        kname1,
        tokens,
        model_dim,
        inter_dim,
        experts,
        topk,
        runtime_swiglu_limit(request.swiglu_limit, "silu"),
        dpsk,
    )
    dev = x.device
    routes = tokens * topk
    max_sorted = (routes + min(routes, experts) * 15 + 15) // 16 * 16
    sorted_ids = torch.empty(max_sorted, dtype=torch.int32, device=dev)
    sorted_weights = torch.empty(max_sorted, dtype=torch.float32, device=dev)
    sorted_expert_ids = torch.empty(max_sorted // 16, dtype=torch.int32, device=dev)
    num_valid_ids = torch.empty(2, dtype=torch.int32, device=dev)
    inter = torch.empty((max_sorted, inter_dim // 2), dtype=torch.uint8, device=dev)
    # One padded native scale chunk per 16-row block (as _mxfp4_a4w4_stage1 sizes it).
    scale_cols = inter_dim // 32
    scale_bytes = max(
        max_sorted * max(64, scale_cols * 2),
        max_sorted // 16 * kas_per_chunk_dw_for(inter_dim) * 4,
    )
    scale_rows = ((scale_bytes + scale_cols - 1) // scale_cols + 31) // 32 * 32
    inter_scale = torch.empty((scale_rows, scale_cols), dtype=torch.uint8, device=dev)
    out = torch.empty_like(x)
    part = flags = 0
    if dpsk > 1:
        part, flags = _dpsk_buffers(dev, max_sorted // 16 * (2 * inter_dim // 128), dpsk)
        part, flags = part.data_ptr(), flags.data_ptr()
    _run_compiled(
        launch,
        *(
            w1.data_ptr(),
            request.w1_scale.data_ptr(),
            sorted_expert_ids.data_ptr(),
            num_valid_ids.data_ptr(),
            inter.data_ptr(),
            inter_scale.data_ptr(),
            x.contiguous().data_ptr(),
            ids.contiguous().data_ptr(),
            weights.contiguous().data_ptr(),
            sorted_ids.data_ptr(),
            sorted_weights.data_ptr(),
            out.data_ptr(),
            part,
            flags,
            torch.cuda.current_stream(dev),
        ),
    )
    return _flydsl_v2_stage2_wrapper(
        inter,
        w1,
        w2,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        out,
        topk,
        kernelName=kname2,
        model_dim=model_dim,
        inter_dim=inter_dim,
        num_experts=experts,
        w2_scale=request.w2_scale,
        a2_scale=inter_scale,
        sorted_weights=sorted_weights,
        block_m=16,
    )
