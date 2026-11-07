# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Whole-graph MXFP4 (a4w4) decode MoE with the routing sort folded into stage 1,
selected by the tuned kernel name ``impl__flydsl_mxfp4_fold2__v1``: two launches
of the kernels in kernels/mxfp4_moe_fold2/.

Stage 1 (g1_fold.py), one CTA per (16-row expert block, 256-column gate/up
n-block): rebuilds the padded per-expert sort of the routed items in LDS,
publishes the sorted tables, zeroes ``out``, quantizes its 16 hidden rows to
MXFP4 inline and writes the MXFP4 SiLU intermediate. Its head issues the first
B stages as soon as the block's expert is known. Stage 2 (g2_port.py): BN256
BK256 down GEMM with bf16x2 atomic adds into ``out``."""

import functools

import torch

from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe_registry import FusedMoeRequest
from aiter.ops.flydsl.kernels.mxfp4_moe_fold2.g1_fold import (
    compile_gemm1_a4w4_fold,
    gemm1_grid,
)
from aiter.ops.flydsl.kernels.mxfp4_moe_fold2.g2_port import compile_gemm2_a4w4_port

_BM = 16
_BN = 256
_BK = 256


def _parse_config(config):
    """"v1" = stage-1 BN256 without XCD swizzle; else "_"-joined bn<N>/xcd<N>.
    Stage-1 kstages stays 2: the fold kernel is wrong at 3 and 4."""
    knobs = {"bn": 256, "xcd": 0}
    if config != "v1":
        for part in config.split("_"):
            key = part.rstrip("0123456789")
            if key not in knobs or key == part:
                raise ValueError(f"mxfp4 fold2 MoE: bad config {config!r}")
            knobs[key] = int(part[len(key):])
    return knobs["bn"], knobs["xcd"]


@functools.cache
def _plan(tokens, model_dim, inter_dim, experts, topk, bn1, xcd1):
    routes = tokens * topk
    max_sorted = (routes + min(routes, experts) * (_BM - 1) + _BM - 1) // _BM * _BM
    n_blocks = max_sorted // _BM
    stage1 = compile_gemm1_a4w4_fold(
        _BM,
        True,
        True,
        D_HIDDEN=model_dim,
        D_INTER=inter_dim,
        NE=experts,
        TOPK=topk,
        BN=bn1,
        BK=_BK,
        interleave=False,
        xcd_swizzle=xcd1,
        kstages=2,
        kastages=0,
        fold=True,
        M_CONST=tokens,
        splitk=1,
        splitk_aux=(4, 5),
        nfuse=1,
        splitk_p16=0,
        splitk_rows=min(tokens, _BM),
        krot=0,
        xbody=0,
        xres=0,
        splitk_rd=0,
        item=0,
    )
    stage2 = compile_gemm2_a4w4_port(
        BM=_BM,
        use_nt=True,
        NE=experts,
        N_OUT=model_dim,
        epilog="atomic",
        D_INTER=inter_dim,
        D_INTER_REAL=None,
        BN=_BN,
        BK=_BK,
        xcd_swizzle=0,
    )
    grid1 = gemm1_grid(tokens, _BM, NE=experts, TOPK=topk, INTER=inter_dim, BN=bn1)
    assert grid1 == n_blocks * (2 * inter_dim // bn1)
    # intermediate e8m0 scales: max_sorted * 64 bytes in rows of inter_dim/32 bytes,
    # the row count rounded up to 32
    scale_cols = inter_dim // 32
    scale_rows = ((max_sorted * 64 + scale_cols - 1) // scale_cols + 31) // 32 * 32
    sizes = (
        max_sorted * 4,  # sorted_token_ids, i32
        max_sorted * 4,  # sorted_weights, f32
        n_blocks * 4,  # sorted_expert_ids, i32
        2 * 4,  # num_valid_ids, i32[2]
        max_sorted * (inter_dim // 2),  # intermediate, MXFP4 bytes
        scale_rows * scale_cols,  # intermediate e8m0 scales
        tokens * (model_dim // 2),  # stage-1 A operand slot (unread under inline quant)
        1,  # stage-2 output-scale slot (unread by the atomic epilogue)
    )
    offsets = []
    total = 0
    for size in sizes:
        offsets.append(total)
        total += (size + 255) // 256 * 256
    return stage1, stage2, grid1, n_blocks, total, tuple(offsets)


def run_mxfp4_moe_fold2_impl(request: FusedMoeRequest, config: str) -> torch.Tensor:
    knobs = _parse_config(config)
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
        and inter_dim == _BN
        and experts <= 257
        and tokens * topk <= 1024
    ):
        raise NotImplementedError(
            "mxfp4 fold2 MoE supports Silu a4w4 per_1x32, inter_dim 256, int32 ids, "
            "fp32 weights, no mask, <= 257 experts"
        )
    stage1, stage2, grid1, n_blocks, ws_bytes, offsets = _plan(
        tokens, model_dim, inter_dim, experts, topk, *knobs
    )
    dev = x.device
    ws = torch.empty(ws_bytes, dtype=torch.uint8, device=dev)
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids = (
        ws.data_ptr() + o for o in offsets[:4]
    )
    inter, inter_scale, a_slot, out_scale_slot = (ws.data_ptr() + o for o in offsets[4:])
    out = torch.empty_like(x)
    stream = torch.cuda.current_stream(dev).cuda_stream
    stage1(
        a_slot,
        0,
        w1.data_ptr(),
        request.w1_scale.data_ptr(),
        sorted_expert_ids,
        num_valid_ids,
        0,
        tokens,
        grid1,
        inter,
        inter_scale,
        x.contiguous().data_ptr(),
        ids.contiguous().data_ptr(),
        weights.contiguous().data_ptr(),
        sorted_ids,
        sorted_weights,
        out.data_ptr(),
        0,
        0,
        stream,
    )
    stage2(
        inter,
        inter_scale,
        w2.data_ptr(),
        request.w2_scale.data_ptr(),
        sorted_expert_ids,
        num_valid_ids,
        sorted_ids,
        sorted_weights,
        tokens,
        n_blocks,
        out.data_ptr(),
        out_scale_slot,
        stream,
    )
    return out
