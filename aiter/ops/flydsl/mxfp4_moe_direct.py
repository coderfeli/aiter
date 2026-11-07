# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Whole-graph MXFP4 (a4w4) MoE for 1-2 tokens, selected by the tuned kernel name
``impl__flydsl_mxfp4_direct__s<split>[w<k_waves>][c<cols>][b<stage2 cols>]``: two launches and no routing sort
(see kernels/mxfp4_moe_direct.py)."""

import re

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch

from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe_registry import FusedMoeRequest
from aiter.ops.flydsl.kernels.mxfp4_moe_direct import (
    STAGE2_BN,
    build_mxfp4_moe_direct_module,
)
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled


def _ptr(t):
    return flyc.from_c_void_p(fx.Uint8, t.data_ptr(), assumed_align=16)


def run_mxfp4_moe_direct_impl(request: FusedMoeRequest, config: str) -> torch.Tensor:
    m = re.fullmatch(r"s(\d+)(?:w(\d+))?(?:c(\d+))?(?:b(\d+))?", config)
    if m is None:
        raise ValueError(f"mxfp4 direct MoE: bad config {config!r}")
    split, k_waves, cols = int(m[1]), int(m[2] or 1), int(m[3] or 64)
    bn = int(m[4] or STAGE2_BN)
    x, w1, w2 = request.hidden_states, request.w1, request.w2
    ids, weights = request.topk_ids, request.topk_weight
    if not (
        request.quant_type == QuantType.per_1x32
        and request.activation == ActivationType.Silu
        and w1.dtype == dtypes.fp4x2
        and w2.dtype == dtypes.fp4x2
        and request.expert_mask is None
        and ids.dtype == torch.int32
        and weights.dtype == torch.float32
        and request.dtype in (None, x.dtype)
    ):
        raise NotImplementedError(
            "mxfp4 direct MoE supports Silu a4w4 per_1x32, int32 ids, fp32 weights, no mask"
        )
    tokens, model_dim = x.shape
    inter_dim = w1.shape[1] // 2
    topk = ids.shape[1]
    launch = build_mxfp4_moe_direct_module(
        tokens, model_dim, inter_dim, topk, split, k_waves, cols, bn
    )
    h = torch.empty((split, tokens * topk, 2 * inter_dim), dtype=torch.float32, device=x.device)
    out = torch.empty_like(x)
    _run_compiled(
        launch,
        *(
            _ptr(t.contiguous())
            for t in (x, w1, request.w1_scale, w2, request.w2_scale, ids, weights, h, out)
        ),
        fx.Stream(torch.cuda.current_stream(x.device)),
    )
    return out
