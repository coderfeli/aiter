# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch

from ..jit.core import compile_ops

MD_NAME = "module_moe_sorting_opus"


@compile_ops("module_moe_sorting_opus", develop=True)
def moe_sorting_opus_get_workspace_size(
    tokens: int,
    num_experts: int,
    topk: int,
    dispatch_policy: int = 0,
    device_id: int = -1,
) -> int: ...


@compile_ops("module_moe_sorting_opus", develop=True)
def moe_sorting_opus_fwd(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    moe_buf: torch.Tensor,
    num_experts: int,
    unit_size: int,
    local_expert_mask: torch.Tensor | None = None,
    num_local_tokens: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
    dispatch_policy: int = 0,
    local_topk_ids: torch.Tensor | None = None,
    m_indices: torch.Tensor | None = None,
    reverse_sorted: torch.Tensor | None = None,
) -> None: ...


@compile_ops("module_moe_sorting_opus", develop=True)
def mxfp4_moe_sort_quant_fwd(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    moe_buf: torch.Tensor,
    activation_quant: torch.Tensor,
    activation_scale_token: torch.Tensor,
    num_experts: int,
    num_local_tokens: torch.Tensor | None = None,
) -> None:
    """Fused route sort + compact MXFP4 activation quantization.

    gfx950 only, ``model_dim=4096``, ``block_m=32``, ``M`` in ``[1, 128]``.
    ``(num_experts, topk)`` must be one of ``(512, 10)`` or ``(513, 11)``
    -- the expert histogram is a kernel template parameter.
    ``num_local_tokens``: as for ``moe_sorting_opus_fwd`` -- only that many
    leading rows are sorted and quantized; ``moe_buf`` is zeroed whole.
    """


@compile_ops("module_moe_sorting_opus", develop=True)
def moe_sorting_opus_quant_fwd(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    moe_buf: torch.Tensor,
    num_experts: int,
    unit_size: int,
    workspace: torch.Tensor,
    m_indices: torch.Tensor,
    reverse_sorted: torch.Tensor,
    hidden_states: torch.Tensor,
    activation_quant: torch.Tensor,
    activation_scale_token: torch.Tensor,
    activation_scale: torch.Tensor,
) -> None:
    """Opus multi-phase sort (P0_v2 + P23) with the MXFP4 activation prep folded in.

    Extra P0_v2 CTAs quantize ``hidden_states`` per token into
    ``activation_quant`` / ``activation_scale_token``; each P23 expert CTA then
    writes its rows' sorted, swizzled E8M0 scales into ``activation_scale``
    (the layout ``mxfp4_moe_sort_fwd`` produces). gfx950, uint8-mesh two-launch
    sort path only.

    An empty ``workspace`` selects a one-launch sort + quant instead (needs an
    empty ``activation_scale``, <= 1024 experts and <= 36864 routes).
    """
