# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Block-ordered pair / single work list over sorted MoE blocks.

Within an expert's run of sort blocks, blocks at even run offsets (0, 2, ...)
lead a pair with the next block when it has the same expert; the follower is
skipped, every other valid block is a single. The run offset looks back at most
three blocks (a longer run only loses pairs, never overlaps them). Layout of
``pw`` (int32):

    pw[0]        number of entries
    pw[1 + i]    (block << 16) | (is_pair << 15) | (is_half << 14) | expert,
                 in block order

``is_half`` marks a single whose valid rows fit in its first half (row
``sort_block_m // 2`` is padding); stage 1 runs it with a half-height tile.

One wave; each of ``CHUNKS`` unrolled chunks covers 64 blocks, so all
expert-id loads of a pass are issued before the first ballot.
"""

import flydsl.compiler as flyc
import torch
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr
from flydsl.expr import rocdl as fly_rocdl
from flydsl.expr.typing import T

from .tensor_shim import _run_compiled

CHUNKS = 16
LOOKBACK = 4


def _buf_iter(tensor):
    return fx.get_iter(fx.rocdl.make_buffer_tensor(tensor, max_size=True))


def _gld(it, idx):
    return fx.ptr_load(it + fx.Int64(idx))


def _gst(it, val, idx):
    fx.ptr_store(val, it + fx.Int64(idx))


_cache = {}


def get_pair_worklist(sort_block_m: int):
    if sort_block_m in _cache:
        return _cache[sort_block_m]
    half_row = sort_block_m // 2

    @flyc.kernel(known_block_size=[64, 1, 1])
    def pair_worklist_kernel(
        pw: fx.Tensor,
        expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        sorted_ids: fx.Tensor,
        i32_size: fx.Int32,
        i32_m: fx.Int32,
    ):
        lane = fx.Int32(gpu.thread_idx.x)
        pw_it = _buf_iter(pw)
        e_it = _buf_iter(expert_ids)
        s_it = _buf_iter(sorted_ids)
        nv = _gld(_buf_iter(num_valid_ids), fx.Int32(0))
        c0 = fx.Int32(0)
        c1 = fx.Int32(1)
        nb = (nv + fx.Int32(sort_block_m - 1)) // fx.Int32(sort_block_m)
        nb = (nb < i32_size).select(nb, i32_size)
        lt_mask = (fx.Int64(1) << fx.Int64(lane)) - fx.Int64(1)
        n_pass = (nb + fx.Int32(64 * CHUNKS - 1)) // fx.Int32(64 * CHUNKS)

        for p, state in range(c0, n_pass, c1, init=[c0]):
            cnt = state[0]
            base = fx.Int32(p) * fx.Int32(64 * CHUNKS) + lane
            ev = []
            hv = []
            for c in range_constexpr(CHUNKS):
                b = base + fx.Int32(64 * c)
                # e[b - LOOKBACK .. b + 1]; indices below 0 are clamped and
                # masked by `same` below.
                row = []
                for d in range_constexpr(-LOOKBACK, 2):
                    i = b + fx.Int32(d)
                    row.append(_gld(e_it, (i > c0).select(i, c0)))
                ev.append(row)
                bc = (b < nb).select(b, c0)
                hv.append(_gld(s_it, bc * fx.Int32(sort_block_m) + fx.Int32(half_row)))
            for c in range_constexpr(CHUNKS):
                b = base + fx.Int32(64 * c)
                row = ev[c]

                def same(d):  # e[b+d] == e[b+d-1], both in [0, nb)
                    i = b + fx.Int32(d)
                    ok = (i >= c1) & (i < nb)
                    return ok & (row[LOOKBACK + d] == row[LOOKBACK + d - 1])

                def even_offset(d):  # run offset of block b+d (capped at 3) is even
                    r1 = same(d)
                    r2 = r1 & same(d - 1)
                    r3 = r2 & same(d - 2)
                    return (~r1) | (r2 & (~r3))

                valid = b < nb
                lead = valid & even_offset(0) & same(1)
                follow = valid & even_offset(-1) & same(0)
                take = valid & (~follow)
                mt = fx.Int64(fly_rocdl.ballot(T.i64, take))
                rt = fx.Int32(fx.math.ctpop(mt & lt_mask))
                e = row[LOOKBACK]
                half = (~lead) & (
                    (hv[c] & fx.Int32(0xFFFFFF)) >= i32_m
                )
                packed = (
                    (b << fx.Int32(16))
                    | (lead.select(fx.Int32(1 << 15), c0))
                    | (half.select(fx.Int32(1 << 14), c0))
                    | e
                )
                if take:
                    _gst(pw_it, packed, c1 + cnt + rt)
                cnt = cnt + fx.Int32(fx.math.ctpop(mt))
            results = yield [cnt]
        cnt = results
        if lane == c0:
            _gst(pw_it, cnt, c0)

    @flyc.jit
    def launch_pair_worklist(
        pw: fx.Tensor,
        expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        sorted_ids: fx.Tensor,
        i32_size: fx.Int32,
        i32_m: fx.Int32,
        stream: fx.Stream = None,
    ):
        stream = stream if stream is not None else fx.Stream(None)
        pair_worklist_kernel(
            pw, expert_ids, num_valid_ids, sorted_ids, i32_size, i32_m
        ).launch(
            grid=(1, 1, 1), block=(64, 1, 1), stream=stream
        )

    _cache[sort_block_m] = launch_pair_worklist
    return launch_pair_worklist


# Stage 1's list, keyed by the sorted_expert_ids pointer, for stage 2 to reuse.
# Stage 2 pops it on every call, so a list never outlives its fused_moe call.
_LAST = {}


def build_pair_worklist(
    sorted_expert_ids, num_valid_ids, sorted_token_ids, num_tokens, sort_block_m, stream
):
    size = int(sorted_expert_ids.shape[0])
    pw = torch.empty(1 + size, dtype=torch.int32, device=sorted_expert_ids.device)
    _run_compiled(
        get_pair_worklist(sort_block_m),
        pw,
        sorted_expert_ids,
        num_valid_ids,
        sorted_token_ids,
        size,
        int(num_tokens),
        stream,
    )
    return pw


def take_pair_worklist(sorted_expert_ids):
    return _LAST.pop(sorted_expert_ids.data_ptr(), None)


class PairUnifiedLaunch:
    """Stage-1 launch as work-list build + one two-body GEMM kernel."""

    def __init__(self, exe, sort_block_m):
        self.exe = exe
        self.sort_block_m = sort_block_m

    def run(
        self,
        args,
        sorted_expert_ids,
        num_valid_ids,
        eid_arg_pos,
        size,
        sorted_token_ids,
        num_tokens,
    ):
        from .tensor_shim import ptr_arg

        size = int(size)
        stream = args[-1]
        pw = torch.empty(1 + size, dtype=torch.int32, device=sorted_expert_ids.device)
        _run_compiled(
            get_pair_worklist(self.sort_block_m),
            pw,
            sorted_expert_ids,
            num_valid_ids,
            sorted_token_ids,
            size,
            int(num_tokens),
            stream,
        )
        _LAST[sorted_expert_ids.data_ptr()] = pw
        args = list(args)
        args[eid_arg_pos] = ptr_arg(pw)
        _run_compiled(self.exe, *args)
