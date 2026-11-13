# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host side of the MiniMax-M3 KV-outer sparse attention (see kernels/m3_kvouter.py).

Stages: invert the per-query top-k into (kv head, block) -> edges and cut it into
tiles, KV-outer attention emitting per-edge partials, LSE combine.
"""

import functools

import torch

from .kernels import m3_kvouter as K
from .kernels.tensor_shim import _run_compiled, current_stream_arg, ptr_arg

__all__ = ["m3_kvouter_plan", "m3_kvouter_attention", "m3_kvouter_attention_only", "m3_kvouter_combine",
           "m3_kvouter_decode", "m3_kvouter_partials"]


@functools.cache
def _cu_count(device_index):
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@functools.cache
def _attn_launcher(hkv, waves_per_eu=K.ATTN_WAVES_PER_EU, contig=False, nt=False, pf=False, u8=False):
    return K.build_attn(hkv, waves_per_eu, contig, nt, pf, u8)


@functools.cache
def _combine_launcher(hkv, u8=False, w16=False):
    return K.build_combine(hkv, u8, w16)


@functools.cache
def _decode_launcher(hkv, q_len, splits, pairs, khalves, xcd=False, htab=0, c16=False, tsplit=1, direct=False,
                     dspec=True):
    return K.build_decode(hkv, q_len, splits, pairs, khalves, xcd, htab, c16, tsplit, direct, dspec)


@functools.cache
def _plan_launcher(lds_count=True, top_scatter=True):
    return K.build_plan(lds_count, top_scatter)


@functools.cache
def _plan_hist_launcher(rpt, ept=1, qpt=1):
    return K.build_plan_hist(rpt, ept, qpt)


_count_bufs = {}
COMBINE_W16_MIN = 8192  # (kv head, query) pairs from which the combine is bandwidth-bound and 16 B loads win
LDS_COUNT_MIN_EDGES = 65536  # below, the per-(kv head, replica) LDS count WGs are a longer chain than atomics
HIST_MAX_EPT = 2  # own edges per thread of the per-head histogram plan
HIST_QWG_PER_CU = 4  # Q-quant WGs per CU of the per-head histogram plan (more chunks per thread beyond)


def _zeroed_counts(n, device):
    """Slot counters; the scan kernels re-zero what they read, so they stay zero."""
    buf = _count_bufs.get(device)
    if buf is None or buf.numel() < n:
        buf = torch.zeros(max(n, 1 << 16), dtype=torch.int32, device=device)
        _count_bufs[device] = buf
    return buf


def m3_kvouter_plan(topk_idx, block_table, query_req_id, query_abs_pos, q, plan_hist=True, lds_count=None,
                    top_scatter=True):
    """Invert topk_idx [Hkv, N, topk] (logical 128-block ids, -1 pad) on device, sync-free,
    and quantize q [N, Hq, 128] bf16 to e4m3 rows once.

    Returns (tiles [T_max, 8] int32, n_tiles [Hkv] int32, edge_q [E_max] int32 (query * 16 + rank),
    inv [Hkv, N, topk] int32 edge id or -1, E_max, q8 [N, Hq, 128] e4m3 bytes, q_scale [N*Hq] f32,
    number of requests, (tile segments, segment stride)).
    Tiles past n_tiles are garbage. With nseg segments, kv head h's n_tiles[h] tiles start
    at tile h * stride (else n_tiles[0] counts all). plan_hist: small problems (N <= 2048 topk
    rows and <= 1024 slots per kv head, at most one plan WG per CU) plan in one launch with
    per-head tile segments (same tiles; False forces the 3-launch plan).
    lds_count: the 4-launch plan counts per (kv head, replica) in LDS instead of with global
    atomics (default from LDS_COUNT_MIN_EDGES edges up). top_scatter: the scatter WGs scan
    the block totals themselves (one launch less) when they fit K.TOP_SCAN_BLOCKS.
    """
    hkv, n, topk = topk_idx.shape
    dev = topk_idx.device
    nreq, maxb = block_table.shape
    assert topk == 16  # edge_q packs query * 16 + rank; partial slots are [Hkv, N, 16]
    nslot = hkv * nreq * maxb
    n_edge = hkv * n * topk
    t_max = -(-n_edge // K.EDGES_PER_TILE) + nslot
    inv = torch.empty((hkv, n, topk), dtype=torch.int32, device=dev)
    edge_q = torch.empty(n_edge, dtype=torch.int32, device=dev)
    tiles = torch.empty((t_max, K.TILE_INTS), dtype=torch.int32, device=dev)
    n_tiles = torch.empty(hkv, dtype=torch.int32, device=dev)
    assert q.is_contiguous() and q.shape[0] == n and q.shape[-1] == K.HEAD_DIM
    qrows = n * q.shape[1]
    q8 = torch.empty(q.shape, dtype=torch.uint8, device=dev)
    qs = torch.empty((qrows,), dtype=torch.float32, device=dev)
    egrid = -(-n_edge // K.NTHREADS)
    cgrid = max(egrid, -(-qrows * 16 // K.NTHREADS))
    nslot_h = nreq * maxb
    # own edges per plan-WG thread: at most one plan WG per CU (two share the LDS atomic rate)
    ept = 1
    while hkv * -(-n * topk // (K.SCAN_T * ept)) > _cu_count(dev.index) and ept < HIST_MAX_EPT:
        ept *= 2
    wph = -(-n * topk // (K.SCAN_T * ept))
    if (plan_hist and nslot_h <= K.HIST_SLOTS and n <= K.SCAN_T * K.HIST_ROWS_PER_THREAD
            and hkv * wph <= _cu_count(dev.index)):
        # small problems: one launch, every plan WG histograms all picks of its kv head in LDS
        # itself; tiles in per-head segments
        th = -(-n * topk // K.EDGES_PER_TILE) + nslot_h
        assert th * hkv <= t_max
        qwg = -(-qrows * 16 // K.SCAN_T)
        qpt = 1
        while qwg > HIST_QWG_PER_CU * _cu_count(dev.index) * qpt and qpt < 8:
            qpt *= 2
        cgrid = hkv * wph + -(-qwg // qpt)  # Q quant on the WGs past the plan WGs
        _run_compiled(
            _plan_hist_launcher(1 << max(0, (-(-n // K.SCAN_T) - 1).bit_length()), ept, qpt),
            ptr_arg(topk_idx), ptr_arg(query_abs_pos), ptr_arg(query_req_id), ptr_arg(block_table),
            ptr_arg(inv), ptr_arg(edge_q), ptr_arg(tiles), ptr_arg(n_tiles), ptr_arg(q), ptr_arg(q8), ptr_arg(qs),
            int(n), int(nreq), int(maxb), int(block_table.stride(0)), int(n_edge), int(nslot),
            int(hkv * wph), int(qrows), int(cgrid), int(wph), int(th), current_stream_arg(),
        )
        return tiles, n_tiles, edge_q, inv, n_edge, q8, qs, nreq, (hkv, th)
    count = _zeroed_counts(nslot * K.COUNT_REPLICAS, dev)
    rbase = torch.empty(nslot * K.COUNT_REPLICAS, dtype=torch.int32, device=dev)
    start = torch.empty(nslot + 1, dtype=torch.int32, device=dev)
    tstart = torch.empty(nslot, dtype=torch.int32, device=dev)
    nblk = -(-nslot // K.SCAN_T)
    bsum = torch.empty(2 * (nblk + 1), dtype=torch.int32, device=dev)
    if lds_count is None:
        lds_count = n_edge >= LDS_COUNT_MIN_EDGES
    if lds_count:
        cgrid = hkv * K.COUNT_REPLICAS + -(-qrows * 16 // K.NTHREADS)
    _run_compiled(
        _plan_launcher(lds_count, top_scatter and nblk <= K.TOP_SCAN_BLOCKS),
        ptr_arg(topk_idx), ptr_arg(query_abs_pos), ptr_arg(query_req_id), ptr_arg(block_table),
        ptr_arg(count), ptr_arg(rbase), ptr_arg(start), ptr_arg(tstart),
        ptr_arg(inv), ptr_arg(edge_q), ptr_arg(tiles), ptr_arg(n_tiles), ptr_arg(bsum),
        ptr_arg(q), ptr_arg(q8), ptr_arg(qs),
        int(n), int(topk), int(nreq), int(maxb), int(block_table.stride(0)), int(n_edge), int(nslot),
        int(egrid), int(nblk), int(qrows), int(cgrid), current_stream_arg(),
    )
    return tiles, n_tiles, edge_q, inv, n_edge, q8, qs, nreq, (1, 0)


def m3_kvouter_plan_torch(topk_idx, block_table, query_req_id, query_abs_pos):
    """Invert topk_idx [Hkv, N, topk] (logical 128-block ids, -1 pad).

    Returns (tiles [T, 8] int32, n_tiles [1] int32, edge_q [E] int32 (query * topk + rank),
    inv [Hkv, N, topk] int32).
    """
    hkv, n, topk = topk_idx.shape
    dev = topk_idx.device
    maxb = block_table.shape[1]
    nreq = block_table.shape[0]
    t = topk_idx.long()
    self_blk = (query_abs_pos.long() // K.BLOCK)[None, :, None]
    valid = (t >= 0) & (t <= self_blk)
    h = torch.arange(hkv, device=dev)[:, None, None]
    slot = (h * nreq + query_req_id.long()[None, :, None]) * maxb + t.clamp(min=0)
    big = hkv * nreq * maxb
    key = torch.where(valid, slot, big).flatten()
    skey, order = torch.sort(key, stable=True)
    n_edge = int(valid.sum().item())
    order = order[:n_edge]
    skey = skey[:n_edge]
    edge_q = (((order // topk) % n) * topk + order % topk).to(torch.int32)
    inv = torch.full((hkv * n * topk,), -1, dtype=torch.int32, device=dev)
    inv[order] = torch.arange(n_edge, dtype=torch.int32, device=dev)
    uniq, counts = torch.unique_consecutive(skey, return_counts=True)
    starts = torch.cumsum(counts, 0) - counts
    ntile_per = (counts + K.EDGES_PER_TILE - 1) // K.EDGES_PER_TILE
    slot_of_tile = torch.repeat_interleave(torch.arange(uniq.numel(), device=dev), ntile_per)
    tile_first = torch.cumsum(ntile_per, 0) - ntile_per
    sub = torch.arange(slot_of_tile.numel(), device=dev) - tile_first[slot_of_tile]
    s_key = uniq[slot_of_tile]
    kvh = s_key // (nreq * maxb)
    req = (s_key // maxb) % nreq
    lblk = s_key % maxb
    tiles = torch.zeros((slot_of_tile.numel(), K.TILE_INTS), dtype=torch.int32, device=dev)
    tiles[:, 0] = kvh
    tiles[:, 1] = block_table[req, lblk]
    tiles[:, 2] = lblk
    tiles[:, 3] = starts[slot_of_tile] + sub * K.EDGES_PER_TILE
    tiles[:, 4] = torch.clamp(counts[slot_of_tile] - sub * K.EDGES_PER_TILE, max=K.EDGES_PER_TILE)
    n_tiles = torch.tensor([tiles.shape[0]], dtype=torch.int32, device=dev)
    return tiles, n_tiles, edge_q, inv.view(hkv, n, topk), n_edge


def m3_kvouter_partials(n_edge, device, u8):
    """Partial buffers (O, lse, scale): O rows are bf16, or u8 codes with two f32 scales per row."""
    rows = max(n_edge, 1) * K.GQA
    po = torch.empty((rows, K.HEAD_DIM), dtype=torch.uint8 if u8 else torch.bfloat16, device=device)
    pl = torch.empty((rows,), dtype=torch.float32, device=device)
    return po, pl, (torch.empty((rows, 2), dtype=torch.float32, device=device) if u8 else None)


def m3_kvouter_attention_only(q, k_cache, v_cache, k_scale, v_scale, plan, query_abs_pos, sm_scale,
                              partial_o, partial_lse, grid=None, pf=None, partial_scale=None):
    """A uint8 partial_o selects u8 partials (needs partial_scale [rows, 2] f32 and pf)."""
    block_table_rows = plan[7]
    tiles, n_tiles, edge_q, inv, n_edge, q8, qs, _ = plan[:8]
    nseg, tseg = plan[8] if len(plan) > 8 else (1, 0)
    hkv = k_cache.shape[1]
    if grid is None:
        grid = max(1, min(tiles.shape[0], _cu_count(q.device.index) * 2))
    # Contiguous tile runs pay off only when a block's edges span tiles, i.e. requests
    # average more query tokens than a tile holds (prefill, not decode / MTP verify).
    contig = bool(q.shape[0] > K.EDGES_PER_TILE * block_table_rows)
    # Partials that cannot stay in MALL until the combine are stored nontemporal.
    nt = bool(partial_o.numel() * partial_o.element_size() >= K.MALL_BYTES)
    # Next tile's K/V by async DMA while this one computes (software-pipelined loop);
    # only for contiguous runs, where tiles of one WG walk consecutive blocks.
    if pf is None:
        pf = contig
    u8 = partial_o.dtype == torch.uint8
    assert u8 == pf and (not u8 or partial_scale is not None)
    # partial slots are query-major [Hkv, N, topk=16] (see kernels/m3_kvouter.py)
    assert partial_o.shape[0] == hkv * q.shape[0] * 16 * K.GQA and inv.shape[2] == 16
    assert partial_o.numel() * partial_o.element_size() < 1 << 31
    _run_compiled(
        _attn_launcher(hkv, contig=contig, nt=nt, pf=pf, u8=u8),
        ptr_arg(q8), ptr_arg(qs), ptr_arg(k_cache), ptr_arg(v_cache), ptr_arg(k_scale), ptr_arg(v_scale),
        ptr_arg(tiles), ptr_arg(n_tiles), ptr_arg(edge_q), ptr_arg(query_abs_pos),
        ptr_arg(partial_o), ptr_arg(partial_lse), ptr_arg(partial_lse if partial_scale is None else partial_scale),
        float(sm_scale * K.LOG2E), int(grid), int(q.shape[0]), int(nseg), int(tseg),
        current_stream_arg(),
    )


def m3_kvouter_attention(
    q, k_cache, v_cache, k_scale, v_scale, plan, query_abs_pos, sm_scale, out=None, grid=None,
    partial_o=None, partial_lse=None, pf=None, partial_scale=None, u8=None,
):
    """q [N, Hq, 128] bf16; caches page-16 SHUFFLE fp8 [nph16, Hkv, ...]; scales [nph16, Hkv, 16].
    u8 (default: with the pipelined prefill loop) halves the partial-O bytes."""
    tiles, n_tiles, edge_q, inv, n_edge = plan[:5]
    n, hq, d = q.shape
    hkv = k_cache.shape[1]
    assert d == K.HEAD_DIM and hq == hkv * K.GQA and q.is_contiguous()
    if out is None:
        out = torch.empty_like(q)
    if partial_o is None:
        if u8 is None:
            u8 = (pf if pf is not None else bool(q.shape[0] > K.EDGES_PER_TILE * plan[7]))
        partial_o, partial_lse, partial_scale = m3_kvouter_partials(n_edge, q.device, u8)
    m3_kvouter_attention_only(q, k_cache, v_cache, k_scale, v_scale, plan, query_abs_pos, sm_scale,
                              partial_o, partial_lse, grid, pf=pf, partial_scale=partial_scale)
    m3_kvouter_combine(partial_o, partial_lse, inv, out, partial_scale)
    return out


def m3_kvouter_combine(partial_o, partial_lse, inv, out, partial_scale=None, w16=None):
    """w16 (u8 only; default from COMBINE_W16_MIN pairs up): 16 B loads, two queries per WG."""
    hkv, n, _ = inv.shape
    u8 = partial_o.dtype == torch.uint8
    if w16 is None:
        w16 = hkv * n >= COMBINE_W16_MIN
    w16 = w16 and u8
    grid = -(-hkv * n // 2) if w16 else hkv * n
    _run_compiled(
        _combine_launcher(hkv, u8, w16),
        ptr_arg(partial_o), ptr_arg(partial_lse), ptr_arg(partial_lse if partial_scale is None else partial_scale),
        ptr_arg(inv), ptr_arg(out), int(n), int(grid),
        current_stream_arg(),
    )
    return out


_req_counters = {}
_decode_partials = {}


def m3_kvouter_decode(q, k_cache, v_cache, k_scale, v_scale, topk_idx, block_table, query_abs_pos, q_len,
                      sm_scale, out=None, splits=None, pairs=K.DEC_PAIRS, khalves=1, tsplit=None, direct=None):
    """Decode / MTP verify in one launch (plan, attention and combine fused).

    Tokens come as block_table.shape[0] requests of q_len (<= 4) consecutive tokens;
    blocks of different requests must not alias. topk_idx [Hkv, N, topk=16].
    The arrival counters and partial scratch are per-device module state: one stream at a time."""
    n, hq, d = q.shape
    hkv = k_cache.shape[1]
    nreq = block_table.shape[0]
    assert d == K.HEAD_DIM and hq == hkv * K.GQA and q.is_contiguous() and topk_idx.is_contiguous()
    assert n == nreq * q_len and topk_idx.shape == (hkv, n, 16) and 1 <= q_len <= 4
    dev = q.device
    ncu = _cu_count(dev.index or 0)
    if splits is None:
        # splits per (request, kv head): the most of 4, 2, 1 that keeps the grid within one WG per CU
        splits = 1
        for sp in (4, 2):
            if nreq * hkv * sp <= ncu:
                splits = sp
                break
    if out is None:
        out = torch.empty_like(q)
    cnt = _req_counters.get(dev)
    if cnt is None or cnt.numel() < nreq * hkv * 4 * K.CNT_STRIDE:
        cnt = torch.zeros(max(nreq * hkv * 4, 256) * K.CNT_STRIDE, dtype=torch.int32, device=dev)
        _req_counters[dev] = cnt
    rows = n * hkv * splits * -(-16 * q_len // (splits * pairs)) * K.GQA
    bufs = _decode_partials.get(dev)
    if bufs is None or bufs[1].numel() < rows:
        bufs = (torch.empty((rows, K.HEAD_DIM), dtype=torch.bfloat16, device=dev),
                torch.empty((rows,), dtype=torch.float32, device=dev))
        _decode_partials[dev] = bufs
    # XCD-local handoff: WGs go round-robin over the XCDs, so it needs whole rounds of 8 groups
    xcd = (nreq * hkv) % K.N_XCD == 0
    # pick dedupe through an LDS table over the request's block ids; at q_len 1 the table init costs more
    maxb = block_table.shape[1]
    htab = 0 if q_len == 1 or maxb > K.DEC_HTAB_MAX else max(64, 1 << (maxb - 1).bit_length())
    # one token's 16 heads per wave on 16x16x128 MFMAs; q_len 3 (3 waves per block) cannot split the block DMA
    c16 = q_len != 3
    if not c16:
        tsplit = 1
    elif tsplit is None:
        # token runs per WG group: the most that keeps the grid within one WG per CU
        tsplit = 1
        for t in (4, 2):
            if q_len % t == 0 and nreq * hkv * splits * t <= ncu:
                tsplit = t
                break
    if direct is None:
        # block operands straight to VGPRs once the grid fills the GPU; below, the LDS copy's chain is shorter
        direct = nreq * hkv * splits * tsplit >= ncu or (q_len == 1 and nreq * hkv * 4 >= ncu)
    direct = bool(direct and c16 and q_len // tsplit == 1 and khalves == 1)
    # speculative round-0 K for direct WGs (304 VGPRs): only while the grid fits one WG per CU
    dspec = nreq * hkv * splits * tsplit <= ncu
    _run_compiled(
        _decode_launcher(hkv, q_len, splits, pairs, khalves, xcd, htab, c16, tsplit, direct, dspec),
        ptr_arg(q), ptr_arg(k_cache), ptr_arg(v_cache), ptr_arg(k_scale), ptr_arg(v_scale),
        ptr_arg(topk_idx), ptr_arg(block_table), ptr_arg(query_abs_pos), ptr_arg(bufs[0]), ptr_arg(bufs[1]),
        ptr_arg(cnt), ptr_arg(out), float(sm_scale * K.LOG2E), int(n), int(block_table.stride(0)),
        int(nreq * hkv * splits * tsplit), current_stream_arg(),
    )
    return out
