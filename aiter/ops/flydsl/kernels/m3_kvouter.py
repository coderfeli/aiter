# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 block-sparse GQA, KV-outer (FlyDSL, gfx950).

The schedule is Fireworks' (fw-ai/minimax-kernels): the per-query top-16 is
inverted into (kv_head, 128-block) -> edges, each edge one (query token, rank).
A tile is up to EDGES_PER_TILE edges of one block; the attention kernel loads
the block's K and V once into LDS and emits, per edge, 16 query-head rows of
normalized partial O (bf16, or u8 codes + 2 scales per row in the pipelined prefill
loop) and their log2-sum-exp, one 2 KB (u8: 2 KB + scales) chunk per edge, query-major
(slot = (kv head, query, rank)) so that the combine reads a query's partials contiguously;
`inv` tells it which picks were kept.

Cache layout is ATOM's page-16 SHUFFLE fp8 with per-token scales:
  K [nph16, Hkv, D//16, 16, 16]   V [nph16, Hkv, 1, D, 16]   scales [nph16, Hkv, 16]
so a 128-block of one kv head is 8 contiguous 2 KB pages in either plane.

MFMA is 32x32x64 fp8 with S^T = K Q^T: A rows are keys, B columns query rows.
A K row r of key tile kt is key kt*32 + 16*((r//4)%2) + 4*(r//8) + r%4, which
makes the accumulator element i of lane half h key kt*32 + 16*h + i -- 16
contiguous keys -- so P feeds the PV MFMA from registers and V is read as whole
16-token page rows.
"""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as raw
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.tensor_shim import buf_base_i64, buf_scalar_load, ptr_buf_tensor

HEAD_DIM = 128
BLOCK = 128
GQA = 16
WAVE = 64
WAVES = 4
NTHREADS = WAVE * WAVES
EDGES_PER_WAVE = 32 // GQA  # a wave's 32 MFMA columns are 2 edges x 16 heads
EDGES_PER_TILE = EDGES_PER_WAVE * WAVES
PAGE_BYTES = 16 * HEAD_DIM  # one 16-token page of one kv head, K or V
BLOCK_BYTES = 8 * PAGE_BYTES
O_ROW_BYTES = HEAD_DIM + 16  # one 64-d bf16 half row, padded 4 banks
TILE_INTS = 8
N_XCD = 8
CPOL_NT = 2  # buffer cache policy bit: nontemporal
CPOL_SC1 = 16  # gfx94x/950 sc1: agent-scope coherent access (what LLVM emits for agent atomics)
MALL_BYTES = 256 << 20  # kvh, phys block, logical block, edge start, edge count, pad
LOG2E = 1.4426950408889634
FP8_MAX = 448.0
P_SCALE = 256.0  # P * v_scale / v_scale_max lands in [0, 256] before e4m3
_UNIT_SCALE = 0x7F7F7F7F
U8_MAGIC = 12582912.0 + 128.0  # 1.5 * 2^23 + bias: fma(x, k, U8_MAGIC) has 128 + round(x * k) in its low byte
ATTN_WAVES_PER_EU = 0  # 3 forces 168 VGPRs and spills


def _lds_at(base, off, width):
    return fx.Tensor(
        fx.make_view(
            fx.recast_iter(fx.Uint8, fx.add_offset(base, fx.make_int_tuple(off))),
            fx.make_layout(width, 1),
        )
    )


def _ld(buf, idx, n, ty=None):
    ty = T.i32 if ty is None else ty
    return fx.add_offset(fx.get_iter(buf), idx).load(T.vec(n, ty) if n > 1 else ty)


def _mfma(a, b, c):
    return fx.rocdl.mfma_scale_f32_32x32x64_f8f6f4(
        T.vec(16, T.f32),
        [raw(a), raw(b), raw(c), 0, 0, 0, raw(fx.Int32(_UNIT_SCALE)), 0,
         raw(fx.Int32(_UNIT_SCALE))],
    )


def _pack8(lo, hi):
    return Vec(lo).shuffle(Vec(hi), list(range(8)))


def _fp8x4(vals):
    """4 f32 -> one i32 of packed e4m3."""
    w = fx.rocdl.cvt_pk_fp8_f32(T.i32, raw(vals[0]), raw(vals[1]), raw(fx.Int32(0)), False)
    w = fx.rocdl.cvt_pk_fp8_f32(T.i32, raw(vals[2]), raw(vals[3]), w, True)
    return fx.Int32(w)


def _pair(v, i):
    """Elements i, i + 1 of vector v as a <2 x f32>."""
    v = Vec(v)
    return v.shuffle(v, [i, i + 1])


def _bc2(x):
    return Vec.from_elements([x, x], fx.Float32)


def _max(a, b, nnan=False):
    # nnan: no NaN operand, so LLVM drops the canonicalizing v_max x, x before each max
    return fx.maxnumf(a, b, fastmath="nnan" if nnan else None)


def _tile_range(bid, n_tiles, grid, contig):
    """(first, end, step) of this WG's tiles.

    contig: a contiguous run per WG, so runs of tiles of one block (hot prefill blocks
    have hundreds) reuse the K/V already in LDS. Runs are ranked XCD-major (WGs are dealt
    round-robin over 8 XCDs) so neighbouring runs share an L2; with fewer tiles than WGs
    the first n_tiles WGs take one each. Otherwise (no block spans tiles, e.g. decode)
    WG b takes tiles b, b + grid, ...
    """
    if not contig:
        return bid, n_tiles, grid
    g_act = (n_tiles < grid).select(n_tiles, grid)
    per_xcd = g_act // fx.Int32(N_XCD)
    xcd_major = bid % fx.Int32(N_XCD) * per_xcd + bid // fx.Int32(N_XCD)
    rank = (g_act % fx.Int32(N_XCD) == fx.Int32(0)).select(xcd_major, bid)
    t_lo = rank * n_tiles // g_act
    t_hi = (bid < g_act).select((rank + fx.Int32(1)) * n_tiles // g_act, t_lo)
    return t_lo, t_hi, fx.Int32(1)


def _tile_segments(nt_buf, nseg, tseg, hkv):
    """Tile count and logical -> stored tile index map. A plan may store the tiles of kv head
    h at h * tseg (nseg = Hkv segments, counts n_tiles[0..nseg)); logical tiles run over
    the segments in order. nseg = 1: one segment, the identity."""
    if hkv == 1:
        return fx.Int32(_ld(nt_buf, fx.Int32(0), 1)), lambda t: t
    cnt = Vec(_ld(nt_buf, fx.Int32(0), hkv))
    n = [(fx.Int32(k) < nseg).select(fx.Int32(cnt[k]), fx.Int32(0)) for k in range(hkv)]
    pre = [fx.Int32(0)]
    for k in range(hkv):
        pre.append(pre[-1] + n[k])

    def phys(t):
        p = t
        for k in range(1, hkv):
            p = (t >= pre[k]).select(t - pre[k] + fx.Int32(k) * tseg, p)
        return p
    return pre[hkv], phys


def _block_attn(k_ptr, v_ptr, k_off, v_off, ks_buf, vs_buf, q_ops, q_sc, qpos, page0, kvh, lblk, k_perm, col,
                half, f32_scale_log2, hkv, nkt=4, kb=None, scl_off=None):
    """One wave, keys [kb, kb + 32 * nkt) of a 128-key block in LDS (kb a multiple of 64,
    default the whole block), 32 query rows (2 edges x 16 heads):
    K / V scales come from global memory, or, given scl_off, from LDS at k_ptr + scl_off
    (128 K then 128 V f32 in key order).
    -> (O^T accumulators [4 x 16 f32], O normalizer, log2-sum-exp)."""

    def scales(plane, kt, sdw):
        if scl_off is None:
            return [Vec(_ld(vs_buf if plane else ks_buf, sdw + fx.Int32(4 * j), 4, T.f32)) for j in range_constexpr(4)]
        o = scl_off + fx.Int32(plane * SCL_V) + (kb + fx.Int32(kt * 32) + half * fx.Int32(16)) * fx.Int32(4)
        return [Vec(_lds_at(k_ptr, o + fx.Int32(16 * j), 16).load()).bitcast(fx.Float32) for j in range_constexpr(4)]

    vred = scl_off is not None and nkt == 4

    def _mx(a, b):
        return _max(a, b, True)

    kts = list(range(nkt))
    js = list(range(nkt // 2))
    kb = fx.Int32(0) if kb is None else kb
    kpg0 = kb // fx.Int32(16)
    neg_inf = fx.Float32(float("-inf"))
    # ---- S^T = K Q^T, 4 key tiles of 32 ----
    zero16 = fx.full(16, 0.0, fx.Float32)
    s_acc = []
    for kt in kts:
        key = kb + fx.Int32(kt * 32) + k_perm
        kpg = key // fx.Int32(16)
        ktk = key % fx.Int32(16)
        acc = zero16
        for ws in range_constexpr(2):
            a0 = kpg * fx.Int32(PAGE_BYTES) + (fx.Int32(ws * 4) + half * fx.Int32(2)) * fx.Int32(256) + ktk * fx.Int32(16)
            lo = Vec(_lds_at(k_ptr, k_off + a0, 16).load()).bitcast(fx.Int32)
            hi = Vec(_lds_at(k_ptr, k_off + a0 + fx.Int32(256), 16).load()).bitcast(fx.Int32)
            acc = _mfma(_pack8(lo, hi), q_ops[ws], acc)
        s_acc.append(Vec(acc))

    # ---- softmax over the block (log2 domain) ----
    # Raw k-scaled scores first; the per-row qk scale is positive, so it is folded
    # into the exp argument. V scales are read twice (max, then use) so that only
    # the 64 scores stay live, and P is packed to e4m3 per key tile.
    qk_scale = q_sc * f32_scale_log2
    # The per-score math runs on <2 x f32> of adjacent accumulator elements, so the
    # packed VALU ops (v_pk_mul / v_pk_fma) read them in place (LLVM's own SLP
    # pairing took elements (i, i+2) and paid ~1.5 v_mov per packed op)
    sdws = []
    s = []
    vmax = fx.Float32(1e-30)
    for ki, kt in enumerate(kts):
        sdw = ((page0 + kpg0 + fx.Int32(2 * kt) + half) * fx.Int32(hkv) + kvh) * fx.Int32(16)
        sdws.append(sdw)
        kscl = scales(0, kt, sdw)
        vscl = None if vred else scales(1, kt, sdw)
        for i in range_constexpr(0, 16, 2):
            s.append(_pair(s_acc[ki], i) * _pair(kscl[i // 4], i % 4))
        if const_expr(not vred):
            for i in range_constexpr(16):
                vmax = _mx(vmax, fx.Float32(vscl[i // 4][i % 4]))

    # causal mask: key kbase + i is visible iff i <= qpos - kbase (one sub per run,
    # constant compares). Only a query's own block has invisible keys, so the selects sit
    # in a divergent branch that the wave skips (s_cbranch_execz) unless one of its rows
    # needs it.
    def flat(v):
        return [fx.Float32(x[u]) for x in v for u in range_constexpr(2)]

    def masked():
        sf = flat(s)
        out = []
        for ki, kt in enumerate(kts):
            lim = qpos - (lblk * fx.Int32(BLOCK) + kb + fx.Int32(kt * 32) + half * fx.Int32(16))
            out += [(fx.Int32(i) <= lim).select(sf[16 * ki + i], neg_inf) for i in range_constexpr(16)]
        return [Vec.from_elements(out[i:i + 2], fx.Float32) for i in range(0, len(out), 2)]

    need = qpos < lblk * fx.Int32(BLOCK) + kb + fx.Int32(32 * nkt - 1)
    if_op = scf.IfOp(raw(need), [T.vec(2, T.f32)] * len(s), has_else=True)
    with ir.InsertionPoint(if_op.then_block):
        scf.YieldOp([raw(x) for x in masked()])
    with ir.InsertionPoint(if_op.else_block):
        scf.YieldOp([raw(x) for x in s])
    s = [Vec(r) for r in if_op.results]
    m = neg_inf
    for x in flat(s):
        m = _mx(m, x)
    m = _mx(m, m.shuffle_xor(fx.Int32(32), fx.Int32(64)))
    if const_expr(vred):
        # block max of the 128 V scales: 2 per lane + a wave reduction (instead of
        # every lane reading all 64 it uses)
        o = scl_off + fx.Int32(SCL_V) + (kb + (half * fx.Int32(32) + col) * fx.Int32(2)) * fx.Int32(4)
        v2 = Vec(_lds_at(k_ptr, o, 8).load()).bitcast(fx.Float32)
        vmax = _mx(vmax, _mx(fx.Float32(v2[0]), fx.Float32(v2[1])))
        for k in range_constexpr(6):
            vmax = _mx(vmax, vmax.shuffle_xor(fx.Int32(1 << k), fx.Int32(64)))
    else:
        vmax = _mx(vmax, vmax.shuffle_xor(fx.Int32(32), fx.Int32(64)))
    mc = m * qk_scale
    # P_SCALE / vmax enters the exp argument, so P' = P * P_SCALE / vmax feeds P' * vscale (one
    # multiply less per score); lsum' = lsum * P_SCALE / vmax. 1 / lsum is v_rcp_f32 and log2
    # skips the denormal pre-scaling (afn: lsum >= 1, vmax a positive normal scale).
    lpm = fx.Float32(math.log2(P_SCALE)) - fx.math.log2(vmax, fastmath="afn")
    neg_mc = lpm - mc
    lsum = fx.Float32(0.0)
    words = []
    for ki in range_constexpr(len(kts)):
        vscl = scales(1, kts[ki], sdws[ki])
        for u in range_constexpr(4):
            quad = []
            for w in range_constexpr(0, 4, 2):
                a = Vec(fx.math.fma(s[8 * ki + 2 * u + w // 2], _bc2(qk_scale), _bc2(neg_mc)))
                pv = Vec.from_elements([fx.Float32(a[j]).exp2(fastmath="afn") for j in range_constexpr(2)],
                                       fx.Float32)
                # serial scalar sum: a 2-wide v_pk_add sum was slower (and not bitwise)
                lsum = lsum + fx.Float32(pv[0])
                lsum = lsum + fx.Float32(pv[1])
                pw = pv * _pair(vscl[u], w)
                quad += [fx.Float32(pw[0]), fx.Float32(pw[1])]
            words.append(_fp8x4(quad))
    lsum = lsum + lsum.shuffle_xor(fx.Int32(32), fx.Int32(64))
    m = mc

    # ---- O^T = V^T P^T: 4 D tiles x 2 key halves ----
    p_ops = [Vec.from_elements(words[8 * ji: 8 * ji + 8], fx.Int32) for ji in range_constexpr(len(js))]
    o_acc = []
    for dt in range_constexpr(4):
        d = fx.Int32(dt * 32) + col
        acc = zero16
        for ji, j in enumerate(js):
            a0 = (kpg0 + fx.Int32(4 * j) + half) * fx.Int32(PAGE_BYTES) + d * fx.Int32(16)
            lo = Vec(_lds_at(v_ptr, v_off + a0, 16).load()).bitcast(fx.Int32)
            hi = Vec(_lds_at(v_ptr, v_off + a0 + fx.Int32(2 * PAGE_BYTES), 16).load()).bitcast(fx.Int32)
            acc = _mfma(_pack8(lo, hi), p_ops[ji], acc)
        o_acc.append(Vec(acc))
    rl = fx.Float32(fx.rocdl.rcp(T.f32, raw(lsum)))
    return o_acc, rl, m + (fx.math.log2(lsum, fastmath="afn") - lpm)


def _mfma16(a, b, c):
    return fx.rocdl.mfma_scale_f32_16x16x128_f8f6f4(
        T.vec(4, T.f32),
        [raw(a), raw(b), raw(c), 0, 0, 0, raw(fx.Int32(_UNIT_SCALE)), 0, raw(fx.Int32(_UNIT_SCALE))],
    )


def _kswz16(pg):
    """Token-slot XOR (in tokens) of K page `pg` within a block, see _block_attn16."""
    return (pg + pg // fx.Int32(4)) % fx.Int32(4) * fx.Int32(4)


def _block_attn16(kv_ptr, k_off, v_off, scl_off, q_op, q_sc, qpos, lblk, lane, f32_scale_log2, khalf=None,
                  regs=None):
    """_block_attn for one wave and 16 query rows (one token's heads) over a whole 128-key block, or
    over keys 64 h + [0, 64) given khalf = h, on 16x16x128 MFMAs (lane l: A row / B column l % 16,
    k = 32 * (l // 16) + [0, 32); C rows 4 * (l // 16) + e). S^T = K Q^T in key tiles of 16: whole
    block, tile kt's A row R is key 32 * (R // 4) + 4 * kt + R % 4 (8 tiles), so lane group g ends up
    holding keys 32 g + [0, 32) in order -- exactly its B slice of P^T for O^T = V^T P^T (one MFMA per
    16 d); half block, key 64 h + 16 * (R // 4) + 4 * kt + R % 4 (4 tiles): lane group g holds keys
    64 h + 16 g + [0, 16), the lower half of its B slice (the upper half P = 0). K must be stored with
    token slot tok ^ _kswz16(page) (_dma_block kswz): the 16 rows of a tile then read 16 different
    bank slots in both cases.
    q_op: the row's e4m3 Q, d 32 g + [0, 32). -> (O^T [8 x 4 f32] of d 16 dt + 4 g + e, 1 / lsum,
    log2-sum-exp).
    regs: a _Direct16 (whole block): operands from VGPRs instead of LDS."""
    whole = khalf is None
    assert regs is None or whole
    NKT = 8 if whole else 4  # key tiles of 16
    KPG = 4 * NKT  # keys per lane group
    g = lane // fx.Int32(16)
    r = lane % fx.Int32(16)
    gq = r // fx.Int32(4)
    iq = r % fx.Int32(4)
    kb = fx.Int32(0) if whole else khalf * fx.Int32(64)
    neg_inf = fx.Float32(float("-inf"))
    zero4 = fx.full(4, 0.0, fx.Float32)
    s_acc = []
    for kt in range_constexpr(NKT):
        if whole:
            pg = gq * fx.Int32(2) + fx.Int32(kt // 4)
        else:
            pg = kb // fx.Int32(16) + gq
        if regs is not None:
            lo, hi = regs.k[kt]
        else:
            tsw = (fx.Int32(kt % 4 * 4) + iq) ^ _kswz16(pg)
            a0 = pg * fx.Int32(PAGE_BYTES) + g * fx.Int32(512) + tsw * fx.Int32(16)
            lo = Vec(_lds_at(kv_ptr, k_off + a0, 16).load()).bitcast(fx.Int32)
            hi = Vec(_lds_at(kv_ptr, k_off + a0 + fx.Int32(256), 16).load()).bitcast(fx.Int32)
        s_acc.append(Vec(_mfma16(_pack8(lo, hi), q_op, zero4)))
    if regs is not None:
        regs.load_v()  # V after the QK MFMAs: K and V operands are never live together

    def scales(plane, kt):
        if regs is not None:
            return regs.vs[kt] if plane else regs.ks[kt]
        o = scl_off + fx.Int32(plane * SCL_V) + (kb + g * fx.Int32(KPG) + fx.Int32(4 * kt)) * fx.Int32(4)
        return Vec(_lds_at(kv_ptr, o, 16).load()).bitcast(fx.Float32)

    qk_scale = q_sc * f32_scale_log2
    s = []
    for kt in range_constexpr(NKT):
        ks = scales(0, kt)
        s += [_pair(s_acc[kt], i) * _pair(ks, i) for i in (0, 2)]

    def flat(v):
        return [fx.Float32(x[u]) for x in v for u in range_constexpr(2)]

    lim = qpos - (lblk * fx.Int32(BLOCK) + kb + g * fx.Int32(KPG))
    need = qpos < lblk * fx.Int32(BLOCK) + kb + fx.Int32(4 * KPG - 1)
    if_op = scf.IfOp(raw(need), [T.vec(2, T.f32)] * len(s), has_else=True)
    with ir.InsertionPoint(if_op.then_block):
        sf = flat(s)
        out = [(fx.Int32(j) <= lim).select(sf[j], neg_inf) for j in range_constexpr(KPG)]
        scf.YieldOp([raw(Vec.from_elements(out[j:j + 2], fx.Float32)) for j in range(0, KPG, 2)])
    with ir.InsertionPoint(if_op.else_block):
        scf.YieldOp([raw(x) for x in s])
    s = [Vec(x) for x in if_op.results]
    m = neg_inf
    for x in flat(s):
        m = _max(m, x, True)
    for k in (16, 32):
        m = _max(m, m.shuffle_xor(fx.Int32(k), fx.Int32(64)), True)
    # block max of the 128 V scales: 2 per lane + a wave reduction
    if regs is not None:
        v2 = regs.vm
    else:
        v2 = Vec(_lds_at(kv_ptr, scl_off + fx.Int32(SCL_V) + lane * fx.Int32(8), 8).load()).bitcast(fx.Float32)
    vmax = _max(fx.Float32(v2[0]), fx.Float32(v2[1]), True)
    for k in range_constexpr(6):
        vmax = _max(vmax, vmax.shuffle_xor(fx.Int32(1 << k), fx.Int32(64)), True)
    mc = m * qk_scale
    lpm = fx.Float32(math.log2(P_SCALE)) - fx.math.log2(vmax, fastmath="afn")
    neg_mc = lpm - mc
    lsum = fx.Float32(0.0)
    words = []
    for kt in range_constexpr(NKT):
        vs = scales(1, kt)
        quad = []
        for w in (0, 2):
            a = Vec(fx.math.fma(s[2 * kt + w // 2], _bc2(qk_scale), _bc2(neg_mc)))
            pv = Vec.from_elements([fx.Float32(a[j]).exp2(fastmath="afn") for j in range_constexpr(2)], fx.Float32)
            lsum = lsum + fx.Float32(pv[0])
            lsum = lsum + fx.Float32(pv[1])
            pw = pv * _pair(vs, w)
            quad += [fx.Float32(pw[0]), fx.Float32(pw[1])]
        words.append(_fp8x4(quad))
    for k in (16, 32):
        lsum = lsum + lsum.shuffle_xor(fx.Int32(k), fx.Int32(64))
    p_op = Vec.from_elements(words + [fx.Int32(0)] * (8 - NKT), fx.Int32)
    o_acc = []
    for dt in range_constexpr(8):
        drow = (fx.Int32(dt * 16) + r) * fx.Int32(16)
        if regs is not None:
            lo, hi = regs.v[dt]
        else:
            if whole:
                a0 = g * fx.Int32(2 * PAGE_BYTES) + drow
            else:
                a0 = (kb // fx.Int32(16) + g) * fx.Int32(PAGE_BYTES) + drow
            lo = Vec(_lds_at(kv_ptr, v_off + a0, 16).load()).bitcast(fx.Int32)
            # half block: the upper 16 k-slots carry P = 0, any finite V will do
            hi = Vec(_lds_at(kv_ptr, v_off + a0 + fx.Int32(PAGE_BYTES), 16).load()).bitcast(fx.Int32) if whole else lo
        o_acc.append(Vec(_mfma16(_pack8(lo, hi), p_op, zero4)))
    rl = fx.Float32(fx.rocdl.rcp(T.f32, raw(lsum)))
    return o_acc, rl, mc + (fx.math.log2(lsum, fastmath="afn") - lpm)


class _Direct16:
    """One wave's _block_attn16 operands of a whole block, loaded from the cache straight into VGPRs
    (no LDS): every operand is a 16 B run of the page-16 layout. K tile kt, lane (g, r): token
    4 (kt % 4) + r % 4 of page 2 (r // 4) + kt // 4, d 32 g + [0, 32); V d-tile dt: d 16 dt + r of
    pages 2 g, 2 g + 1; scales of keys 32 g + 4 kt + [0, 4); 2 V scales per lane for the block max.
    K and its scales load at construction, V and its scales at load_v() (after the QK MFMAs, so
    both sets are never live at once)."""

    def __init__(self, k_base, v_base, ks_base, vs_base, page0, kvh, hkv, lane, k=None):
        self.args = (v_base, vs_base, page0, kvh, hkv, lane)
        g = lane // fx.Int32(16)
        r = lane % fx.Int32(16)
        if k is not None:
            self.k, self.ks = k
            return
        k_rs = _block_rsrc(k_base, page0, kvh, hkv, PAGE_BYTES)
        ks_rs = _block_rsrc(ks_base, page0, kvh, hkv, 64)
        self.k = []
        for kt in range_constexpr(8):
            pg = r // fx.Int32(4) * fx.Int32(2) + fx.Int32(kt // 4)
            b0 = pg * fx.Int32(hkv * PAGE_BYTES) + g * fx.Int32(512) + (fx.Int32(kt % 4 * 4) + r % fx.Int32(4)) * fx.Int32(16)
            self.k.append(tuple(Vec(buffer_ops.buffer_load(k_rs, (b0 + fx.Int32(256 * h)) // fx.Int32(4), 4, T.i32))
                                for h in range_constexpr(2)))
        self.ks = [Vec(buffer_ops.buffer_load(ks_rs, _Direct16._sc(g, kt, hkv), 4, T.f32)) for kt in range_constexpr(8)]

    @staticmethod
    def _sc(g, kt, hkv):
        return ((g * fx.Int32(2) + fx.Int32(kt // 4)) * fx.Int32(hkv * 64) + fx.Int32(kt % 4 * 16)) // fx.Int32(4)

    def load_v(self):
        v_base, vs_base, page0, kvh, hkv, lane = self.args
        g = lane // fx.Int32(16)
        r = lane % fx.Int32(16)
        pstr = hkv * PAGE_BYTES
        v_rs = _block_rsrc(v_base, page0, kvh, hkv, PAGE_BYTES)
        vs_rs = _block_rsrc(vs_base, page0, kvh, hkv, 64)
        self.vm = Vec(buffer_ops.buffer_load(vs_rs, (lane // fx.Int32(8) * fx.Int32(hkv * 64) + lane % fx.Int32(8) * fx.Int32(8))
                                             // fx.Int32(4), 2, T.f32))
        self.vs = [Vec(buffer_ops.buffer_load(vs_rs, _Direct16._sc(g, kt, hkv), 4, T.f32)) for kt in range_constexpr(8)]
        self.v = []
        for dt in range_constexpr(8):
            b0 = g * fx.Int32(2 * pstr) + (fx.Int32(dt * 16) + r) * fx.Int32(16)
            self.v.append(tuple(Vec(buffer_ops.buffer_load(v_rs, (b0 + fx.Int32(pstr * h)) // fx.Int32(4), 4, T.i32))
                                for h in range_constexpr(2)))

    def reload_k(self, cond, k_base, ks_base, page0):
        """K operands of `page0` instead if cond (wave-uniform: a wrong speculative load)."""
        old = [x for kt in self.k for x in kt] + self.ks
        if_op = scf.IfOp(raw(cond), [raw(x).type for x in old], has_else=True)
        with ir.InsertionPoint(if_op.then_block):
            v_base, vs_base, _, kvh, hkv, lane = self.args
            n = _Direct16(k_base, v_base, ks_base, vs_base, page0, kvh, hkv, lane)
            scf.YieldOp([raw(x) for x in [x for kt in n.k for x in kt] + n.ks])
        with ir.InsertionPoint(if_op.else_block):
            scf.YieldOp([raw(x) for x in old])
        res = [Vec(x) for x in if_op.results]
        v_base, vs_base, _, kvh, hkv, lane = self.args
        return _Direct16(None, v_base, None, vs_base, page0, kvh, hkv, lane,
                         k=([tuple(res[2 * i: 2 * i + 2]) for i in range(8)], res[16:24]))


def _load_block(lds, k_base, v_base, tid, page0, kvh, hkv, nthreads=NTHREADS):
    """K, V of one 128-block of kv head `kvh` -> LDS (identical byte layout, 8 pages of 2 KB).
    Buffer resources start at the block (_block_rsrc): offsets stay small for any cache size."""
    k_rsrc = _block_rsrc(k_base, page0, kvh, hkv, PAGE_BYTES)
    v_rsrc = _block_rsrc(v_base, page0, kvh, hkv, PAGE_BYTES)
    for p in range_constexpr(BLOCK_BYTES // 16 // nthreads):
        ci = tid + fx.Int32(p * nthreads)
        pg = ci // fx.Int32(PAGE_BYTES // 16)
        within = ci % fx.Int32(PAGE_BYTES // 16)
        dw = pg * fx.Int32(hkv * PAGE_BYTES // 4) + within * fx.Int32(4)
        kw = Vec(buffer_ops.buffer_load(k_rsrc, dw, 4, T.i32))
        vw = Vec(buffer_ops.buffer_load(v_rsrc, dw, 4, T.i32))
        _lds_at(lds.k.ptr, ci * fx.Int32(16), 16).store(kw.bitcast(fx.Uint8))
        _lds_at(lds.v.ptr, ci * fx.Int32(16), 16).store(vw.bitcast(fx.Uint8))


def _stage_o_half(lds, o_lds, o_acc, o_mul, hh, col, half, o_ptr=None):
    """Normalized bf16 d-half `hh` of the wave's 32 rows -> LDS rows of O_ROW_BYTES
    (at lds.o, or at o_ptr)."""
    o_ptr = lds.o.ptr if o_ptr is None else o_ptr
    for dl in range_constexpr(2):
        for g in range_constexpr(4):
            vals = [(fx.Float32(o_acc[2 * hh + dl][g * 4 + u]) * o_mul).to(fx.BFloat16) for u in range_constexpr(4)]
            pk = Vec.from_elements(vals, fx.BFloat16).bitcast(fx.Uint8)
            off = o_lds + col * fx.Int32(O_ROW_BYTES) + fx.Int32(dl * 64 + g * 16) + half * fx.Int32(8)
            _lds_at(o_ptr, off, 8).store(pk)


def _stage_o_u8(o_ptr, o_lds, o_acc, col, half):
    """The lane's 64 d of its row (d % 8 // 4 == half) -> u8 codes 128 + round(127 * o / amax)
    at byte d of an LDS row (O_ROW_BYTES stride). Returns the lane's raw-accumulator amax."""
    amax = fx.Float32(1e-30)
    for dt in range_constexpr(4):
        for i in range_constexpr(16):
            amax = _max(amax, fx.absf(fx.Float32(o_acc[dt][i])), True)
    k = fx.Float32(127.0) * fx.Float32(fx.rocdl.rcp(T.f32, raw(amax)))
    for dt in range_constexpr(4):
        for g in range_constexpr(4):
            f = []
            for u in range_constexpr(0, 4, 2):
                fv = Vec(fx.math.fma(_pair(o_acc[dt], 4 * g + u), _bc2(k), _bc2(fx.Float32(U8_MAGIC)))).bitcast(fx.Int32)
                f += [raw(fx.Int32(fv[0])), raw(fx.Int32(fv[1]))]
            # low bytes of f0..f3 -> one word (byte selectors: 0-3 src_lo, 4-7 src_hi, 0x0c zero)
            lo = fx.rocdl.perm_b32(f[1], f[0], raw(fx.Int32(0x0C0C0400)))
            hi = fx.rocdl.perm_b32(f[3], f[2], raw(fx.Int32(0x0C0C0400)))
            word = fx.Int32(fx.rocdl.perm_b32(hi, lo, raw(fx.Int32(0x05040100))))
            off = o_lds + col * fx.Int32(O_ROW_BYTES) + fx.Int32(dt * 32 + g * 8) + half * fx.Int32(4)
            _lds_at(o_ptr, off, 4).store(Vec.from_elements([word], fx.Int32).bitcast(fx.Uint8))
    return amax


def build_attn(hkv: int, waves_per_eu=0, contig=False, nt=False, pf=False, u8=False):
    """pf: software-pipelined tile loop -- the next tile's K/V and scales (when its block
    differs) are copied global -> LDS by async DMA into the second of two stages while this
    tile computes, and its record and Q rows are loaded before this tile's partial stores.
    u8 (pf only): partial O rows as u8 codes (see _stage_o_u8) plus two f32 scales per row
    (one per lane half) at arg_psc, half the bytes of bf16 rows."""
    assert pf == u8, "the pipelined loop stores u8 partials; the plain loop bf16"
    hq = hkv * GQA
    if pf:
        # two K/V(+scales) stages; O is staged inside one of them
        lds_fields = {"kv": fx.Array[fx.Uint8, 2 * KV_STAGE, 16]}
    else:
        lds_fields = {
            "k": fx.Array[fx.Uint8, BLOCK_BYTES, 16],
            "v": fx.Array[fx.Uint8, BLOCK_BYTES, 16],
            "o": fx.Array[fx.Uint8, WAVES * 32 * O_ROW_BYTES, 16],
        }
    SharedStorage = fx.struct(type("M3KvOuterLds", (), {"__annotations__": lds_fields}))

    @flyc.kernel(name=f"m3_kvouter_attn_hkv{hkv}{'_contig' if contig else ''}{'_nt' if nt else ''}{'_pf' if pf else ''}{'_u8' if u8 else ''}",
                 known_block_size=[NTHREADS, 1, 1])
    def attn_kernel(
        arg_q: fx.Pointer,
        arg_qs: fx.Pointer,
        arg_k: fx.Pointer,
        arg_v: fx.Pointer,
        arg_ks: fx.Pointer,
        arg_vs: fx.Pointer,
        arg_tiles: fx.Pointer,
        arg_ntiles: fx.Pointer,
        arg_eq: fx.Pointer,
        arg_qpos: fx.Pointer,
        arg_po: fx.Pointer,
        arg_plse: fx.Pointer,
        arg_psc: fx.Pointer,
        f32_scale_log2: fx.Float32,
        grid: fx.Int32,
        i32_nq: fx.Int32,
        i32_nseg: fx.Int32,
        i32_tseg: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        bid = fx.Int32(gpu.block_id("x"))
        wave = tid // fx.Int32(WAVE)
        lane = tid % fx.Int32(WAVE)
        col = lane % fx.Int32(32)
        half = lane // fx.Int32(32)
        head16 = col % fx.Int32(GQA)

        q_buf = ptr_buf_tensor(arg_q, fx.Int32)
        qs_buf = ptr_buf_tensor(arg_qs, fx.Float32)
        ks_buf = ptr_buf_tensor(arg_ks, fx.Float32)
        vs_buf = ptr_buf_tensor(arg_vs, fx.Float32)
        t_buf = ptr_buf_tensor(arg_tiles, fx.Int32)
        nt_buf = ptr_buf_tensor(arg_ntiles, fx.Int32)
        eq_buf = ptr_buf_tensor(arg_eq, fx.Int32)
        qp_buf = ptr_buf_tensor(arg_qpos, fx.Int32)
        lse_buf = ptr_buf_tensor(arg_plse, fx.Float32)
        psc_buf = ptr_buf_tensor(arg_psc, fx.Float32)
        po_base = buf_base_i64(arg_po)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        n_tiles, tphys = _tile_segments(nt_buf, i32_nseg, i32_tseg, hkv)

        # K row r of a key tile -> key offset in the tile (see module doc).
        r = col
        k_perm = (r // fx.Int32(4)) % fx.Int32(2) * fx.Int32(16) + (r // fx.Int32(8)) * fx.Int32(4) + r % fx.Int32(4)

        t_lo, t_hi, t_step = _tile_range(bid, n_tiles, grid, contig)
        if const_expr(not pf):
            for t_iv, held in range(t_lo, t_hi, t_step, init=[fx.Int32(-1)]):
                t = tphys(fx.Int32(t_iv))
                rec_lo = Vec(_ld(t_buf, t * fx.Int32(TILE_INTS), 4))
                rec_hi = Vec(_ld(t_buf, t * fx.Int32(TILE_INTS) + fx.Int32(4), 4))
                kvh = fx.Int32(rec_lo[0])
                phys = fx.Int32(rec_lo[1])
                lblk = fx.Int32(rec_lo[2])
                e_start = fx.Int32(rec_lo[3])
                n_edges = fx.Int32(rec_hi[0])
                page0 = phys * fx.Int32(8)
                blk_key = phys * fx.Int32(hkv) + kvh
                fresh = (blk_key != fx.Int32(held[0])) if contig else True

                # ---- K, V block -> LDS (identical byte layout, 8 pages of 2 KB), unless held ----
                if fresh:
                    gpu.barrier()
                    _load_block(lds, buf_base_i64(arg_k), buf_base_i64(arg_v), tid, page0, kvh, hkv)

                # ---- this lane's query row: edge, token, absmax-scaled e4m3 Q ----
                ed = wave * fx.Int32(EDGES_PER_WAVE) + col // fx.Int32(GQA)
                col_ok = ed < n_edges
                e = e_start + col_ok.select(ed, fx.Int32(0))
                eqv = fx.Int32(_ld(eq_buf, e, 1))  # query * 16 + rank
                qi = eqv >> fx.Int32(4)
                qpos = fx.Int32(_ld(qp_buf, qi, 1))
                # e4m3 Q row (quantized once per token by build_qquant) + its scale
                q_row = qi * fx.Int32(hq) + kvh * fx.Int32(GQA) + head16
                q_sc = fx.Float32(_ld(qs_buf, q_row, 1, T.f32))
                q_ops = []
                for ws in range_constexpr(2):
                    dw = q_row * fx.Int32(HEAD_DIM // 4) + fx.Int32(ws * 16) + half * fx.Int32(8)
                    lo = Vec(_ld(q_buf, dw, 4))
                    hi = Vec(_ld(q_buf, dw + fx.Int32(4), 4))
                    q_ops.append(_pack8(lo, hi))

                if fresh:
                    gpu.barrier()

                o_acc, o_mul, lse = _block_attn(lds.k.ptr, lds.v.ptr, fx.Int32(0), fx.Int32(0), ks_buf, vs_buf, q_ops,
                                                q_sc, qpos, page0, kvh, lblk, k_perm, col, half, f32_scale_log2, hkv)

                # ---- epilogue: normalized bf16 partial rows + lse, edge-contiguous ----
                # Stage the wave's 32 rows in LDS one 64-d half at a time, then write each
                # half-row as a whole 128 B line (rows e*16+head of two consecutive edges are
                # contiguous): 16 B per lane instead of 8 B scattered over 32 rows.
                o_lds = wave * fx.Int32(32 * O_ROW_BYTES)
                wave_rows = (n_edges - wave * fx.Int32(EDGES_PER_WAVE)) * fx.Int32(GQA)
                # partials are query-major: edge (query, rank) owns partial slot (kvh, query, rank)
                pslot = kvh * i32_nq * fx.Int32(16) + eqv
                pbufs = [ptr_buf_tensor(po_base + fx.Int64(fx.Int32(fx.rocdl.readlane(T.i32, raw(pslot), j * GQA)))
                                        * fx.Int64(GQA * HEAD_DIM * 2), fx.Int32) for j in range_constexpr(EDGES_PER_WAVE)]
                for hh in range_constexpr(2):
                    _stage_o_half(lds, o_lds, o_acc, o_mul, hh, col, half)
                    for p in range_constexpr(32 * HEAD_DIM // 16 // WAVE):
                        idx = lane + fx.Int32(p * WAVE)
                        orow = idx // fx.Int32(8)
                        chunk = idx % fx.Int32(8)
                        w4 = Vec(_lds_at(lds.o.ptr, o_lds + orow * fx.Int32(O_ROW_BYTES) + chunk * fx.Int32(16), 16).load()).bitcast(fx.Int32)
                        po_buf = pbufs[p * 8 // GQA]  # rows p*8 .. p*8+7 belong to one edge
                        po_rsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(po_buf)) if nt else None
                        po_dw = (orow % fx.Int32(GQA)) * fx.Int32(HEAD_DIM // 2) + fx.Int32(hh * 32) + chunk * fx.Int32(4)
                        if orow < wave_rows:
                            if const_expr(nt):
                                # partials too big to stay in MALL for the combine are streamed
                                # past the caches, which keeps L2 for K/V and Q
                                buffer_ops.buffer_store(w4, po_rsrc, po_dw, cache_modifier=CPOL_NT)
                            else:
                                fx.add_offset(fx.get_iter(po_buf), po_dw).store(w4)
                if col_ok & (half == fx.Int32(0)):
                    fx.add_offset(fx.get_iter(lse_buf), pslot * fx.Int32(GQA) + head16).store(lse)
                results = yield [blk_key]

        if const_expr(pf):
            # Software-pipelined: everything tile t+1 needs (its record, edge -> query, Q
            # rows, K/V by async DMA) is issued before tile t's partial stores, so that with
            # in-order vmcnt nothing ever waits for the stores. LDS: two K/V(+scales) stages.
            kv_ptr = lds.kv.ptr
            k_base, v_base, ks_base, vs_base = (buf_base_i64(a) for a in (arg_k, arg_v, arg_ks, arg_vs))
            STG = KV_STAGE
            ed = wave * fx.Int32(EDGES_PER_WAVE) + col // fx.Int32(GQA)
            # The record two tiles ahead stays in VGPRs until the tile that consumes it (a
            # readfirstlane at the load waits there, behind this tile's partial stores), and the
            # u8 partial stores are bounded by their resources instead of branched around.
            wave_s = fx.Int32(fx.rocdl.readfirstlane(T.i32, raw(wave)))
            # whole-buffer resources: an offset of -16 elements is out of range (dropped)
            lse_rs = buffer_ops.create_buffer_resource_from_addr(buf_base_i64(arg_plse), num_records_bytes=0x7FFFFFF0)
            psc_rs = buffer_ops.create_buffer_resource_from_addr(buf_base_i64(arg_psc), num_records_bytes=0x7FFFFFF0)

            def rfl(r):
                return [fx.Int32(fx.rocdl.readfirstlane(T.i32, raw(x))) for x in r]

            def rec(t, now=True):
                t = tphys(t)
                lo = Vec(_ld(t_buf, t * fx.Int32(TILE_INTS), 4))
                hi = Vec(_ld(t_buf, t * fx.Int32(TILE_INTS) + fx.Int32(4), 4))
                r = [fx.Int32(lo[i]) for i in range_constexpr(4)] + [fx.Int32(hi[0])]  # kvh phys lblk e0 n
                if const_expr(now):
                    # WG-uniform: SGPR copies give scalar partial-store resources (no
                    # readfirstlane waterfall around every store) and free VGPRs
                    r = rfl(r)
                return r

            def key_of(r):
                return r[1] * fx.Int32(hkv) + r[0]

            def edge_q(r):
                e = r[3] + (ed < r[4]).select(ed, fx.Int32(0))
                return e, fx.Int32(_ld(eq_buf, e, 1))  # query * 16 + rank

            def q_rows(r, eqv):
                qi = eqv >> fx.Int32(4)
                q_row = qi * fx.Int32(hq) + r[0] * fx.Int32(GQA) + head16
                vals = [fx.Int32(_ld(qp_buf, qi, 1)), fx.Int32(_ld(qs_buf, q_row, 1))]
                for ws in range_constexpr(2):
                    dw = q_row * fx.Int32(HEAD_DIM // 4) + fx.Int32(ws * 16) + half * fx.Int32(8)
                    for h4 in range_constexpr(2):
                        v4 = Vec(_ld(q_buf, dw + fx.Int32(4 * h4), 4))
                        vals += [fx.Int32(v4[i]) for i in range_constexpr(4)]
                return vals  # qpos, q scale bits, 16 Q words

            def fetch(r, stage):
                # every lane writes its 16 B; completes with wait_asyncmark
                _dma_block(kv_ptr, stage * fx.Int32(STG), k_base, v_base, ks_base, vs_base, wave, lane, NTHREADS,
                           r[1] * fx.Int32(8), r[0], hkv)
                fx.rocdl.asyncmark()

            have = t_lo < t_hi
            t0 = have.select(t_lo, fx.Int32(0))
            r0 = rec(t0)
            t1 = t0 + t_step
            r1 = rec((t1 < t_hi).select(t1, t0))
            if have:
                fetch(r0, fx.Int32(0))
            _, qi0 = edge_q(r0)
            q0 = q_rows(r0, qi0)
            # as many (empty) stores after Q as an iteration issues, so that at the loop header
            # LLVM's waitcnt merge does not count Q as just loaded (and drain the stores for it)
            nul = buffer_ops.create_buffer_resource_from_addr(po_base, num_records_bytes=0)
            for i in range_constexpr(4):  # distinct, non-adjacent offsets: no DSE / merging
                buffer_ops.buffer_store(Vec.filled(4, 0, fx.Int32), nul, fx.Int32(64 * i))
            for i in range_constexpr(2):
                buffer_ops.buffer_store(fx.Float32(0.0), nul, fx.Int32(64 * (4 + i)))
            for t_iv, st in range(t_lo, t_hi, t_step, init=r0 + r1 + [qi0] + q0 + [fx.Int32(0), fx.Int32(1)]):
                t = fx.Int32(t_iv)
                rc = [fx.Int32(st[i]) for i in range_constexpr(5)]
                rn = [fx.Int32(st[5 + i]) for i in range_constexpr(5)]
                rn = rfl(rn)
                eqv = fx.Int32(st[10])
                qv = [fx.Int32(st[11 + i]) for i in range_constexpr(18)]
                b = fx.Int32(st[29])
                landed = fx.Int32(st[30]) != fx.Int32(0)  # this tile's K/V was just fetched
                key = key_of(rc)
                tn = t + t_step
                has_next = tn < t_hi
                nfresh = has_next & (key_of(rn) != key)

                # this tile's K/V has landed / every wave is past the last tile's epilogue
                # (whose O may sit in the stage about to be fetched into)
                if landed | nfresh:
                    fx.rocdl.wait_asyncmark(0)
                    gpu.barrier()
                t2 = tn + t_step
                r2 = rec((t2 < t_hi).select(t2, t), False)
                rn_c = [has_next.select(rn[i], rc[i]) for i in range_constexpr(5)]
                _, qi_n = edge_q(rn_c)
                if nfresh:
                    fetch(rn, fx.Int32(1) - b)

                q_ops = [Vec.from_elements(qv[2 + 8 * ws: 10 + 8 * ws], fx.Int32) for ws in range_constexpr(2)]
                k_off = b * fx.Int32(STG)
                o_acc, o_mul, lse = _block_attn(kv_ptr, kv_ptr, k_off, k_off + fx.Int32(BLOCK_BYTES), ks_buf, vs_buf,
                                                q_ops, qv[1].bitcast(fx.Float32), qv[0], rc[1] * fx.Int32(8), rc[0],
                                                rc[2], k_perm, col, half, f32_scale_log2, hkv,
                                                scl_off=k_off + fx.Int32(2 * BLOCK_BYTES))
                q_n = q_rows(rn_c, qi_n)

                # O is staged over this tile's stage when the next one is being fetched into
                # the other (so all K/V reads must be done), else over the idle other stage
                # (each wave touches only its own rows there)
                if nfresh:
                    gpu.barrier()
                n_edges = rc[4]
                o_lds = nfresh.select(b, fx.Int32(1) - b) * fx.Int32(STG) + wave * fx.Int32(32 * O_ROW_BYTES)
                # Partials are query-major: edge (query, rank) owns partial slot (kvh, query, rank),
                # so a query's 16 partials are contiguous for the combine. The stores are
                # unconditional: a missing edge gets an empty O resource / an out-of-range offset.
                pslot = rc[0] * i32_nq * fx.Int32(16) + eqv
                amax = _stage_o_u8(kv_ptr, o_lds, o_acc, col, half)
                po_rs = []
                for j in range_constexpr(EDGES_PER_WAVE):
                    sj = fx.Int32(fx.rocdl.readlane(T.i32, raw(pslot), j * GQA))
                    okj = wave_s * fx.Int32(EDGES_PER_WAVE) + fx.Int32(j) < n_edges
                    po_rs.append(buffer_ops.create_buffer_resource_from_addr(
                        po_base + fx.Int64(sj) * fx.Int64(GQA * HEAD_DIM),
                        num_records_bytes=okj.select(fx.Int32(GQA * HEAD_DIM), fx.Int32(0))))
                for p in range_constexpr(32 * HEAD_DIM // 16 // WAVE):
                    idx = lane + fx.Int32(p * WAVE)
                    orow = idx // fx.Int32(8)
                    chunk = idx % fx.Int32(8)
                    w4 = Vec(_lds_at(kv_ptr, o_lds + orow * fx.Int32(O_ROW_BYTES) + chunk * fx.Int32(16), 16).load()).bitcast(fx.Int32)
                    # rows p*8 .. p*8+7 belong to one edge
                    buffer_ops.buffer_store(w4, po_rs[p * 8 // GQA],
                                            (orow % fx.Int32(GQA)) * fx.Int32(HEAD_DIM // 4) + chunk * fx.Int32(4),
                                            cache_modifier=CPOL_NT if nt else 0)
                erow = pslot * fx.Int32(GQA) + head16
                e_ok = ed < n_edges
                buffer_ops.buffer_store(amax * o_mul * fx.Float32(1.0 / 127.0), psc_rs,
                                        e_ok.select(erow * fx.Int32(2) + half, fx.Int32(-16)))
                # lane half 1 holds the same lse
                buffer_ops.buffer_store(lse, lse_rs, (e_ok & (half == fx.Int32(0))).select(erow, fx.Int32(-16)))
                results = yield (rn_c + r2 + [qi_n] + q_n
                                 + [nfresh.select(fx.Int32(1) - b, b), nfresh.select(fx.Int32(1), fx.Int32(0))])

    @flyc.jit
    def launch(
        arg_q: fx.Pointer,
        arg_qs: fx.Pointer,
        arg_k: fx.Pointer,
        arg_v: fx.Pointer,
        arg_ks: fx.Pointer,
        arg_vs: fx.Pointer,
        arg_tiles: fx.Pointer,
        arg_ntiles: fx.Pointer,
        arg_eq: fx.Pointer,
        arg_qpos: fx.Pointer,
        arg_po: fx.Pointer,
        arg_plse: fx.Pointer,
        arg_psc: fx.Pointer,
        f32_scale_log2: fx.Float32,
        i32_grid: fx.Int32,
        i32_nq: fx.Int32,
        i32_nseg: fx.Int32,
        i32_tseg: fx.Int32,
        stream: fx.Stream,
    ):
        attn_kernel(
            arg_q, arg_qs, arg_k, arg_v, arg_ks, arg_vs, arg_tiles, arg_ntiles, arg_eq,
            arg_qpos, arg_po, arg_plse, arg_psc, f32_scale_log2, i32_grid, i32_nq, i32_nseg, i32_tseg,
        ).launch(grid=(fx.Int64(i32_grid), 1, 1), block=(NTHREADS, 1, 1), stream=stream)

    if waves_per_eu:
        launch.compile_hints = {"waves_per_eu": waves_per_eu}
    return launch


def _wg_q_quant(raws, cpt, lpr):
    """A Q row spread over `lpr` adjacent lanes of `cpt` 16 B bf16 chunks each, scaled to the row
    absmax -> (cpt * 2 e4m3 words of this lane's 8 * cpt d, row absmax)."""
    qf = []
    for v in raws:
        bf = v.bitcast(fx.BFloat16)
        qf.extend(fx.Float32(bf[u]) for u in range(8))
    amax = fx.Float32(1e-30)
    for x in qf:
        amax = _max(amax, fx.absf(x))
    k = 1
    while k < lpr:
        amax = _max(amax, amax.shuffle_xor(fx.Int32(k), fx.Int32(64)))
        k *= 2
    q_inv = fx.Float32(FP8_MAX) / amax
    return [_fp8x4([qf[4 * j + u] * q_inv for u in range(4)]) for j in range(2 * cpt)], amax


CNT_STRIDE = 64  # one 256 B line per request counter: same-line atomics serialize
SCL_V = 2 * BLOCK * 4  # V scales follow K scales in a block's LDS scale window
SCL_BYTES = 2 * SCL_V
KV_STAGE = 2 * BLOCK_BYTES + SCL_BYTES  # K, V, then their scales, of one block
DEC_PAIRS = 4  # blocks a decode WG attends at once (LDS: 4 x 32 KB)
DEC_HTAB_MAX = 2048  # block ids up to which the decode pick dedupe uses an LDS table (2 x 4 B each)
DEC_ROW_BYTES = HEAD_DIM * 2 + 16  # one staged bf16 O row, padded 4 banks


def _block_rsrc(base_i64, page0, kvh, hkv, unit_bytes):
    """Buffer resource at the first `unit_bytes` unit of 16-page `page0` (wave-uniform) of kv
    head `kvh`, so per-lane offsets stay small however large the cache is."""
    pg = fx.Int64(fx.Int32(fx.rocdl.readfirstlane(T.i32, raw(page0))))
    addr = base_i64 + (pg * fx.Int64(hkv) + fx.Int64(kvh)) * fx.Int64(unit_bytes)
    return buffer_ops.create_buffer_resource_from_addr(addr, num_records_bytes=8 * hkv * unit_bytes)


def _dma16(rsrc, dst, voff):
    """Per lane 16 B at byte offset `voff` of `rsrc` -> LDS at wave-uniform `dst` + 16 * lane."""
    fx.rocdl.raw_ptr_buffer_load_async_lds(
        rsrc, fx.to_llvm_ptr(dst), raw(fx.Int32(16)), raw(voff), raw(fx.Int32(0)), raw(fx.Int32(0)),
        aux=ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 0))


def _dma_block(kv_ptr, st_off, k_base, v_base, ks_base, vs_base, wave, lane, nthreads, page0, kvh, hkv, kswz=False):
    """Async global -> LDS copy of one block's K, V (by `nthreads` threads, `wave` counted from
    the first of them) and, per wave, its 128 K and 128 V f32 scales; no VGPRs, completes
    with asyncmark / wait_asyncmark. LDS: K, V, K scales, V scales (key order, at + SCL_V).
    kswz: K page p's 16 B chunk of token t goes to token slot t ^ _kswz16(p) (_block_attn16)."""
    k_rsrc = _block_rsrc(k_base, page0, kvh, hkv, PAGE_BYTES)
    v_rsrc = _block_rsrc(v_base, page0, kvh, hkv, PAGE_BYTES)
    for p in range_constexpr(BLOCK_BYTES // 16 // nthreads):
        c0 = fx.Int32(p * nthreads) + wave * fx.Int32(WAVE)  # wave-uniform first chunk
        ci = c0 + lane
        pg = ci // fx.Int32(PAGE_BYTES // 16)
        within = ci % fx.Int32(PAGE_BYTES // 16)
        voff = pg * fx.Int32(hkv * PAGE_BYTES) + within * fx.Int32(16)
        koff = voff
        if const_expr(kswz):
            koff = voff ^ (_kswz16(pg) * fx.Int32(16))
        for plane, rsrc, off in ((0, k_rsrc, koff), (BLOCK_BYTES, v_rsrc, voff)):
            _dma16(rsrc, kv_ptr + (st_off + fx.Int32(plane) + c0 * fx.Int32(16)), off)
    # Scales: lanes 32-63 repeat lanes 0-31 into the unused second half of each 512 B
    # window (every lane of an LDS DMA writes).
    c = lane % fx.Int32(32)  # 16 B chunk: page c // 4, tokens 4 * (c % 4) ..
    voff = c // fx.Int32(4) * fx.Int32(hkv * 64) + c % fx.Int32(4) * fx.Int32(16)
    sc_off = st_off + fx.Int32(2 * BLOCK_BYTES)
    _dma16(_block_rsrc(ks_base, page0, kvh, hkv, 64), kv_ptr + sc_off, voff)
    _dma16(_block_rsrc(vs_base, page0, kvh, hkv, 64), kv_ptr + (sc_off + fx.Int32(SCL_V)), voff)


def _dedupe_all(pick_ptr, x, lane, np_):
    """(earlier, token mask) of pick `x` = pick[lane] against all `np_` picks of the request:
    earlier = some lower lane holds the same block, mask bit t = token t picked it. All-pairs
    compare against LDS broadcast reads (a readlane loop keeps 64 SGPR masks live and spills)."""
    earlier = fx.Int32(0)
    mask = fx.Int32(0)
    for j4 in range(np_ // 4):
        # 4 picks; lds.pick is Int32, so the offset counts elements
        xs = Vec(_lds_at(pick_ptr, fx.Int32(4 * j4), 16).load()).bitcast(fx.Int32)
        for u in range(4):
            jj = 4 * j4 + u
            eq = (x == fx.Int32(xs[u])).select(fx.Int32(1), fx.Int32(0))
            earlier = earlier | (eq & (fx.Int32(jj) < lane).select(fx.Int32(1), fx.Int32(0)))
            mask = mask | (eq << fx.Int32(jj // 16))
    return earlier, mask


def _dedupe_htab(tab_ptr, x, lane, hb):
    """_dedupe_all through a direct-mapped LDS table over block ids [0, hb) (one wave; its LDS
    ops complete in order): min lane and OR of token bits per block."""
    xi = (x >= fx.Int32(0)).select(x, fx.Int32(hb))
    fx.atomic_min(fx.add_offset(tab_ptr, xi), lane, syncscope="workgroup")
    fx.atomic_or(fx.add_offset(tab_ptr, xi + fx.Int32(hb + 1)), fx.Int32(1) << (lane // fx.Int32(16)),
                 syncscope="workgroup")
    owner = fx.Int32(fx.add_offset(tab_ptr, xi).load())
    mask = fx.Int32(fx.add_offset(tab_ptr, xi + fx.Int32(hb + 1)).load())
    return (owner != lane).select(fx.Int32(1), fx.Int32(0)), mask



def build_decode(hkv: int, q_len: int, splits: int, pairs: int = DEC_PAIRS, khalves: int = 1, xcd: bool = False,
                 htab: int = 0, c16: bool = False, tsplit: int = 1, direct: bool = False, dspec: bool = True):
    """Decode / MTP verify in one launch: requests of `q_len` (<= 4) consecutive tokens whose
    picks never alias another request's blocks.

    WG (request, kv head, split): wave 0 inverts the request's q_len x 16 picks into its U
    unique blocks (first-pick order) and the token mask of each. Per round the WG loads
    `pairs` unique blocks once each (async, straight to LDS); a group of ceil(q_len / 2)
    waves attends one block (or one 64-key half of it, khalves=2) for every token of the
    request that picked it (a half-wave per token). The groups' results are LSE-merged
    through LDS into one partial per token and round, stored at
    [request, kv head, token, round, split]. The request's last split to arrive merges
    those into `out` and re-zeroes the counter.

    The WG's Q rows are loaded and quantized once into LDS; htab > 0 (block ids < htab)
    dedupes the picks through an LDS table (ds_min first lane, ds_or token mask) instead of
    all-pairs compares.

    xcd (needs a multiple of 8 groups): the splits of a group run on one XCD, so partials,
    counter and combine loads stay in its L2 instead of agent-coherent (sc1) memory traffic.

    c16: a wave attends one token's 16 heads on 16x16x128 MFMAs (_block_attn16) instead of
    two tokens on 32x32x64 ones (half of which idle at q_len 1).

    tsplit (c16): the request's tokens are cut into `tsplit` runs of q_len / tsplit, one WG per (run,
    split): less math per CU at the same split count, and each run has its own counter and last-arriver
    combine (of only its tokens). All S * tsplit WGs of a group still dedupe the whole request's picks.

    direct (c16, one wave per block, one round): that wave loads its block's operands from the cache
    straight into VGPRs (_Direct16) instead of an LDS copy, so the WG needs LDS only for Q and the
    merge, and several WGs share a CU. dspec: round 0's K is loaded speculatively (token 0's pick) before
    the dedupe, as the LDS path's DMA; costs VGPRs (304 vs 203: one WG per CU).
    """
    assert 1 <= q_len <= 4
    TS = tsplit
    assert q_len % TS == 0 and (TS == 1 or c16)
    QT = q_len // TS  # tokens per WG
    S = splits
    P = pairs
    KH = khalves
    G = P * KH  # wave groups, each one (block slot, key half)
    EPW = 1 if c16 else EDGES_PER_WAVE  # tokens per wave
    ROWS = EPW * GQA  # query rows per wave
    wpb = -(-QT // EPW)  # waves per group
    assert BLOCK_BYTES // 16 % (KH * wpb * WAVE) == 0, "a group's waves must split the block DMA evenly"
    NW = G * wpb
    NT = NW * WAVE
    assert not direct or (c16 and wpb == 1 and KH == 1)
    # direct: round 0's K (token 0's pick) loaded before the dedupe (VGPRs live across it: 304 vs 203)
    dspec = direct and dspec
    NP = q_len * 16
    NPQ = QT * GQA  # the WG's Q rows
    SS = S * TS  # WGs per group
    R_MAX = -(-NP // (S * P))
    SR = S * R_MAX  # partial slots per (request, kv head, token)
    hq = hkv * GQA
    DCH = HEAD_DIM // 8
    HB = htab
    QCPT = 1
    while QCPT * NT < NPQ * 16:
        QCPT *= 2  # 16 B Q chunks per thread
    QLPR = 16 // QCPT  # lanes per Q row
    assert QCPT <= 16 and NT // QLPR >= NPQ
    lds_fields = {
        "kv": fx.Array[fx.Uint8, NW * ROWS * DEC_ROW_BYTES if direct else max(P * KV_STAGE, NW * ROWS * DEC_ROW_BYTES), 16],
        "lse": fx.Array[fx.Float32, NW * ROWS],
        "pick": fx.Array[fx.Int32, WAVE, 16],
        "ppag": fx.Array[fx.Int32, WAVE],
        "ublk": fx.Array[fx.Int32, NP],
        "uphys": fx.Array[fx.Int32, NP],
        "umask": fx.Array[fx.Int32, NP],
        "misc": fx.Array[fx.Int32, 2],
        "htab": fx.Array[fx.Int32, 2 * max(HB, 1) + 4, 16],
        "qs": fx.Array[fx.Uint8, NPQ * HEAD_DIM, 16],
        "qsc": fx.Array[fx.Float32, NPQ],
    }
    SharedStorage = fx.struct(type(f"M3DecodeLds{q_len}", (), {"__annotations__": lds_fields}))
    agent = fx.rocdl.SyncScope.Workgroup if xcd else fx.rocdl.SyncScope.Agent
    st_pol = 0 if xcd else CPOL_SC1  # partial stores: L2 (same XCD) or agent-coherent
    ld_pol = 1 if xcd else CPOL_SC1  # partial loads: sc0 (past L1) or agent-coherent
    neg_big = -1.0e30
    neg_inf = float("-inf")

    @flyc.kernel(name=f"m3_kvouter_decode_hkv{hkv}_q{q_len}_s{S}p{P}h{KH}{'x' if xcd else ''}{f't{htab}' if htab else ''}{'c' if c16 else ''}{f'q{TS}' if TS > 1 else ''}{('ds' if dspec else 'd') if direct else ''}", known_block_size=[NT, 1, 1])
    def decode_kernel(
        arg_q: fx.Pointer,
        arg_k: fx.Pointer,
        arg_v: fx.Pointer,
        arg_ks: fx.Pointer,
        arg_vs: fx.Pointer,
        arg_topk: fx.Pointer,
        arg_bt: fx.Pointer,
        arg_qpos: fx.Pointer,
        arg_po: fx.Pointer,
        arg_plse: fx.Pointer,
        arg_cnt: fx.Pointer,
        arg_out: fx.Pointer,
        f32_scale_log2: fx.Float32,
        i32_nq: fx.Int32,
        i32_bt_stride: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        bid = fx.Int32(gpu.block_id("x"))
        wave = tid // fx.Int32(WAVE)
        lane = tid % fx.Int32(WAVE)
        col = lane % fx.Int32(32)
        half = lane // fx.Int32(32)
        head16 = col % fx.Int32(GQA)
        grp_w = wave // fx.Int32(wpb)
        pair = grp_w // fx.Int32(KH)  # block slot
        khalf = grp_w % fx.Int32(KH)
        wip = wave % fx.Int32(wpb)
        if const_expr(xcd):
            # the SS WGs of a group share one XCD (WGs go round-robin over the XCDs): bid
            # = (grp // 8 * SS + j) * 8 + grp % 8; the handoff then stays in that XCD's L2
            xj = bid // fx.Int32(N_XCD)
            grp = xj // fx.Int32(SS) * fx.Int32(N_XCD) + bid % fx.Int32(N_XCD)
            sj = xj % fx.Int32(SS)
        else:
            grp = bid // fx.Int32(SS)  # request * hkv + kv head
            sj = bid % fx.Int32(SS)
        th = sj // fx.Int32(S)  # token run
        split = sj % fx.Int32(S)
        tb = th * fx.Int32(QT)  # the run's first token within the request
        cgrp = grp * fx.Int32(TS) + th  # arrival counter
        req = grp // fx.Int32(hkv)
        kvh = grp % fx.Int32(hkv)
        tok0 = req * fx.Int32(q_len)

        q_buf = ptr_buf_tensor(arg_q, fx.Int32)
        ks_buf = ptr_buf_tensor(arg_ks, fx.Float32)
        vs_buf = ptr_buf_tensor(arg_vs, fx.Float32)
        tk_buf = ptr_buf_tensor(arg_topk, fx.Int32)
        bt_buf = ptr_buf_tensor(arg_bt, fx.Int32)
        qp_buf = ptr_buf_tensor(arg_qpos, fx.Int32)
        out_buf = ptr_buf_tensor(arg_out, fx.Int32)
        k_base, v_base, ks_base, vs_base = (buf_base_i64(a) for a in (arg_k, arg_v, arg_ks, arg_vs))
        po_rsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(ptr_buf_tensor(arg_po, fx.Int32)))
        lse_rsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(ptr_buf_tensor(arg_plse, fx.Float32)))

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        lse_t = lds.lse.view(fx.make_layout(NW * ROWS, 1))
        pick_t = lds.pick.view(fx.make_layout(WAVE, 1))
        ppag_t = lds.ppag.view(fx.make_layout(WAVE, 1))
        ublk_t = lds.ublk.view(fx.make_layout(NP, 1))
        uphys_t = lds.uphys.view(fx.make_layout(NP, 1))
        umask_t = lds.umask.view(fx.make_layout(NP, 1))
        misc_t = lds.misc.view(fx.make_layout(2, 1))
        qsc_t = lds.qsc.view(fx.make_layout(NPQ, 1))
        kv_ptr = lds.kv.ptr

        if const_expr(HB > 0):
            # dedupe table: [0, HB] first lane per block (HB: invalid picks), then token masks
            for c in range_constexpr(-(-(2 * HB + 2) // NT)):
                hi = tid + fx.Int32(c * NT)
                if hi < fx.Int32(2 * HB + 2):
                    fx.add_offset(lds.htab.ptr, hi).store((hi <= fx.Int32(HB)).select(fx.Int32(WAVE), fx.Int32(0)))
        # ---- wave 0: the request's picks -> unique blocks (first-pick order) + token masks ----
        if wave == fx.Int32(0):
            ln = (lane < fx.Int32(NP)).select(lane, fx.Int32(0))
            t = ln // fx.Int32(16)
            b = fx.Int32(_ld(tk_buf, (kvh * i32_nq + tok0 + t) * fx.Int32(16) + ln % fx.Int32(16), 1))
            qp = fx.Int32(_ld(qp_buf, tok0 + t, 1))
            ok = (lane < fx.Int32(NP)) & (b >= fx.Int32(0)) & (b <= qp // fx.Int32(BLOCK))
            pick_t[lane] = ok.select(b, fx.Int32(-1))
            ppag_t[lane] = fx.Int32(_ld(bt_buf, req * i32_bt_stride + ok.select(b, fx.Int32(0)), 1))
        # the WG's q_len x 16 Q rows, once (not once per block group): thread = (row, QCPT
        # 16 B chunks); waves 1.. issue these while wave 0 runs the pick chain
        q_r = tid // fx.Int32(QLPR)
        q_j = tid % fx.Int32(QLPR)
        q_rok = q_r < fx.Int32(NPQ)
        q_rc = q_rok.select(q_r, fx.Int32(0))
        q_dw = ((tok0 + tb + q_rc // fx.Int32(GQA)) * fx.Int32(hq) + kvh * fx.Int32(GQA) + q_rc % fx.Int32(GQA)) * fx.Int32(HEAD_DIM // 2)
        q_raw = [Vec(_ld(q_buf, q_dw + q_j * fx.Int32(QCPT * 4) + fx.Int32(4 * c), 4)) for c in range_constexpr(QCPT)]
        gpu.barrier()

        # Speculative round-0 load: unique block s * P + p is token 0's pick s * P + p whenever
        # token 0's picks are valid and distinct (MTP: tokens share their picks). Checked below.
        kv_off = pair * fx.Int32(KV_STAGE)
        wig = wave - pair * fx.Int32(KH * wpb)  # wave within the block's groups
        spec_j = split * fx.Int32(P) + pair
        spec_page = fx.Int32(ppag_t[(spec_j < fx.Int32(NP)).select(spec_j, fx.Int32(0))]) * fx.Int32(8)

        def load_block(page0):
            # every wave of the group also copies the (same) 1 KB of scales: no divergent branch
            _dma_block(kv_ptr, kv_off, k_base, v_base, ks_base, vs_base, wig, lane, KH * wpb * WAVE, page0, kvh, hkv,
                       kswz=c16)
            fx.rocdl.asyncmark()

        # this lane's query row (token `ed` of the request, head) is the same for every block
        r = col
        k_perm = (r // fx.Int32(4)) % fx.Int32(2) * fx.Int32(16) + (r // fx.Int32(8)) * fx.Int32(4) + r % fx.Int32(4)
        # c16: lane = (k group g16, head); rowi = the lane's row among the wave's ROWS
        g16 = lane // fx.Int32(GQA)
        rowi = head16 if c16 else col
        hsel = g16 if c16 else half  # the lanes that store a row's lse
        ed = wip * fx.Int32(EPW) + (fx.Int32(0) if c16 else col // fx.Int32(GQA))
        ed_ok = ed < fx.Int32(QT)
        ed_c = ed_ok.select(ed, fx.Int32(0))  # token within the run
        et = tb + ed_c  # token within the request
        tok = tok0 + et

        if const_expr(direct):
            spec_regs = _Direct16(k_base, v_base, ks_base, vs_base, spec_page, kvh, hkv, lane) if dspec else None
        else:
            load_block(spec_page)
        if wave == fx.Int32(0):
            x = fx.Int32(pick_t[lane])
            if const_expr(HB > 0):
                earlier, mask = _dedupe_htab(lds.htab.ptr, x, lane, HB)
            else:
                earlier, mask = _dedupe_all(lds.pick.ptr, x, lane, NP)
            first = (x >= fx.Int32(0)) & (earlier == fx.Int32(0))
            bal = fx.Int64(fx.rocdl.ballot(T.i64, raw(first)))
            urank = fx.Int32(fx.rocdl.mbcnt_hi(T.i32, raw(fx.Int32(bal >> fx.Int64(32))),
                                               raw(fx.rocdl.mbcnt_lo(T.i32, raw(fx.Int32(bal)), raw(fx.Int32(0))))))
            if first:
                uphys_t[urank] = fx.Int32(ppag_t[lane])
                ublk_t[urank] = x
                umask_t[urank] = mask
            if lane == fx.Int32(WAVE - 1):
                total = urank + first.select(fx.Int32(1), fx.Int32(0))
                misc_t[0] = total
                if total == fx.Int32(0):
                    uphys_t[0] = fx.Int32(0)  # keeps the (discarded) clamped load in bounds

        qpos = fx.Int32(_ld(qp_buf, tok, 1))
        q_words, q_amax = _wg_q_quant(q_raw, QCPT, QLPR)
        if q_rok:
            _lds_at(lds.qs.ptr, q_r * fx.Int32(HEAD_DIM) + q_j * fx.Int32(QCPT * 8), QCPT * 8).store(
                Vec.from_elements(q_words, fx.Int32).bitcast(fx.Uint8))
            if q_j == fx.Int32(0):
                qsc_t[q_r] = q_amax * fx.Float32(1.0 / FP8_MAX)
        gpu.barrier()

        q_row = ed_c * fx.Int32(GQA) + head16
        if const_expr(c16):
            q_ops = Vec(_lds_at(lds.qs.ptr, q_row * fx.Int32(HEAD_DIM) + g16 * fx.Int32(32), 32).load()).bitcast(fx.Int32)
        else:
            q_ops = [Vec(_lds_at(lds.qs.ptr, q_row * fx.Int32(HEAD_DIM) + fx.Int32(ws * 64) + half * fx.Int32(32), 32).load()
                         ).bitcast(fx.Int32) for ws in range_constexpr(2)]
        q_sc = fx.Float32(qsc_t[q_row])
        n_u = fx.Int32(misc_t[0])
        u_last = (n_u > fx.Int32(0)).select(n_u - fx.Int32(1), fx.Int32(0))

        def rounds(sp):
            lo = sp * fx.Int32(P)
            return (n_u > lo).select((n_u - lo + fx.Int32(S * P - 1)) // fx.Int32(S * P), fx.Int32(0))

        n_r = rounds(split)
        stg = wave * fx.Int32(ROWS * DEC_ROW_BYTES)  # this wave's staged O rows (over the K/V area)
        for r_iv in range(fx.Int32(0), n_r, fx.Int32(1)):
            rd = fx.Int32(r_iv)
            u = (rd * fx.Int32(S) + split) * fx.Int32(P) + pair
            act = u < n_u
            uc = act.select(u, u_last)
            page0 = fx.Int32(uphys_t[uc]) * fx.Int32(8)
            if const_expr(direct):
                if const_expr(R_MAX > 1):
                    gpu.barrier()  # last round's staged rows are consumed
                if const_expr(dspec):
                    regs = spec_regs.reload_k((rd != fx.Int32(0)) | (page0 != spec_page), k_base, ks_base, page0)
                else:
                    regs = _Direct16(k_base, v_base, ks_base, vs_base, page0, kvh, hkv, lane)
            else:
                regs = None
                gpu.barrier()  # last round's staged rows are consumed
                if (rd != fx.Int32(0)) | (page0 != spec_page):
                    fx.rocdl.wait_asyncmark(0)  # a wrong speculative copy must land before it is overwritten
                    load_block(page0)
                fx.rocdl.wait_asyncmark(0)
                gpu.barrier()

            member = (fx.Int32(umask_t[uc]) >> et) & fx.Int32(1)
            if const_expr(c16):
                o_acc, o_mul, lse_b = _block_attn16(kv_ptr, kv_off, kv_off + fx.Int32(BLOCK_BYTES),
                                                    kv_off + fx.Int32(2 * BLOCK_BYTES), q_ops, q_sc, qpos,
                                                    fx.Int32(ublk_t[uc]), lane, f32_scale_log2,
                                                    khalf=None if KH == 1 else khalf, regs=regs)
            else:
                o_acc, o_mul, lse_b = _block_attn(kv_ptr, kv_ptr, kv_off, kv_off + fx.Int32(BLOCK_BYTES), ks_buf, vs_buf,
                                                  q_ops, q_sc, qpos, page0, kvh, fx.Int32(ublk_t[uc]), k_perm, col, half,
                                                  f32_scale_log2, hkv, nkt=4 // KH,
                                                  kb=None if KH == 1 else khalf * fx.Int32(64),
                                                  scl_off=kv_off + fx.Int32(2 * BLOCK_BYTES))
            lse_b = (act & (member != fx.Int32(0))).select(lse_b, fx.Float32(neg_inf))

            # ---- LSE-merge the P blocks' rows through LDS ----
            lse_t[wave * fx.Int32(ROWS) + rowi] = lse_b
            gpu.barrier()  # lses visible, and every K/V read is done
            m = fx.Float32(neg_big)
            lses = []
            for p2 in range_constexpr(G):
                l2 = fx.Float32(lse_t[fx.Int32(p2 * wpb * ROWS) + wip * fx.Int32(ROWS) + rowi])
                lses.append(l2)
                m = _max(m, l2)
            den = fx.Float32(0.0)
            for l2 in lses:
                den = den + (l2 - m).exp2(fastmath="afn")
            has = den > fx.Float32(0.0)
            scale = has.select((lse_b - m).exp2(fastmath="afn") / den * o_mul, fx.Float32(0.0))
            if const_expr(c16):
                # d = 16 dt + 4 g16 + e
                for dt in range_constexpr(8):
                    vals = [(fx.Float32(o_acc[dt][e]) * scale).to(fx.BFloat16) for e in range_constexpr(4)]
                    off = stg + head16 * fx.Int32(DEC_ROW_BYTES) + fx.Int32(dt * 32) + g16 * fx.Int32(8)
                    _lds_at(kv_ptr, off, 8).store(Vec.from_elements(vals, fx.BFloat16).bitcast(fx.Uint8))
            else:
                for dt in range_constexpr(4):
                    for g in range_constexpr(4):
                        vals = [(fx.Float32(o_acc[dt][g * 4 + e]) * scale).to(fx.BFloat16) for e in range_constexpr(4)]
                        off = stg + col * fx.Int32(DEC_ROW_BYTES) + fx.Int32(dt * 64 + g * 16) + half * fx.Int32(8)
                        _lds_at(kv_ptr, off, 8).store(Vec.from_elements(vals, fx.BFloat16).bitcast(fx.Uint8))
            slot = rd * fx.Int32(S) + split
            if (grp_w == fx.Int32(0)) & ed_ok & (hsel == fx.Int32(0)):
                buffer_ops.buffer_store(has.select(m + fx.math.log2(den), fx.Float32(neg_inf)), lse_rsrc,
                                        ((grp * fx.Int32(q_len) + tb + ed) * fx.Int32(SR) + slot) * fx.Int32(GQA) + head16,
                                        cache_modifier=st_pol)
            gpu.barrier()
            # sum the P staged copies of each (token, head) row -> one bf16 partial row
            for c in range_constexpr(-(-QT * GQA * DCH // NT)):
                unit = tid + fx.Int32(c * NT)
                row = unit // fx.Int32(DCH)  # run token * 16 + head
                chunk = unit % fx.Int32(DCH)
                t = row // fx.Int32(GQA)
                head = row % fx.Int32(GQA)
                srow = (t // fx.Int32(EPW)) * fx.Int32(ROWS) + (t % fx.Int32(EPW)) * fx.Int32(GQA) + head
                acc = [fx.Float32(0.0)] * 8
                for p2 in range_constexpr(G):
                    pv = Vec(_lds_at(kv_ptr, (fx.Int32(p2 * wpb * ROWS) + srow) * fx.Int32(DEC_ROW_BYTES) + chunk * fx.Int32(16),
                                     16).load()).bitcast(fx.BFloat16)
                    acc = [acc[e] + fx.Float32(pv[e]) for e in range_constexpr(8)]
                vals = [acc[e].to(fx.BFloat16) for e in range_constexpr(8)]
                prow = ((grp * fx.Int32(q_len) + tb + t) * fx.Int32(SR) + slot) * fx.Int32(GQA) + head
                if unit < fx.Int32(QT * GQA * DCH):
                    buffer_ops.buffer_store(Vec.from_elements(vals, fx.BFloat16).bitcast(fx.Int32), po_rsrc,
                                            prow * fx.Int32(HEAD_DIM // 2) + chunk * fx.Int32(4), cache_modifier=st_pol)
        fx.rocdl.wait_asyncmark(0)  # a split with no block still has its speculative copy in flight
        # sc1 (agent-coherent) stores need only to have landed before the arrive; an agent
        # release fence would also write back the whole L2 (buffer_wbl2)
        fx.rocdl.s_waitcnt(vmcnt=0)

        # ---- arrive; the request's last split merges ----
        gpu.barrier()
        if tid == fx.Int32(0):
            cnt_ptr = fx.add_offset(_gptr_i32(arg_cnt), cgrp * fx.Int32(CNT_STRIDE))
            last = fx.atomic_add(cnt_ptr, fx.Int32(1), syncscope=agent) == fx.Int32(S - 1)
            misc_t[1] = last.select(fx.Int32(1), fx.Int32(0))
            if last:
                fx.add_offset(fx.get_iter(ptr_buf_tensor(arg_cnt, fx.Int32)), cgrp * fx.Int32(CNT_STRIDE)).store(fx.Int32(0))
        gpu.barrier()
        if fx.Int32(misc_t[1]) == fx.Int32(1):
            # (token, head, 8 d) units; slot (round, split) holds data iff round < rounds(split)
            n_rs = [rounds(fx.Int32(sp)) for sp in range_constexpr(S)]
            nunit = -(-QT * GQA * DCH // NT)
            uinfo = []
            for c in range_constexpr(nunit):
                unit = tid + fx.Int32(c * NT)
                uok = unit < fx.Int32(QT * GQA * DCH)
                unit = uok.select(unit, fx.Int32(0))
                row = unit // fx.Int32(DCH)
                t = tb + row // fx.Int32(GQA)
                head = row % fx.Int32(GQA)
                chunk = unit % fx.Int32(DCH)
                uinfo.append((uok, t, head, chunk, (grp * fx.Int32(q_len) + t) * fx.Int32(SR * GQA) + head))
            init = []
            for c in range_constexpr(nunit):
                init += [fx.Float32(neg_big), fx.Float32(0.0)] + [fx.Float32(0.0)] * 8
            for r_iv, st in range(fx.Int32(0), n_rs[0], fx.Int32(1), init=init):
                rd = fx.Int32(r_iv)
                new = []
                for c in range_constexpr(nunit):
                    _, t, head, chunk, row0 = uinfo[c]
                    valid = [rd < n_rs[sp] for sp in range_constexpr(S)]
                    rows = [row0 + (rd * fx.Int32(S) + fx.Int32(sp)) * fx.Int32(GQA) for sp in range_constexpr(S)]
                    # a split without round rd reads split 0's (written, finite) row: w = 0 cancels it
                    # without a select per element
                    rows = [valid[sp].select(rows[sp], rows[0]) for sp in range_constexpr(S)]
                    parts = [Vec(buffer_ops.buffer_load(po_rsrc, rows[sp] * fx.Int32(HEAD_DIM // 2) + chunk * fx.Int32(4), 4,
                                                        T.i32, cache_modifier=ld_pol)).bitcast(fx.BFloat16)
                             for sp in range_constexpr(S)]
                    lses = [fx.Float32(buffer_ops.buffer_load(lse_rsrc, rows[sp], 1, T.f32, cache_modifier=ld_pol))
                            for sp in range_constexpr(S)]
                    lses = [valid[sp].select(lses[sp], fx.Float32(neg_inf)) for sp in range_constexpr(S)]
                    mx = fx.Float32(st[10 * c])
                    m_new = mx
                    for x in lses:
                        m_new = _max(m_new, x)
                    a = (mx - m_new).exp2(fastmath="afn")
                    den = fx.Float32(st[10 * c + 1]) * a
                    acc = [fx.Float32(st[10 * c + 2 + e]) * a for e in range_constexpr(8)]
                    for sp in range_constexpr(S):
                        w = (lses[sp] - m_new).exp2(fastmath="afn")
                        den = den + w
                        for e in range_constexpr(8):
                            acc[e] = fx.math.fma(fx.Float32(parts[sp][e]), w, acc[e])
                    new += [m_new, den] + acc
                st = yield new
            for c in range_constexpr(nunit):
                uok, t, head, chunk, _ = uinfo[c]
                den = fx.Float32(st[10 * c + 1])
                rden = (den > fx.Float32(0.0)).select(fx.Float32(1.0) / den, fx.Float32(0.0))
                vals = [(fx.Float32(st[10 * c + 2 + e]) * rden).to(fx.BFloat16) for e in range_constexpr(8)]
                obase = ((tok0 + t) * fx.Int32(hq) + kvh * fx.Int32(GQA) + head) * fx.Int32(HEAD_DIM // 2) + chunk * fx.Int32(4)
                if uok:
                    fx.add_offset(fx.get_iter(out_buf), obase).store(Vec.from_elements(vals, fx.BFloat16).bitcast(fx.Int32))

    @flyc.jit
    def launch(
        arg_q: fx.Pointer,
        arg_k: fx.Pointer,
        arg_v: fx.Pointer,
        arg_ks: fx.Pointer,
        arg_vs: fx.Pointer,
        arg_topk: fx.Pointer,
        arg_bt: fx.Pointer,
        arg_qpos: fx.Pointer,
        arg_po: fx.Pointer,
        arg_plse: fx.Pointer,
        arg_cnt: fx.Pointer,
        arg_out: fx.Pointer,
        f32_scale_log2: fx.Float32,
        i32_nq: fx.Int32,
        i32_bt_stride: fx.Int32,
        i32_grid: fx.Int32,
        stream: fx.Stream,
    ):
        decode_kernel(
            arg_q, arg_k, arg_v, arg_ks, arg_vs, arg_topk, arg_bt, arg_qpos, arg_po, arg_plse, arg_cnt,
            arg_out, f32_scale_log2, i32_nq, i32_bt_stride,
        ).launch(grid=(fx.Int64(i32_grid), 1, 1), block=(NT, 1, 1), stream=stream)

    return launch


def build_combine(hkv: int, u8=False, w16=False):
    """One WG per (kv head, query): thread = (head, 8 d), so each thread puts all 16 of its
    partial loads (16 B bf16, or 8 B u8 + 2 scales) in flight at once.
    w16 (u8): thread = (head, 16 d) with 16 B code loads; a WG merges two
    (kv head, query) pairs (grid = ceil(Hkv * N / 2))."""
    assert u8 or not w16
    hq = hkv * GQA
    DCH = HEAD_DIM // 8  # 8-d chunks per row

    if w16:
        return _build_combine_w16(hkv)

    @flyc.kernel(name=f"m3_kvouter_combine_hkv{hkv}{'_u8' if u8 else ''}", known_block_size=[NTHREADS, 1, 1])
    def combine_kernel(
        arg_po: fx.Pointer,
        arg_plse: fx.Pointer,
        arg_psc: fx.Pointer,
        arg_inv: fx.Pointer,
        arg_out: fx.Pointer,
        i32_nq: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        pid = fx.Int32(gpu.block_id("x"))  # kvh * nq + q
        head = tid // fx.Int32(DCH)
        dch = tid % fx.Int32(DCH)
        po_buf = ptr_buf_tensor(arg_po, fx.Int32)
        lse_buf = ptr_buf_tensor(arg_plse, fx.Float32)
        psc_buf = ptr_buf_tensor(arg_psc, fx.Float32)
        inv_buf = ptr_buf_tensor(arg_inv, fx.Int32)
        out_buf = ptr_buf_tensor(arg_out, fx.Int32)
        kvh = pid // i32_nq
        qi = pid % i32_nq
        # partials are query-major (slot = pid * 16 + rank); inv only says which picks were kept
        es = [fx.Int32(buf_scalar_load(inv_buf, pid * fx.Int32(16) + fx.Int32(rk))) for rk in range_constexpr(16)]  # WG-uniform
        neg_inf = fx.Float32(float("-inf"))
        lses = []
        parts = []
        scs = []
        oks = []
        for rk in range_constexpr(16):
            ok = es[rk] >= fx.Int32(0)
            oks.append(ok)
            row = (pid * fx.Int32(16) + fx.Int32(rk)) * fx.Int32(GQA) + head
            if const_expr(u8):
                parts.append(Vec(_ld(po_buf, row * fx.Int32(HEAD_DIM // 4) + dch * fx.Int32(2), 2)))
                scs.append(Vec(_ld(psc_buf, row * fx.Int32(2), 2, T.f32)))
            else:
                parts.append(Vec(_ld(po_buf, row * fx.Int32(HEAD_DIM // 2) + dch * fx.Int32(4), 4)).bitcast(fx.BFloat16))
            lses.append(ok.select(fx.Float32(_ld(lse_buf, row, 1, T.f32)), neg_inf))
        mx = lses[0]
        for rk in range_constexpr(1, 16):
            mx = _max(mx, lses[rk])
        acc = [fx.Float32(0.0)] * 8
        den = fx.Float32(0.0)
        bias = [fx.Float32(0.0)] * 2  # sum of w * scale per 4-d group: the u8 codes carry +128
        for rk in range_constexpr(16):
            w = (lses[rk] - mx).exp2(fastmath="afn")
            den = den + w
            if const_expr(u8):
                for h in range_constexpr(2):
                    # a dropped pick's slot is never written: its weight is 0 but its bytes are stale
                    ws = oks[rk].select(w * fx.Float32(scs[rk][h]), fx.Float32(0.0))
                    bias[h] = bias[h] + ws
                    word = fx.Int32(parts[rk][h])
                    for u in range_constexpr(4):
                        code = ((word >> fx.Int32(8 * u)) & fx.Int32(0xFF)) if u < 3 else (word >> fx.Int32(24)) & fx.Int32(0xFF)
                        acc[4 * h + u] = acc[4 * h + u] + ws * code.to(fx.Float32)
            else:
                for u in range_constexpr(8):
                    acc[u] = acc[u] + oks[rk].select(w * fx.Float32(parts[rk][u]), fx.Float32(0.0))
        if const_expr(u8):
            acc = [acc[u] - fx.Float32(128.0) * bias[u // 4] for u in range_constexpr(8)]
        # a query with every pick dropped has den = NaN (mx = -inf): output 0, as the reference
        rden = (den > fx.Float32(0.0)).select(fx.Float32(1.0) / den, fx.Float32(0.0))
        obase = (qi * fx.Int32(hq) + kvh * fx.Int32(GQA) + head) * fx.Int32(HEAD_DIM // 2) + dch * fx.Int32(4)
        vals = [(acc[u] * rden).to(fx.BFloat16) for u in range_constexpr(8)]
        fx.add_offset(fx.get_iter(out_buf), obase).store(Vec.from_elements(vals, fx.BFloat16).bitcast(fx.Int32))

    @flyc.jit
    def launch(
        arg_po: fx.Pointer,
        arg_plse: fx.Pointer,
        arg_psc: fx.Pointer,
        arg_inv: fx.Pointer,
        arg_out: fx.Pointer,
        i32_nq: fx.Int32,
        i32_grid: fx.Int32,
        stream: fx.Stream,
    ):
        combine_kernel(arg_po, arg_plse, arg_psc, arg_inv, arg_out, i32_nq).launch(
            grid=(fx.Int64(i32_grid), 1, 1), block=(NTHREADS, 1, 1), stream=stream
        )

    return launch


def _build_combine_w16(hkv: int):
    hq = hkv * GQA
    DCH = HEAD_DIM // 16  # 16-d chunks per row
    HALF = NTHREADS // 2

    @flyc.kernel(name=f"m3_kvouter_combine_hkv{hkv}_u8w16", known_block_size=[NTHREADS, 1, 1])
    def combine_kernel(
        arg_po: fx.Pointer,
        arg_plse: fx.Pointer,
        arg_psc: fx.Pointer,
        arg_inv: fx.Pointer,
        arg_out: fx.Pointer,
        i32_nq: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        bid = fx.Int32(gpu.block_id("x"))
        t = tid % fx.Int32(HALF)
        head = t // fx.Int32(DCH)
        dch = t % fx.Int32(DCH)
        pid = bid * fx.Int32(2) + tid // fx.Int32(HALF)  # kvh * nq + q
        pok = pid < fx.Int32(hkv) * i32_nq
        pid = pok.select(pid, fx.Int32(0))
        po_buf = ptr_buf_tensor(arg_po, fx.Int32)
        lse_buf = ptr_buf_tensor(arg_plse, fx.Float32)
        psc_buf = ptr_buf_tensor(arg_psc, fx.Float32)
        inv_buf = ptr_buf_tensor(arg_inv, fx.Int32)
        out_buf = ptr_buf_tensor(arg_out, fx.Int32)
        kvh = pid // i32_nq
        qi = pid % i32_nq
        es = []
        for j in range_constexpr(4):
            v4 = Vec(_ld(inv_buf, pid * fx.Int32(16) + fx.Int32(4 * j), 4))
            es += [fx.Int32(v4[u]) for u in range_constexpr(4)]
        neg_inf = fx.Float32(float("-inf"))
        lses = []
        parts = []
        scs = []
        for rk in range_constexpr(16):
            row = (pid * fx.Int32(16) + fx.Int32(rk)) * fx.Int32(GQA) + head
            parts.append(Vec(_ld(po_buf, row * fx.Int32(HEAD_DIM // 4) + dch * fx.Int32(4), 4)))
            scs.append(Vec(_ld(psc_buf, row * fx.Int32(2), 2, T.f32)))  # code dwords alternate scales
            lses.append((es[rk] >= fx.Int32(0)).select(fx.Float32(_ld(lse_buf, row, 1, T.f32)), neg_inf))
        mx = lses[0]
        for rk in range_constexpr(1, 16):
            mx = _max(mx, lses[rk])
        acc = [fx.Float32(0.0)] * 16
        den = fx.Float32(0.0)
        bias = [fx.Float32(0.0)] * 2  # sum of w * scale per scale: the u8 codes carry +128
        for rk in range_constexpr(16):
            w = (lses[rk] - mx).exp2(fastmath="afn")
            den = den + w
            # a dropped pick's slot is never written: its weight is 0 but its bytes are stale
            wss = [(es[rk] >= fx.Int32(0)).select(w * fx.Float32(scs[rk][h]), fx.Float32(0.0))
                   for h in range_constexpr(2)]
            bias = [bias[h] + wss[h] for h in range_constexpr(2)]
            for k in range_constexpr(4):
                ws = wss[k % 2]
                word = fx.Int32(parts[rk][k])
                for u in range_constexpr(4):
                    code = ((word >> fx.Int32(8 * u)) & fx.Int32(0xFF)) if u < 3 else (word >> fx.Int32(24)) & fx.Int32(0xFF)
                    acc[4 * k + u] = acc[4 * k + u] + ws * code.to(fx.Float32)
        # a query with every pick dropped has den = NaN (mx = -inf): output 0, as the reference
        rden = (den > fx.Float32(0.0)).select(fx.Float32(1.0) / den, fx.Float32(0.0))
        obase = (qi * fx.Int32(hq) + kvh * fx.Int32(GQA) + head) * fx.Int32(HEAD_DIM // 2) + dch * fx.Int32(8)
        vals = [((acc[u] - fx.Float32(128.0) * bias[(u // 4) % 2]) * rden).to(fx.BFloat16) for u in range_constexpr(16)]
        if pok:
            for k in range_constexpr(2):
                fx.add_offset(fx.get_iter(out_buf), obase + fx.Int32(4 * k)).store(
                    Vec.from_elements(vals[8 * k: 8 * k + 8], fx.BFloat16).bitcast(fx.Int32))

    @flyc.jit
    def launch(
        arg_po: fx.Pointer,
        arg_plse: fx.Pointer,
        arg_psc: fx.Pointer,
        arg_inv: fx.Pointer,
        arg_out: fx.Pointer,
        i32_nq: fx.Int32,
        i32_grid: fx.Int32,
        stream: fx.Stream,
    ):
        combine_kernel(arg_po, arg_plse, arg_psc, arg_inv, arg_out, i32_nq).launch(
            grid=(fx.Int64(i32_grid), 1, 1), block=(NTHREADS, 1, 1), stream=stream
        )

    return launch


# ---------------------------------------------------------------------------
# Plan: invert topk_idx [Hkv, N, topk] into edge CSR + tiles, on device, no sync;
# Q -> e4m3 + per-row scale rides along in the count kernel.
# Slot = (kv head, request, logical block). count -> scan -> scatter.
# ---------------------------------------------------------------------------
def _gptr_i32(arg):
    pt = fx.PointerType.get(fx.Int32.ir_type, address_space=fx.AddressSpace.Global, alignment=4)
    return fx.inttoptr(pt, buf_base_i64(arg))


def _edge_slot(g, topk_buf, qpos_buf, req_buf, i32_nq, i32_topk, i32_nreq, i32_maxb):
    """Slot of flat edge g of topk_idx [Hkv, N, topk], or -1 when the pick is dropped."""
    h = g // (i32_nq * i32_topk)
    qi = (g // i32_topk) % i32_nq
    blk = fx.Int32(_ld(topk_buf, g, 1))
    qp = fx.Int32(_ld(qpos_buf, qi, 1))
    rq = fx.Int32(_ld(req_buf, qi, 1))
    ok = (blk >= fx.Int32(0)) & (blk <= qp // fx.Int32(BLOCK))
    slot = (h * i32_nreq + rq) * i32_maxb + blk
    return ok.select(slot, fx.Int32(-1)), qi


def _qquant_row(q_buf, g, i32_rows, raw=None):
    """Thread g of 16 per row: 8 bf16 of Q row g//16 -> e4m3 scaled to the row absmax.
    Returns (row, ok, 2 packed words, scale = absmax / 448). raw: the 4 words, loaded
    already (_qquant_load)."""
    row = g // fx.Int32(16)
    ok = row < i32_rows
    r = ok.select(row, fx.Int32(0))
    if raw is None:
        raw = _qquant_load(q_buf, g, i32_rows)
    bf = Vec(raw).bitcast(fx.BFloat16)
    qf = [fx.Float32(bf[j]) for j in range_constexpr(8)]
    amax = fx.Float32(1e-30)
    for x in qf:
        amax = _max(amax, fx.absf(x))
    for k in range_constexpr(4):
        amax = _max(amax, amax.shuffle_xor(fx.Int32(1 << k), fx.Int32(64)))
    q_inv = fx.Float32(FP8_MAX) / amax
    w = Vec.from_elements([_fp8x4([qf[4 * j + u] * q_inv for u in range_constexpr(4)]) for j in range_constexpr(2)], fx.Int32)
    return r, ok, w, amax * fx.Float32(1.0 / FP8_MAX)


def _qquant_load(q_buf, g, i32_rows):
    row = g // fx.Int32(16)
    r = (row < i32_rows).select(row, fx.Int32(0))
    return Vec(_ld(q_buf, r * fx.Int32(HEAD_DIM // 2) + g % fx.Int32(16) * fx.Int32(4), 4))


COUNT_REPLICAS = 16  # privatized slot counters: hot blocks take every query's atomic


SCAN_T = 256
SCAN_LOG = 8


def _block_scan(lds, tid, se, st):
    """Inclusive Hillis-Steele scan of (se, st) over the SCAN_T threads of the block."""
    a_t = lds.a.view(fx.make_layout(SCAN_T, 1))
    b_t = lds.b.view(fx.make_layout(SCAN_T, 1))
    a_t[tid] = se
    b_t[tid] = st
    gpu.barrier()
    for k in range_constexpr(SCAN_LOG):
        off = 1 << k
        has = tid >= fx.Int32(off)
        src = has.select(tid - fx.Int32(off), tid)
        xa = has.select(fx.Int32(a_t[src]), fx.Int32(0))
        xb = has.select(fx.Int32(b_t[src]), fx.Int32(0))
        gpu.barrier()
        a_t[tid] = fx.Int32(a_t[tid]) + xa
        b_t[tid] = fx.Int32(b_t[tid]) + xb
        gpu.barrier()
    return a_t, b_t


def _wave_incl_scan(lane, a, b):
    """Inclusive scan of (a, b) over the 64 lanes of the wave."""
    for k in range_constexpr(6):
        has = lane >= fx.Int32(1 << k)
        xa = fx.Int32(gpu.shuffle_up(a, 1 << k, WAVE))
        xb = fx.Int32(gpu.shuffle_up(b, 1 << k, WAVE))
        a = a + has.select(xa, fx.Int32(0))
        b = b + has.select(xb, fx.Int32(0))
    return a, b


def _tile_record(tl_buf, bt_buf, slot, e, pos, cnt, tile, i32_nreq, i32_maxb, i32_bt_stride, phys=None):
    """Record of the tile holding edges pos .. of `slot` (edge e = its first; cnt = slot edges;
    phys = its physical block, loaded here if not given)."""
    rest = cnt - pos
    lblk = slot % i32_maxb
    rq = (slot // i32_maxb) % i32_nreq
    kvh = slot // (i32_maxb * i32_nreq)
    if phys is None:
        phys = fx.Int32(_ld(bt_buf, rq * i32_bt_stride + lblk, 1))
    tb = tile * fx.Int32(TILE_INTS)
    rec = Vec.from_elements([kvh, phys, lblk, e], fx.Int32)
    rec2 = Vec.from_elements(
        [(rest < fx.Int32(EDGES_PER_TILE)).select(rest, fx.Int32(EDGES_PER_TILE)),
         fx.Int32(0), fx.Int32(0), fx.Int32(0)], fx.Int32)
    fx.add_offset(fx.get_iter(tl_buf), tb).store(rec)
    fx.add_offset(fx.get_iter(tl_buf), tb + fx.Int32(4)).store(rec2)


def _slot_start(st_buf, bs_buf, slot, i32_nslot):
    """Global exclusive edge start of `slot` (block-local scan + scanned block total);
    slot == nslot is the edge total, kept in start[nslot]."""
    loc = fx.Int32(_ld(st_buf, slot, 1))
    off = fx.Int32(_ld(bs_buf, slot // fx.Int32(SCAN_T) * fx.Int32(2), 1))
    return (slot < i32_nslot).select(loc + off, loc)


LDS_COUNT_SLOTS = 8192  # slot range one LDS count WG holds (32 KB); wider ranges count with global atomics
COUNT_EPT = 8  # edges per thread per pass of the LDS count (their loads are issued together)
TOP_SCAN_BLOCKS = 4096  # scan blocks (nslot / SCAN_T) the scatter WGs scan themselves (32 KB LDS)


def build_plan(lds_count=True, top_scatter=True):
    """count + scan_local + scan_top + scatter (small problems: build_plan_hist).

    lds_count: one count WG per (kv head, replica) histograms that replica's contiguous
    query range in LDS and stores the counts (no agent-scope atomics per edge). Q
    quantization runs in the WGs after those.
    top_scatter (nslot <= TOP_SCAN_BLOCKS * SCAN_T): no scan_top launch; wave 0 of every
    scatter WG scans the scan_local block totals itself."""
    R = COUNT_REPLICAS
    NSL = LDS_COUNT_SLOTS
    EPT = COUNT_EPT

    @fx.struct
    class CountLds:
        cnt: fx.Array[fx.Int32, NSL]

    @flyc.kernel(name="m3_kvouter_plan_count_lds", known_block_size=[NTHREADS, 1, 1])
    def count_lds_kernel(
        arg_topk: fx.Pointer, arg_qpos: fx.Pointer, arg_req: fx.Pointer,
        arg_count: fx.Pointer, arg_inv: fx.Pointer, arg_q: fx.Pointer, arg_q8: fx.Pointer, arg_qs: fx.Pointer,
        i32_nq: fx.Int32, i32_nreq: fx.Int32, i32_maxb: fx.Int32, i32_nslot: fx.Int32, i32_qrows: fx.Int32,
        i32_nedge: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        bid = fx.Int32(gpu.block_id("x"))
        i32_ngrp = i32_nedge // (i32_nq * fx.Int32(16)) * fx.Int32(R)  # Hkv * R
        if bid >= i32_ngrp:
            g = (bid - i32_ngrp) * fx.Int32(NTHREADS) + tid
            qr, q_ok, q_w, q_sc = _qquant_row(ptr_buf_tensor(arg_q, fx.Int32), g, i32_qrows)
            if q_ok:
                fx.add_offset(fx.get_iter(ptr_buf_tensor(arg_q8, fx.Int32)),
                              qr * fx.Int32(HEAD_DIM // 4) + g % fx.Int32(16) * fx.Int32(2)).store(q_w)
                if g % fx.Int32(16) == fx.Int32(0):
                    fx.add_offset(fx.get_iter(ptr_buf_tensor(arg_qs, fx.Float32)), qr).store(q_sc)
        if bid < i32_ngrp:
            topk_buf = ptr_buf_tensor(arg_topk, fx.Int32)
            qpos_buf = ptr_buf_tensor(arg_qpos, fx.Int32)
            req_buf = ptr_buf_tensor(arg_req, fx.Int32)
            inv_buf = ptr_buf_tensor(arg_inv, fx.Int32)
            cnt_buf = ptr_buf_tensor(arg_count, fx.Int32)
            lds = fx.SharedAllocator().allocate(CountLds).peek()
            cnt_t = lds.cnt.view(fx.make_layout(NSL, 1))
            h = bid // fx.Int32(R)
            rep = bid % fx.Int32(R)
            # queries qi with qi * R // nq == rep; their slots span whole requests
            q_lo = (rep * i32_nq + fx.Int32(R - 1)) // fx.Int32(R)
            q_hi = ((rep + fx.Int32(1)) * i32_nq + fx.Int32(R - 1)) // fx.Int32(R)
            has_q = q_lo < q_hi
            rq_lo = fx.Int32(_ld(req_buf, has_q.select(q_lo, fx.Int32(0)), 1))
            rq_hi = fx.Int32(_ld(req_buf, has_q.select(q_hi - fx.Int32(1), fx.Int32(0)), 1))
            s_lo = (h * i32_nreq + rq_lo) * i32_maxb
            nrange = (rq_hi - rq_lo + fx.Int32(1)) * i32_maxb
            use_lds = nrange <= fx.Int32(NSL)
            n_used = use_lds.select(nrange, fx.Int32(0))
            for z_iv in range(tid, n_used, fx.Int32(NTHREADS)):
                cnt_t[fx.Int32(z_iv)] = fx.Int32(0)
            gpu.barrier()
            e_lo = (h * i32_nq + q_lo) * fx.Int32(16)
            e_hi = (h * i32_nq + q_hi) * fx.Int32(16)
            for b_iv in range(e_lo, e_hi, fx.Int32(NTHREADS * EPT)):
                base = fx.Int32(b_iv)
                eds = []
                for i in range_constexpr(EPT):
                    gi = base + tid + fx.Int32(i * NTHREADS)
                    g_ok = gi < e_hi
                    gc = g_ok.select(gi, e_lo)
                    qi = (gc // fx.Int32(16)) % i32_nq
                    eds.append((gi, g_ok, fx.Int32(_ld(topk_buf, gc, 1)), fx.Int32(_ld(qpos_buf, qi, 1)),
                                fx.Int32(_ld(req_buf, qi, 1))))
                for gi, g_ok, blk, qp, rq in eds:
                    ok = g_ok & (blk >= fx.Int32(0)) & (blk <= qp // fx.Int32(BLOCK))
                    slot = (h * i32_nreq + rq) * i32_maxb + blk
                    rank = fx.Int32(-1)
                    if ok & use_lds:
                        rank = fx.atomic_add(fx.add_offset(lds.cnt.ptr, slot - s_lo), fx.Int32(1), syncscope="workgroup")
                    if ok & (use_lds == False):  # noqa: E712
                        rank = fx.atomic_add(fx.add_offset(_gptr_i32(arg_count), rep * i32_nslot + slot), fx.Int32(1),
                                             syncscope=fx.rocdl.SyncScope.Agent)
                    if g_ok:
                        fx.add_offset(fx.get_iter(inv_buf), gi).store(rank)
            gpu.barrier()
            for w_iv in range(tid, n_used, fx.Int32(NTHREADS)):
                idx = fx.Int32(w_iv)
                c = fx.Int32(cnt_t[idx])
                # counters stay zero between plans (scan_local re-zeroes what it reads)
                if c != fx.Int32(0):
                    fx.add_offset(fx.get_iter(cnt_buf), rep * i32_nslot + s_lo + idx).store(c)

    @flyc.kernel(name="m3_kvouter_plan_count", known_block_size=[NTHREADS, 1, 1])
    def count_kernel(
        arg_topk: fx.Pointer, arg_qpos: fx.Pointer, arg_req: fx.Pointer,
        arg_count: fx.Pointer, arg_inv: fx.Pointer, arg_q: fx.Pointer, arg_q8: fx.Pointer, arg_qs: fx.Pointer,
        i32_nq: fx.Int32, i32_topk: fx.Int32, i32_nreq: fx.Int32, i32_maxb: fx.Int32, i32_nedge: fx.Int32,
        i32_nslot: fx.Int32, i32_qrows: fx.Int32,
    ):
        """Count edges per slot; also quantize Q to e4m3 once per token (16 threads per row)."""
        g = fx.Int32(gpu.block_id("x")) * fx.Int32(NTHREADS) + fx.Int32(gpu.thread_id("x"))
        qr, q_ok, q_w, q_sc = _qquant_row(ptr_buf_tensor(arg_q, fx.Int32), g, i32_qrows)
        if q_ok:
            fx.add_offset(fx.get_iter(ptr_buf_tensor(arg_q8, fx.Int32)),
                          qr * fx.Int32(HEAD_DIM // 4) + g % fx.Int32(16) * fx.Int32(2)).store(q_w)
            if g % fx.Int32(16) == fx.Int32(0):
                fx.add_offset(fx.get_iter(ptr_buf_tensor(arg_qs, fx.Float32)), qr).store(q_sc)
        if g < i32_nedge:
            topk_buf = ptr_buf_tensor(arg_topk, fx.Int32)
            qpos_buf = ptr_buf_tensor(arg_qpos, fx.Int32)
            req_buf = ptr_buf_tensor(arg_req, fx.Int32)
            inv_buf = ptr_buf_tensor(arg_inv, fx.Int32)
            slot, qi = _edge_slot(g, topk_buf, qpos_buf, req_buf, i32_nq, i32_topk, i32_nreq, i32_maxb)
            rank = fx.Int32(-1)
            if slot >= fx.Int32(0):
                rep = qi * fx.Int32(R) // i32_nq  # contiguous query ranges keep a tile q-local
                rank = fx.atomic_add(fx.add_offset(_gptr_i32(arg_count), rep * i32_nslot + slot), fx.Int32(1),
                                     syncscope=fx.rocdl.SyncScope.Agent)
            fx.add_offset(fx.get_iter(inv_buf), g).store(rank)

    @fx.struct
    class ScanLds:
        a: fx.Array[fx.Int32, SCAN_T]
        b: fx.Array[fx.Int32, SCAN_T]

    @flyc.kernel(name="m3_kvouter_plan_scan_local", known_block_size=[SCAN_T, 1, 1])
    def scan_local_kernel(
        arg_count: fx.Pointer, arg_rbase: fx.Pointer, arg_start: fx.Pointer, arg_tstart: fx.Pointer,
        arg_bsum: fx.Pointer, i32_nslot: fx.Int32,
    ):
        """One slot per thread: reduce the replicas (each gets its base within the slot,
        counters re-zeroed), block-local exclusive scan of edges and tiles, block totals."""
        tid = fx.Int32(gpu.thread_id("x"))
        bid = fx.Int32(gpu.block_id("x"))
        s = bid * fx.Int32(SCAN_T) + tid
        cnt_buf = ptr_buf_tensor(arg_count, fx.Int32)
        rb_buf = ptr_buf_tensor(arg_rbase, fx.Int32)
        st_buf = ptr_buf_tensor(arg_start, fx.Int32)
        ts_buf = ptr_buf_tensor(arg_tstart, fx.Int32)
        bs_buf = ptr_buf_tensor(arg_bsum, fx.Int32)
        lds = fx.SharedAllocator().allocate(ScanLds).peek()
        c = fx.Int32(0)
        if s < i32_nslot:
            xs = [fx.Int32(_ld(cnt_buf, fx.Int32(rp) * i32_nslot + s, 1)) for rp in range_constexpr(R)]
            for rp in range_constexpr(R):
                idx = fx.Int32(rp) * i32_nslot + s
                fx.add_offset(fx.get_iter(rb_buf), idx).store(c)
                c = c + xs[rp]
            for rp in range_constexpr(R):
                if xs[rp] != fx.Int32(0):
                    fx.add_offset(fx.get_iter(cnt_buf), fx.Int32(rp) * i32_nslot + s).store(fx.Int32(0))
        t = (c + fx.Int32(EDGES_PER_TILE - 1)) // fx.Int32(EDGES_PER_TILE)
        a_t, b_t = _block_scan(lds, tid, c, t)
        if s < i32_nslot:
            fx.add_offset(fx.get_iter(st_buf), s).store(fx.Int32(a_t[tid]) - c)
            fx.add_offset(fx.get_iter(ts_buf), s).store(fx.Int32(b_t[tid]) - t)
        if tid == fx.Int32(SCAN_T - 1):
            fx.add_offset(fx.get_iter(bs_buf), bid * fx.Int32(2)).store(fx.Int32(a_t[tid]))
            fx.add_offset(fx.get_iter(bs_buf), bid * fx.Int32(2) + fx.Int32(1)).store(fx.Int32(b_t[tid]))

    @flyc.kernel(name="m3_kvouter_plan_scan_top", known_block_size=[SCAN_T, 1, 1])
    def scan_top_kernel(arg_bsum: fx.Pointer, arg_start: fx.Pointer, arg_ntiles: fx.Pointer,
                        i32_nblk: fx.Int32, i32_nslot: fx.Int32):
        """Exclusive scan of the block totals in place; totals -> start[nslot], n_tiles."""
        tid = fx.Int32(gpu.thread_id("x"))
        bs_buf = ptr_buf_tensor(arg_bsum, fx.Int32)
        st_buf = ptr_buf_tensor(arg_start, fx.Int32)
        nt_buf = ptr_buf_tensor(arg_ntiles, fx.Int32)
        lds = fx.SharedAllocator().allocate(ScanLds).peek()
        per = (i32_nblk + fx.Int32(SCAN_T - 1)) // fx.Int32(SCAN_T)
        lo = tid * per
        hi = lo + per
        hi = (hi < i32_nblk).select(hi, i32_nblk)
        se = fx.Int32(0)
        st = fx.Int32(0)
        for i in range(lo, hi, fx.Int32(1)):
            se = se + fx.Int32(_ld(bs_buf, i * fx.Int32(2), 1))
            st = st + fx.Int32(_ld(bs_buf, i * fx.Int32(2) + fx.Int32(1), 1))
        a_t, b_t = _block_scan(lds, tid, se, st)
        e_run = fx.Int32(a_t[tid]) - se
        t_run = fx.Int32(b_t[tid]) - st
        if tid == fx.Int32(SCAN_T - 1):
            fx.add_offset(fx.get_iter(nt_buf), fx.Int32(0)).store(fx.Int32(b_t[tid]))
            fx.add_offset(fx.get_iter(st_buf), i32_nslot).store(fx.Int32(a_t[tid]))
        for i in range(lo, hi, fx.Int32(1)):
            xe = fx.Int32(_ld(bs_buf, i * fx.Int32(2), 1))
            xt = fx.Int32(_ld(bs_buf, i * fx.Int32(2) + fx.Int32(1), 1))
            fx.add_offset(fx.get_iter(bs_buf), i * fx.Int32(2)).store(e_run)
            fx.add_offset(fx.get_iter(bs_buf), i * fx.Int32(2) + fx.Int32(1)).store(t_run)
            e_run = e_run + xe
            t_run = t_run + xt

    @flyc.kernel(name="m3_kvouter_plan_scatter", known_block_size=[NTHREADS, 1, 1])
    def scatter_kernel(
        arg_topk: fx.Pointer, arg_qpos: fx.Pointer, arg_req: fx.Pointer, arg_bt: fx.Pointer,
        arg_rbase: fx.Pointer, arg_start: fx.Pointer, arg_tstart: fx.Pointer,
        arg_inv: fx.Pointer, arg_eq: fx.Pointer, arg_tiles: fx.Pointer, arg_bsum: fx.Pointer,
        i32_nq: fx.Int32, i32_topk: fx.Int32, i32_nreq: fx.Int32, i32_maxb: fx.Int32, i32_nedge: fx.Int32,
        i32_nslot: fx.Int32, i32_bt_stride: fx.Int32,
    ):
        """Edge -> its CSR position; the first edge of every 8 writes that tile's record."""
        g = fx.Int32(gpu.block_id("x")) * fx.Int32(NTHREADS) + fx.Int32(gpu.thread_id("x"))
        if g < i32_nedge:
            topk_buf = ptr_buf_tensor(arg_topk, fx.Int32)
            qpos_buf = ptr_buf_tensor(arg_qpos, fx.Int32)
            req_buf = ptr_buf_tensor(arg_req, fx.Int32)
            bt_buf = ptr_buf_tensor(arg_bt, fx.Int32)
            rb_buf = ptr_buf_tensor(arg_rbase, fx.Int32)
            st_buf = ptr_buf_tensor(arg_start, fx.Int32)
            ts_buf = ptr_buf_tensor(arg_tstart, fx.Int32)
            inv_buf = ptr_buf_tensor(arg_inv, fx.Int32)
            eq_buf = ptr_buf_tensor(arg_eq, fx.Int32)
            tl_buf = ptr_buf_tensor(arg_tiles, fx.Int32)
            bs_buf = ptr_buf_tensor(arg_bsum, fx.Int32)
            slot, qi = _edge_slot(g, topk_buf, qpos_buf, req_buf, i32_nq, i32_topk, i32_nreq, i32_maxb)
            if slot >= fx.Int32(0):
                rep = qi * fx.Int32(R) // i32_nq  # contiguous query ranges keep a tile q-local
                pos = fx.Int32(_ld(rb_buf, rep * i32_nslot + slot, 1)) + fx.Int32(_ld(inv_buf, g, 1))
                s0 = _slot_start(st_buf, bs_buf, slot, i32_nslot)
                e = s0 + pos
                fx.add_offset(fx.get_iter(inv_buf), g).store(e)
                fx.add_offset(fx.get_iter(eq_buf), e).store(qi * fx.Int32(16) + g % i32_topk)
                if pos % fx.Int32(EDGES_PER_TILE) == fx.Int32(0):
                    cnt = _slot_start(st_buf, bs_buf, slot + fx.Int32(1), i32_nslot) - s0
                    tile = (fx.Int32(_ld(ts_buf, slot, 1)) + fx.Int32(_ld(bs_buf, slot // fx.Int32(SCAN_T) * fx.Int32(2) + fx.Int32(1), 1))
                            + pos // fx.Int32(EDGES_PER_TILE))
                    _tile_record(tl_buf, bt_buf, slot, e, pos, cnt, tile, i32_nreq, i32_maxb, i32_bt_stride)

    TOPB = TOP_SCAN_BLOCKS

    @fx.struct
    class TopLds:
        ex: fx.Array[fx.Int32, 2 * TOPB + 2]  # exclusive (edge, tile) prefix per scan block; then the totals

    @flyc.kernel(name="m3_kvouter_plan_scatter_top", known_block_size=[NTHREADS, 1, 1])
    def scatter_top_kernel(
        arg_topk: fx.Pointer, arg_qpos: fx.Pointer, arg_req: fx.Pointer, arg_bt: fx.Pointer,
        arg_rbase: fx.Pointer, arg_start: fx.Pointer, arg_tstart: fx.Pointer,
        arg_inv: fx.Pointer, arg_eq: fx.Pointer, arg_tiles: fx.Pointer, arg_bsum: fx.Pointer, arg_ntiles: fx.Pointer,
        i32_nq: fx.Int32, i32_topk: fx.Int32, i32_nreq: fx.Int32, i32_maxb: fx.Int32, i32_nedge: fx.Int32,
        i32_nslot: fx.Int32, i32_bt_stride: fx.Int32, i32_nblk: fx.Int32,
    ):
        """scan_top + scatter: wave 0 scans the block totals into LDS (exclusive prefixes),
        then edge -> its CSR position; the first edge of every 8 writes that tile's record."""
        tid = fx.Int32(gpu.thread_id("x"))
        bid = fx.Int32(gpu.block_id("x"))
        g = bid * fx.Int32(NTHREADS) + tid
        lane = tid % fx.Int32(WAVE)
        bs_buf = ptr_buf_tensor(arg_bsum, fx.Int32)
        lds = fx.SharedAllocator().allocate(TopLds).peek()
        ex_t = lds.ex.view(fx.make_layout(2 * TOPB + 2, 1))
        if tid < fx.Int32(WAVE):
            per = (i32_nblk + fx.Int32(WAVE - 1)) // fx.Int32(WAVE)
            lo = lane * per
            hi = lo + per
            hi = (hi < i32_nblk).select(hi, i32_nblk)
            lo = (lo < hi).select(lo, hi)
            se = fx.Int32(0)
            st = fx.Int32(0)
            for i in range(lo, hi, fx.Int32(1)):
                se = se + fx.Int32(_ld(bs_buf, fx.Int32(i) * fx.Int32(2), 1))
                st = st + fx.Int32(_ld(bs_buf, fx.Int32(i) * fx.Int32(2) + fx.Int32(1), 1))
            ie, it = _wave_incl_scan(lane, se, st)
            e_run = ie - se
            t_run = it - st
            for i in range(lo, hi, fx.Int32(1)):
                ii = fx.Int32(i)
                xe = fx.Int32(_ld(bs_buf, ii * fx.Int32(2), 1))
                xt = fx.Int32(_ld(bs_buf, ii * fx.Int32(2) + fx.Int32(1), 1))
                ex_t[ii * fx.Int32(2)] = e_run
                ex_t[ii * fx.Int32(2) + fx.Int32(1)] = t_run
                e_run = e_run + xe
                t_run = t_run + xt
            if lane == fx.Int32(WAVE - 1):
                ex_t[fx.Int32(2 * TOPB)] = ie
                if bid == fx.Int32(0):
                    fx.add_offset(fx.get_iter(ptr_buf_tensor(arg_ntiles, fx.Int32)), fx.Int32(0)).store(it)
        gpu.barrier()
        if g < i32_nedge:
            topk_buf = ptr_buf_tensor(arg_topk, fx.Int32)
            qpos_buf = ptr_buf_tensor(arg_qpos, fx.Int32)
            req_buf = ptr_buf_tensor(arg_req, fx.Int32)
            bt_buf = ptr_buf_tensor(arg_bt, fx.Int32)
            rb_buf = ptr_buf_tensor(arg_rbase, fx.Int32)
            st_buf = ptr_buf_tensor(arg_start, fx.Int32)
            ts_buf = ptr_buf_tensor(arg_tstart, fx.Int32)
            inv_buf = ptr_buf_tensor(arg_inv, fx.Int32)
            eq_buf = ptr_buf_tensor(arg_eq, fx.Int32)
            tl_buf = ptr_buf_tensor(arg_tiles, fx.Int32)

            def slot_start(sl):
                # slot == nslot is the edge total
                sc = (sl < i32_nslot).select(sl, fx.Int32(0))
                loc = fx.Int32(_ld(st_buf, sc, 1)) + fx.Int32(ex_t[sc // fx.Int32(SCAN_T) * fx.Int32(2)])
                return (sl < i32_nslot).select(loc, fx.Int32(ex_t[fx.Int32(2 * TOPB)]))

            slot, qi = _edge_slot(g, topk_buf, qpos_buf, req_buf, i32_nq, i32_topk, i32_nreq, i32_maxb)
            if slot >= fx.Int32(0):
                rep = qi * fx.Int32(R) // i32_nq  # contiguous query ranges keep a tile q-local
                pos = fx.Int32(_ld(rb_buf, rep * i32_nslot + slot, 1)) + fx.Int32(_ld(inv_buf, g, 1))
                s0 = slot_start(slot)
                e = s0 + pos
                fx.add_offset(fx.get_iter(inv_buf), g).store(e)
                fx.add_offset(fx.get_iter(eq_buf), e).store(qi * fx.Int32(16) + g % i32_topk)
                if pos % fx.Int32(EDGES_PER_TILE) == fx.Int32(0):
                    cnt = slot_start(slot + fx.Int32(1)) - s0
                    tile = (fx.Int32(_ld(ts_buf, slot, 1)) + fx.Int32(ex_t[slot // fx.Int32(SCAN_T) * fx.Int32(2) + fx.Int32(1)])
                            + pos // fx.Int32(EDGES_PER_TILE))
                    _tile_record(tl_buf, bt_buf, slot, e, pos, cnt, tile, i32_nreq, i32_maxb, i32_bt_stride)

    @flyc.jit
    def launch(
        arg_topk: fx.Pointer, arg_qpos: fx.Pointer, arg_req: fx.Pointer, arg_bt: fx.Pointer,
        arg_count: fx.Pointer, arg_rbase: fx.Pointer, arg_start: fx.Pointer, arg_tstart: fx.Pointer,
        arg_inv: fx.Pointer, arg_eq: fx.Pointer, arg_tiles: fx.Pointer, arg_ntiles: fx.Pointer,
        arg_bsum: fx.Pointer, arg_q: fx.Pointer, arg_q8: fx.Pointer, arg_qs: fx.Pointer, i32_nq: fx.Int32, i32_topk: fx.Int32, i32_nreq: fx.Int32, i32_maxb: fx.Int32,
        i32_bt_stride: fx.Int32, i32_nedge: fx.Int32, i32_nslot: fx.Int32, i32_egrid: fx.Int32,
        i32_nblk: fx.Int32, i32_qrows: fx.Int32, i32_cgrid: fx.Int32, stream: fx.Stream,
    ):
        if const_expr(lds_count):
            # i32_cgrid: Hkv * R count WGs, then the Q-quant WGs
            count_lds_kernel(arg_topk, arg_qpos, arg_req, arg_count, arg_inv, arg_q, arg_q8, arg_qs,
                             i32_nq, i32_nreq, i32_maxb, i32_nslot, i32_qrows, i32_nedge).launch(
                grid=(fx.Int64(i32_cgrid), 1, 1), block=(NTHREADS, 1, 1), stream=stream)
        else:
            count_kernel(arg_topk, arg_qpos, arg_req, arg_count, arg_inv, arg_q, arg_q8, arg_qs,
                         i32_nq, i32_topk, i32_nreq, i32_maxb, i32_nedge, i32_nslot, i32_qrows).launch(
                grid=(fx.Int64(i32_cgrid), 1, 1), block=(NTHREADS, 1, 1), stream=stream)
        scan_local_kernel(arg_count, arg_rbase, arg_start, arg_tstart, arg_bsum, i32_nslot).launch(
            grid=(fx.Int64(i32_nblk), 1, 1), block=(SCAN_T, 1, 1), stream=stream)
        if const_expr(top_scatter):
            scatter_top_kernel(arg_topk, arg_qpos, arg_req, arg_bt, arg_rbase, arg_start, arg_tstart,
                               arg_inv, arg_eq, arg_tiles, arg_bsum, arg_ntiles,
                               i32_nq, i32_topk, i32_nreq, i32_maxb, i32_nedge, i32_nslot, i32_bt_stride, i32_nblk).launch(
                grid=(fx.Int64(i32_egrid), 1, 1), block=(NTHREADS, 1, 1), stream=stream)
        else:
            scan_top_kernel(arg_bsum, arg_start, arg_ntiles, i32_nblk, i32_nslot).launch(
                grid=(1, 1, 1), block=(SCAN_T, 1, 1), stream=stream)
            scatter_kernel(arg_topk, arg_qpos, arg_req, arg_bt, arg_rbase, arg_start, arg_tstart,
                           arg_inv, arg_eq, arg_tiles, arg_bsum,
                           i32_nq, i32_topk, i32_nreq, i32_maxb, i32_nedge, i32_nslot, i32_bt_stride).launch(
                grid=(fx.Int64(i32_egrid), 1, 1), block=(NTHREADS, 1, 1), stream=stream)

    return launch


HIST_SLOTS = 1024  # slots per kv head the one-launch histogram plan holds in LDS
HIST_ROWS_PER_THREAD = 8  # topk rows of one kv head each thread histograms: N <= 2048


def _rot_picks(tk, tid):
    """16 picks of a row (4 quads) with each quad rotated by (tid // 4) % 4 (selects)."""
    v = [fx.Int32(tk[u // 4][u % 4]) for u in range(16)]
    er = tid // fx.Int32(4) % fx.Int32(4)
    out = []
    for j in range(4):
        for e in range(4):
            x = v[4 * j + e]
            for k in range(1, 4):
                x = (er == fx.Int32(k)).select(v[4 * j + (e + k) % 4], x)
            out.append(x)
    return out


def _hist_seg(bid, i32_wph, i32_nq, i32_nreq, i32_maxb, i32_th):
    """Segment (kv head h) of a histogram-plan WG: (h, WG in h, first row, rows, first slot,
    first edge, first tile). Head h's WGs histogram only its rows into its own slots and
    place edges / tiles in its ranges (h * N * 16, h * th)."""
    h = bid // i32_wph
    return (h, bid % i32_wph, h * i32_nq, i32_nq, h * i32_nreq * i32_maxb, h * i32_nq * fx.Int32(16),
            h * i32_th)


def build_plan_hist(rpt=HIST_ROWS_PER_THREAD, ept=1, qpt=1):
    """Whole plan in ONE launch for small problems (per kv head: slots <= HIST_SLOTS, N rows <=
    SCAN_T * HIST_ROWS_PER_THREAD): every plan WG histograms ALL picks of its kv head into LDS
    itself (no global counters), scans them, and places its own SCAN_T * ept edges after those
    of earlier WGs. Slot counters are packed: low 16 bits all edges, high 16 bits edges of
    earlier WGs. Head h's WGs write its edges from h * N * 16 and its tiles from h * th,
    n_tiles[h] = its count (segments, see _tile_segments).
    ept: own edges per thread (fewer plan WGs, each with the whole histogram to build).
    Only the WGs past the plan grid quantize Q (qpt chunks per thread, loads first), so the
    plan WGs' load chain does not queue behind the Q traffic."""
    NS = HIST_SLOTS
    P = NS // SCAN_T  # slots per thread in the scan
    RPT = rpt
    EPT = ept
    ROWS_PER_WG = SCAN_T * EPT // 16  # topk rows of one WG's own edges

    @fx.struct
    class HistLds:
        cnt: fx.Array[fx.Int32, NS + 1]  # packed (earlier << 16) + total per slot; [NS] takes dropped picks
        own: fx.Array[fx.Int32, NS + 1]  # own edge ranks; [NS] takes dropped edges
        es: fx.Array[fx.Int32, NS]  # slot edge start
        ts: fx.Array[fx.Int32, NS]  # slot tile start
        wsum: fx.Array[fx.Int32, 2 * WAVES]

    @flyc.kernel(name=f"m3_kvouter_plan_hist{rpt}{f'_e{ept}' if ept > 1 else ''}{f'_q{qpt}' if qpt > 1 else ''}", known_block_size=[SCAN_T, 1, 1])
    def hist_kernel(
        arg_topk: fx.Pointer, arg_qpos: fx.Pointer, arg_req: fx.Pointer, arg_bt: fx.Pointer,
        arg_inv: fx.Pointer, arg_eq: fx.Pointer, arg_tiles: fx.Pointer, arg_ntiles: fx.Pointer,
        arg_q: fx.Pointer, arg_q8: fx.Pointer, arg_qs: fx.Pointer,
        i32_nq: fx.Int32, i32_nreq: fx.Int32, i32_maxb: fx.Int32, i32_nedge: fx.Int32,
        i32_nslot: fx.Int32, i32_bt_stride: fx.Int32, i32_egrid: fx.Int32, i32_qrows: fx.Int32,
        i32_wph: fx.Int32, i32_th: fx.Int32, i32_cgrid: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        bid = fx.Int32(gpu.block_id("x"))
        qbid = bid - i32_egrid
        qgrid = i32_cgrid - i32_egrid
        if qbid >= fx.Int32(0):
            # all chunk loads first: vmcnt is in order and counts the stores
            q_tb = ptr_buf_tensor(arg_q, fx.Int32)
            gqs = [(qbid + fx.Int32(qk) * qgrid) * fx.Int32(SCAN_T) + tid for qk in range(qpt)]
            raws = [_qquant_load(q_tb, x, i32_qrows) for x in gqs]
            for qk in range_constexpr(qpt):
                gq = gqs[qk]
                qr, q_ok, q_w, q_sc = _qquant_row(q_tb, gq, i32_qrows, raws[qk])
                if q_ok:
                    fx.add_offset(fx.get_iter(ptr_buf_tensor(arg_q8, fx.Int32)),
                                  qr * fx.Int32(HEAD_DIM // 4) + gq % fx.Int32(16) * fx.Int32(2)).store(q_w)
                    if gq % fx.Int32(16) == fx.Int32(0):
                        fx.add_offset(fx.get_iter(ptr_buf_tensor(arg_qs, fx.Float32)), qr).store(q_sc)
        seg, sbid, row0, rows, slot0, e0, t0 = _hist_seg(bid, i32_wph, i32_nq, i32_nreq, i32_maxb, i32_th)
        gl = [sbid * fx.Int32(SCAN_T * EPT) + fx.Int32(k * SCAN_T) + tid for k in range(EPT)]
        if bid < i32_egrid:
            lds = fx.SharedAllocator().allocate(HistLds).peek()
            cnt_t = lds.cnt.view(fx.make_layout(NS, 1))
            own_t = lds.own.view(fx.make_layout(NS, 1))
            es_t = lds.es.view(fx.make_layout(NS, 1))
            ts_t = lds.ts.view(fx.make_layout(NS, 1))
            ws_t = lds.wsum.view(fx.make_layout(2 * WAVES, 1))
            topk_buf = ptr_buf_tensor(arg_topk, fx.Int32)
            qpos_buf = ptr_buf_tensor(arg_qpos, fx.Int32)
            req_buf = ptr_buf_tensor(arg_req, fx.Int32)
            bt_buf = ptr_buf_tensor(arg_bt, fx.Int32)
            inv_buf = ptr_buf_tensor(arg_inv, fx.Int32)
            eq_buf = ptr_buf_tensor(arg_eq, fx.Int32)
            tl_buf = ptr_buf_tensor(arg_tiles, fx.Int32)
            # all loads first: this thread's own edge, then its RPT topk rows
            own = []
            for k in range_constexpr(EPT):
                g = row0 * fx.Int32(16) + gl[k]
                g_ok = gl[k] < rows * fx.Int32(16)
                slot, qi = _edge_slot(g_ok.select(g, fx.Int32(0)), topk_buf, qpos_buf, req_buf, i32_nq, fx.Int32(16),
                                      i32_nreq, i32_maxb)
                # the own edge's physical block, in case it opens a tile (off the post-scan path)
                phys = fx.Int32(_ld(bt_buf, (slot >= fx.Int32(0)).select(
                    slot // i32_maxb % i32_nreq * i32_bt_stride + slot % i32_maxb, fx.Int32(0)), 1))
                own.append([g, g_ok, slot, qi, phys])
            rws = []
            for j in range_constexpr(RPT):
                r = tid + fx.Int32(j * SCAN_T)
                r_ok = r < rows
                rc = row0 + r_ok.select(r, fx.Int32(0))
                rq_i = rc % i32_nq
                # quad u of the row at register u - rot, and each quad rotated (_rot_picks): sorted rows
                # put e.g. block 0 in the same pick of every row; 16 rotations spread those same-slot
                # LDS atomics over instructions
                rot = tid % fx.Int32(4)
                tk = [Vec(_ld(topk_buf, rc * fx.Int32(16) + (fx.Int32(u) + rot) % fx.Int32(4) * fx.Int32(4), 4))
                      for u in range_constexpr(4)]
                rws.append((r, r_ok, rc, tk, fx.Int32(_ld(qpos_buf, rq_i, 1)), fx.Int32(_ld(req_buf, rq_i, 1))))
            for s in range_constexpr(P):
                cnt_t[tid * fx.Int32(P) + fx.Int32(s)] = fx.Int32(0)
                own_t[tid * fx.Int32(P) + fx.Int32(s)] = fx.Int32(0)
            gpu.barrier()
            lo_row = sbid * fx.Int32(ROWS_PER_WG)
            for j in range_constexpr(RPT):
                r, r_ok, rc, tk, qp, rq = rws[j]
                h = rc // i32_nq
                lim = qp // fx.Int32(BLOCK)
                sbase = (h * i32_nreq + rq) * i32_maxb - slot0
                inc = (r < lo_row).select(fx.Int32(0x10001), fx.Int32(1))
                picks = _rot_picks(tk, tid)
                for u in range_constexpr(16):
                    blk = picks[u]
                    take = r_ok & (blk >= fx.Int32(0)) & (blk <= lim)
                    # branch-free: dropped picks add 0 to the pad counter
                    fx.atomic_add(fx.add_offset(lds.cnt.ptr, take.select(sbase + blk, fx.Int32(NS))),
                                  take.select(inc, fx.Int32(0)), syncscope="workgroup")
            for k in range_constexpr(EPT):
                g, g_ok, slot, qi, phys = own[k]
                ok = g_ok & (slot >= fx.Int32(0))
                sc = ok.select(slot - slot0, fx.Int32(0))
                if const_expr(EPT == 1):
                    rank = fx.Int32(0)
                    if ok:
                        rank = fx.atomic_add(fx.add_offset(lds.own.ptr, sc), fx.Int32(1), syncscope="workgroup")
                else:
                    # branch-free: dropped edges add 0 to the pad counter
                    rank = fx.Int32(fx.atomic_add(fx.add_offset(lds.own.ptr, ok.select(sc, fx.Int32(NS))),
                                                  ok.select(fx.Int32(1), fx.Int32(0)), syncscope="workgroup"))
                own[k] += [ok, sc, rank]
            gpu.barrier()
            # exclusive scan of slot edge / tile counts: P slots per thread
            s0 = tid * fx.Int32(P)
            tots = [fx.Int32(cnt_t[s0 + fx.Int32(u)]) & fx.Int32(0xFFFF) for u in range_constexpr(P)]
            ntl = [(c + fx.Int32(EDGES_PER_TILE - 1)) // fx.Int32(EDGES_PER_TILE) for c in tots]
            se = tots[0]
            st = ntl[0]
            for u in range_constexpr(1, P):
                se = se + tots[u]
                st = st + ntl[u]
            lane = tid % fx.Int32(WAVE)
            wave = tid // fx.Int32(WAVE)
            a, b = _wave_incl_scan(lane, se, st)
            if lane == fx.Int32(WAVE - 1):
                ws_t[wave] = a
                ws_t[WAVES + wave] = b
            gpu.barrier()
            e_run = a - se
            t_run = b - st
            n_til = fx.Int32(0)
            for w in range_constexpr(WAVES):
                wa = fx.Int32(ws_t[w])
                wb = fx.Int32(ws_t[WAVES + w])
                e_run = e_run + (fx.Int32(w) < wave).select(wa, fx.Int32(0))
                t_run = t_run + (fx.Int32(w) < wave).select(wb, fx.Int32(0))
                n_til = n_til + wb
            for u in range_constexpr(P):
                es_t[s0 + fx.Int32(u)] = e_run
                ts_t[s0 + fx.Int32(u)] = t_run
                e_run = e_run + tots[u]
                t_run = t_run + ntl[u]
            if (sbid == fx.Int32(0)) & (tid == fx.Int32(0)):
                fx.add_offset(fx.get_iter(ptr_buf_tensor(arg_ntiles, fx.Int32)), seg).store(n_til)
            gpu.barrier()
            for k in range_constexpr(EPT):
                g, g_ok, slot, qi, phys, ok, sc, rank = own[k]
                if g_ok:
                    e = fx.Int32(-1)
                    if ok:
                        c = fx.Int32(cnt_t[sc])
                        pos = (c >> fx.Int32(16)) + rank
                        e = e0 + fx.Int32(es_t[sc]) + pos
                        fx.add_offset(fx.get_iter(eq_buf), e).store(qi * fx.Int32(16) + g % fx.Int32(16))
                        if pos % fx.Int32(EDGES_PER_TILE) == fx.Int32(0):
                            _tile_record(tl_buf, bt_buf, sc + slot0, e, pos, c & fx.Int32(0xFFFF),
                                         t0 + fx.Int32(ts_t[sc]) + pos // fx.Int32(EDGES_PER_TILE),
                                         i32_nreq, i32_maxb, i32_bt_stride, phys)
                    fx.add_offset(fx.get_iter(inv_buf), g).store(e)

    @flyc.jit
    def launch(
        arg_topk: fx.Pointer, arg_qpos: fx.Pointer, arg_req: fx.Pointer, arg_bt: fx.Pointer,
        arg_inv: fx.Pointer, arg_eq: fx.Pointer, arg_tiles: fx.Pointer, arg_ntiles: fx.Pointer,
        arg_q: fx.Pointer, arg_q8: fx.Pointer, arg_qs: fx.Pointer,
        i32_nq: fx.Int32, i32_nreq: fx.Int32, i32_maxb: fx.Int32, i32_bt_stride: fx.Int32,
        i32_nedge: fx.Int32, i32_nslot: fx.Int32, i32_egrid: fx.Int32, i32_qrows: fx.Int32, i32_cgrid: fx.Int32,
        i32_wph: fx.Int32, i32_th: fx.Int32, stream: fx.Stream,
    ):
        hist_kernel(arg_topk, arg_qpos, arg_req, arg_bt, arg_inv, arg_eq, arg_tiles, arg_ntiles,
                    arg_q, arg_q8, arg_qs, i32_nq, i32_nreq, i32_maxb, i32_nedge, i32_nslot, i32_bt_stride,
                    i32_egrid, i32_qrows, i32_wph, i32_th, i32_cgrid).launch(
            grid=(fx.Int64(i32_cgrid), 1, 1), block=(SCAN_T, 1, 1), stream=stream)

    return launch
