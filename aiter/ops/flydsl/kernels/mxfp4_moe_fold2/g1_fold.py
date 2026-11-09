# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

# The _HPF/_HPB/_TKW/_AFB/_BEF/_HD2/_KAP knobs below are fixed constants.

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from . import dpp_utils
from .communication_ops_utils import (
    atomic_add_agent,
    fence_acquire,
    fence_release,
)
from .layout_utils import crd2idx
from .mxfp4_gemm_common import (
    _e8m0_from_amax,
    _e8m0_roundup,
    _fabs_f32,
    _inline_dpp_quad_amax,
    _lds_swizzle_mask,
    _raw,
    _umax_i32,
    bq_bytes_for,
    bscale_bytes_for,
    k_half_for,
    k_tiles_total_for,
    kas_per_chunk_dw_for,
    kbs_per_expert_dw_for,
    kBS_stride_k0_dw,
    kbs_stride_n0_dw_for,
    kmchunks_for,
    lds_acc_bytes_for,
    num_n_blocks_for,
)


def _udiv(a, c):
    return fx.Int32(fx.Uint32(a) // fx.Uint32(c))


def _umod(a, c):
    return fx.Int32(fx.Uint32(a) % fx.Uint32(c))


def _global_i32_ptr(addr_i64):
    ptr_ty = fx.PointerType.get(
        T.i32, address_space=fx.AddressSpace.Global, alignment=4
    )
    return fx.inttoptr(ptr_ty, fx.Int64(addr_i64))


def _global_i32_at(addr_i64, idx):
    # fx.ptr_load/add_offset for a plain scalar read -- no tiling needed, so
    # skip the Tensor/tile/register-fragment machinery fx.copy requires.
    return _global_i32_ptr(addr_i64)[idx]


def _global_i32_load(tiles, idx):
    # Must build atom/types here, not as module globals -- they need an active
    # MLIR trace context.
    atom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32)
    reg_lay = fx.make_layout(1, 1)
    r = fx.make_rmem_tensor(reg_lay, fx.Int32)
    fx.copy_atom_call(atom, fx.slice(tiles, (None, idx)), r)
    return r.load()[0]


def _global_scalar_tiles(addr_i64, numeric_cls, num_elems):
    ptr_ty = fx.PointerType.get(
        numeric_cls.ir_type,
        address_space=fx.AddressSpace.Global,
        alignment=numeric_cls.width // 8,
    )
    ptr = fx.inttoptr(ptr_ty, fx.Int64(addr_i64))
    flat = fx.make_view(ptr, fx.make_layout(num_elems, 1))
    return fx.logical_divide(flat, fx.make_layout(1, 1))


def _scalar_store(tiles, idx, value, numeric_cls):
    atom = fx.make_copy_atom(fx.UniversalCopy(numeric_cls.width), numeric_cls)
    reg_lay = fx.make_layout(1, 1)
    r = fx.make_rmem_tensor(reg_lay, numeric_cls)
    r.store(fx.Vector.from_elements([numeric_cls(value)], numeric_cls))
    fx.copy_atom_call(atom, r, fx.slice(tiles, (None, idx)))


def _layout_idx(layout, *coords):
    idx_coords = [fx.Int64(c) for c in coords]
    flat = crd2idx(idx_coords, layout)
    return fx.Int32(flat)


def n_out_for(inter):
    return 2 * inter


LOG2E = 1.4426950408889634


def _silu_mul_batch(gs, us):
    e = [fx.Float32(rocdl.exp2(T.f32, _raw(g * fx.Float32(-LOG2E)))) for g in gs]
    sig = [fx.Float32(rocdl.rcp(T.f32, _raw(fx.Float32(1.0) + ei))) for ei in e]
    return [gs[i] * sig[i] * us[i] for i in range(len(gs))]


def _pkmax_u16(a_i32, b_i32):
    _v2i16 = T.vec(2, T.i16)
    va = llvm.BitcastOp(_v2i16, _raw(a_i32)).result
    vb = llvm.BitcastOp(_v2i16, _raw(b_i32)).result
    vm = arith.MaxUIOp(va, vb).result
    out = llvm.BitcastOp(T.i32, vm).result
    return fx.Int32(out)


def _inline_e8m0(amax_u16_i32):
    f32 = fx.Float32(
        _raw((fx.Int32(_raw(amax_u16_i32)) & fx.Int32(0xFFFF)) << fx.Int32(16)).bitcast(
            T.f32
        )
    )
    return _e8m0_roundup(f32)


# ---------------------------------------------------------------------------
# PROLOGUE FOLD helpers (r1_d0, wave 10)
# ---------------------------------------------------------------------------
# These implement, inside stage-1's own block head, the work that used to be a
# separate `moe_sort_quant` graph node: the 257-expert padded sort and the
# [M, D_HIDDEN] bf16 moe_buf zero-init.  Every stage-1 CTA recomputes the whole
# sort redundantly (~400 VALU ops over M*TOPK <= 378 items) so that NO
# inter-CTA communication and NO grid-wide barrier is needed.


# sort-scratch LDS layout, in i32 words
# WAVE-31 r1_d2 (compute): MEMORY-LEVEL PARALLELISM on the inline-quant A path.
# ISA evidence (gemm1_a4w4_fold_..._iq_sep_ks6_as7_bsa2_spk2_sa45_rw6_fold6,
# 21_final_isa.s): despite kStages=6 REGISTER stages on B, the steady-state
# k-tile body ends with a recurring `s_waitcnt vmcnt(9)` / `vmcnt(8)` pair that
# gates the LDS write of the freshly quantised A tile.  Those waits exist
# because the hidden (A) global load for k-tile K_C is issued at the TOP of the
# SAME body that consumes it -- prefetch distance ONE body, i.e. only the 8
# B `buffer_load_dwordx4` issues of that body cover its latency.  The drain
# caps the wave's outstanding-load window at ~8-18, far below the 63 vmcnt
# slots and far below the 6 x 10 = 60 the B register ladder nominally buys.
# GEAK_G1_HPF=1 pushes the hidden load ONE body further ahead (distance 2), so
# the recurring wait should relax from vmcnt(8) to ~vmcnt(18).  Costs 8 VGPRs
# (decode runs 1 wave/SIMD at 268 VGPRs, so VGPRs are free here).
# GEAK_G1_HPF=0 restores the incumbent byte-for-byte (same JIT symbol name).
#
# MEASURED (wave 31, this lane, ONE container, process-interleaved ABBA-balanced,
# 8 pairs + 2 discarded warm-ups, separate clone + FLYDSL_RUNTIME_CACHE_DIR per arm,
# null = this same source with GEAK_G1_HPF=0 so the arms are SOURCE-IDENTICAL and
# differ only in the traced ISA):
#   ISA (rocprofv3-free, FLYDSL_DUMP_IR final_isa): the 5x vmcnt(9) + 5x vmcnt(8)
#     recurring body waits DISAPPEAR and are replaced by vmcnt(18..24).  VGPR
#     268 -> 276, AGPR 12 -> 20, occupancy unchanged at 1 wave/SIMD.  LIVE.
#   gemm1 device duration at dec_bs1 (rocprofv3, 23 eager dispatches, mechanism
#     attribution ONLY -- never a speedup claim): 17136 ns -> 16642 ns mean
#     (-2.9%), 16200 -> 15640 ns min (-3.5%).  Against the 6150 GB/s
#     parallelism-matched roof on 56 m-blocks x 1,671,168 B = 93.585 MB
#     (T_roof = 15.22 us) that is 88.8% -> 91.5% of roof on the mean and
#     93.9% -> 97.3% on the min.  NOTE: the incumbent gemm1 KERNEL was never at
#     73% of roof; the wave-29 73% figure is a BUCKET-level number and its
#     shortfall lives in the graph-replay/dispatch floor, not in this k-loop.
#   HEADLINE: raw medians A 1.38368 vs null 1.37887 (+0.35%); mode-matched
#     (base dec_bs1 >= 0.0455) A 1.38526 (n=7) vs 1.38137 (n=6) (+0.28%);
#     dec_bs1 cand_ms 0.03585 vs 0.03600 (-0.42%); paired sign 5/8 on both.
#     THAT IS BELOW the 0.88% per-bucket observability floor and below this
#     window's 1.47% clean headline span.  DO NOT BANK IT AS A WIN.
# WAVE-33 r2_d0 measured (rocprofv3 --kernel-trace, 6 ABBA-alternated profile
# processes per arm, 30 eager iters each, gemm1 DISPATCH DURATION medians):
#   depth  dec_bs1(spk2,224 CTA)  dec_bs2(216 CTA)  dec_bs3(324 CTA)
#     0        16380 ns              25981 ns          33620 ns
#     1        16441 ns (+0.0%)      26000 ns          33320 ns   <- w31 artifact
#     2        15800 ns (-3.5%)      25840 ns (-0.5%)  32960 ns (-2.0%)
#     3        17521 ns (+6.9%)      26200 ns          32880 ns
#     4        17800 ns              26480 ns          33080 ns
# ISA (21_final_isa.s, spk2 kernel): depth 0 has 5x vmcnt(9)+5x vmcnt(8)
# recurring body waits, max threshold 23, VGPR 268 / AGPR 12.  Depth 1 ->
# vmcnt(18..23), VGPR 276 / AGPR 20 (this REPRODUCES wave-31's reported ISA
# exactly, but NOT its -2.9% device delta -- depth 1 measures 0.0% here).
# Depth 2 -> recurring vmcnt(11)/(10) with a max threshold of 35, VGPR 279 /
# AGPR 23, still 1 wave/SIMD, scratch 0.  Depth >=3 falls off a cliff at
# dec_bs1 (kUnroll is only 6 there, so the prologue swallows half the loop).
# DEFAULT = 2.  GEAK_G1_HPF=0 reverts to the incumbent BYTE-FOR-BYTE.
_HPF = 2
_HPB = 1
# 0923 r1_d1 (memory): FOLD-HEAD / PROLOGUE vmcnt de-serialisation.
#   _TKW=1 : gather the fold head's topk_weights (step f) into registers at
#            step (b), next to the topk_ids load -> removes one dependent global
#            round trip + its vmcnt(0) drain.  Same bytes/values -> bit-exact.
#   _AFB=1 : A-FIRST prologue batch.  Issue ALL prologue hidden(A) loads (and
#            the hpf queue) back to back, THEN the B / B-scale stages, THEN the
#            inline quant of each stage.  vmcnt retires in order, so each quant
#            wait becomes vmcnt(#B in flight) instead of a chain of ~3 serial A
#            round trips interleaved with B.  Same loads, same math -> bit-exact.
# 0 == incumbent emitted body and JIT symbol.
_TKW = 1
_AFB = 0
# 0926 r1_d0 (memory): FOLD-HEAD INTERCEPT.  Per-CTA s_memrealtime probe
# (graph replay, cold L2) at dec_bs1: head = 1.9 us p50 during which HBM is
# idle for EVERY CTA; the B prologue (bandwidth-bound, ~34 MB at bs1) only
# starts after the whole 257-expert sort, rank (f) and publish (g) finish.
#   _BEF=N : issue the first N B/B-scale prologue stages right after the
#            expert id (fold_e) is resolved in step (e), BEFORE the rank (f)
#            and publish (g) steps (LDS-only + barriers), then issue every
#            prologue hidden(A) load, then the remaining B stages, then
#            quantise.  Same loads, same math -> bit-exact.  n-fusion off only.
#   _HD2=1 : head v2 -- topk_ids loads issued BEFORE the moe_buf zero-init
#            stores (gfx9 vmcnt is in-order and counts stores, so the topk
#            wait no longer includes the store acks); topk ids stay in VGPRs
#            (each thread only ever reads back its own item), histogram zeroed
#            before the topk wait, one head barrier fewer.
# 0 == incumbent emitted body and JIT symbol.
_BEF = 3
_HD2 = 1
#   _KAP=1 : kernarg PIN.  LLVM sinks the s_loads of arg_bq/arg_bscale/ntok and
#            the split-K part/flag pointers into the `_pid < bound` block, each
#            followed by an s_waitcnt lgkmcnt(0) on the path to the first B /
#            hidden load (a cold-L2 scalar round trip per replay).  An empty
#            side-effecting inline asm consuming them at entry keeps every
#            kernarg s_load in the entry batch, in parallel with the topk path.
_KAP = 1

_W_CNT = 0      # 272: LDS histogram, tokens per expert
_W_WT = 272     # 16 : wave totals + [8] = expert-256 block count
_W_MISC = 288   # 16 : [0] = this block's expert, [1] = block index within it
_W_ROWS = 304   # 16 : m_indices for this block's 16 rows
_W_STOK = 320   # 16 : sorted_token_ids for this block
_W_WGT = 336    # 16 : sorted_weights (f32 bits) for this block
_W_TK = 352     # NREP*256: cached topk_ids


def _sort_words(mt):
    nrep = (mt + 255) // 256
    return _W_TK + nrep * 256


def _ashr(x, n):
    return fx.Int32(arith.ShRSIOp(_raw(x), _raw(fx.Int32(n))).result)


def _is_eq(a, b):
    """1 when a == b else 0, branch-free (a, b >= 0)."""
    return fx.Int32(0) - _ashr((a ^ b) - fx.Int32(1), 31)


def _dpp_scan_wave(x):
    """64-lane inclusive add-scan, mirroring the HIP dpp_inclusive_scan_wave."""
    for ctrl in (0x111, 0x112, 0x114, 0x118):
        t = dpp_utils.update_dpp_i32(0, _raw(x), ctrl, 0xF, 0xF, True)
        x = x + fx.Int32(t)
    t = dpp_utils.update_dpp_i32(0, _raw(x), 0x142, 0xA, 0xF, True)
    x = x + fx.Int32(t)
    t = dpp_utils.update_dpp_i32(0, _raw(x), 0x143, 0xC, 0xF, True)
    x = x + fx.Int32(t)
    return x


def _lds_atomic_add(tiles, idx, val):
    atom = fx.make_copy_atom(fx.UniversalAtomicAdd(fx.Int32), fx.Int32)
    r = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)
    r.store(fx.Vector.from_elements([fx.Int32(val)], fx.Int32))
    fx.copy_atom_call(atom, r, fx.slice(tiles, (None, idx)))


def _lds_i32_tiles(base_ptr, word_off, n_words):
    flat = fx.make_view(
        fx.recast_iter(fx.Int32, fx.add_offset(base_ptr, word_off * 4)),
        fx.make_layout(n_words, 1),
    )
    return fx.logical_divide(flat, fx.make_layout(1, 1))


def _global_f32x4_buffer_tiles(addr_i64, num_elems):
    ptr_ty = fx.PointerType.get(
        T.f32, address_space=fx.AddressSpace.Global, alignment=16
    )
    ptr = fx.inttoptr(ptr_ty, fx.Int64(addr_i64))
    view = fx.Tensor(fx.make_view(ptr, fx.make_layout(num_elems, 1)))
    bt = fx.rocdl.make_buffer_tensor(view, max_size=True)
    return fx.logical_divide(bt, fx.make_layout(4, 1))


def _global_i32x4_buffer_tiles(addr_i64, num_elems):
    ptr_ty = fx.PointerType.get(
        T.i32, address_space=fx.AddressSpace.Global, alignment=16
    )
    ptr = fx.inttoptr(ptr_ty, fx.Int64(addr_i64))
    view = fx.Tensor(fx.make_view(ptr, fx.make_layout(num_elems, 1)))
    bt = fx.rocdl.make_buffer_tensor(view, max_size=True)
    return fx.logical_divide(bt, fx.make_layout(4, 1))


def _bf16_pack2(x, y):
    """Pack two Float32 into one Int32 as (lo=bf16(x), hi=bf16(y)), RNE."""
    ux = fx.Int32(_raw(x).bitcast(T.i32))
    uy = fx.Int32(_raw(y).bitcast(T.i32))
    ux = ux + fx.Int32(0x7FFF) + (ux.shrui(fx.Int32(16)) & fx.Int32(1))
    uy = uy + fx.Int32(0x7FFF) + (uy.shrui(fx.Int32(16)) & fx.Int32(1))
    return (ux.shrui(fx.Int32(16)) & fx.Int32(0xFFFF)) | (uy & fx.Int32(-65536))


def _bf16_unpack2(w):
    lo = fx.Float32(_raw(w << fx.Int32(16)).bitcast(T.f32))
    hi = fx.Float32(_raw(w & fx.Int32(-65536)).bitcast(T.f32))
    return lo, hi


def gemm1_grid(n_tokens, BM, *, NE, TOPK, INTER, BN=256, SK=1, NF=1):
    num_n_blocks = num_n_blocks_for(n_out_for(INTER), BN)
    if BM == 128:
        max_m_blocks = (n_tokens * TOPK + NE * (BM - 1) + BM - 1) // BM
    else:
        active = min(n_tokens * TOPK, NE)
        max_m_blocks = (n_tokens * TOPK + active * (BM - 1) + BM - 1) // BM
    tiles = max_m_blocks * (num_n_blocks // NF)
    if SK > 1:
        tiles = (tiles + 7) // 8 * 8
    return tiles * SK


@flyc.jit
def _gemm1_body_fold(
    lds_raw_ptr,
    sort_lds_ptr,
    fold_e,
    arg_aq,
    arg_ascale,
    arg_bq,
    arg_bscale,
    arg_eids,
    arg_mind,
    arg_aqout,
    arg_ascaleout,
    arg_hidden,
    bx_i32,
    lane,
    wave,
    use_nt,
    i32_ntok,
    i32_total_m_blocks,
    arg_part,
    arg_flag,
    sk_i32,
    *,
    SK=1,
    SK_ST_AUX=0,
    SK_LD_AUX=1,
    SK_P16=0,
    SK_ROWS=0,
    SK_RD=0,
    KROT=0,
    BM,
    BN,
    BK,
    inline_quant=False,
    K,
    N_OUT,
    NE,
    interleave=False,
    kStages=2,
    kAStages=None,
    fold=False,
    head_hook=None,
    xb_next=None,
    xb_pre=None,
    xb_out=None,
    xb_dist=0,
    a_res=0,
    a_skip=False,
    krot_tile=None,
):
    KH_TILE = BK // 2
    K_HALF = k_half_for(K)
    K_TILES_TOTAL = k_tiles_total_for(K, BK)
    assert K_TILES_TOTAL % SK == 0
    KTP = K_TILES_TOTAL // SK
    kUnroll = KTP - kStages
    kAS_per_chunk_dw = kas_per_chunk_dw_for(K)
    kBS_stride_n0_dw = kbs_stride_n0_dw_for(K)
    kBS_per_expert_dw = kbs_per_expert_dw_for(N_OUT, K)
    BQ_BYTES = bq_bytes_for(NE, N_OUT, K)
    BSCALE_BYTES = bscale_bytes_for(NE, N_OUT, K)
    NUM_N_BLOCKS = num_n_blocks_for(N_OUT, BN)
    inter = N_OUT // 2
    OUT_AS_PER_CHUNK_DW = kas_per_chunk_dw_for(inter)
    K_G2_HALF = k_half_for(inter)
    _ARES = int(bool(a_res))
    if _ARES:
        kAStages = KTP
    kAStages, kSubBlocks, kMChunks, _lds_bytes = _bm_constants(
        BM, BN, KH_TILE, KTP, kAStages, SK, _ARES
    )
    _ACC_OFF = (
        (kAStages * BM * KH_TILE + kSubBlocks * KTP * 256 + 15) // 16 * 16
        if _ARES else 0
    )
    _SK_SCRATCH_OFF = _lds_bytes - 64

    BN_INT = BN // 2
    # NJ = 16-col B tiles per wave (BN=256 -> 4: 2 gate + 2 up; BN=128 -> 2).
    NJ = BN // 64
    W8 = BN // 8          # gate columns owned by one mfma wave
    NTH = (BN // 2) // 8  # epilogue threads per output row
    # r2_d0: use_nt may carry an explicit weight-tile aux (int > 1).
    b_aux = (
        int(use_nt)
        if (use_nt is not True and use_nt is not False and int(use_nt) > 1)
        else (2 if use_nt else 0)
    )
    M_REPS = BM // 16

    n_block_idx = bx_i32 % fx.Int32(NUM_N_BLOCKS)
    m_block_idx = bx_i32 // fx.Int32(NUM_N_BLOCKS)
    if const_expr(fold):
        e = fold_e
    else:
        e = rocdl.readfirstlane(T.i32, _raw(_global_i32_at(arg_eids, m_block_idx)))
    m_row = m_block_idx * fx.Int32(BM)
    if const_expr(SK > 1):
        sk_b_byte = sk_i32 * fx.Int32(KTP * 2048)
        sk_h_byte = sk_i32 * fx.Int32(KTP * BK * 2)
        sk_bs_byte = sk_i32 * fx.Int32(KTP * kBS_stride_k0_dw * 4)

    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)
    lane_div_8 = lane // fx.Int32(8)
    lane_mod_8 = lane % fx.Int32(8)

    # ---- WAVE-74 r1_d0: PER-CTA k-START ROTATION (KROT) --------------------
    # The w1 expert stride is N_OUT*K_HALF = 512*3072 = 1.5 MiB = 3*2**19, so
    # every expert's weight tile base carries the IDENTICAL low-order address
    # phase.  Every CTA also walks k from tile 0 in lockstep, so at any instant
    # the whole co-resident request set sits inside one ~2 KiB window of the
    # 48 KiB (= KTP * 2048 B) column tile.  KROT rotates the k-walk START by
    # rot = (tile_id * KROT) mod KTP, so co-resident CTAs spread over the
    # column tile instead of stacking on one phase.
    #
    # ARITHMETIC (this is what picks KROT, not a blind sweep): the walk visits
    # rot in the cyclic subgroup <KROT> of Z_KTP, whose size is KTP/gcd(KROT,
    # KTP).  KTP = 24 here, so KROT=3 (the w73 first-try value) reaches only
    # 24/3 = 8 of the 24 phases; ONLY KROT coprime to 24 -- {1,5,7,11,13,...}
    # -- reaches all 24.  KROT=1 is rejected on a second ground: adjacent tile
    # ids (the two n-blocks of one m-block) then differ by a single 2 KiB step
    # and stay inside the same window.  {5,7,11,13} are the candidates.
    #
    # COST: every k offset in this body is LINEAR in the k-tile index, so the
    # rotation folds entirely into the wave-uniform SGPR base of each stream
    # (B, B-scale, hidden) -- zero extra VGPR work in the inner loop, one
    # branchless conditional subtract per unrolled step, hoisted to the top.
    # Every k tile is still visited exactly once, so the byte traffic and the
    # arithmetic are unchanged; only the fp32 accumulation ORDER rotates.
    # Disabled under split-K (the SK offsets already re-base the k window).
    _KROT = int(KROT) if (KROT and SK == 1) else 0

    def _rot_list(tile_i32, n):
        _r0 = rocdl.readfirstlane(
            T.i32, _raw(_umod(tile_i32 * fx.Int32(_KROT), fx.Int32(KTP)))
        )
        _out = []
        for _kt in range_constexpr(n):
            _g = _r0 + fx.Int32(_kt)
            _g = _g - (fx.Int32(KTP) & _ashr(fx.Int32(KTP) - _g - fx.Int32(1), 31))
            _out.append(rocdl.readfirstlane(T.i32, _raw(_g)))
        return _out

    _krt = bx_i32 if krot_tile is None else krot_tile
    if const_expr(_KROT):
        _rot_g = _rot_list(_krt, KTP)

    def _k_b_byte(kt, rg=None):
        # byte offset of k-tile `kt` inside this wave's 16-column B tile
        if const_expr(_KROT):
            return (_rot_g if rg is None else rg)[kt] * fx.Int32(2048)
        return fx.Int32(kt * 2048)

    def _k_bs_byte(kt, rg=None):
        if const_expr(_KROT):
            return (_rot_g if rg is None else rg)[kt] * fx.Int32(kBS_stride_k0_dw * 4)
        return fx.Int32(kt * kBS_stride_k0_dw * 4)

    def _k_h_byte(kt):
        if const_expr(_KROT):
            return _rot_g[kt] * fx.Int32(BK * 2)
        return fx.Int32(kt * (BK * 2))

    aq_num_records = fx.Int64(i32_ntok * fx.Int32(K_HALF))
    _asc_per_mb = max(BM // 32, 1) * kAS_per_chunk_dw * 4
    ascale_num = fx.Int64(i32_total_m_blocks) * fx.Int64(_asc_per_mb)

    # fx.copy's BufferCopy/BufferCopyLDS atoms take soffset as an element count,
    # not the bytes buffer_ops.buffer_load's soffset_bytes expects.
    def _global_i32_buffer_view(addr_i64, num_bytes):
        # make_layout's dynamic-shape leaf must be i32/i64, not fx.Index.
        num_bytes_i64 = fx.Int64(num_bytes)
        ptr_ty = fx.PointerType.get(
            T.i32, address_space=fx.AddressSpace.Global, alignment=4
        )
        ptr = fx.inttoptr(ptr_ty, fx.Int64(addr_i64))
        view = fx.Tensor(
            fx.make_view(ptr, fx.make_layout(num_bytes_i64 // fx.Int64(4), 1))
        )
        return fx.rocdl.make_buffer_tensor(
            view, max_size=False, num_records_bytes=num_bytes_i64
        )

    def _global_i32_buffer_tiles(addr_i64, num_bytes, tile_elems):
        return fx.logical_divide(
            _global_i32_buffer_view(addr_i64, num_bytes), fx.make_layout(tile_elems, 1)
        )

    bq_tiles = _global_i32_buffer_tiles(arg_bq, BQ_BYTES, 4)
    bq_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(b_aux), fx.Int32)
    bq_reg_lay = fx.make_layout(4, 1)

    bscale_tiles = _global_i32_buffer_tiles(arg_bscale, BSCALE_BYTES, 1)
    bscale_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(2), fx.Int32)
    bscale_reg_lay = fx.make_layout(1, 1)

    # aq/ascale: global->LDS async DMA (no register fragment), via BufferCopyLDS.
    aq_buf = _global_i32_buffer_view(arg_aq, aq_num_records)
    aq_dma_tiles4 = fx.logical_divide(aq_buf, fx.make_layout(4, 1))
    aq_dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), fx.Int32)

    ascale_buf = _global_i32_buffer_view(arg_ascale, ascale_num)
    ascale_dma_tiles4 = fx.logical_divide(ascale_buf, fx.make_layout(4, 1))
    ascale_dma_tiles1 = fx.logical_divide(ascale_buf, fx.make_layout(1, 1))
    ascale_dma_atom16 = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), fx.Int32)
    ascale_dma_atom4 = fx.make_copy_atom(fx.rocdl.BufferCopyLDS32b(), fx.Int32)

    if const_expr(inline_quant):
        hidden_num = fx.Int64(i32_ntok * fx.Int32(K * 2))
        hidden_tiles = _global_i32_buffer_tiles(arg_hidden, hidden_num, 4)
        hidden_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(0), fx.Int32)
        hidden_reg_lay = fx.make_layout(4, 1)

    # Union LDS region [s_aq | s_asc], reused as lds_acc (f32 accumulator) in
    # the epilogue. s_aq/lds_acc start at lds_raw_ptr; s_asc follows at
    # +kAStages*BM*KH_TILE.

    cached_actual_row = []
    cached_row_inline = None
    if const_expr(inline_quant):
        rcls = wave * fx.Int32(4) + lane_div_16
        if const_expr(fold and head_hook is not None and _BEF and not _AFB):
            cached_row_inline = None    # loaded after head_hook (see _BEF)
        elif const_expr(fold):
            # The 16 gather rows for THIS m-block were computed in the block
            # head and live in the private sort-scratch LDS region.
            _rw_tiles = _lds_i32_tiles(sort_lds_ptr, _W_ROWS, 16)
            cached_row_inline = _global_i32_load(_rw_tiles, rcls)
        else:
            cached_row_inline = _global_i32_at(arg_mind, m_row + rcls)
    else:
        for sub in range_constexpr(kSubBlocks):
            idx = m_row + wave * fx.Int32(BM // 4) + fx.Int32(sub * 8) + lane_div_8
            cached_actual_row.append(_global_i32_at(arg_mind, idx))

    # -- b_load_s_base[j] (HIP 412-416), readfirstlane'd uniform per wave ------
    N0_HALF = N_OUT // 32

    def _b_bases_for(nbi):
        # n-block-only re-derivation of b_load_s_base / b_scale_s_base(_hi)
        # for the NEXT fused n-block (XB).  Gate/up-separated, BN=256 only.
        _sb = []
        for j in range_constexpr(NJ):
            g = fx.Int32(j & 1)
            n0 = nbi * fx.Int32(BN // 32) + wave * fx.Int32(BN // 128) + fx.Int32(j >> 1)
            col = (g * fx.Int32(N0_HALF) + n0) * fx.Int32(16)
            _sb.append(rocdl.readfirstlane(T.i32, (e * fx.Int32(N_OUT) + col) * fx.Int32(K_HALF)))
        _nb0 = nbi * fx.Int32(BN // 32) + wave * fx.Int32(BN // 128)
        _npg = _nb0 // fx.Int32(2)
        _npl = [_npg, _npg + fx.Int32(N_OUT // 64)]
        _bs, _bsh = [], []
        for mw in range_constexpr(2):
            base = (
                e * fx.Int32(kBS_per_expert_dw) + _npl[mw] * fx.Int32(kBS_stride_n0_dw)
            ) * fx.Int32(4)
            base = rocdl.readfirstlane(T.i32, base)
            _bs.append(base)
            _bsh.append(base + fx.Int32(16 * kBS_stride_k0_dw * 4))
        return _sb, _bs, _bsh

    b_load_s_base = []
    for j in range_constexpr(NJ):
        if const_expr(interleave):
            col = (
                n_block_idx * fx.Int32(BN) + wave * fx.Int32(BN // 4) + fx.Int32(j * 16)
            )
        else:
            # n0 = index of this wave's 16-column gate tile; g picks gate/up.
            # BN=256: 2 n0 per wave (j>>1); BN=128: 1 n0 per wave.
            g = fx.Int32(j & 1)
            n0 = (
                n_block_idx * fx.Int32(BN // 32)
                + wave * fx.Int32(BN // 128)
                + fx.Int32(j >> 1)
            )
            col = (g * fx.Int32(N0_HALF) + n0) * fx.Int32(16)
        v = (e * fx.Int32(N_OUT) + col) * fx.Int32(K_HALF)
        b_load_s_base.append(rocdl.readfirstlane(T.i32, v))

    # -- b_scale_s_base / _hi (HIP 418-429) -----------------------------------
    if const_expr(interleave):
        mni_base = n_block_idx * fx.Int32(BN // 32) + wave * fx.Int32(BN // 128)
        np_list = [mni_base, mni_base + fx.Int32(1)]
    else:
        # A b_scale dword covers TWO adjacent 16-col n0 tiles (opsel `in_b`
        # picks the byte).  At BN=256 a wave owns both, so in_b is the compile
        # -time j>>1.  At BN=128 a wave owns exactly one, and which of the two
        # it is depends on `wave` -- a runtime value, while the MFMA opsel must
        # be constexpr.  Fix: keep opsel at in_b=0 and byte-rotate the scale
        # dword right by 8 when this wave owns the odd tile (bytes are
        # [n_sub0_k0, n_sub1_k0, n_sub0_k1, n_sub1_k1], so >>8 moves the odd
        # sub-tile's bytes onto opsel 0 and 2).
        _n0_base = n_block_idx * fx.Int32(BN // 32) + wave * fx.Int32(BN // 128)
        np_gate = _n0_base // fx.Int32(2)
        np_list = [np_gate, np_gate + fx.Int32(N_OUT // 64)]
        bs_shift = (_n0_base & fx.Int32(1)) * fx.Int32(8)
    b_scale_s_base, b_scale_s_base_hi = [], []
    for mw in range_constexpr(2):
        base = (
            e * fx.Int32(kBS_per_expert_dw) + np_list[mw] * fx.Int32(kBS_stride_n0_dw)
        ) * fx.Int32(4)
        if const_expr(SK > 1):
            base = base + sk_bs_byte
        base = rocdl.readfirstlane(T.i32, base)
        b_scale_s_base.append(base)
        b_scale_s_base_hi.append(base + fx.Int32(16 * kBS_stride_k0_dw * 4))

    accm = [[None] * NJ for _ in range(kMChunks)]
    b = [[[None, None] for _ in range(NJ)] for _ in range(kStages)]
    b_scale_v = [[None, None] for _ in range(kStages)]

    def issue_a_load_lds(slot, kt):
        for sub in range_constexpr(kSubBlocks):
            lds_row = wave * fx.Int32(BM // 4) + fx.Int32(sub * 8)
            mask = _lds_swizzle_mask(lds_row + lane_div_8)
            voffset = ((lane_mod_8 * fx.Int32(16)) ^ mask) + cached_actual_row[
                sub
            ] * fx.Int32(K_HALF)
            off = fx.Int32(slot * (BM * KH_TILE)) + lds_row * fx.Int32(KH_TILE)
            fx.copy(
                aq_dma_atom,
                fx.slice(aq_dma_tiles4, (None, voffset // fx.Int32(16))),
                fx.slice(s_aq_i32x4_tiles, (None, off // fx.Int32(16))),
                soffset=fx.Int32(kt * KH_TILE) // fx.Int32(4),
            )

    # s_aq as flat i32, divided into 4-element (128-bit) and 1-element tiles.
    s_aq_i32_flat = fx.make_view(
        fx.recast_iter(fx.Int32, lds_raw_ptr),
        fx.make_layout(kAStages * BM * KH_TILE // 4, 1),
    )
    s_aq_i32x4_tiles = fx.logical_divide(s_aq_i32_flat, fx.make_layout(4, 1))
    s_aq_i32x1_tiles = fx.logical_divide(s_aq_i32_flat, fx.make_layout(1, 1))
    i32x4_copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
    i32x4_reg_lay = fx.make_layout(4, 1)

    def _lds_i32x4_load(tile_idx):
        r = fx.make_rmem_tensor(i32x4_reg_lay, fx.Int32)
        fx.copy_atom_call(
            i32x4_copy_atom, fx.slice(s_aq_i32x4_tiles, (None, tile_idx)), r
        )
        return r.load()

    def issue_a_ds_read(slot):
        mask = _lds_swizzle_mask(lane_mod_16)
        a = [[None, None] for _ in range(kMChunks)]
        for k in range_constexpr(2):
            lds_col = (lane_div_16 * fx.Int32(16) + fx.Int32(k * 64)) ^ mask
            for i in range_constexpr(kMChunks):
                lds_row = lane_mod_16 + fx.Int32(i * 16)
                off = (
                    fx.Int32(slot * (BM * KH_TILE))
                    + lds_row * fx.Int32(KH_TILE)
                    + lds_col
                )
                a[i][k] = _lds_i32x4_load(off // fx.Int32(16))
        return a

    def issue_a_scale_load():
        chunk_base = m_row // fx.Int32(32)
        v16 = (wave * fx.Int32(64) + lane) * fx.Int32(16)
        v4 = (wave * fx.Int32(64) + lane) * fx.Int32(4)
        for sub in range_constexpr(kSubBlocks):
            s_chunk = rocdl.readfirstlane(
                T.i32, (chunk_base + fx.Int32(sub)) * fx.Int32(kAS_per_chunk_dw * 4)
            )
            lds_sub = fx.Int32(sub * kAS_per_chunk_dw * 4)
            fx.copy(
                ascale_dma_atom16,
                fx.slice(ascale_dma_tiles4, (None, v16 // fx.Int32(16))),
                fx.slice(
                    asc_i32x4_tiles,
                    (None, (lds_sub + wave * fx.Int32(1024)) // fx.Int32(16)),
                ),
                soffset=s_chunk // fx.Int32(4),
            )
            for d in range_constexpr(3):
                byte_off = 4096 + d * 1024
                s_off = rocdl.readfirstlane(T.i32, s_chunk + fx.Int32(byte_off))
                fx.copy(
                    ascale_dma_atom4,
                    fx.slice(ascale_dma_tiles1, (None, v4 // fx.Int32(4))),
                    fx.slice(
                        asc_i32_tiles,
                        (
                            None,
                            (lds_sub + fx.Int32(byte_off) + wave * fx.Int32(256))
                            // fx.Int32(4),
                        ),
                    ),
                    soffset=s_off // fx.Int32(4),
                )

    # s_asc as flat i32.
    s_asc_i32_flat = fx.make_view(
        fx.recast_iter(fx.Int32, fx.add_offset(lds_raw_ptr, kAStages * BM * KH_TILE)),
        fx.make_layout(kSubBlocks * KTP * 64, 1),
    )
    asc_i32_tiles = fx.logical_divide(s_asc_i32_flat, fx.make_layout(1, 1))
    asc_i32x4_tiles = fx.logical_divide(s_asc_i32_flat, fx.make_layout(4, 1))

    def issue_a_scale_ds_read(kt):
        out = []
        for sub in range_constexpr(kSubBlocks):
            lds_dw = (
                fx.Int32(sub * kAS_per_chunk_dw)
                + fx.Int32(kt * 64)
                + lane_div_16 * fx.Int32(16)
                + lane_mod_16
            )
            out.append(_raw(_global_i32_load(asc_i32_tiles, lds_dw)))
        return out

    lib = lane & fx.Int32(3)
    lane_shr2_and3 = (lane >> fx.Int32(2)) & fx.Int32(3)
    r_in_chunk = wave * fx.Int32(4) + lane_div_16

    def inline_quant_load_kt(B128_IDX, kt, row_token):
        v_voff = (
            row_token * fx.Int32(K * 2)
            + lane_shr2_and3 * fx.Int32(64)
            + lib * fx.Int32(16)
        )
        if const_expr(_KROT):
            _so = _k_h_byte(kt) + fx.Int32(B128_IDX * 256)
        else:
            _so = fx.Int32(kt * (BK * 2) + B128_IDX * 256)
        if const_expr(SK > 1):
            _so = _so + sk_h_byte
        s_soff = rocdl.readfirstlane(T.i32, _so)
        r = fx.make_rmem_tensor(hidden_reg_lay, fx.Int32)
        fx.copy(
            hidden_copy_atom,
            fx.slice(hidden_tiles, (None, v_voff // fx.Int32(16))),
            r,
            soffset=s_soff // fx.Int32(4),
        )
        return r.load()

    def _inline_quant_core_batch(specs, slot, scale_accum):
        n = len(specs)
        h_dw = [
            [fx.Int32(_raw(h_v[j])) for j in range_constexpr(4)]
            for (_b, _s, h_v) in specs
        ]
        la = [None] * n
        for i in range_constexpr(n):
            hm = [h_dw[i][j] & fx.Int32(0x7FFF7FFF) for j in range_constexpr(4)]
            m01 = _pkmax_u16(hm[0], hm[1])
            m23 = _pkmax_u16(hm[2], hm[3])
            m0123 = _pkmax_u16(m01, m23)
            lo = m0123 & fx.Int32(0xFFFF)
            hi = m0123.shrui(fx.Int32(16)) & fx.Int32(0xFFFF)
            la[i] = _umax_i32(lo, hi)
        a = [fx.Int32(_raw(la[i])) for i in range_constexpr(n)]
        s1 = [
            fx.Int32(
                dpp_utils.update_dpp_i32(_raw(a[i]), _raw(a[i]), 0xB1, 0xF, 0xF, True)
            )
            for i in range_constexpr(n)
        ]
        a = [_umax_i32(a[i], s1[i]) for i in range_constexpr(n)]
        s2 = [
            fx.Int32(
                dpp_utils.update_dpp_i32(_raw(a[i]), _raw(a[i]), 0x4E, 0xF, 0xF, True)
            )
            for i in range_constexpr(n)
        ]
        a = [_umax_i32(a[i], s2[i]) for i in range_constexpr(n)]
        e8 = [_inline_e8m0(a[i]) for i in range_constexpr(n)]
        for i in range_constexpr(n):
            B128_IDX, SUB, _hv = specs[i]
            qs_raw = _raw(fx.Float32(_raw(e8[i] << fx.Int32(23)).bitcast(T.f32)))
            pk = _raw(fx.Int32(0))
            for j in range_constexpr(4):
                src_bf16x2 = _raw(
                    fx.Vector.from_elements([h_dw[i][j]], fx.Int32).bitcast(fx.BFloat16)
                )
                pk = rocdl.cvt_scalef32_pk_fp4_bf16(T.i32, pk, src_bf16x2, qs_raw, j)
            pk = fx.Int32(pk)
            r = fx.Int32(SUB * 16) + r_in_chunk
            kb_in_kt = fx.Int32(B128_IDX * 4) + lane_shr2_and3
            mask_r = _lds_swizzle_mask(r)
            b_off = lib * fx.Int32(4)
            off = (
                fx.Int32(slot * (BM * KH_TILE))
                + r * fx.Int32(KH_TILE)
                + ((kb_in_kt * fx.Int32(16)) ^ mask_r)
                + b_off
            )
            _scalar_store(s_aq_i32x1_tiles, off // fx.Int32(4), pk, fx.Int32)
            pack_byte = B128_IDX * 2 + SUB
            scale_accum = scale_accum | (e8[i] << fx.Int32(pack_byte * 8))
        return scale_accum

    def inline_quant_kt(B128_IDX, SUB, slot, kt, row_token, scale_accum):
        h_v = inline_quant_load_kt(B128_IDX, kt, row_token)
        return _inline_quant_core_batch([(B128_IDX, SUB, h_v)], slot, scale_accum)

    def inline_quant_pack_write(kt, scale_accum):
        lane_tgt = lane_shr2_and3 * fx.Int32(16) + r_in_chunk
        off = fx.Int32(kt * 256) + lane_tgt * fx.Int32(4)
        _scalar_store(asc_i32_tiles, off // fx.Int32(4), scale_accum, fx.Int32)

    # A-resident XB, second n-block: every A tile / A-scale for this m-block
    # is already in LDS (written by n-block 0 in the same k order), so all
    # hidden loads, quantisation and LDS A writes are skipped.
    _askip = bool(a_skip)
    assert not _askip or _ARES

    def _nl_load(B128_IDX, kt, row_token):
        return None

    def _nl_core(specs, slot, scale_accum):
        return scale_accum

    def _nl_pack(kt, scale_accum):
        return None

    inline_quant_load_kt = _nl_load if _askip else inline_quant_load_kt
    _inline_quant_core_batch = _nl_core if _askip else _inline_quant_core_batch
    inline_quant_pack_write = _nl_pack if _askip else inline_quant_pack_write

    def issue_b_load_j(b_slot, K_C, j, alt=None):
        _sbl = b_load_s_base if alt is None else alt[0]
        _rg = None if alt is None else alt[3]
        v = (lane_div_16 * fx.Int32(256)) + (lane_mod_16 * fx.Int32(16))
        if const_expr(_KROT):
            s_b = rocdl.readfirstlane(
                T.i32, _raw(_sbl[j] + _k_b_byte(K_C, _rg))
            )
        else:
            v = v + fx.Int32(K_C * 2048)
            s_b = _sbl[j]
        if const_expr(SK > 1):
            v = v + sk_b_byte
        for half in range_constexpr(2):
            tile_idx = (v + fx.Int32(half * 1024)) // fx.Int32(16)
            r = fx.make_rmem_tensor(bq_reg_lay, fx.Int32)
            fx.copy(
                bq_copy_atom,
                fx.slice(bq_tiles, (None, tile_idx)),
                r,
                soffset=s_b // fx.Int32(4),
            )
            b_slot[j][half] = r.load()

    def issue_b_scale_load(bs_slot, K_C, alt=None):
        _bsl = b_scale_s_base if alt is None else alt[1]
        _bshl = b_scale_s_base_hi if alt is None else alt[2]
        _rg = None if alt is None else alt[3]
        v = ((lane_div_16 * fx.Int32(16)) + lane_mod_16) * fx.Int32(4)
        K_C_HI = K_C // 16
        imm = 0 if _KROT else (K_C - K_C_HI * 16) * (kBS_stride_k0_dw * 4)
        for mw in range_constexpr(2):
            if const_expr(_KROT):
                s_off = rocdl.readfirstlane(
                    T.i32, _raw(_bsl[mw] + _k_bs_byte(K_C, _rg))
                )
            else:
                s_off = _bsl[mw] if K_C_HI == 0 else _bshl[mw]
            idx = (v + fx.Int32(imm)) // fx.Int32(4)
            r = fx.make_rmem_tensor(bscale_reg_lay, fx.Int32)
            fx.copy(
                bscale_copy_atom,
                fx.slice(bscale_tiles, (None, idx)),
                r,
                soffset=s_off // fx.Int32(4),
            )
            _v = r.load()[0]
            if const_expr(NJ < 4):
                _v = fx.Int32(fx.Uint32(_v) >> fx.Uint32(bs_shift))
            bs_slot[mw] = _v

    mfma_ty = T.f32x4
    zero4 = fx.Vector.filled(4, 0.0, fx.Float32)

    def mfma_cluster(b_slot, a, a_scale, bs_slot, J, init):
        if const_expr(interleave):
            mni = J // 2
            in_b = J % 2
        else:
            mni = J % 2
            in_b = J // 2
        sb = bs_slot[mni]
        bJ0, bJ1 = b_slot[J][0], b_slot[J][1]
        if const_expr(kMChunks == 1):
            sa = a_scale[0]
            if const_expr(init):
                accm[0][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                    mfma_ty, [a[0][0], bJ0, zero4, 4, 4, 0, sa, 0 + in_b, sb]
                )
            else:
                accm[0][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                    mfma_ty, [a[0][0], bJ0, accm[0][J], 4, 4, 0, sa, 0 + in_b, sb]
                )
            accm[0][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                mfma_ty, [a[0][1], bJ1, accm[0][J], 4, 4, 2, sa, 2 + in_b, sb]
            )
        else:
            for sub in range_constexpr(kSubBlocks):
                i0 = sub * 2 + 0
                i1 = sub * 2 + 1
                sa = a_scale[sub]
                if const_expr(init):
                    accm[i0][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        mfma_ty, [a[i0][0], bJ0, zero4, 4, 4, 0, sa, 0 + in_b, sb]
                    )
                    accm[i1][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        mfma_ty, [a[i1][0], bJ0, zero4, 4, 4, 1, sa, 0 + in_b, sb]
                    )
                else:
                    accm[i0][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        mfma_ty, [a[i0][0], bJ0, accm[i0][J], 4, 4, 0, sa, 0 + in_b, sb]
                    )
                    accm[i1][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        mfma_ty, [a[i1][0], bJ0, accm[i1][J], 4, 4, 1, sa, 0 + in_b, sb]
                    )
                accm[i0][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                    mfma_ty, [a[i0][1], bJ1, accm[i0][J], 4, 4, 2, sa, 2 + in_b, sb]
                )
                accm[i1][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                    mfma_ty, [a[i1][1], bJ1, accm[i1][J], 4, 4, 3, sa, 2 + in_b, sb]
                )

    _relax_prologue = (BM == 128) and not inline_quant
    # --- deepened hidden(A) prefetch depth (queue itself is built below) ---
    _hpf = bool(_HPF) and bool(inline_quant) and (kUnroll > 1)
    _HPD = min(int(_HPF), kUnroll - 1) if _hpf else 0
    _afb = bool(_AFB) and bool(inline_quant)
    _bef = (
        min(int(_BEF), kStages)
        if (head_hook is not None and inline_quant and fold and not _afb)
        else 0
    )
    if const_expr(not inline_quant):
        issue_a_scale_load()
    if const_expr(_bef):
        # B-EARLY-FETCH (see _BEF): B only needs fold_e, which the head
        # resolved in step (e); start the bandwidth-bound B stream now and run
        # the LDS-only rank/publish steps under its latency.
        for _kc in range_constexpr(_bef):
            for _j in range_constexpr(NJ):
                issue_b_load_j(b[_kc], _kc, _j)
            issue_b_scale_load(b_scale_v[_kc], _kc)
        rocdl.sched_barrier(0)
        _hk_mb = head_hook()
        if const_expr(_hk_mb is not None):
            # ITEM-DIRECT: the sorted m-block is only known after the hook.
            m_block_idx = _hk_mb
            m_row = m_block_idx * fx.Int32(BM)
        rocdl.sched_barrier(0)
        cached_row_inline = _global_i32_load(
            _lds_i32_tiles(sort_lds_ptr, _W_ROWS, 16), rcls
        )
        h_pro = [
            [
                inline_quant_load_kt(0, _kc, cached_row_inline),
                inline_quant_load_kt(1, _kc, cached_row_inline),
            ]
            for _kc in range(kStages)
        ]
        for _kc in range_constexpr(_bef, kStages):
            for _j in range_constexpr(NJ):
                issue_b_load_j(b[_kc], _kc, _j)
            issue_b_scale_load(b_scale_v[_kc], _kc)
        if const_expr(_hpf):
            h_pend = [
                [
                    inline_quant_load_kt(0, kStages + _d, cached_row_inline),
                    inline_quant_load_kt(1, kStages + _d, cached_row_inline),
                ]
                for _d in range(_HPD)
            ]
        rocdl.sched_barrier(0)
        for _kc in range_constexpr(kStages):
            _sacc = _inline_quant_core_batch(
                [(0, 0, h_pro[_kc][0]), (1, 0, h_pro[_kc][1])], _kc, fx.Int32(0)
            )
            inline_quant_pack_write(_kc, _sacc)
    if const_expr(_afb):
        # A-FIRST prologue batch (see _AFB).  Phase 1: every prologue hidden
        # load + the hpf queue, back to back.  Phase 2: all B / B-scale stages.
        # Phase 3: quantise stage by stage; each wait only needs the A loads
        # issued before the B burst, so B flies under the quant VALU.
        h_pro = [
            [
                inline_quant_load_kt(0, _kc, cached_row_inline),
                inline_quant_load_kt(1, _kc, cached_row_inline),
            ]
            for _kc in range(kStages)
        ]
        if const_expr(_hpf):
            h_pend = [
                [
                    inline_quant_load_kt(0, kStages + _d, cached_row_inline),
                    inline_quant_load_kt(1, kStages + _d, cached_row_inline),
                ]
                for _d in range(_HPD)
            ]
        rocdl.sched_barrier(0)
        for _kc in range_constexpr(kStages):
            for _j in range_constexpr(NJ):
                issue_b_load_j(b[_kc], _kc, _j)
            issue_b_scale_load(b_scale_v[_kc], _kc)
            if const_expr(_AFB >= 2):
                # pin stage order: LLVM otherwise sinks stage 0 (first
                # consumed) to the END of the burst, so the first MFMA's wait
                # drains every prologue B stage.
                rocdl.sched_barrier(0)
        rocdl.sched_barrier(0)
        for _kc in range_constexpr(kStages):
            _sacc = _inline_quant_core_batch(
                [(0, 0, h_pro[_kc][0]), (1, 0, h_pro[_kc][1])], _kc, fx.Int32(0)
            )
            inline_quant_pack_write(_kc, _sacc)
    _xbp = xb_pre is not None
    if const_expr(_xbp):
        # XB: this n-block's first kStages B / B-scale stages were issued by
        # the PREVIOUS fused n-block's drain (same expert, same m-block).
        assert inline_quant and not _afb and not _bef and not _relax_prologue
        for K_C in range_constexpr(kStages):
            b[K_C] = xb_pre[0][K_C]
            b_scale_v[K_C] = xb_pre[1][K_C]
    for K_C in range_constexpr(0 if (_afb or _bef) else kStages):
        if const_expr(inline_quant):
            scale_accum = fx.Int32(0)
            scale_accum = inline_quant_kt(
                0, 0, K_C, K_C, cached_row_inline, scale_accum
            )
            for _j in range_constexpr(0 if _xbp else NJ // 2):
                issue_b_load_j(b[K_C], K_C, _j)
            scale_accum = inline_quant_kt(
                1, 0, K_C, K_C, cached_row_inline, scale_accum
            )
            for _j in range_constexpr(NJ // 2, NJ // 2 if _xbp else NJ):
                issue_b_load_j(b[K_C], K_C, _j)
            inline_quant_pack_write(K_C, scale_accum)
        else:
            issue_a_load_lds(K_C, K_C)
            if const_expr(not _relax_prologue):
                for j in range_constexpr(NJ):
                    issue_b_load_j(b[K_C], K_C, j)
        if const_expr(not _relax_prologue and not _xbp):
            issue_b_scale_load(b_scale_v[K_C], K_C)
    if const_expr(_relax_prologue):
        rocdl.sched_barrier(0)
        for K_C in range_constexpr(kStages):
            for j in range_constexpr(NJ):
                issue_b_load_j(b[K_C], K_C, j)
            issue_b_scale_load(b_scale_v[K_C], K_C)

    # --- deepened hidden(A) prefetch: issue K_C's load _HPD bodies earlier ---
    # WAVE-33 r2_d0: generalised wave-31's fixed distance-1 queue into a queue
    # of _HPD k-tile pairs, so GEAK_G1_HPF=N issues the hidden(A) global load
    # N loop bodies ahead of the body that quantises it.  N=1 reproduces the
    # wave-31 artifact BYTE-FOR-BYTE (same emitted order, same symbol suffix);
    # N=0 reverts to the incumbent.  Cost is 8 VGPRs per extra step.
    if const_expr(_hpf and not _afb and not _bef):
        h_pend = [
            [
                inline_quant_load_kt(0, kStages + _d, cached_row_inline),
                inline_quant_load_kt(1, kStages + _d, cached_row_inline),
            ]
            for _d in range(_HPD)
        ]

    _xbn = xb_next is not None
    if const_expr(_xbn):
        assert BN == 256 and not interleave and SK == 1
        _nsb, _nbs, _nbsh = _b_bases_for(xb_next % fx.Int32(NUM_N_BLOCKS))
        _alt = (
            _nsb, _nbs, _nbsh,
            (_rot_list(xb_next if krot_tile is None else krot_tile, kStages) if _KROT else None),
        )
        _xb_b = [[[None, None] for _ in range(NJ)] for _ in range(kStages)]
        _xb_bs = [[None, None] for _ in range(kStages)]
    # XB distance: next n-block stage s is issued at virtual step
    # kUnroll - _XBD + s (steady loop if < kUnroll, else drain step).
    _XBD = max(int(xb_dist), 0)

    for OFFSET in range_constexpr(kUnroll):
        K_C = kStages + OFFSET
        read_slot = OFFSET % kAStages
        write_slot = K_C % kAStages
        slot_b = OFFSET % kStages
        gpu.barrier()
        if const_expr(BM == 128):
            asc_cur = issue_a_scale_ds_read(K_C - kStages)
            a_cur = issue_a_ds_read(read_slot)
        else:
            a_cur = issue_a_ds_read(read_slot)
            asc_cur = issue_a_scale_ds_read(K_C - kStages)
        if const_expr(not inline_quant):
            issue_a_load_lds(write_slot, K_C)
        if const_expr(inline_quant):
            if const_expr(_hpf):
                h_v0, h_v1 = h_pend[0][0], h_pend[0][1]
                h_pend = h_pend[1:]
                if const_expr(OFFSET + _HPD < kUnroll):
                    h_pend = h_pend + [
                        [
                            inline_quant_load_kt(0, K_C + _HPD, cached_row_inline),
                            inline_quant_load_kt(1, K_C + _HPD, cached_row_inline),
                        ]
                    ]
            else:
                h_v0 = inline_quant_load_kt(0, K_C, cached_row_inline)
                h_v1 = inline_quant_load_kt(1, K_C, cached_row_inline)
            if const_expr(not (_hpf and _HPB == 0)):
                rocdl.sched_barrier(0)
        for J in range_constexpr(NJ):
            if const_expr(BM != 128):
                rocdl.sched_barrier(0)
                rocdl.s_setprio(1)
            mfma_cluster(
                b[slot_b], a_cur, asc_cur, b_scale_v[slot_b], J, init=(OFFSET == 0)
            )
            if const_expr(BM != 128):
                rocdl.s_setprio(0)
            rocdl.sched_barrier(0)
            issue_b_load_j(b[slot_b], K_C, J)
            rocdl.sched_barrier(0)
        issue_b_scale_load(b_scale_v[slot_b], K_C)
        _xs = OFFSET - (kUnroll - _XBD) if _xbn else -1
        if const_expr(0 <= _xs < kStages):
            rocdl.sched_barrier(0)
            for _j in range_constexpr(NJ):
                issue_b_load_j(_xb_b[_xs], _xs, _j, _alt)
            issue_b_scale_load(_xb_bs[_xs], _xs, _alt)
            rocdl.sched_barrier(0)
        if const_expr(inline_quant):
            scale_accum = _inline_quant_core_batch(
                [(0, 0, h_v0), (1, 0, h_v1)], write_slot, fx.Int32(0)
            )
            inline_quant_pack_write(K_C, scale_accum)

    for S in range_constexpr(kStages):
        kt = KTP - kStages + S
        gpu.barrier()
        if const_expr(BM == 128):
            asc_cur = issue_a_scale_ds_read(kt)
            a_cur = issue_a_ds_read(kt % kAStages)
        else:
            a_cur = issue_a_ds_read(kt % kAStages)
            asc_cur = issue_a_scale_ds_read(kt)
        _xs = S + _XBD if _xbn else kStages
        for J in range_constexpr(NJ):
            if const_expr(_xs < kStages):
                rocdl.sched_barrier(0)
            mfma_cluster(
                b[kt % kStages], a_cur, asc_cur, b_scale_v[kt % kStages], J, init=False
            )
            if const_expr(_xs < kStages):
                rocdl.sched_barrier(0)
                issue_b_load_j(_xb_b[_xs], _xs, J, _alt)
                rocdl.sched_barrier(0)
        if const_expr(_xs < kStages):
            issue_b_scale_load(_xb_bs[_xs], _xs, _alt)
    if const_expr(_xbn):
        xb_out["b"] = (_xb_b, _xb_bs)

    # 0926 r1_d1 (compute): REGISTER-DIRECT SPLIT-K EXCHANGE (SK_RD).
    # s_memrealtime timeline at dec_bs1 (SK=2): after the weight stream ends
    # the critical path is acc->LDS (0.20 us) -> barrier -> LDS->regs ->
    # partial store + release (0.48 us incl. the flag atomic) -> peer load
    # (0.64 us) -> LDS -> epilogue.  Here each K-partition stores its fp32
    # partial straight from the MFMA accumulators (register layout, private
    # scratch so any consistent slot permutation is legal) BEFORE the drain
    # barrier; the LDS accumulator is staged only by the LAST arriver, after
    # it has added the peer partial in registers.  Row-limited like SK_ROWS:
    # lanes 32..63 hold rows 8..15, which are exactly +0.0 in every partition
    # when SK_ROWS <= 8 (cnt_e <= M), so they are neither sent nor added.
    # Rows [SK_ROWS, 8) are sent and added as +0.0 + +0.0.  mine + peer in
    # the same f32 op order as the LDS path -> bit-identical.
    _skrd = bool(SK_RD) and SK > 1 and not SK_P16
    if const_expr(_skrd):
        assert kMChunks == 1 and BM == 16
        assert min(int(SK_ROWS) or BM, BM) <= 8
        _rd_st_atom = fx.make_copy_atom(
            fx.rocdl.BufferCopy128b(SK_ST_AUX), fx.Float32
        )
        _rd_ld_atom = fx.make_copy_atom(
            fx.rocdl.BufferCopy128b(SK_LD_AUX), fx.Float32
        )
        _rd_tiles = _global_f32x4_buffer_tiles(arg_part, 1 << 26)
        _rd_lay = fx.make_layout(4, 1)
        _RD_STRIDE = NJ * 128
        _rd_slot = wave * fx.Int32(32) + lane
        _rd_base = (bx_i32 * fx.Int32(SK) + sk_i32) * fx.Int32(_RD_STRIDE)
        _rd_regs = []
        for J in range_constexpr(NJ):
            _vv = fx.Vector(accm[0][J])
            _rr = fx.make_rmem_tensor(_rd_lay, fx.Float32)
            _rr.store(
                fx.Vector.from_elements(
                    [fx.Float32(_vv[_e]) for _e in range_constexpr(4)], fx.Float32
                )
            )
            _rd_regs.append(_rr)
        if lane < fx.Int32(32):
            for J in range_constexpr(NJ):
                fx.copy_atom_call(
                    _rd_st_atom,
                    _rd_regs[J],
                    fx.slice(
                        _rd_tiles, (None, _rd_base + fx.Int32(J * 128) + _rd_slot)
                    ),
                )
        # b2 r1_d0: see the LDS-staged exchange below -- drain the partial
        # stores before the (workgroup-scope) release + flag atomic.
        rocdl.s_waitcnt(vmcnt=0)
        fence_release(fx.rocdl.SyncScope.WorkgroupOneAs)

    gpu.barrier()

    # lds_acc reuses the s_aq region (offset 0) as an f32 accumulator.
    acc_layout = fx.make_layout((BM, BN), (BN, 1))

    def acc_idx(row, col):
        return _layout_idx(acc_layout, row, col)

    acc_copy_atom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
    acc_reg_lay = fx.make_layout(1, 1)
    acc_flat_view = fx.make_view(
        fx.recast_iter(
            fx.Float32,
            (fx.add_offset(lds_raw_ptr, _ACC_OFF) if _ARES else lds_raw_ptr),
        ),
        fx.make_layout(BM * BN, 1),
    )
    acc_flat_tiles = fx.logical_divide(acc_flat_view, fx.make_layout(1, 1))
    acc_f32x4_tiles = fx.logical_divide(acc_flat_view, fx.make_layout(4, 1))

    def acc_store(idx, value):
        r = fx.make_rmem_tensor(acc_reg_lay, fx.Float32)
        r.store(fx.Vector.from_elements([fx.Float32(value)], fx.Float32))
        fx.copy_atom_call(acc_copy_atom, r, fx.slice(acc_flat_tiles, (None, idx)))

    def acc_load(idx):
        r = fx.make_rmem_tensor(acc_reg_lay, fx.Float32)
        fx.copy_atom_call(acc_copy_atom, fx.slice(acc_flat_tiles, (None, idx)), r)
        return r.load()[0]

    def _acc_stage(_getv):
        for i in range_constexpr(kMChunks):
            row_base = fx.Int32(i * 16) + lane_div_16 * fx.Int32(4)
            for J in range_constexpr(NJ):
                is_up = (J % 2) == 1
                J_local = J // 2
                col_local = wave * fx.Int32(W8) + fx.Int32(J_local * 16) + lane_mod_16
                lds_col = (fx.Int32(BN // 2) + col_local) if is_up else col_local
                vec = _getv(i, J)
                for v in range_constexpr(4):
                    idx = acc_idx(row_base + fx.Int32(v), lds_col)
                    acc_store(idx, vec[v])

    if const_expr(not _skrd):
        _acc_stage(lambda i, J: fx.Vector(accm[i][J]))
        gpu.barrier()

    tx_i32 = fx.Int32(gpu.thread_id("x"))

    # ---------------- split-K combine (no extra graph node) ----------------
    # Each K-partition CTA dumps its fp32 partial tile to global, fences at
    # WORKGROUP scope (an agent-scope fence lowers to buffer_wbl2/buffer_inv
    # and costs more than the whole kernel), then bumps a per-output-tile
    # arrival counter.  The LAST arrival (no spinning, so residency is
    # irrelevant) adds the peer partials into its LDS accumulator, resets the
    # counter with atomic_add(-SK) and runs the stock epilogue.  Visibility is
    # carried by the DATA cache-policy aux bits on the buffer copies.
    if const_expr(SK > 1 and not _skrd):
        _ELEM4 = (BM * BN) // 4
        assert _ELEM4 % 256 == 0
        _NVEC = _ELEM4 // 256
        # ---- ROW-LIMITED EXCHANGE (exact, bit-identical) ----------------
        # Thread tx / register-vector v owns LDS acc row 4*v + wave (the acc
        # layout is plain row-major (BM,BN):(BN,1), so f32x4 chunk v*256+tx
        # sits at row (v*256+tx)/64 = 4*v + wave).  A BM=16 tile can never
        # hold more than min(BM, M) real rows, because one expert can appear
        # at most ONCE per token, so cnt_e <= M.  The remaining rows gather
        # token index M_CONST, which is out of range, so their A fragment is
        # an all-zero buffer read and BOTH K-partitions accumulate exactly
        # 0.0 there -- skipping them in the exchange is bit-exact, not an
        # approximation.  At dec_bs1 that is 6 of 16 rows = 2.67x fewer
        # bytes on the round trip, at full fp32 precision.
        _RW = int(SK_ROWS) if SK_ROWS else BM
        _RW = min(_RW, BM)
        _FULLV = _RW // 4
        _PARTV = 1 if (_RW % 4) else 0
        _NVU = _FULLV + _PARTV
        assert 1 <= _NVU <= _NVEC
        _PSTRIDE = _NVU * 256
        p_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float32)
        p_lay = fx.make_layout(4, 1)

        def _p_load(tiles, idx):
            r = fx.make_rmem_tensor(p_lay, fx.Float32)
            fx.copy_atom_call(p_atom, fx.slice(tiles, (None, idx)), r)
            return r

        def _p_store(tiles, idx, r):
            fx.copy_atom_call(p_atom, r, fx.slice(tiles, (None, idx)))

        _mine = []
        for _v in range_constexpr(_NVU):
            _mine.append(_p_load(acc_f32x4_tiles, fx.Int32(_v * 256) + tx_i32))

        if const_expr(SK_P16):
            # HALF-WIDTH EXCHANGE: the partials round-trip as bf16 (the own
            # partial stays fp32 in registers, so only the PEER half is
            # rounded; stage-1's output is re-quantised to fp4 downstream so
            # 8 mantissa bits on one addend is far below the output LSB).
            # Bytes moved per CTA drop 2x.  The buffer is private scratch, so
            # any consistent thread->slot permutation is legal; we pair
            # register vector 2p/2p+1 into one 128b chunk of 8 bf16.
            assert _NVU % 2 == 0
            _NPAIR = _NVU // 2
            _CHUNKS = _NPAIR * 256
            part_tiles = _global_i32x4_buffer_tiles(arg_part, 1 << 26)
            pg_store_atom = fx.make_copy_atom(
                fx.rocdl.BufferCopy128b(SK_ST_AUX), fx.Int32
            )
            pg_load_atom = fx.make_copy_atom(
                fx.rocdl.BufferCopy128b(SK_LD_AUX), fx.Int32
            )
            pw_lay = fx.make_layout(4, 1)

            def _pk_pair(p):
                a = _mine[2 * p].load()
                b = _mine[2 * p + 1].load()
                return [
                    _bf16_pack2(fx.Float32(a[0]), fx.Float32(a[1])),
                    _bf16_pack2(fx.Float32(a[2]), fx.Float32(a[3])),
                    _bf16_pack2(fx.Float32(b[0]), fx.Float32(b[1])),
                    _bf16_pack2(fx.Float32(b[2]), fx.Float32(b[3])),
                ]

            _my_cbase = (bx_i32 * fx.Int32(SK) + sk_i32) * fx.Int32(_CHUNKS)
            for _p in range_constexpr(_NPAIR):
                _w = _pk_pair(_p)
                _r = fx.make_rmem_tensor(pw_lay, fx.Int32)
                _r.store(fx.Vector.from_elements(_w, fx.Int32))
                fx.copy_atom_call(
                    pg_store_atom,
                    _r,
                    fx.slice(
                        part_tiles,
                        (None, _my_cbase + fx.Int32(_p * 256) + tx_i32),
                    ),
                )
        else:
            part_tiles = _global_f32x4_buffer_tiles(arg_part, 1 << 26)
            pg_store_atom = fx.make_copy_atom(
                fx.rocdl.BufferCopy128b(SK_ST_AUX), fx.Float32
            )
            pg_load_atom = fx.make_copy_atom(
                fx.rocdl.BufferCopy128b(SK_LD_AUX), fx.Float32
            )

            def _pg_load(idx):
                r = fx.make_rmem_tensor(p_lay, fx.Float32)
                fx.copy_atom_call(pg_load_atom, fx.slice(part_tiles, (None, idx)), r)
                return r

            def _pg_store(idx, r):
                fx.copy_atom_call(pg_store_atom, r, fx.slice(part_tiles, (None, idx)))

            _my_base = (bx_i32 * fx.Int32(SK) + sk_i32) * fx.Int32(_PSTRIDE)
            for _v in range_constexpr(_FULLV):
                _pg_store(_my_base + fx.Int32(_v * 256) + tx_i32, _mine[_v])
            if const_expr(_PARTV):
                if wave < fx.Int32(_RW - 4 * _FULLV):
                    _pg_store(
                        _my_base + fx.Int32(_FULLV * 256) + tx_i32, _mine[_FULLV]
                    )

        # b2 r1_d0 (0927) CORRECTNESS: drain this wave's partial stores
        # (vmcnt(0)) BEFORE the release/barrier/flag atomic.  A WORKGROUP-scope
        # release fence does not wait on vmcnt outside tgsplit mode, so the
        # thread-0 `atomic_add_agent` on the arrival counter could become
        # visible while other waves' partial stores were still in flight; the
        # last arriver then summed a stale peer partial.  Invisible when the
        # kernel owns the GPU, exposed by ANY concurrent kernel (TBO / overlap
        # streams): 4-6 corrupted M=6 outputs (rel_norm 0.014-0.074) per 1200
        # graph-replayed calls with a concurrent stream, 0/2400 with this wait.
        # The stores already carry sc1 (SK_ST_AUX=4) so no L2 writeback
        # (buffer_wbl2, the measured 2x agent-scope cost) is needed.
        rocdl.s_waitcnt(vmcnt=0)
        fence_release(fx.rocdl.SyncScope.WorkgroupOneAs)
        gpu.barrier()

        _scr_view = fx.make_view(
            fx.recast_iter(
                fx.Int32, fx.add_offset(lds_raw_ptr, _SK_SCRATCH_OFF)
            ),
            fx.make_layout(16, 1),
        )
        _scr_tiles = fx.logical_divide(_scr_view, fx.make_layout(1, 1))
        _flag_addr = fx.Int64(arg_flag) + fx.Int64(bx_i32) * fx.Int64(4)
        if tx_i32 == fx.Int32(0):
            _old = fx.Int32(atomic_add_agent(_flag_addr, fx.Int32(1)))
            _scalar_store(_scr_tiles, 0, _old, fx.Int32)
        gpu.barrier()
        _is_last = _global_i32_load(_scr_tiles, 0) == fx.Int32(SK - 1)

        if _is_last:
            fence_acquire(fx.rocdl.SyncScope.WorkgroupOneAs)
            for _s in range_constexpr(1, SK):
                _peer = (sk_i32 + fx.Int32(_s)) % fx.Int32(SK)
                if const_expr(SK_P16):
                    _pc = (bx_i32 * fx.Int32(SK) + _peer) * fx.Int32(_CHUNKS)
                    for _p in range_constexpr(_NPAIR):
                        _r = fx.make_rmem_tensor(pw_lay, fx.Int32)
                        fx.copy_atom_call(
                            pg_load_atom,
                            fx.slice(
                                part_tiles,
                                (None, _pc + fx.Int32(_p * 256) + tx_i32),
                            ),
                            _r,
                        )
                        _wv = _r.load()
                        _f = []
                        for _q in range_constexpr(4):
                            _lo, _hi = _bf16_unpack2(fx.Int32(_wv[_q]))
                            _f.append(_lo)
                            _f.append(_hi)
                        for _h in range_constexpr(2):
                            _a = _mine[2 * _p + _h].load()
                            _mine[2 * _p + _h].store(
                                fx.Vector.from_elements(
                                    [
                                        fx.Float32(_a[_e]) + _f[4 * _h + _e]
                                        for _e in range_constexpr(4)
                                    ],
                                    fx.Float32,
                                )
                            )
                else:
                    _pb = (bx_i32 * fx.Int32(SK) + _peer) * fx.Int32(_PSTRIDE)

                    def _acc_v(_v):
                        _o = _pg_load(_pb + fx.Int32(_v * 256) + tx_i32)
                        _a = _mine[_v].load()
                        _b = _o.load()
                        _mine[_v].store(
                            fx.Vector.from_elements(
                                [
                                    fx.Float32(_a[_e]) + fx.Float32(_b[_e])
                                    for _e in range_constexpr(4)
                                ],
                                fx.Float32,
                            )
                        )

                    for _v in range_constexpr(_FULLV):
                        _acc_v(_v)
                    if const_expr(_PARTV):
                        if wave < fx.Int32(_RW - 4 * _FULLV):
                            _acc_v(_FULLV)
            for _v in range_constexpr(_FULLV):
                _p_store(
                    acc_f32x4_tiles, fx.Int32(_v * 256) + tx_i32, _mine[_v]
                )
            if const_expr(_PARTV):
                if wave < fx.Int32(_RW - 4 * _FULLV):
                    _p_store(
                        acc_f32x4_tiles,
                        fx.Int32(_FULLV * 256) + tx_i32,
                        _mine[_FULLV],
                    )
            if tx_i32 == fx.Int32(0):
                atomic_add_agent(_flag_addr, fx.Int32(-SK))
            gpu.barrier()
    if const_expr(_skrd):
        # partials were stored + released before the drain barrier above.
        _scr_view = fx.make_view(
            fx.recast_iter(
                fx.Int32, fx.add_offset(lds_raw_ptr, _SK_SCRATCH_OFF)
            ),
            fx.make_layout(16, 1),
        )
        _scr_tiles = fx.logical_divide(_scr_view, fx.make_layout(1, 1))
        _flag_addr = fx.Int64(arg_flag) + fx.Int64(bx_i32) * fx.Int64(4)
        if tx_i32 == fx.Int32(0):
            _old = fx.Int32(atomic_add_agent(_flag_addr, fx.Int32(1)))
            _scalar_store(_scr_tiles, 0, _old, fx.Int32)
        gpu.barrier()
        _is_last = _global_i32_load(_scr_tiles, 0) == fx.Int32(SK - 1)

        if _is_last:
            fence_acquire(fx.rocdl.SyncScope.WorkgroupOneAs)
            for _s in range_constexpr(1, SK):
                _peer = (sk_i32 + fx.Int32(_s)) % fx.Int32(SK)
                _pb = (bx_i32 * fx.Int32(SK) + _peer) * fx.Int32(_RD_STRIDE)
                if lane < fx.Int32(32):
                    for J in range_constexpr(NJ):
                        _o = fx.make_rmem_tensor(_rd_lay, fx.Float32)
                        fx.copy_atom_call(
                            _rd_ld_atom,
                            fx.slice(
                                _rd_tiles,
                                (None, _pb + fx.Int32(J * 128) + _rd_slot),
                            ),
                            _o,
                        )
                        _a = _rd_regs[J].load()
                        _b = _o.load()
                        _rd_regs[J].store(
                            fx.Vector.from_elements(
                                [
                                    fx.Float32(_a[_e]) + fx.Float32(_b[_e])
                                    for _e in range_constexpr(4)
                                ],
                                fx.Float32,
                            )
                        )
            _acc_stage(lambda i, J: _rd_regs[J].load())
            if tx_i32 == fx.Int32(0):
                atomic_add_agent(_flag_addr, fx.Int32(-SK))
            gpu.barrier()

    m_lane = tx_i32 // fx.Int32(NTH)
    n_lane = tx_i32 % fx.Int32(NTH)
    wave_grp = n_lane // fx.Int32(4)
    kk = n_lane % fx.Int32(4)

    aqout_layout = fx.make_layout((BM, K_G2_HALF), (K_G2_HALF, 1))
    # UniversalCopy has no nontemporal/cache-hint knob; dropped (perf-neutral).
    aqout_tiles = _global_scalar_tiles(arg_aqout, fx.Int32, 1 << 24)
    scales_per_mr = [None] * M_REPS

    # At BN<256 only BM*NTH of the 256 threads own an output row; the rest
    # must not store (they would scribble into the NEXT expert block's rows).
    def _epilogue_tail():

        for mr in range_constexpr(M_REPS):
            row_local = fx.Int32(mr * 16) + m_lane

            gate_vs = [None] * 8
            up_vs = [None] * 8
            for ee in range_constexpr(8):
                col_in_grp = fx.Int32(8) * kk + fx.Int32(ee)
                gate_col = wave_grp * fx.Int32(32) + col_in_grp
                up_col = fx.Int32(BN // 2) + gate_col
                gate_vs[ee] = acc_load(acc_idx(row_local, gate_col))
                up_vs[ee] = acc_load(acc_idx(row_local, up_col))
            # Round to bf16 before MXFP4, as the bf16-intermediate reference does.
            result = [
                fx.Float32(fx.BFloat16(v)) for v in _silu_mul_batch(gate_vs, up_vs)
            ]

            local_max = _fabs_f32(result[0])
            for ee in range_constexpr(1, 8):
                local_max = local_max.maximumf(_fabs_f32(result[ee]))
            lm_i = _inline_dpp_quad_amax(fx.Int32(_raw(local_max).bitcast(T.i32)))
            local_max = fx.Float32(_raw(lm_i).bitcast(T.f32))

            e8m0, qscale = _e8m0_from_amax(local_max)
            scales_per_mr[mr] = e8m0

            packed_i32 = _raw(fx.Int32(0))
            qscale_raw = _raw(qscale)
            for w in range_constexpr(4):
                packed_i32 = rocdl.cvt_scalef32_pk_fp4_f32(
                    T.i32,
                    packed_i32,
                    _raw(result[2 * w]),
                    _raw(result[2 * w + 1]),
                    qscale_raw,
                    w,
                )
            packed = fx.Int32(packed_i32)

            byte_pos = (
                n_block_idx * fx.Int32(BN_INT // 2)
                + wave_grp * fx.Int32(16)
                + kk * fx.Int32(4)
            )
            out_row = m_row + row_local
            store_off = _layout_idx(aqout_layout, out_row, byte_pos)
            _scalar_store(aqout_tiles, store_off // fx.Int32(4), packed, fx.Int32)

        # (chunk, ku, wave_grp, m_lane) -> dword index; shape is a placeholder.
        ascaleout_layout = fx.make_layout(
            (1 << 20, 2, 4, 16), (OUT_AS_PER_CHUNK_DW, 64, 16, 1)
        )
        ascaleout_i8_tiles = _global_scalar_tiles(arg_ascaleout, fx.Int8, 1 << 26)
        ascaleout_i16_tiles = _global_scalar_tiles(arg_ascaleout, fx.Int16, 1 << 25)
        if kk == fx.Int32(0):
            # Global index of this thread's 32-wide mxfp4 group along `inter`.
            # BN=256 -> grp = nb*4 + wave_grp (identical to the stock kernel);
            # BN=128 -> grp = nb*2 + wave_grp.  The shuffled e8m0 byte address is a
            # function of grp only, so re-deriving (nb_old, wave_grp_old) from grp
            # keeps the layout gemm2 reads byte-for-byte identical.
            grp = n_block_idx * fx.Int32(BN // 64) + wave_grp
            nb_old = grp >> fx.Int32(2)
            wg_old = grp & fx.Int32(3)
            ku = nb_old >> fx.Int32(1)
            ikxdl = nb_old & fx.Int32(1)
            if const_expr(BM == 16):
                chunk = m_block_idx
                dword_off = _layout_idx(ascaleout_layout, chunk, ku, wg_old, m_lane)
                addr = dword_off * fx.Int32(4) + ikxdl * fx.Int32(2)
                _scalar_store(ascaleout_i8_tiles, addr, scales_per_mr[0], fx.Int8)
            else:
                for sub in range_constexpr(kSubBlocks):
                    chunk = m_block_idx * fx.Int32(kSubBlocks) + fx.Int32(sub)
                    dword_off = _layout_idx(ascaleout_layout, chunk, ku, wg_old, m_lane)
                    pair_i32 = scales_per_mr[sub * 2 + 0] | (
                        scales_per_mr[sub * 2 + 1] << fx.Int32(8)
                    )
                    addr = dword_off * fx.Int32(4) + ikxdl * fx.Int32(2)
                    _scalar_store(
                        ascaleout_i16_tiles, addr // fx.Int32(2), pair_i32, fx.Int16
                    )


    def _epilogue_guarded():
        if const_expr(BM * NTH < 256):
            if tx_i32 < fx.Int32(BM * NTH):
                _epilogue_tail()
        else:
            _epilogue_tail()

    if const_expr(SK > 1):
        if _is_last:
            _epilogue_guarded()
    else:
        _epilogue_guarded()

def _bm_constants(BM, BN, KH_TILE, K_TILES_TOTAL, kAStages=None, SK=1, A_RES=0):
    if kAStages is None:
        kAStages = 2 if BM == 128 else 3
    kSubBlocks = 1 if BM < 32 else BM // 32
    kMChunks = kmchunks_for(BM)
    s_aq_bytes = kAStages * BM * KH_TILE
    s_asc_bytes = kSubBlocks * K_TILES_TOTAL * 256
    lds_acc_bytes = lds_acc_bytes_for(BM, BN)
    lds_bytes = max(s_aq_bytes + s_asc_bytes, lds_acc_bytes)
    if A_RES:
        # A-resident (XB): lds_acc gets its OWN region after [s_aq | s_asc] so
        # the epilogue of n-block 0 does not clobber the quantised A tiles
        # that n-block 1 re-reads.
        lds_bytes = (s_aq_bytes + s_asc_bytes + 15) // 16 * 16 + lds_acc_bytes
    if SK > 1:
        # 64B tail scratch for the split-K arrival-order broadcast.
        lds_bytes += 64
    return kAStages, kSubBlocks, kMChunks, lds_bytes


def compile_gemm1_a4w4_fold(
    BM=32,
    use_nt=True,
    inline_quant=False,
    *,
    D_HIDDEN,
    D_INTER,
    NE,
    TOPK,
    BN=256,
    BK=256,
    interleave=False,
    xcd_swizzle=0,
    kstages=2,
    kastages=0,
    fold=False,
    M_CONST=0,
    splitk=1,
    splitk_aux=(0, 1),
    nfuse=1,
    splitk_p16=0,
    splitk_rows=0,
    splitk_rd=0,
    krot=0,
    xbody=0,
    xres=0,
    item=0,
):
    assert splitk >= 1
    assert nfuse >= 1
    # r2_d0: explicit weight-tile aux (int > 1) -> treated as an nt-class variant.
    _WAUX = (
        int(use_nt)
        if (use_nt is not True and use_nt is not False and int(use_nt) > 1)
        else 0
    )
    if _WAUX:
        use_nt = True
    if (BM, use_nt, inline_quant) not in {
        (32, True, False),
        (32, False, False),
        (64, False, False),
        (128, False, False),
        (16, True, True),
        (16, False, True),   # vendored: inline-quant with CACHED (non-nt) B loads
    }:
        raise AssertionError(
            f"unsupported gemm1 variant (BM={BM}, use_nt={use_nt}, inline_quant={inline_quant})"
        )

    assert BN in (128, 256) and BK == 256, f"only BN in (128,256), BK==256, got BN={BN} BK={BK}"
    KH_TILE = BK // 2
    _K = D_HIDDEN
    assert _K % BK == 0, f"D_HIDDEN (K) must be a multiple of {BK}, got {_K}"
    _INTER = D_INTER
    _N_OUT = n_out_for(_INTER)
    assert (
        _N_OUT % BN == 0
    ), f"2*D_INTER (N_OUT) must be a multiple of {BN}, got {_N_OUT}"
    _NE = NE
    _K_TILES_TOTAL = k_tiles_total_for(_K, BK)
    _SK = int(splitk)
    assert _K_TILES_TOTAL % _SK == 0
    _KTP = _K_TILES_TOTAL // _SK
    assert _SK == 1 or inline_quant, "split-K is only wired for the inline-quant port"
    _NUM_N_BLOCKS = num_n_blocks_for(_N_OUT, BN)
    # ---- WAVE-18 r1_d0: N-FUSION -----------------------------------------
    # One CTA sequentially owns _NF adjacent n-blocks of the SAME m-block.
    # Byte-neutral on B (each (m,n) tile is still read exactly once) and the
    # per-CTA fold block head (257-expert sort replication) is paid ONCE per
    # _NF output tiles instead of once per tile.  CTA count drops by _NF.
    _NF = int(nfuse)
    assert _NUM_N_BLOCKS % _NF == 0, "nfuse must divide num_n_blocks"
    assert _NF == 1 or _SK == 1, "n-fusion and split-K are mutually exclusive"
    _NNB = _NUM_N_BLOCKS // _NF

    _kas = int(kastages) if kastages else (2 if BM == 128 else 3)
    assert _kas >= 2
    assert _kas + 1 <= _KTP or _SK == 1
    assert kstages < _KTP, "kStages must be < k-tiles per partition"
    _, _, _, lds_bytes = _bm_constants(
        BM, BN, KH_TILE, _KTP, _kas, _SK
    )

    variant_tag = "iq" if inline_quant else ("nt" if use_nt else "cached")
    # Tag with H/INTER/NE so different shape specializations get distinct
    # kernel/smem symbols (so KIMI and non-KIMI instances never collide).
    gu_tag = "il" if interleave else "sep"
    name_suffix = f"h{_K}_i{_INTER}_ne{_NE}_bm{BM}_bn{BN}_{variant_tag}_{gu_tag}_ks{kstages}_as{_kas}"
    name_suffix += "_bsa2"
    if inline_quant and not use_nt:
        # r2_d0: cached (non-nt) w13 stream -- distinct JIT symbol/launcher.
        name_suffix += "_wc"
    if _WAUX and _WAUX != 2:
        name_suffix += f"_wa{_WAUX}"
    if _HPF and inline_quant:
        name_suffix += f"_hpf{_HPF}" + ("" if _HPB else "nb")
    _SKP16 = int(bool(splitk_p16))
    _SKROWS = min(int(splitk_rows) or BM, BM)
    if _SK > 1:
        name_suffix += f"_spk{_SK}_sa{splitk_aux[0]}{splitk_aux[1]}"
        name_suffix += "_vw0"  # b2 r1_d0: vmcnt(0) before the split-K release
        if _SKP16:
            name_suffix += "_p16"
        if _SKROWS != BM:
            name_suffix += f"_rw{_SKROWS}"
    # 0926 r1_d1: register-direct split-K exchange (see SK_RD in the body).
    _SKRD = (
        int(bool(splitk_rd))
        if (_SK > 1 and not _SKP16 and BM == 16 and _SKROWS <= 8 and inline_quant)
        else 0
    )
    if _SKRD:
        name_suffix += "_skrd"
    if _NF > 1:
        name_suffix += f"_nf{_NF}"
    # XB: cross-n-block B prefetch under n-fusion (see _gemm1_body_fold).
    _XBN = int(xbody) if (_NF > 1 and inline_quant and BN == 256 and not interleave) else 0
    if _XBN:
        name_suffix += f"_xb{_XBN}"
    _XRN = int(bool(xres)) if (_XBN and _SK == 1) else 0
    if _XRN:
        name_suffix += "_xr"
        _, _, _, lds_bytes = _bm_constants(BM, BN, KH_TILE, _KTP, _KTP, _SK, 1)
    if xcd_swizzle > 0:
        name_suffix += f"_xcd{xcd_swizzle}"
    # WAVE-74 r1_d0: per-CTA k-start rotation.  krot==0 (or split-K) leaves the
    # emitted body AND the JIT symbol byte-identical to the incumbent.
    _KROT = int(krot) % _KTP if (krot and _SK == 1) else 0
    if _KROT:
        name_suffix += f"_kr{_KROT}"
    _TKWN = int(bool(_TKW)) if fold else 0
    if _TKWN:
        name_suffix += "_tw"
    if _AFB and inline_quant:
        name_suffix += f"_afb{int(_AFB)}"
    _BEFN = int(_BEF) if (fold and inline_quant and not _AFB and int(nfuse) == 1) else 0
    if _BEFN:
        name_suffix += f"_bef{min(_BEFN, kstages)}"
    _HD2N = int(bool(_HD2)) if fold else 0
    if _HD2N:
        name_suffix += "_hd2"
    _KAPN = int(bool(_KAP)) if fold else 0
    if _KAPN:
        name_suffix += "_kap"

    # ---- prologue-fold compile constants -------------------------------
    _FOLD = bool(fold)
    assert _NF == 1 or _FOLD, "n-fusion is only wired for the fold port"
    _TOPK = TOPK
    if _FOLD:
        assert inline_quant and BM == 16, "fold is decode-only (BM=16 inline-quant)"
        assert M_CONST > 0, "fold needs the ACTUAL M as a compile constant"
        name_suffix += f"_fold{M_CONST}"
    _MT = int(M_CONST) * int(TOPK) if _FOLD else 0
    _NREP = (_MT + 255) // 256 if _FOLD else 0
    # ITEM-DIRECT HEAD (see _ITEM_DOC below).  Legal only where every expert
    # owns at most ONE m-block (M <= BM) and MT <= NE, so m-block count ==
    # distinct experts <= MT and the grid is exactly MT * NNB tiles.
    _ITEM = int(
        bool(item)
        and _FOLD
        and int(M_CONST) <= BM
        and _MT <= _NE
        and _NREP == 1
        and _NF == 1
        and xcd_swizzle == 0
        and _BEFN > 0
        and _HD2N > 0
    )
    if _ITEM:
        name_suffix += "_it"
    if _FOLD:
        _active = min(_MT, _NE)
        _max_sorted = ((_MT + _active * (BM - 1)) + BM - 1) // BM * BM
        _NB_MAX = _max_sorted // BM
        _GRID = _NB_MAX * _NNB
        _NDW = int(M_CONST) * (_K // 2)         # bf16 [M, K] as i32 words
        _NREP_Z = (_NDW + _GRID * 256 - 1) // (_GRID * 256)
        sort_bytes = _sort_words(_MT) * 4
    else:
        _GRID = 0
        _NDW = 0
        _NREP_Z = 0
        sort_bytes = 16

    @fx.struct
    class SharedStorage:
        raw: fx.Array[fx.Uint8, lds_bytes, 16]
        srt: fx.Array[fx.Uint8, sort_bytes, 16]

    @flyc.kernel(name=f"gemm1_a4w4_fold_{name_suffix}", known_block_size=[256, 1, 1])
    def gemm1_kernel_fold(
        arg_aq: fx.Int64,
        arg_ascale: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_mind: fx.Int64,
        i32_ntok: fx.Int32,
        arg_aqout: fx.Int64,
        arg_ascaleout: fx.Int64,
        arg_hidden: fx.Int64,
        arg_topk: fx.Int64,
        arg_topkw: fx.Int64,
        arg_stok: fx.Int64,
        arg_swgt: fx.Int64,
        arg_moebuf: fx.Int64,
        arg_part: fx.Int64,
        arg_flag: fx.Int64,
    ):
        _smem = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_raw_ptr = _smem.raw.ptr
        sort_ptr = _smem.srt.ptr
        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        tx_i32 = fx.Int32(tx)
        bx_i32 = fx.Int32(bx)
        lane = tx_i32 % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tx_i32 // fx.Int32(64))
        if const_expr(_KAPN):
            _pins = [
                _raw(fx.Int64(arg_bq)),
                _raw(fx.Int64(arg_bscale)),
                _raw(fx.Int64(arg_part)),
                _raw(fx.Int64(arg_flag)),
                _raw(fx.Int32(i32_ntok)),
            ]
            llvm.InlineAsmOp(
                None, _pins, "", ",".join(["s"] * len(_pins)), has_side_effects=True
            )
        fold_excl_a = fx.Int32(0)
        fold_nb_a = fx.Int32(0)
        fold_t256 = fx.Int32(0)
        if const_expr(_FOLD):
            cnt_tiles = _lds_i32_tiles(sort_ptr, _W_CNT, 272)
            wt_tiles = _lds_i32_tiles(sort_ptr, _W_WT, 16)
            mi_tiles = _lds_i32_tiles(sort_ptr, _W_MISC, 16)
            rw_tiles = _lds_i32_tiles(sort_ptr, _W_ROWS, 16)
            st_tiles = _lds_i32_tiles(sort_ptr, _W_STOK, 16)
            wg_tiles = _lds_i32_tiles(sort_ptr, _W_WGT, 16)
            tk_tiles = _lds_i32_tiles(sort_ptr, _W_TK, _NREP * 256)

            _tkw_regs = []
            _tk_regs = []
            if const_expr(_HD2N):
                # ---- head v2 (see _HD2): topk loads FIRST ------------------
                if const_expr(_ITEM):
                    # ITEM-DIRECT: tile -> (item j, n-block).  The expert is
                    # topk_ids[j]; one uniform load, no sort needed for B.
                    if const_expr(_SK > 1):
                        _it_grp = 8 * _SK
                        _it_r = _umod(bx_i32, _it_grp)
                        _it_pid = _udiv(bx_i32, _it_grp) * fx.Int32(8) + _umod(_it_r, 8)
                    else:
                        _it_pid = bx_i32
                    _it_j = _it_pid // fx.Int32(_NNB)
                    _it_jc = fx.Int32(arith.minsi(_raw(_it_j), _raw(fx.Int32(_MT - 1))))
                    _it_e_raw = _global_i32_at(arg_topk, _it_jc)
                for _r in range_constexpr(_NREP):
                    _i = tx_i32 + fx.Int32(_r * 256)
                    _m = _ashr(_i - fx.Int32(_MT), 31)      # -1 iff _i < MT
                    _tk_regs.append((_global_i32_at(arg_topk, _i & _m), _m))
                    if const_expr(_TKWN):
                        _tkw_regs.append(_global_i32_at(arg_topkw, _i & _m))
                # (a) moe_buf zero-init, issued behind the topk loads
                mb_tiles = _global_scalar_tiles(arg_moebuf, fx.Int32, _NDW)
                _gtid = bx_i32 * fx.Int32(256) + tx_i32
                for _rz in range_constexpr(_NREP_Z):
                    _zi = _gtid + fx.Int32(_rz * _GRID * 256)
                    if _zi < fx.Int32(_NDW):
                        _scalar_store(mb_tiles, _zi, fx.Int32(0), fx.Int32)
                # (c0) zero the histogram while the topk loads fly
                _e_a = tx_i32
                for _r in range_constexpr(2):
                    _ci = tx_i32 + fx.Int32(_r * 256)
                    if _ci < fx.Int32(272):
                        _scalar_store(cnt_tiles, _ci, fx.Int32(0), fx.Int32)
                _tk_regs = [
                    (_tv & _m) | (fx.Int32(-1) ^ _m) for (_tv, _m) in _tk_regs
                ]
                if const_expr(_ITEM):
                    if tx_i32 == fx.Int32(0):
                        _scalar_store(mi_tiles, fx.Int32(2), fx.Int32(0), fx.Int32)
                    _it_e = rocdl.readfirstlane(T.i32, _raw(_it_e_raw))
                gpu.barrier()
                for _r in range_constexpr(_NREP):
                    _i = tx_i32 + fx.Int32(_r * 256)
                    if _i < fx.Int32(_MT):
                        _lds_atomic_add(cnt_tiles, _tk_regs[_r], 1)
                if const_expr(_ITEM):
                    # duplicate item (an earlier item routes to the same
                    # expert) or padded pid -> this CTA owns no tile.
                    _it_dup = _is_eq(_tk_regs[0], _it_e) & _ashr(tx_i32 - _it_j, 31)
                    _it_dup = _it_dup | (
                        fx.Int32(0) - _ashr(fx.Int32(_MT * _NNB - 1) - _it_pid, 31)
                    )
                    if _it_dup != fx.Int32(0):
                        _scalar_store(mi_tiles, fx.Int32(2), fx.Int32(1), fx.Int32)
                gpu.barrier()
            else:
                # ---- (a) moe_buf zero-init, striped over the whole grid ---
                mb_tiles = _global_scalar_tiles(arg_moebuf, fx.Int32, _NDW)
                _gtid = bx_i32 * fx.Int32(256) + tx_i32
                for _rz in range_constexpr(_NREP_Z):
                    _zi = _gtid + fx.Int32(_rz * _GRID * 256)
                    if _zi < fx.Int32(_NDW):
                        _scalar_store(mb_tiles, _zi, fx.Int32(0), fx.Int32)

                # ---- (b) cache topk_ids in LDS (pad slots read as -1) -----
                for _r in range_constexpr(_NREP):
                    _i = tx_i32 + fx.Int32(_r * 256)
                    _m = _ashr(_i - fx.Int32(_MT), 31)          # -1 iff _i < MT
                    _v = (_global_i32_at(arg_topk, _i & _m) & _m) | (fx.Int32(-1) ^ _m)
                    if const_expr(_TKWN):
                        _tkw_regs.append(_global_i32_at(arg_topkw, _i & _m))
                    _scalar_store(tk_tiles, _i, _v, fx.Int32)
                gpu.barrier()

                # ---- (c) per-expert token counts: LDS histogram, O(1)/thread
                _e_a = tx_i32
                for _r in range_constexpr(2):
                    _ci = tx_i32 + fx.Int32(_r * 256)
                    if _ci < fx.Int32(272):
                        _scalar_store(cnt_tiles, _ci, fx.Int32(0), fx.Int32)
                gpu.barrier()
                for _r in range_constexpr(_NREP):
                    _i = tx_i32 + fx.Int32(_r * 256)
                    if _i < fx.Int32(_MT):
                        _lds_atomic_add(cnt_tiles, _global_i32_load(tk_tiles, _i), 1)
                gpu.barrier()
            def _scan_d():
                _cnt_a = _global_i32_load(cnt_tiles, _e_a)
                _nb256 = (
                    _global_i32_load(cnt_tiles, fx.Int32(256)) + fx.Int32(BM - 1)
                ) >> fx.Int32(4)
                _nb_a = (_cnt_a + fx.Int32(BM - 1)) >> fx.Int32(4)

                # ---- (d) block-wide inclusive scan of the padded block counts --
                _inc = _dpp_scan_wave(_nb_a)
                gpu.barrier()
                if lane == fx.Int32(63):
                    _scalar_store(wt_tiles, wave, _inc, fx.Int32)
                gpu.barrier()
                _w0 = _global_i32_load(wt_tiles, fx.Int32(0))
                _w1 = _global_i32_load(wt_tiles, fx.Int32(1))
                _w2 = _global_i32_load(wt_tiles, fx.Int32(2))
                _w3 = _global_i32_load(wt_tiles, fx.Int32(3))
                _pref = (
                    (_w0 & _ashr(fx.Int32(0) - wave, 31))
                    + (_w1 & _ashr(fx.Int32(1) - wave, 31))
                    + (_w2 & _ashr(fx.Int32(2) - wave, 31))
                )
                _t256 = _w0 + _w1 + _w2 + _w3
                _excl = _inc + _pref - _nb_a
                gpu.barrier()
                return _nb_a, _excl, _t256, _t256 + _nb256

            if const_expr(_ITEM):
                # sort scan deferred into the head hook (after the B burst)
                _it_dead = rocdl.readfirstlane(
                    T.i32, _raw(_global_i32_load(mi_tiles, fx.Int32(2)))
                )
                total_m_blocks = fx.Int32(_NB_MAX)
            else:
                fold_nb_a, fold_excl_a, fold_t256, total_m_blocks = _scan_d()
        else:
            cumsum0 = _global_i32_at(arg_cumsum, fx.Int32(0))
            total_m_blocks = cumsum0 // fx.Int32(BM)
        bound = total_m_blocks * fx.Int32(_NNB)

        _NXCD = 8
        _xq = _udiv(bound, _NXCD)
        _xr = _umod(bound, _NXCD)
        _SW = xcd_swizzle

        def _xcd(pid):
            xc = _umod(pid, _NXCD)
            wgid = (
                xc * _xq
                + fx.Int32(arith.minsi(_raw(xc), _raw(_xr)))
                + _udiv(pid, _NXCD)
            )
            _ng = fx.Int32(_SW * _NNB)
            group_id = wgid // _ng
            first_pid_m = group_id * fx.Int32(_SW)
            remaining_m = total_m_blocks - first_pid_m
            group_size_m = fx.Int32(arith.minsi(_raw(remaining_m), _raw(fx.Int32(_SW))))
            wig = wgid % _ng
            m_block = first_pid_m + (wig % group_size_m)
            n_block = wig // group_size_m
            return m_block * fx.Int32(_NNB) + n_block

        # split-K block map: keep the SK partitions of one output tile on the
        # SAME XCD (blocks go to XCDs round-robin as bx % 8), so the partial
        # exchange lands in a shared L2 instead of crossing the fabric.  This
        # demux MUST run before the fold head derives its tile from _pid; the
        # raw blockIdx stays in use for the grid-strided zero-init above.
        if const_expr(_ITEM):
            # live iff _it_dead == 0  (bound = 1)
            bound = fx.Int32(1)
            _pid = _it_dead
            _skid = _udiv(_umod(bx_i32, 8 * _SK), 8) if _SK > 1 else fx.Int32(0)
        elif const_expr(_SK > 1):
            _grp = _NXCD * _SK
            _chunk = _udiv(bx_i32, _grp)
            _r = _umod(bx_i32, _grp)
            _skid = _udiv(_r, _NXCD)
            _pid = _chunk * fx.Int32(_NXCD) + _umod(_r, _NXCD)
        else:
            _pid = bx_i32
            _skid = fx.Int32(0)
        if fx.Int32(_pid) < bound:
            if const_expr(_ITEM):
                _tile = _it_pid
            elif const_expr(_SW > 0):
                _tile = _xcd(_pid)
            else:
                _tile = _pid
            fold_e = fx.Int32(0)
            if const_expr(_FOLD):
                if const_expr(_ITEM):
                    _nbi = _tile % fx.Int32(_NNB)
                    _mbi = None
                    _lo = fx.Int32(0)
                    fold_e = _it_e
                else:
                    _mbi = _tile // fx.Int32(_NNB)
                    _nbi = _tile % fx.Int32(_NNB)
                    # (e) which expert owns this m-block?
                    if _mbi >= fold_excl_a:
                        if _mbi < fold_excl_a + fold_nb_a:
                            _scalar_store(mi_tiles, fx.Int32(0), _e_a, fx.Int32)
                            _scalar_store(
                                mi_tiles, fx.Int32(1), _mbi - fold_excl_a, fx.Int32
                            )
                    if tx_i32 == fx.Int32(0):
                        if _mbi >= fold_t256:
                            _scalar_store(mi_tiles, fx.Int32(0), fx.Int32(256), fx.Int32)
                            _scalar_store(
                                mi_tiles, fx.Int32(1), _mbi - fold_t256, fx.Int32
                            )
                    if tx_i32 < fx.Int32(16):
                        _scalar_store(rw_tiles, tx_i32, fx.Int32(M_CONST), fx.Int32)
                        _scalar_store(st_tiles, tx_i32, fx.Int32(M_CONST), fx.Int32)
                        _scalar_store(wg_tiles, tx_i32, fx.Int32(0), fx.Int32)
                    gpu.barrier()
                    fold_e = rocdl.readfirstlane(
                        T.i32, _raw(_global_i32_load(mi_tiles, fx.Int32(0)))
                    )
                    _lo = _global_i32_load(mi_tiles, fx.Int32(1)) * fx.Int32(BM)

                _hs = {"mbi": _mbi, "lo": _lo, "tot": total_m_blocks}

                def _head_fg():
                    _mbi = _hs["mbi"]
                    _lo = _hs["lo"]
                    total_m_blocks = _hs["tot"]
                    # (f) rank the items of this expert and keep rows [lo, lo+16)
                    _base = fx.Int32(0)
                    for _r in range_constexpr(_NREP):
                        _i = tx_i32 + fx.Int32(_r * 256)
                        if const_expr(_HD2N):
                            _v = _tk_regs[_r]
                        else:
                            _v = _global_i32_load(tk_tiles, _i)
                        _x = _is_eq(_v, fold_e)
                        _inc2 = _dpp_scan_wave(_x)
                        gpu.barrier()
                        if lane == fx.Int32(63):
                            _scalar_store(wt_tiles, wave, _inc2, fx.Int32)
                        gpu.barrier()
                        _u0 = _global_i32_load(wt_tiles, fx.Int32(0))
                        _u1 = _global_i32_load(wt_tiles, fx.Int32(1))
                        _u2 = _global_i32_load(wt_tiles, fx.Int32(2))
                        _u3 = _global_i32_load(wt_tiles, fx.Int32(3))
                        _p2 = (
                            (_u0 & _ashr(fx.Int32(0) - wave, 31))
                            + (_u1 & _ashr(fx.Int32(1) - wave, 31))
                            + (_u2 & _ashr(fx.Int32(2) - wave, 31))
                        )
                        _d = _base + _p2 + _inc2 - _x - _lo
                        if _x == fx.Int32(1):
                            if fx.Uint32(_d) < fx.Uint32(BM):
                                _tok = _i // fx.Int32(_TOPK)
                                _kk = _i % fx.Int32(_TOPK)
                                _scalar_store(rw_tiles, _d, _tok, fx.Int32)
                                _scalar_store(
                                    st_tiles, _d, _tok | (_kk << fx.Int32(24)), fx.Int32
                                )
                                if const_expr(_TKWN):
                                    _scalar_store(wg_tiles, _d, _tkw_regs[_r], fx.Int32)
                                else:
                                    _scalar_store(
                                        wg_tiles, _d, _global_i32_at(arg_topkw, _i), fx.Int32
                                    )
                        _base = _base + _u0 + _u1 + _u2 + _u3
                    gpu.barrier()

                    # (g) publish this m-block's slice of the stage-2 sort tables
                    if _nbi == fx.Int32(0):
                        _stok_g = _global_scalar_tiles(arg_stok, fx.Int32, 1 << 24)
                        _swgt_g = _global_scalar_tiles(arg_swgt, fx.Int32, 1 << 24)
                        _seid_g = _global_scalar_tiles(arg_eids, fx.Int32, 1 << 22)
                        if tx_i32 < fx.Int32(BM):
                            _sp = _mbi * fx.Int32(BM) + tx_i32
                            _scalar_store(
                                _stok_g,
                                _sp,
                                _global_i32_load(st_tiles, tx_i32),
                                fx.Int32,
                            )
                            _scalar_store(
                                _swgt_g,
                                _sp,
                                _global_i32_load(wg_tiles, tx_i32),
                                fx.Int32,
                            )
                        if tx_i32 == fx.Int32(0):
                            _scalar_store(_seid_g, _mbi, fold_e, fx.Int32)
                            if _mbi == fx.Int32(0):
                                _nv_g = _global_scalar_tiles(arg_cumsum, fx.Int32, 2)
                                _scalar_store(
                                    _nv_g,
                                    fx.Int32(0),
                                    total_m_blocks * fx.Int32(BM),
                                    fx.Int32,
                                )
                                _scalar_store(
                                    _nv_g, fx.Int32(1), fx.Int32(M_CONST), fx.Int32
                                )

                if const_expr(not _BEFN):
                    _head_fg()

                def _head_item():
                    # ITEM-DIRECT head tail, run under the B burst: the sort
                    # scan (d), this expert's m-block (e'), then (f)/(g).
                    _nb_a, _excl, _t256, _tot = _scan_d()
                    if tx_i32 == fold_e:
                        _scalar_store(mi_tiles, fx.Int32(1), _excl, fx.Int32)
                    if tx_i32 == fx.Int32(0):
                        if fold_e == fx.Int32(256):
                            _scalar_store(mi_tiles, fx.Int32(1), _t256, fx.Int32)
                    if tx_i32 < fx.Int32(16):
                        _scalar_store(rw_tiles, tx_i32, fx.Int32(M_CONST), fx.Int32)
                        _scalar_store(st_tiles, tx_i32, fx.Int32(M_CONST), fx.Int32)
                        _scalar_store(wg_tiles, tx_i32, fx.Int32(0), fx.Int32)
                    gpu.barrier()
                    _mb = rocdl.readfirstlane(
                        T.i32, _raw(_global_i32_load(mi_tiles, fx.Int32(1)))
                    )
                    _hs["mbi"] = _mb
                    _hs["tot"] = _tot
                    _head_fg()
                    return _mb

            _xb_prev = None
            _xb_box = {}
            for _fj in range_constexpr(_NF):
                if const_expr(_NF > 1):
                    # fused tile -> real (m_block, n_block) tile index
                    _btile = (
                        _mbi * fx.Int32(_NUM_N_BLOCKS)
                        + _nbi * fx.Int32(_NF)
                        + fx.Int32(_fj)
                    )
                else:
                    _btile = _tile
                if const_expr(_NF > 1 and _fj > 0):
                    # previous n-block's epilogue still reading lds_acc, which
                    # the next n-block's prologue overwrites as s_aq.
                    gpu.barrier()
                _gemm1_body_fold(
                    lds_raw_ptr,
                    sort_ptr,
                    fold_e,
                    arg_aq,
                    arg_ascale,
                    arg_bq,
                    arg_bscale,
                    arg_eids,
                    arg_mind,
                    arg_aqout,
                    arg_ascaleout,
                    arg_hidden,
                    _btile,
                    lane,
                    wave,
                    (_WAUX if _WAUX else use_nt),
                    i32_ntok,
                    total_m_blocks,
                    arg_part,
                    arg_flag,
                    _skid,
                    SK=_SK,
                    SK_ST_AUX=int(splitk_aux[0]),
                    SK_LD_AUX=int(splitk_aux[1]),
                    SK_P16=_SKP16,
                    SK_ROWS=_SKROWS,
                    SK_RD=_SKRD,
                    KROT=_KROT,
                    BM=BM,
                    BN=BN,
                    BK=BK,
                    inline_quant=inline_quant,
                    K=_K,
                    N_OUT=_N_OUT,
                    NE=_NE,
                    interleave=interleave,
                    kStages=kstages,
                    kAStages=_kas,
                    fold=_FOLD,
                    head_hook=(
                        (_head_item if _ITEM else _head_fg) if (_BEFN and _FOLD) else None
                    ),
                    xb_next=(
                        (_btile + fx.Int32(1)) if (_XBN and _fj + 1 < _NF) else None
                    ),
                    xb_pre=_xb_prev,
                    xb_out=_xb_box,
                    xb_dist=max(_XBN - 1, 0),
                    a_res=_XRN,
                    a_skip=bool(_XRN and _fj > 0),
                    krot_tile=(
                        (_mbi * fx.Int32(_NUM_N_BLOCKS) + _nbi * fx.Int32(_NF))
                        if _XRN else None
                    ),
                )
                _xb_prev = _xb_box.get("b") if (_XBN and _fj + 1 < _NF) else None

    @flyc.jit
    def launch_gemm1_fold(
        arg_aq: fx.Int64,
        arg_ascale: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_mind: fx.Int64,
        i32_ntok: fx.Int32,
        i32_grid: fx.Int32,
        arg_aqout: fx.Int64,
        arg_ascaleout: fx.Int64,
        arg_hidden: fx.Int64,
        arg_topk: fx.Int64,
        arg_topkw: fx.Int64,
        arg_stok: fx.Int64,
        arg_swgt: fx.Int64,
        arg_moebuf: fx.Int64,
        arg_part: fx.Int64,
        arg_flag: fx.Int64,
        stream: fx.Stream,
    ):
        grid_x = fx.Int64(i32_grid)
        gemm1_kernel_fold(
            arg_aq,
            arg_ascale,
            arg_bq,
            arg_bscale,
            arg_eids,
            arg_cumsum,
            arg_mind,
            i32_ntok,
            arg_aqout,
            arg_ascaleout,
            arg_hidden,
            arg_topk,
            arg_topkw,
            arg_stok,
            arg_swgt,
            arg_moebuf,
            arg_part,
            arg_flag,
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    # FlyDSL 0.3.2 keys a launcher's JIT cache on the scalar closure values it
    # reaches through gemm1_kernel_fold (M_CONST, xcd_swizzle, SK, ...), so
    # each specialization gets its own cache entry without renaming the launcher.
    return launch_gemm1_fold
