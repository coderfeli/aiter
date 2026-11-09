# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from .. import dpp_utils
from ..layout_utils import crd2idx
from ..mxfp4_gemm_common import (
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
# PROLOGUE FOLD helpers
# ---------------------------------------------------------------------------
# These implement, inside stage-1's own block head, the work that used to be a
# separate `moe_sort_quant` graph node: the 257-expert padded sort and the
# [M, D_HIDDEN] bf16 moe_buf zero-init.  Every stage-1 CTA recomputes the whole
# sort redundantly (~400 VALU ops over M*TOPK <= 378 items) so that NO
# inter-CTA communication and NO grid-wide barrier is needed.

# Hidden(A) global loads are issued _HPF k-loop bodies ahead of the body that
# quantises them, so the per-body vmcnt wait covers more B loads in flight.
_HPF = 2
# The first _BEF B/B-scale prologue stages (capped at kStages) are issued as
# soon as the block's expert is known, before the LDS-only rank/publish head
# steps, so HBM is not idle while the sort finishes.
_BEF = 3

# sort-scratch LDS layout, in i32 words
_W_CNT = 0      # 272: LDS histogram, tokens per expert
_W_WT = 272     # 16 : wave totals + [8] = expert-256 block count
_W_MISC = 288   # 16 : [0] = this block's expert, [1] = block index within it
_W_ROWS = 304   # 16 : m_indices for this block's 16 rows
_W_STOK = 320   # 16 : sorted_token_ids for this block
_W_WGT = 336    # 16 : sorted_weights (f32 bits) for this block
_W_TK = 352     # NREP*256: unused tail (former topk_ids cache); sizes the region


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


def gemm1_grid(n_tokens, BM, *, NE, TOPK, INTER, BN=256):
    num_n_blocks = num_n_blocks_for(n_out_for(INTER), BN)
    active = min(n_tokens * TOPK, NE)
    max_m_blocks = (n_tokens * TOPK + active * (BM - 1) + BM - 1) // BM
    return max_m_blocks * num_n_blocks


@flyc.jit
def _gemm1_body_fold(
    lds_raw_ptr,
    sort_lds_ptr,
    fold_e,
    arg_bq,
    arg_bscale,
    arg_aqout,
    arg_ascaleout,
    arg_hidden,
    bx_i32,
    lane,
    wave,
    i32_ntok,
    *,
    BM,
    BN,
    BK,
    K,
    N_OUT,
    NE,
    kStages=2,
    kAStages,
    head_hook,
):
    KH_TILE = BK // 2
    K_HALF = k_half_for(K)
    KTP = k_tiles_total_for(K, BK)
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
    kAStages, kSubBlocks, kMChunks, _lds_bytes = _bm_constants(
        BM, BN, KH_TILE, KTP, kAStages
    )

    BN_INT = BN // 2
    # NJ = 16-col B tiles per wave (BN=256 -> 4: 2 gate + 2 up; BN=128 -> 2).
    NJ = BN // 64
    W8 = BN // 8          # gate columns owned by one mfma wave
    NTH = (BN // 2) // 8  # epilogue threads per output row
    b_aux = 2  # nt weight stream
    M_REPS = BM // 16

    n_block_idx = bx_i32 % fx.Int32(NUM_N_BLOCKS)
    m_block_idx = bx_i32 // fx.Int32(NUM_N_BLOCKS)
    e = fold_e
    m_row = m_block_idx * fx.Int32(BM)

    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)

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

    hidden_num = fx.Int64(i32_ntok * fx.Int32(K * 2))
    hidden_tiles = _global_i32_buffer_tiles(arg_hidden, hidden_num, 4)
    hidden_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(0), fx.Int32)
    hidden_reg_lay = fx.make_layout(4, 1)

    rcls = wave * fx.Int32(4) + lane_div_16

    # -- b_load_s_base[j] (HIP 412-416), readfirstlane'd uniform per wave ------
    N0_HALF = N_OUT // 32

    b_load_s_base = []
    for j in range_constexpr(NJ):
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
        base = rocdl.readfirstlane(T.i32, base)
        b_scale_s_base.append(base)
        b_scale_s_base_hi.append(base + fx.Int32(16 * kBS_stride_k0_dw * 4))

    accm = [[None] * NJ for _ in range(kMChunks)]
    b = [[[None, None] for _ in range(NJ)] for _ in range(kStages)]
    b_scale_v = [[None, None] for _ in range(kStages)]

    # Union LDS region [s_aq | s_asc], reused as lds_acc (f32 accumulator) in
    # the epilogue. s_aq/lds_acc start at lds_raw_ptr; s_asc follows at
    # +kAStages*BM*KH_TILE.

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

    # s_asc as flat i32.
    s_asc_i32_flat = fx.make_view(
        fx.recast_iter(fx.Int32, fx.add_offset(lds_raw_ptr, kAStages * BM * KH_TILE)),
        fx.make_layout(kSubBlocks * KTP * 64, 1),
    )
    asc_i32_tiles = fx.logical_divide(s_asc_i32_flat, fx.make_layout(1, 1))

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
        _so = fx.Int32(kt * (BK * 2) + B128_IDX * 256)
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

    def inline_quant_pack_write(kt, scale_accum):
        lane_tgt = lane_shr2_and3 * fx.Int32(16) + r_in_chunk
        off = fx.Int32(kt * 256) + lane_tgt * fx.Int32(4)
        _scalar_store(asc_i32_tiles, off // fx.Int32(4), scale_accum, fx.Int32)

    def issue_b_load_j(b_slot, K_C, j):
        v = (lane_div_16 * fx.Int32(256)) + (lane_mod_16 * fx.Int32(16))
        v = v + fx.Int32(K_C * 2048)
        s_b = b_load_s_base[j]
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

    def issue_b_scale_load(bs_slot, K_C):
        v = ((lane_div_16 * fx.Int32(16)) + lane_mod_16) * fx.Int32(4)
        K_C_HI = K_C // 16
        imm = (K_C - K_C_HI * 16) * (kBS_stride_k0_dw * 4)
        for mw in range_constexpr(2):
            s_off = b_scale_s_base[mw] if K_C_HI == 0 else b_scale_s_base_hi[mw]
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
        mni = J % 2
        in_b = J // 2
        sb = bs_slot[mni]
        bJ0, bJ1 = b_slot[J][0], b_slot[J][1]
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

    # Hidden(A) prefetch depth: the hidden load for k-tile K_C is issued _HPD
    # loop bodies ahead of the body that quantises it.
    _HPD = min(int(_HPF), kUnroll - 1)
    _bef = min(int(_BEF), kStages)
    # B early fetch: B only needs fold_e, which the head resolved in step (e);
    # start the bandwidth-bound B stream now and run the LDS-only rank/publish
    # steps (head_hook) under its latency.
    for _kc in range_constexpr(_bef):
        for _j in range_constexpr(NJ):
            issue_b_load_j(b[_kc], _kc, _j)
        issue_b_scale_load(b_scale_v[_kc], _kc)
    rocdl.sched_barrier(0)
    head_hook()
    rocdl.sched_barrier(0)
    # The 16 gather rows for THIS m-block were computed by the head hook and
    # live in the private sort-scratch LDS region.
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

    for OFFSET in range_constexpr(kUnroll):
        K_C = kStages + OFFSET
        read_slot = OFFSET % kAStages
        write_slot = K_C % kAStages
        slot_b = OFFSET % kStages
        gpu.barrier()
        a_cur = issue_a_ds_read(read_slot)
        asc_cur = issue_a_scale_ds_read(K_C - kStages)
        h_v0, h_v1 = h_pend[0][0], h_pend[0][1]
        h_pend = h_pend[1:]
        if const_expr(OFFSET + _HPD < kUnroll):
            h_pend = h_pend + [
                [
                    inline_quant_load_kt(0, K_C + _HPD, cached_row_inline),
                    inline_quant_load_kt(1, K_C + _HPD, cached_row_inline),
                ]
            ]
        rocdl.sched_barrier(0)
        for J in range_constexpr(NJ):
            rocdl.sched_barrier(0)
            rocdl.s_setprio(1)
            mfma_cluster(
                b[slot_b], a_cur, asc_cur, b_scale_v[slot_b], J, init=(OFFSET == 0)
            )
            rocdl.s_setprio(0)
            rocdl.sched_barrier(0)
            issue_b_load_j(b[slot_b], K_C, J)
            rocdl.sched_barrier(0)
        issue_b_scale_load(b_scale_v[slot_b], K_C)
        scale_accum = _inline_quant_core_batch(
            [(0, 0, h_v0), (1, 0, h_v1)], write_slot, fx.Int32(0)
        )
        inline_quant_pack_write(K_C, scale_accum)

    for S in range_constexpr(kStages):
        kt = KTP - kStages + S
        gpu.barrier()
        a_cur = issue_a_ds_read(kt % kAStages)
        asc_cur = issue_a_scale_ds_read(kt)
        for J in range_constexpr(NJ):
            mfma_cluster(
                b[kt % kStages], a_cur, asc_cur, b_scale_v[kt % kStages], J, init=False
            )

    gpu.barrier()

    # lds_acc reuses the s_aq region (offset 0) as an f32 accumulator.
    acc_layout = fx.make_layout((BM, BN), (BN, 1))

    def acc_idx(row, col):
        return _layout_idx(acc_layout, row, col)

    acc_copy_atom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
    acc_reg_lay = fx.make_layout(1, 1)
    acc_flat_view = fx.make_view(
        fx.recast_iter(fx.Float32, lds_raw_ptr),
        fx.make_layout(BM * BN, 1),
    )
    acc_flat_tiles = fx.logical_divide(acc_flat_view, fx.make_layout(1, 1))

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

    _acc_stage(lambda i, J: fx.Vector(accm[i][J]))
    gpu.barrier()

    tx_i32 = fx.Int32(gpu.thread_id("x"))

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
            chunk = m_block_idx
            dword_off = _layout_idx(ascaleout_layout, chunk, ku, wg_old, m_lane)
            addr = dword_off * fx.Int32(4) + ikxdl * fx.Int32(2)
            _scalar_store(ascaleout_i8_tiles, addr, scales_per_mr[0], fx.Int8)

    if const_expr(BM * NTH < 256):
        if tx_i32 < fx.Int32(BM * NTH):
            _epilogue_tail()
    else:
        _epilogue_tail()


def _bm_constants(BM, BN, KH_TILE, K_TILES_TOTAL, kAStages):
    kSubBlocks = 1 if BM < 32 else BM // 32
    kMChunks = kmchunks_for(BM)
    s_aq_bytes = kAStages * BM * KH_TILE
    s_asc_bytes = kSubBlocks * K_TILES_TOTAL * 256
    lds_acc_bytes = lds_acc_bytes_for(BM, BN)
    lds_bytes = max(s_aq_bytes + s_asc_bytes, lds_acc_bytes)
    return kAStages, kSubBlocks, kMChunks, lds_bytes


def compile_gemm1_a4w4_fold(
    *,
    D_HIDDEN,
    D_INTER,
    NE,
    TOPK,
    BN=256,
    xcd_swizzle=0,
    M_CONST,
):
    BM = 16
    BK = 256
    kstages = 2
    assert BN in (128, 256), f"only BN in (128,256), got BN={BN}"
    KH_TILE = BK // 2
    _K = D_HIDDEN
    assert _K % BK == 0, f"D_HIDDEN (K) must be a multiple of {BK}, got {_K}"
    _INTER = D_INTER
    _N_OUT = n_out_for(_INTER)
    assert (
        _N_OUT % BN == 0
    ), f"2*D_INTER (N_OUT) must be a multiple of {BN}, got {_N_OUT}"
    _NE = NE
    _KTP = k_tiles_total_for(_K, BK)
    _NNB = num_n_blocks_for(_N_OUT, BN)

    _kas = 3
    assert kstages < _KTP, "kStages must be < k-tiles"
    _, _, _, lds_bytes = _bm_constants(BM, BN, KH_TILE, _KTP, _kas)

    # Tag with H/INTER/NE so different shape specializations get distinct
    # kernel/smem symbols (so KIMI and non-KIMI instances never collide).
    name_suffix = f"h{_K}_i{_INTER}_ne{_NE}_bm{BM}_bn{BN}_iq_sep_ks{kstages}_as{_kas}"
    name_suffix += f"_bsa2_hpf{_HPF}"
    if xcd_swizzle > 0:
        name_suffix += f"_xcd{xcd_swizzle}"
    name_suffix += f"_tw_bef{min(_BEF, kstages)}_hd2_kap"

    # ---- prologue-fold compile constants -------------------------------
    _TOPK = TOPK
    assert M_CONST > 0, "fold needs the ACTUAL M as a compile constant"
    name_suffix += f"_fold{M_CONST}"
    _MT = int(M_CONST) * int(TOPK)
    _NREP = (_MT + 255) // 256
    _active = min(_MT, _NE)
    _max_sorted = ((_MT + _active * (BM - 1)) + BM - 1) // BM * BM
    _NB_MAX = _max_sorted // BM
    _GRID = _NB_MAX * _NNB
    _NDW = int(M_CONST) * (_K // 2)         # bf16 [M, K] as i32 words
    _NREP_Z = (_NDW + _GRID * 256 - 1) // (_GRID * 256)
    sort_bytes = _sort_words(_MT) * 4

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
        # Kernarg pin: LLVM would otherwise sink the s_loads of arg_bq/arg_bscale/
        # ntok (and the unused arg_part/arg_flag) into the `_pid < bound` block,
        # each with an s_waitcnt lgkmcnt(0) on the path to the first B / hidden
        # load.  An empty side-effecting asm consuming them at entry keeps every
        # kernarg s_load in the entry batch, in parallel with the topk path.
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
        cnt_tiles = _lds_i32_tiles(sort_ptr, _W_CNT, 272)
        wt_tiles = _lds_i32_tiles(sort_ptr, _W_WT, 16)
        mi_tiles = _lds_i32_tiles(sort_ptr, _W_MISC, 16)
        rw_tiles = _lds_i32_tiles(sort_ptr, _W_ROWS, 16)
        st_tiles = _lds_i32_tiles(sort_ptr, _W_STOK, 16)
        wg_tiles = _lds_i32_tiles(sort_ptr, _W_WGT, 16)

        # ---- (b) topk_ids / topk_weights loads FIRST: gfx9 vmcnt is in-order
        # and counts stores, so issuing them before the moe_buf zero-init keeps
        # the store acks out of the topk wait.  Each thread only ever reads
        # back its own items, so they stay in VGPRs (pad slots read as -1).
        _tkw_regs = []
        _tk_regs = []
        for _r in range_constexpr(_NREP):
            _i = tx_i32 + fx.Int32(_r * 256)
            _m = _ashr(_i - fx.Int32(_MT), 31)      # -1 iff _i < MT
            _tk_regs.append((_global_i32_at(arg_topk, _i & _m), _m))
            _tkw_regs.append(_global_i32_at(arg_topkw, _i & _m))
        # (a) moe_buf zero-init, striped over the whole grid, issued behind
        # the topk loads
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
        gpu.barrier()
        # (c) per-expert token counts: LDS histogram, O(1)/thread
        for _r in range_constexpr(_NREP):
            _i = tx_i32 + fx.Int32(_r * 256)
            if _i < fx.Int32(_MT):
                _lds_atomic_add(cnt_tiles, _tk_regs[_r], 1)
        gpu.barrier()

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
        fold_nb_a, fold_excl_a, fold_t256, total_m_blocks = (
            _nb_a, _excl, _t256, _t256 + _nb256
        )
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

        _pid = bx_i32
        if fx.Int32(_pid) < bound:
            if const_expr(_SW > 0):
                _tile = _xcd(_pid)
            else:
                _tile = _pid
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

            def _head_fg():
                # (f) rank the items of this expert and keep rows [lo, lo+16)
                _base = fx.Int32(0)
                for _r in range_constexpr(_NREP):
                    _i = tx_i32 + fx.Int32(_r * 256)
                    _v = _tk_regs[_r]
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
                            _scalar_store(wg_tiles, _d, _tkw_regs[_r], fx.Int32)
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

            _gemm1_body_fold(
                lds_raw_ptr,
                sort_ptr,
                fold_e,
                arg_bq,
                arg_bscale,
                arg_aqout,
                arg_ascaleout,
                arg_hidden,
                _tile,
                lane,
                wave,
                i32_ntok,
                BM=BM,
                BN=BN,
                BK=BK,
                K=_K,
                N_OUT=_N_OUT,
                NE=_NE,
                kStages=kstages,
                kAStages=_kas,
                head_hook=_head_fg,
            )

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
    # reaches through gemm1_kernel_fold (M_CONST, xcd_swizzle, BN, ...), so
    # each specialization gets its own cache entry without renaming the launcher.
    return launch_gemm1_fold
