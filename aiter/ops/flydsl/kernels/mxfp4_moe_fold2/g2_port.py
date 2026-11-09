# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .. import buffer_ops

from ..mxfp4_gemm_common import (
    _buffer_rsrc,
    _gep1,
    _gep3,
    _global_base_ptr1,
    _global_ptr1,
    _lds_ptr3,
    _lds_swizzle_mask,
    _raw,
    bq_bytes_for,
    bscale_bytes_for,
    k_half_for,
    k_tiles_total_for,
    kas_per_chunk_dw_for,
    kbs_per_expert_dw_for,
    kBS_stride_k0_dw,
    kbs_stride_n0_dw_for,
    kmchunks_for,
    kStages,
    kunroll_for,
    lds_acc_bytes_for,
    num_n_blocks_for,
)


def saq_slot_bytes(BM, KH_TILE):
    return BM * KH_TILE


def tiling(BM):
    n_load_waves = min(4, BM // 8)
    rows_per_wave = BM // n_load_waves
    return n_load_waves, rows_per_wave, rows_per_wave // 8


def _udiv(a, c):
    cc = fx.Int32(c) if isinstance(c, int) else c
    return fx.Int32(arith.divui(_raw(a), _raw(cc)))


def _issue_a_load_lds(
    aq_rsrc, saq_base_i32, slot, kt, car, lane, slot_bytes, lds_row, KH_TILE, k_half
):
    lane_mod_8 = lane % fx.Int32(8)
    mask = _lds_swizzle_mask(lds_row + (lane // fx.Int32(8)))
    voffset = ((lane_mod_8 * fx.Int32(16)) ^ mask) + car * fx.Int32(k_half)
    off_i32 = fx.Int32(slot * slot_bytes) + lds_row * fx.Int32(KH_TILE)
    lds_ptr = _lds_ptr3(saq_base_i32, off_i32)
    rocdl.raw_ptr_buffer_load_lds(
        aq_rsrc,
        lds_ptr,
        fx.Int32(16),
        voffset,
        fx.Int32(kt * KH_TILE),
        fx.Int32(0),
        fx.Int32(0),
    )


def compile_gemm2_a4w4_port(*, NE, N_OUT, D_INTER):
    BM = 16
    BN = 256
    BK = 256
    KH_TILE = BK // 2
    _K = D_INTER
    assert _K % BK == 0, (
        f"D_INTER (gemm2 contraction K = inter_dim) must be a multiple of {BK}, "
        f"got {_K}; inter_dim not divisible by {BK} (e.g. 384/192) is not "
        f"supported by this BK={BK} kernel"
    )
    _K_HALF = k_half_for(_K)
    _K_TILES_TOTAL = k_tiles_total_for(_K, BK)
    _slot_bytes = saq_slot_bytes(BM, KH_TILE)
    _aStages = kStages if _K_TILES_TOTAL <= kStages else 3
    _lds_bytes = lds_acc_bytes_for(BM, BN) + _aStages * _slot_bytes
    _num_n_blocks = num_n_blocks_for(N_OUT, BN)
    _n_load_waves, _rows_per_wave, _kSubBlocks = tiling(BM)
    _name = f"gemm2_a4w4_port_ne{NE}_h{N_OUT}_i{_K}_bm{BM}_nt_atomic_g2bs2_pro125"

    @fx.struct
    class SharedStorage:
        raw: fx.Array[fx.Uint8, _lds_bytes, 16]

    @flyc.kernel(name=_name, known_block_size=[256, 1, 1])
    def gemm2_kernel(
        arg_aq: fx.Int64,
        arg_ascale: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_stids: fx.Int64,
        arg_sweights: fx.Int64,
        i32_M: fx.Int32,
        i32_max_m_blocks: fx.Int32,
        arg_out: fx.Int64,
        arg_out_scale: fx.Int64,
    ):
        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        tx_i32 = fx.Int32(tx)
        bx_i32 = fx.Int32(bx)

        lane = tx_i32 % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tx_i32 // fx.Int32(64))

        aq_rsrc = _buffer_rsrc(arg_aq, fx.Index(0x7FFFFFF0))
        lds_raw_ptr = fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr
        saq_base_i32 = fx.Int32(fx.ptrtoint(lds_raw_ptr))

        def _issue_all_a_loads(m_row0):
            for slot in range_constexpr(min(kStages, _K_TILES_TOTAL)):
                for sub in range_constexpr(_kSubBlocks):
                    lds_row = wave * fx.Int32(_rows_per_wave) + fx.Int32(sub * 8)
                    car = m_row0 + lds_row + (lane // fx.Int32(8))
                    _issue_a_load_lds(
                        aq_rsrc,
                        saq_base_i32,
                        slot,
                        slot,
                        car,
                        lane,
                        _slot_bytes,
                        lds_row,
                        KH_TILE=KH_TILE,
                        k_half=_K_HALF,
                    )

        cumsum0 = llvm.load(T.i32, _global_ptr1(arg_cumsum, fx.Int32(0)))
        total_m_blocks = _udiv(cumsum0, BM)
        bound = total_m_blocks * fx.Int32(_num_n_blocks)

        if bx_i32 < bound:
            _sp_m, _sp_n = _g2_spart_tile(bx_i32, total_m_blocks, _num_n_blocks)
            _e_pre = llvm.load(
                T.i32,
                _global_ptr1(arg_eids, _sp_m * fx.Int32(4)),
                invariant=True,
            )
            _e_pre = rocdl.readfirstlane(T.i32, _e_pre)
            m_row0 = _sp_m * fx.Int32(BM)
            if wave < fx.Int32(_n_load_waves):
                _issue_all_a_loads(m_row0)
            rocdl.sched_barrier(0)
            llvm.InlineAsmOp(None, [_raw(_e_pre)], "", "s", has_side_effects=True)
            _gemm2_body_g2bs2_pro125(
                lds_raw_ptr,
                arg_ascale,
                arg_bq,
                arg_bscale,
                arg_stids,
                arg_sweights,
                i32_M,
                arg_out,
                _sp_m * fx.Int32(_num_n_blocks) + _sp_n,
                lane,
                wave,
                BM,
                NE,
                N_OUT,
                aq_rsrc=aq_rsrc,
                D_INTER=_K,
                aStages=_aStages,
                BN=BN,
                BK=BK,
                KH_TILE=KH_TILE,
                e_pre=_e_pre,
            )

    @flyc.jit
    def launch_gemm2_g2bs2_pro125(
        arg_aq: fx.Int64,
        arg_ascale: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_stids: fx.Int64,
        arg_sweights: fx.Int64,
        i32_M: fx.Int32,
        i32_max_m_blocks: fx.Int32,
        arg_out: fx.Int64,
        arg_out_scale: fx.Int64,
        stream: fx.Stream,
    ):
        grid_x = arith.index_cast(T.index, i32_max_m_blocks) * fx.Index(
            _num_n_blocks
        )
        gemm2_kernel(
            arg_aq,
            arg_ascale,
            arg_bq,
            arg_bscale,
            arg_eids,
            arg_cumsum,
            arg_stids,
            arg_sweights,
            i32_M,
            i32_max_m_blocks,
            arg_out,
            arg_out_scale,
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    launch_gemm2_g2bs2_pro125.compile_hints["llvm_options"] = {"enable-post-misched": False, "amdgpu-kernarg-preload-count": 16}

    return launch_gemm2_g2bs2_pro125


@flyc.jit
def _gemm2_body_g2bs2_pro125(
    lds_raw_ptr,
    arg_ascale,
    arg_bq,
    arg_bscale,
    arg_stids,
    arg_sweights,
    i32_M,
    arg_out,
    bx_i32,
    lane,
    wave,
    BM,
    NE,
    N_OUT,
    *,
    aq_rsrc,
    D_INTER,
    aStages=kStages,
    BN,
    BK,
    KH_TILE,
    e_pre,
):
    _aStages = aStages
    _kMChunks = kmchunks_for(BM)
    _slot_bytes = saq_slot_bytes(BM, KH_TILE)
    _K = D_INTER
    _K_HALF = k_half_for(_K)
    _K_TILES_TOTAL = k_tiles_total_for(_K, BK)
    _kUnroll = kunroll_for(_K, BK)
    _kAS_per_chunk_dw = kas_per_chunk_dw_for(_K)
    _kBS_stride_n0_dw = kbs_stride_n0_dw_for(_K)
    _bq_bytes = bq_bytes_for(NE, N_OUT, _K)
    _bscale_bytes = bscale_bytes_for(NE, N_OUT, _K)
    _kbs_per_expert_dw = kbs_per_expert_dw_for(N_OUT, _K)
    _num_n_blocks = num_n_blocks_for(N_OUT, BN)
    _n_load_waves, _rows_per_wave, _kSubBlocks = tiling(BM)
    b_aux = 2

    m_block_idx = _udiv(bx_i32, _num_n_blocks)
    n_block_idx = bx_i32 - m_block_idx * fx.Int32(_num_n_blocks)
    e = e_pre
    m_row = m_block_idx * fx.Int32(BM)

    ascale_rsrc = _buffer_rsrc(arg_ascale, fx.Index(0x7FFFFFF0))
    bq_rsrc = _buffer_rsrc(arg_bq, fx.Index(_bq_bytes))
    bscale_rsrc = _buffer_rsrc(arg_bscale, fx.Index(_bscale_bytes))

    # Sequential LDS layout: saq bytes at offset 0, f32 accumulator after them.
    saq_base_i32 = fx.Int32(fx.ptrtoint(lds_raw_ptr))
    lds_acc_base_i32 = saq_base_i32 + fx.Int32(_aStages * _slot_bytes)

    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)

    b_load_s_base = []
    for j in range_constexpr(4):
        v = (
            e * fx.Int32(N_OUT)
            + n_block_idx * fx.Int32(BN)
            + wave * fx.Int32(BN // 4)
            + fx.Int32(j * 16)
        ) * fx.Int32(_K_HALF)
        b_load_s_base.append(rocdl.readfirstlane(T.i32, v))

    mni_base = n_block_idx * fx.Int32(BN // 16 // 2) + wave * fx.Int32(BN // 64 // 2)
    b_scale_s_base = []
    for mw in range_constexpr(2):
        v = (
            e * fx.Int32(_kbs_per_expert_dw)
            + (mni_base + fx.Int32(mw)) * fx.Int32(_kBS_stride_n0_dw)
        ) * fx.Int32(4)
        b_scale_s_base.append(rocdl.readfirstlane(T.i32, v))

    chunk_base = m_row // fx.Int32(16)
    a_scale_s_base = [
        rocdl.readfirstlane(
            T.i32,
            (chunk_base + fx.Int32(sub)) * fx.Int32(_kAS_per_chunk_dw) * fx.Int32(4),
        )
        for sub in range_constexpr(_kSubBlocks)
    ]

    v_voff_scale = ((lane_div_16 * fx.Int32(16)) + lane_mod_16) * fx.Int32(4)

    def load_a_scale_tile(kt):
        out = [None] * _kSubBlocks
        for sub in range_constexpr(_kSubBlocks):
            out[sub] = buffer_ops.buffer_load(
                ascale_rsrc,
                (v_voff_scale + fx.Int32(kt * 256)) // fx.Int32(4),
                vec_width=1,
                dtype=T.i32,
                cache_modifier=0,
                soffset_bytes=a_scale_s_base[sub],
            )
        return out

    def load_b_scale_tile(kt):
        imm = kt * (kBS_stride_k0_dw * 4)
        out = [None, None]
        for mw in range_constexpr(2):
            out[mw] = buffer_ops.buffer_load(
                bscale_rsrc,
                (v_voff_scale + fx.Int32(imm)) // fx.Int32(4),
                vec_width=1,
                dtype=T.i32,
                cache_modifier=2,
                soffset_bytes=b_scale_s_base[mw],
            )
        return out

    def load_b_tile(kt):
        v_voff_b = (
            (lane_div_16 * fx.Int32(256))
            + (lane_mod_16 * fx.Int32(16))
            + fx.Int32(kt * 2048)
        )
        out = [[None, None] for _ in range(4)]
        for j in range_constexpr(4):
            for half in range_constexpr(2):
                frag = buffer_ops.buffer_load(
                    bq_rsrc,
                    (v_voff_b + fx.Int32(half * 1024)) // fx.Int32(4),
                    vec_width=4,
                    dtype=T.i32,
                    cache_modifier=b_aux,
                    soffset_bytes=b_load_s_base[j],
                )
                out[j][half] = Vec(frag)
        return out

    def issue_a_load_lds(slot, kt):
        for sub in range_constexpr(_kSubBlocks):
            lds_row = wave * fx.Int32(_rows_per_wave) + fx.Int32(sub * 8)
            car = m_row + lds_row + (lane // fx.Int32(8))
            _issue_a_load_lds(
                aq_rsrc,
                saq_base_i32,
                slot,
                kt,
                car,
                lane,
                _slot_bytes,
                lds_row,
                KH_TILE=KH_TILE,
                k_half=_K_HALF,
            )

    def issue_a_ds_read(slot):
        lane_row = lane_mod_16
        lane_col = lane_div_16 * fx.Int32(16)
        mask = _lds_swizzle_mask(lane_row)
        base_ptr = _lds_ptr3(saq_base_i32, fx.Int32(0))
        a = [[None, None] for _ in range(_kMChunks)]
        for k in range_constexpr(2):
            lds_col = (lane_col + fx.Int32(k * 64)) ^ mask
            for i in range_constexpr(_kMChunks):
                lds_row = lane_row + fx.Int32(i * 16)
                byte_off = (
                    fx.Int32(slot * _slot_bytes) + lds_row * fx.Int32(KH_TILE) + lds_col
                )
                a[i][k] = llvm.load(T.vec(4, T.i32), _gep3(base_ptr, byte_off))
        return a

    mfma_res_ty = T.f32x4
    zero4 = Vec.filled(4, 0.0, fx.Float32)
    accm = [[None, None, None, None] for _ in range(_kMChunks)]

    def mfma_cluster(b_tile, a, a_scale_sub, b_scale_slot, init):
        for J in range_constexpr(4):
            mni = J // 2
            in_b = J % 2
            sb = b_scale_slot[mni]
            b_J0 = b_tile[J][0]
            b_J1 = b_tile[J][1]
            for sub in range_constexpr(_kSubBlocks):
                sa = a_scale_sub[sub]
                i0 = sub * 2
                if const_expr(init):
                    accm[i0][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        mfma_res_ty, [a[i0][0], b_J0, zero4, 4, 4, 0, sa, 0 + in_b, sb]
                    )
                else:
                    accm[i0][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        mfma_res_ty,
                        [a[i0][0], b_J0, accm[i0][J], 4, 4, 0, sa, 0 + in_b, sb],
                    )
                accm[i0][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                    mfma_res_ty,
                    [a[i0][1], b_J1, accm[i0][J], 4, 4, 2, sa, 2 + in_b, sb],
                )

    def _kloop_fence():
        gpu.barrier()

    _epi_pre = None
    if const_expr(_K_TILES_TOTAL <= kStages):
        a_scale_v = [load_a_scale_tile(kt) for kt in range_constexpr(_K_TILES_TOTAL)]
        b_scale_v = [load_b_scale_tile(kt) for kt in range_constexpr(_K_TILES_TOTAL)]
        b = [load_b_tile(kt) for kt in range_constexpr(_K_TILES_TOTAL)]
        _epi_pre = _atomic_prefetch(arg_stids, arg_sweights, m_row, BM)
        for S in range_constexpr(_K_TILES_TOTAL):
            kt = S
            slot = kt % kStages
            _kloop_fence()
            a = issue_a_ds_read(slot)
            a_scale_sub = [a_scale_v[kt][sub] for sub in range_constexpr(_kSubBlocks)]
            mfma_cluster(b[slot], a, a_scale_sub, b_scale_v[slot], init=(S == 0))
    else:
        a_scale_v = [load_a_scale_tile(kt) for kt in range_constexpr(_K_TILES_TOTAL)]
        b_scale_v = [load_b_scale_tile(kt) for kt in range_constexpr(_K_TILES_TOTAL)]
        b = [load_b_tile(kt) for kt in range_constexpr(_K_TILES_TOTAL)]

        for OFFSET in range_constexpr(_kUnroll):
            kt = OFFSET
            slot = kt % _aStages
            next_kt = kStages + OFFSET
            write_slot = next_kt % _aStages
            _kloop_fence()
            a = issue_a_ds_read(slot)
            issue_a_load_lds(write_slot, next_kt)
            a_scale_sub = [a_scale_v[kt][sub] for sub in range_constexpr(_kSubBlocks)]
            mfma_cluster(b[kt], a, a_scale_sub, b_scale_v[kt], init=(OFFSET == 0))

        for S in range_constexpr(kStages):
            kt = _K_TILES_TOTAL - kStages + S
            slot = kt % _aStages
            _kloop_fence()
            a = issue_a_ds_read(slot)
            a_scale_sub = [a_scale_v[kt][sub] for sub in range_constexpr(_kSubBlocks)]
            mfma_cluster(b[kt], a, a_scale_sub, b_scale_v[kt], init=False)

    _atomic_bf16_epilog(
        lds_acc_base_i32,
        accm,
        arg_out,
        arg_stids,
        arg_sweights,
        m_row,
        n_block_idx,
        wave,
        lane,
        i32_M,
        BM,
        N_OUT,
        BN,
        pre=_epi_pre,
    )


def _atomic_prefetch(arg_stids, arg_sweights, m_row, BM):
    M_REPS = BM // 8
    m_lane = fx.Int32(gpu.thread_id("x")) // fx.Int32(32)
    stids_base = _global_base_ptr1(arg_stids)
    sweights_base = _global_base_ptr1(arg_sweights)
    packed = []
    weight = []
    for mr in range_constexpr(M_REPS):
        sorted_pos = m_row + fx.Int32(mr * 8) + m_lane
        packed.append(
            llvm.load(
                T.i32, _gep1(stids_base, sorted_pos * fx.Int32(4)), invariant=True
            )
        )
        weight.append(
            llvm.load(
                T.f32, _gep1(sweights_base, sorted_pos * fx.Int32(4)), invariant=True
            )
        )
    return packed, weight


@flyc.jit
def _atomic_bf16_epilog(
    lds_acc_base_i32,
    accm,
    arg_out,
    arg_stids,
    arg_sweights,
    m_row,
    n_block_idx,
    wave,
    lane,
    i32_M,
    BM,
    N_OUT,
    BN,
    pre=None,
):
    _kMChunks = kmchunks_for(BM)
    M_REPS = BM // 8
    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)
    lds_base = _lds_ptr3(lds_acc_base_i32, fx.Int32(0))

    tx_i32 = fx.Int32(gpu.thread_id("x"))
    m_lane = tx_i32 // fx.Int32(32)
    n_lane = tx_i32 % fx.Int32(32)
    col_start = n_lane * fx.Int32(2)
    out_base = _global_base_ptr1(arg_out)

    if const_expr(pre is not None):
        packed, weight = pre
    else:
        packed, weight = _atomic_prefetch(arg_stids, arg_sweights, m_row, BM)

    for i in range_constexpr(_kMChunks):
        row_base = fx.Int32(i * 16) + lane_div_16 * fx.Int32(4)
        for J in range_constexpr(4):
            col = wave * fx.Int32(64) + fx.Int32(J * 16) + lane_mod_16
            vec = Vec(accm[i][J])
            for v in range_constexpr(4):
                idx = (row_base + fx.Int32(v)) * fx.Int32(BN) + col
                llvm.StoreOp(_raw(vec[v]), _gep3(lds_base, idx * fx.Int32(4)))

    gpu.barrier()
    _pin_all = [_raw(w) for w in weight] + [_raw(p) for p in packed]
    llvm.InlineAsmOp(
        None, _pin_all, "", ",".join(["v"] * len(_pin_all)),
        has_side_effects=True,
    )

    _pk_all = []
    for mr in range_constexpr(M_REPS):
        row_in_block = fx.Int32(mr * 8) + m_lane
        _pk_row = []
        for s in range_constexpr(4):
            idx0 = row_in_block * fx.Int32(BN) + col_start + fx.Int32(s * 64)
            v2 = Vec(
                llvm.load(T.vec(2, T.f32), _gep3(lds_base, idx0 * fx.Int32(4)))
            )
            pk = Vec.from_elements(
                [v2[0] * weight[mr], v2[1] * weight[mr]], fx.Float32
            ).to(fx.BFloat16)
            _pk_row.append(pk)
        _pk_all.append(_pk_row)
    _pk_flat = [_raw(pk) for _r in _pk_all for pk in _r]
    llvm.InlineAsmOp(
        None, _pk_flat, "", ",".join(["v"] * len(_pk_flat)),
        has_side_effects=True,
    )

    for mr in range_constexpr(M_REPS):
        token_id = packed[mr] & fx.Int32(0x00FFFFFF)
        if token_id < i32_M:
            row_base_addr = (
                token_id * fx.Int32(N_OUT) + n_block_idx * fx.Int32(BN) + col_start
            )
            for s in range_constexpr(4):
                off = (row_base_addr + fx.Int32(s * 64)) * fx.Int32(2)
                out_ptr = _gep1(out_base, off)
                llvm.AtomicRMWOp(
                    llvm.AtomicBinOp.fadd,
                    out_ptr,
                    _raw(_pk_all[mr][s]),
                    llvm.AtomicOrdering.monotonic,
                    syncscope="agent",
                    alignment=4,
                )


def _g2_spart_tile(block_1d_id, M0, N0):
    gn = fx.Int32(4)
    n0 = fx.Int32(N0)
    m01c = fx.Int32(2)
    mn = M0 * n0
    group_size = _udiv(mn + gn - fx.Int32(1), gn)
    big_group_num = gn - (group_size * gn - mn)
    group_id_y = _udiv(block_1d_id, gn)
    group_id_x = block_1d_id - group_id_y * gn
    remap_a = group_id_x * group_size + group_id_y
    remap_b = group_id_x * group_size + big_group_num - group_id_x + group_id_y
    remap = (group_id_x <= big_group_num).select(remap_a, remap_b)
    idx_M0 = _udiv(remap, n0)
    idx_N0 = remap - idx_M0 * n0
    M0_tmp = _udiv(M0, m01c)
    M0_mod = M0 - M0_tmp * m01c
    M01_adapt = (idx_M0 < (M0 - M0_mod)).select(m01c, M0_mod)
    idx_M00 = _udiv(idx_M0, m01c)
    idx_M01 = idx_M0 - idx_M00 * m01c
    idx_local = idx_N0 + idx_M01 * n0
    N_out = _udiv(idx_local, M01_adapt)
    loc_mod = idx_local - N_out * M01_adapt
    return loc_mod + idx_M00 * m01c, N_out
