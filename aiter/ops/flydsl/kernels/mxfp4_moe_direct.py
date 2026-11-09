# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Two-launch MXFP4 (a4w4) MoE for 1-2 tokens: no routing sort.

Stage 1 gives every (route, column block, K part) its own workgroup, reads the
expert from ``topk_ids``, quantizes the token's K part into LDS and stores one fp32
gate/up partial per K part; its waves can split the K part further and add their
partials in LDS in wave order, so stage 2 reads fewer partials. Stage 2 runs one
wave per route: it adds the partials in part order, applies SiLU * up, quantizes
to MXFP4 in registers, multiplies by w2 and adds the routing-weighted rows of a
token in LDS in route order. Weights are aiter's 16x16 preshuffle with
``e8m0_shuffle`` scales.
"""

from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels.tensor_shim import AITER_FLYDSL_KERNARG_PRELOAD

STAGE2_BN = 32
# Weights and scales are read once: stream them nt (cold-stream rate ~6.8 vs ~5.4 TB/s cached).
_NT = 2


def _raw(x):
    return x.ir_value() if hasattr(x, "ir_value") else x


def _p(p, dtype):
    return fx.recast_iter(dtype, p)


def _vec4(p, offset, aux=0):
    view = fx.make_view(_p(p, fx.Int32), fx.make_layout((4, 268435456), (1, 4)))
    buf = fx.rocdl.make_buffer_tensor(view)
    reg = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
    atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(aux), fx.Int32)
    fx.copy_atom_call(atom, fx.slice(buf, (None, offset >> 2)), reg)
    return reg.load()


def _bscalar(p, offset, dtype, aux=0):
    view = fx.make_view(_p(p, dtype), fx.make_layout((1, 1073741824), (1, 1)))
    buf = fx.rocdl.make_buffer_tensor(view)
    reg = fx.make_rmem_tensor(fx.make_layout(1, 1), dtype)
    atom = fx.make_copy_atom(fx.rocdl.BufferCopy(dtype.width, aux), dtype)
    fx.copy_atom_call(atom, fx.slice(buf, (None, offset)), reg)
    return reg.load()[0]


def _uniform(x):
    return fx.Int32(rocdl.readfirstlane(T.i32, _raw(x)))


def _scale_index(e, n, k, N, K):
    # Byte offset of the E8M0 scale of (expert e, row n, 32-group k) after e8m0_shuffle.
    G = K // 32
    return (
        e * N * G
        + (n >> 5) * (G * 32)
        + (k >> 3) * 256
        + (k & 3) * 64
        + (n & 15) * 4
        + ((k >> 2) & 1) * 2
        + ((n >> 4) & 1)
    )


def _weight_index(e, n, k, N, K):
    # Byte offset of row n's 32-group k after shuffle_weight(layout=(16, 16)).
    return e * (N * K // 8) + (n >> 4) * (K * 2) + k * 64 + (n & 15) * 4


def _e8m0_exp(peak):
    # ceil(log2(amax / 6)) as a biased exponent, without a rounded division.
    amax = peak.bitcast(fx.Float32)
    amax = (amax > fx.Float32(1e-10)).select(amax, fx.Float32(1e-10))
    bits = amax.bitcast(fx.Int32)
    return (
        ((bits >> 23) & 255)
        - 2
        + ((bits & 0x7FFFFF) > 0x400000).select(fx.Int32(1), fx.Int32(0))
    )


def _quant8(vals):
    # Four adjacent lanes cover one 32-value group.
    peak = fx.Int32(0)
    for j in range_constexpr(8):
        bits = vals[j].bitcast(fx.Int32) & fx.Int32(0x7FFFFFFF)
        peak = (fx.Uint32(peak) > fx.Uint32(bits)).select(peak, bits)
    for mask in (0x80B1, 0x804E):
        other = fx.Int32(rocdl.ds_swizzle(T.i32, _raw(peak), _raw(fx.Int32(mask))))
        peak = (fx.Uint32(peak) > fx.Uint32(other)).select(peak, other)
    exp = _e8m0_exp(peak)
    scale = (exp << 23).bitcast(fx.Float32)
    q = fx.Int32(0)
    for j in range_constexpr(4):
        q = fx.Int32(
            rocdl.cvt_scalef32_pk_fp4_f32(
                T.i32, _raw(q), _raw(vals[2 * j]), _raw(vals[2 * j + 1]), _raw(scale), j
            )
        )
    return q, exp


def _quant4(vals):
    # Eight adjacent lanes cover one 32-value group; even lanes return its dwords.
    peak = fx.Int32(0)
    for j in range_constexpr(4):
        bits = vals[j].bitcast(fx.Int32) & fx.Int32(0x7FFFFFFF)
        peak = (fx.Uint32(bits) > fx.Uint32(peak)).select(bits, peak)
    for mask in (0x80B1, 0x804E):
        other = fx.Int32(rocdl.ds_swizzle(T.i32, _raw(peak), _raw(fx.Int32(mask))))
        peak = (fx.Uint32(other) > fx.Uint32(peak)).select(other, peak)
    lane = fx.Int32(fx.gpu.thread_idx.x) % 64
    other = fx.Int32(rocdl.ds_bpermute(T.i32, _raw((lane ^ 4) * 4), _raw(peak)))
    peak = (fx.Uint32(other) > fx.Uint32(peak)).select(other, peak)
    exp = _e8m0_exp(peak)
    scale = (exp << 23).bitcast(fx.Float32)
    q = fx.Int32(0)
    for j in range_constexpr(2):
        q = fx.Int32(
            rocdl.cvt_scalef32_pk_fp4_f32(
                T.i32, _raw(q), _raw(vals[2 * j]), _raw(vals[2 * j + 1]), _raw(scale), j
            )
        )
    other_q = fx.Int32(rocdl.ds_swizzle(T.i32, _raw(q), _raw(fx.Int32(0x80B1))))
    return q | (other_q << 16), exp


@cache
def build_mxfp4_moe_direct_module(
    tokens, model_dim, inter_dim, topk, split, k_waves=1, cols=64, bn=STAGE2_BN
):
    D, I, TOP, SPLIT, KW = model_dim, inter_dim, topk, split, k_waves
    R = tokens * topk
    BN = bn
    NW = cols // 16
    THREADS = NW * KW * 64
    if not (I % 128 == 0 and I <= 256 and D % (128 * SPLIT * KW) == 0 and D % BN == 0):
        raise ValueError(
            f"mxfp4 direct MoE: unsupported D={D} I={I} split={SPLIT} k_waves={KW}"
        )
    if not (cols % 16 == 0 and (I * 2) % cols == 0 and THREADS <= 1024):
        raise ValueError(f"mxfp4 direct MoE: unsupported cols={cols} k_waves={KW}")
    if TOP * 64 > 1024:
        raise ValueError(f"mxfp4 direct MoE: topk={TOP} exceeds one workgroup")
    DK = D // SPLIT
    WK = DK // KW

    @flyc.kernel(name=f"mxfp4_moe_direct_s1_d{D}_i{I}_r{R}_k{TOP}_s{SPLIT}_w{KW}_c{cols}")
    def stage1(
        X: fx.Pointer,
        W: fx.Pointer,
        WS: fx.Pointer,
        IDS: fx.Pointer,
        H: fx.Pointer,
    ):
        tid = fx.Int32(fx.gpu.thread_idx.x)
        lane = tid % 64
        wave = tid // 64
        kw = wave // NW
        n = fx.Int32(fx.gpu.block_idx.x) * cols + (wave % NW) * 16 + lane % 16
        r = fx.Int32(fx.gpu.block_idx.y)
        part = fx.Int32(fx.gpu.block_idx.z)
        e = _uniform(_p(IDS, fx.Int32)[r])
        token = fx.Int32(fx.Uint32(r) // fx.Uint32(TOP))
        shared = fx.SharedAllocator()
        local_q = shared.allocate(fx.Array[fx.Int32, DK // 8]).peek()
        local_s = shared.allocate(fx.Array[fx.Int32, DK // 32]).peek()
        partials = shared.allocate(fx.Array[fx.Float32, KW * cols]).peek()
        # Quantize this K part of the token once for all waves.
        for c in range_constexpr((DK // 8 + THREADS - 1) // THREADS):
            q_tid = tid + c * THREADS
            if q_tid < DK // 8:
                xbase = token * D + part * DK + q_tid * 8
                vals = [fx.Float32(_p(X, fx.BFloat16)[xbase + j]) for j in range_constexpr(8)]
                q, s = _quant8(vals)
                local_q[q_tid] = q
                if q_tid % 4 == 0:
                    local_s[q_tid // 4] = s
        fx.gpu.barrier()
        acc = fx.Vector.filled(4, 0.0, fx.Float32)
        for it in range(WK // 128):
            lk = kw * (WK // 32) + it * 4 + lane // 16
            k = part * (DK // 32) + lk
            av = fx.Vector.from_elements(
                [local_q[lk * 4 + j] for j in range_constexpr(4)], fx.Int32
            )
            sa = local_s[lk]
            bv = _vec4(W, _weight_index(e, n, k, I * 2, D), _NT)
            sb = fx.Int32(_bscalar(WS, _scale_index(e, n, k, I * 2, D), fx.Uint8, _NT))
            acc = fx.Vector(
                rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                    T.f32x4, [av, bv, acc, 4, 4, 0, sa, 0, sb]
                )
            )
        # Every MFMA row is the same token: keep row 0.
        if const_expr(KW == 1):
            if lane < 16:
                _p(H, fx.Float32)[(part * R + r) * (I * 2) + n] = acc[0]
        else:
            if lane < 16:
                partials[kw * cols + (wave % NW) * 16 + lane] = acc[0]
            fx.gpu.barrier()
            # Add the wave partials in K order, as stage 2 adds the K parts.
            if tid < cols:
                total = fx.Float32(0.0)
                for w in range_constexpr(KW):
                    total = total + partials[w * cols + tid]
                _p(H, fx.Float32)[
                    (part * R + r) * (I * 2) + fx.Int32(fx.gpu.block_idx.x) * cols + tid
                ] = total

    @flyc.kernel(name=f"mxfp4_moe_direct_s2_d{D}_i{I}_r{R}_k{TOP}_s{SPLIT}_b{BN}")
    def stage2(
        W: fx.Pointer,
        WS: fx.Pointer,
        IDS: fx.Pointer,
        RW: fx.Pointer,
        OUT: fx.Pointer,
        H: fx.Pointer,
    ):
        tid = fx.Int32(fx.gpu.thread_idx.x)
        lane = tid % 64
        wave = tid // 64
        token = fx.Int32(fx.gpu.block_idx.y)
        r = token * TOP + wave
        n = fx.Int32(fx.gpu.block_idx.x) * BN + lane % 16
        e = _uniform(_p(IDS, fx.Int32)[r])
        partials = fx.SharedAllocator().allocate(fx.Array[fx.Float32, TOP * BN]).peek()
        packed = fx.Int32(0)
        exponent = fx.Int32(0)
        if lane < I // 4:
            # Add the K parts in order, SiLU(gate) * up, round to bf16, MXFP4.
            # Keep "+ j" outermost so the four loads of a part vectorize.
            c = lane * 4
            vals = []
            for j in range_constexpr(4):
                g = fx.Float32(0.0)
                u = fx.Float32(0.0)
                for s in range_constexpr(SPLIT):
                    g = g + _p(H, fx.Float32)[(s * R + r) * (2 * I) + c + j]
                    u = u + _p(H, fx.Float32)[(s * R + r) * (2 * I) + I + c + j]
                ep = fx.Float32(
                    rocdl.exp2(T.f32, _raw(g * fx.Float32(-1.4426950408889634)))
                )
                sig = fx.Float32(rocdl.rcp(T.f32, _raw(fx.Float32(1.0) + ep)))
                vals.append(fx.Float32(fx.BFloat16(g * sig * u)))
            packed, exponent = _quant4(vals)
        accs = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range_constexpr(BN // 16)]
        for it in range_constexpr(I // 128):
            k = fx.Int32(it * 4) + lane // 16
            # Move the wave's packed dwords into MFMA operand order.
            av = fx.Vector.from_elements(
                [
                    fx.Int32(
                        rocdl.ds_bpermute(T.i32, _raw((k * 4 + j) * 8), _raw(packed))
                    )
                    for j in range_constexpr(4)
                ],
                fx.Int32,
            )
            sa = fx.Int32(rocdl.ds_bpermute(T.i32, _raw(k * 32), _raw(exponent)))
            for b in range_constexpr(BN // 16):
                bv = _vec4(W, _weight_index(e, n + b * 16, k, D, I), _NT)
                sb = fx.Int32(
                    _bscalar(WS, _scale_index(e, n + b * 16, k, D, I), fx.Uint8, _NT)
                )
                accs[b] = fx.Vector(
                    rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        T.f32x4, [av, bv, accs[b], 4, 4, 0, sa, 0, sb]
                    )
                )
        if lane < 16:
            for b in range_constexpr(BN // 16):
                partials[wave * BN + b * 16 + lane] = accs[b][0]
        fx.gpu.barrier()
        # Routing-weighted sum of the token's routes, in route order.
        if tid < BN:
            total = fx.Float32(0.0)
            for j in range_constexpr(TOP):
                total = total + partials[j * BN + tid] * _p(RW, fx.Float32)[token * TOP + j]
            _p(OUT, fx.BFloat16)[
                token * D + fx.Int32(fx.gpu.block_idx.x) * BN + tid
            ] = fx.BFloat16(total)

    @flyc.jit
    def launch(
        X: fx.Pointer,
        W1: fx.Pointer,
        WS1: fx.Pointer,
        W2: fx.Pointer,
        WS2: fx.Pointer,
        IDS: fx.Pointer,
        RW: fx.Pointer,
        H: fx.Pointer,
        OUT: fx.Pointer,
        stream: fx.Stream,
    ):
        stage1(X, W1, WS1, IDS, H).launch(
            grid=(I * 2 // cols, R, SPLIT), block=(THREADS,), stream=stream
        )
        stage2(W2, WS2, IDS, RW, OUT, H).launch(
            grid=(D // BN, tokens), block=(TOP * 64,), stream=stream
        )

    launch.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": 6,
        }
    }
    return launch
