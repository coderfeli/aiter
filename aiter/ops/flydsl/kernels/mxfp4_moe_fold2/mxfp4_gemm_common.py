"""Shared MXFP4 GEMM helpers: pointer/buffer setup, LDS swizzle, e8m0 scales, tile sizes."""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from . import buffer_ops
from . import dpp_utils

kStages = 2
kBS_stride_k0_dw = 64


def _raw(v):
    if not isinstance(v, ir.Value) and hasattr(v, "ir_value"):
        return v.ir_value()
    return v


def _lds_ptr3(base_i32, byte_off_i32):
    ptr_ty = fx.PointerType.get(T.i8, fx.AddressSpace.Shared)
    return fx.to_llvm_ptr(fx.inttoptr(ptr_ty, fx.Int64(base_i32 + byte_off_i32)))


def _gep3(base_ptr, byte_off_i32):
    return buffer_ops.get_element_ptr(
        base_ptr, byte_offset=byte_off_i32, elem_type=T.i8
    )


def _global_base_ptr1(addr_i64):
    ptr_ty = fx.PointerType.get(T.i8, fx.AddressSpace.Global)
    return fx.to_llvm_ptr(fx.inttoptr(ptr_ty, fx.Int64(addr_i64)))


def _gep1(base_ptr, byte_off_i32):
    return buffer_ops.get_element_ptr(
        base_ptr, byte_offset=byte_off_i32, elem_type=T.i8
    )


def _global_ptr1(arg, byte_off_i32):
    return _gep1(_global_base_ptr1(arg), byte_off_i32)


def _buffer_rsrc(addr_i64, num_records_bytes):
    return buffer_ops.create_buffer_resource_from_addr(
        fx.Int64(addr_i64), num_records_bytes=num_records_bytes
    )


def _lds_swizzle_mask(row, row_bytes=128):
    """XOR16 swizzle for an FP4 LDS row of `row_bytes`; permutes its 16-byte columns."""
    assert row_bytes in (64, 128), f"unsupported FP4 LDS row width {row_bytes}"
    return (row & fx.Int32(2 * (row_bytes // 16) - 2)) << fx.Int32(3)


def _fabs_f32(x):
    return fmath.absf(x)


def _e8m0_roundup(amax_f32):
    wi = (amax_f32 * fx.Float32(1.0 / 6.0)).bitcast(fx.Int32)
    bexp = (wi + fx.Int32(0x7FFFFF)).shrui(fx.Int32(23)) & fx.Int32(0xFF)
    lt = fx.Uint32(bexp) < fx.Uint32(254)
    return lt.select(bexp, fx.Int32(254))


def _e8m0_from_amax(amax_f32):
    e8m0 = _e8m0_roundup(amax_f32)
    qscale = (e8m0 << fx.Int32(23)).bitcast(fx.Float32)
    return e8m0, qscale


def _umax_i32(a, b):
    is_gt = fx.Uint32(a) > fx.Uint32(b)
    return is_gt.select(a, b)


def _dpp_umax_step(a32, dpp_ctrl):
    swapped = dpp_utils.update_dpp_i32(a32, a32, dpp_ctrl, 0xF, 0xF, True)
    return _umax_i32(a32, fx.Int32(swapped))


def _inline_dpp_quad_amax(a32):
    return _dpp_umax_step(_dpp_umax_step(a32, 0xB1), 0x4E)


def k_half_for(k):
    return k // 2


def k_tiles_total_for(k, BK):
    return k // BK


def kunroll_for(k, BK):
    return k_tiles_total_for(k, BK) - kStages


def kas_c_k1_for(k):
    return (k // 32) // 4 // 2


def kbs_c_k1_for(k):
    return (k // 32) // 4 // 2


def kbs_stride_n0_dw_for(k):
    return kbs_c_k1_for(k) * 64


def kas_per_chunk_dw_for(k):
    return kas_c_k1_for(k) * 64


def num_n_blocks_for(n, BN):
    return n // BN


def kbs_c_n1_for(n):
    return n // 16 // 2


def kbs_per_expert_dw_for(n, k):
    return kbs_c_n1_for(n) * kbs_stride_n0_dw_for(k)


def bq_bytes_for(ne, n, k):
    return ne * n * k_half_for(k)


def bscale_bytes_for(ne, n, k):
    return ne * kbs_per_expert_dw_for(n, k) * 4


def kmchunks_for(BM):
    return BM // 16


def lds_acc_bytes_for(rows, BN):
    return rows * BN * 4
