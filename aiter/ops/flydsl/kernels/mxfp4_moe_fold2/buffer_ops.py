# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""AMD buffer resource descriptors and buffer loads over raw device addresses.

Buffer instructions are an AMD hardware feature (buffer resource descriptor
plus ROCDL intrinsics) providing out-of-bounds protection and better memory
throughput. Only published flydsl APIs are used.
"""

from __future__ import annotations

from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as std_arith
from flydsl._mlir.dialects import llvm, rocdl
from flydsl._mlir.extras import types as T
from flydsl.expr.meta import dsl_loc_tracing
from flydsl.runtime.device import is_rdna_arch


def _get_buffer_flags(arch=None):
    """Get AMD buffer resource descriptor (V#) flags word (bits 127:96).

    CDNA (gfx9xx):    (7 << 12) | (4 << 15)                         = 0x20070
    RDNA (gfx10+):    (7 << 12) | (4 << 15) | (1 << 24) | (2 << 28) = 0x21020070
      - bit 24 set to 1 (required on RDNA)
      - OOB_SELECT=2 (no bounds checking, matching LLVM boundsCheck=false)

    With ``arch=None`` the arch comes from ``is_rdna_arch``'s own detection.
    """
    flags = (7 << 12) | (4 << 15)
    if is_rdna_arch(arch):
        flags |= 1 << 24  # reserved bit, must be 1 on RDNA
        flags |= 2 << 28  # OOB_SELECT = 2 (no bounds checking)
    return flags


def _unwrap_value(value):
    """Recursively unwrap ArithValue or similar wrappers to get the actual MLIR value.

    Handles:
    - FlyDSL ArithValue (has ._value)
    - flyc DSL Numeric like fx.Int32 (has .ir_value() method)
    - flyc ArithValue (is already ir.Value subclass)
    """
    # DSL Numeric (Int32, Float32, etc.) — use ir_value() to materialize
    if hasattr(value, "ir_value") and not isinstance(value, ir.Value):
        return value.ir_value()
    max_depth = 10  # Safety limit
    depth = 0
    while depth < max_depth and not isinstance(value, ir.Value):
        if hasattr(value, "_value"):
            value = value._value
        elif hasattr(value, "value"):
            value = value.value
        else:
            break
        depth += 1
    return value


@dsl_loc_tracing
def _create_i32_constant(value: int) -> ir.Value:
    """Create i32 constant using standard MLIR arith dialect."""
    i32_type = T.i32()
    if value > 0x7FFFFFFF:
        value = int(value - 2**32)
    attr = ir.IntegerAttr.get(i32_type, value)
    op = std_arith.ConstantOp(i32_type, attr)
    return _unwrap_value(op.result)


def _to_i32_offset(offset: ir.Value) -> ir.Value:
    """Normalize an already-unwrapped offset value to i32.

    Accepts index (index_cast), i32 (as-is), wider ints e.g. i64 (trunc), and
    narrower ints (sign-extend). Lets callers pass fx.Int64 offsets (element or
    byte) without hitting the ``index_cast i64<->i32`` incompatibility.
    """
    ot = offset.type
    if isinstance(ot, ir.IntegerType):
        if ot.width == 32:
            return offset
        if ot.width > 32:
            return _unwrap_value(std_arith.TruncIOp(T.i32(), offset).result)
        return _unwrap_value(std_arith.ExtSIOp(T.i32(), offset).result)
    return _unwrap_value(std_arith.IndexCastOp(T.i32(), offset).result)


@dsl_loc_tracing
def _create_i16_constant(value: int) -> ir.Value:
    """Create i16 constant using standard MLIR arith dialect."""
    i16_type = T.i16()
    attr = ir.IntegerAttr.get(i16_type, value)
    op = std_arith.ConstantOp(i16_type, attr)
    return _unwrap_value(op.result)


@dsl_loc_tracing
def _create_i64_constant(value: int) -> ir.Value:
    """Create i64 constant using standard MLIR arith dialect."""
    i64_type = T.i64()
    attr = ir.IntegerAttr.get(i64_type, value)
    op = std_arith.ConstantOp(i64_type, attr)
    return _unwrap_value(op.result)


@dsl_loc_tracing
def _ptr8_to_v4i32(ptr8_val) -> ir.Value:
    """Reinterpret a buffer resource (!llvm.ptr<8>) as a <4 x i32> vector.

    Required by the scalar ``s.buffer.load`` intrinsic, whose resource operand is
    a v4i32 rather than the opaque buffer pointer used by the vector path.
    """
    i128_ty = ir.IntegerType.get_signless(128)
    v4i32_ty = ir.VectorType.get([4], ir.IntegerType.get_signless(32))
    i128_val = llvm.ptrtoint(i128_ty, _unwrap_value(ptr8_val))
    return llvm.bitcast(v4i32_ty, i128_val)


@dsl_loc_tracing
def get_element_ptr(
    base_ptr,
    byte_offset: int | ir.Value | None = None,
    static_byte_offset: int = 0,
    elem_type: ir.Type | None = None,
    no_wrap_flags=None,
) -> ir.Value:
    """Build an LLVM GEP from a base pointer plus byte offsets."""
    _gep_dynamic_index_sentinel = -(2**31)

    base_ptr = _unwrap_value(base_ptr)
    if not isinstance(static_byte_offset, int):
        raise TypeError(
            f"static_byte_offset must be int, got {type(static_byte_offset)!r}"
        )
    if elem_type is None:
        elem_type = T.i8()
    elif callable(elem_type):
        elem_type = elem_type()

    if byte_offset is None:
        dynamic_indices = []
        raw_constant_indices = [int(static_byte_offset)]
    elif isinstance(byte_offset, int):
        dynamic_indices = []
        raw_constant_indices = [int(byte_offset) + int(static_byte_offset)]
    else:
        offset_val = _unwrap_value(byte_offset)
        if isinstance(offset_val.type, ir.IndexType):
            i64_type = T.i64()
            offset_val = _unwrap_value(
                std_arith.IndexCastOp(i64_type, offset_val).result
            )
        elif not isinstance(offset_val.type, ir.IntegerType):
            raise TypeError(
                "byte_offset must be int, index, or integer-typed MLIR value; "
                f"got {offset_val.type}"
            )

        if static_byte_offset != 0:
            static_type = offset_val.type
            static_attr = ir.IntegerAttr.get(static_type, int(static_byte_offset))
            static_const = _unwrap_value(
                std_arith.ConstantOp(static_type, static_attr).result
            )
            offset_val = _unwrap_value(
                std_arith.AddIOp(offset_val, static_const).result
            )

        dynamic_indices = [offset_val]
        raw_constant_indices = [_gep_dynamic_index_sentinel]

    return llvm.GEPOp(
        base_ptr.type,
        base_ptr,
        dynamic_indices,
        raw_constant_indices,
        elem_type,
        no_wrap_flags,
    ).result


@dsl_loc_tracing
def create_buffer_resource_from_addr(
    addr_i64: ir.Value,
    *,
    num_records_bytes: int | ir.Value | None = None,
) -> ir.Value:
    """Create AMD buffer resource descriptor from a raw i64 device address.

    Useful when working with runtime pointer arrays (e.g. IPC-mapped addresses
    or device-side pointer tables) where no fly.memref is available.
    The full address is encoded as the buffer base; callers should pass
    byte offset 0 to buffer_load / buffer_store.

    Args:
        addr_i64: Raw 64-bit device address (i64 MLIR value).
        num_records_bytes: Optional buffer size in bytes for hardware OOB checking.

    Returns:
        ROCDL buffer resource descriptor (!llvm.ptr<8>).

    Example:
        >>> rsrc = create_buffer_resource_from_addr(raw_addr_i64)
        >>> data = buffer_load(rsrc, i32_zero, vec_width=4, dtype=T.i32)
    """
    addr_i64 = _unwrap_value(addr_i64)
    ptr_type = ir.Type.parse("!llvm.ptr")
    base_ptr = llvm.IntToPtrOp(ptr_type, addr_i64).result
    flags = _create_i32_constant(_get_buffer_flags())
    stride = _create_i16_constant(0)
    if num_records_bytes is None:
        num_records = _create_i64_constant(0xFFFFFFFF)
    elif isinstance(num_records_bytes, int):
        nbytes = max(0, min(int(num_records_bytes), 0xFFFFFFFF))
        num_records = _create_i64_constant(nbytes)
    else:
        num_records = _unwrap_value(num_records_bytes)
        i64_type = T.i64()
        if (
            not isinstance(num_records.type, ir.IntegerType)
            or num_records.type.width != 64
        ):
            if isinstance(num_records.type, ir.IndexType):
                num_records = _unwrap_value(
                    std_arith.IndexCastOp(i64_type, num_records).result
                )
            else:
                num_records = _unwrap_value(
                    std_arith.ExtSIOp(i64_type, num_records).result
                )
    rsrc_type = ir.Type.parse("!llvm.ptr<8>")
    return rocdl.MakeBufferRsrcOp(
        rsrc_type, base_ptr, stride, num_records, flags
    ).result


@dsl_loc_tracing
def buffer_load(
    rsrc: ir.Value,
    offset: ir.Value,
    vec_width: int = 4,
    dtype=None,
    mask: ir.Value | None = None,
    cache_modifier: int = 0,
    soffset_bytes: int | ir.Value | None = None,
    is_scalar: bool = False,
) -> ir.Value:
    """AMD buffer load operation.

    Load data from global memory using buffer descriptor and offset.
    Uses hardware-level bounds checking and vectorization.

    Args:
        rsrc: Buffer resource descriptor (!llvm.ptr<8>)
        offset: Offset in elements (i32 type)
        vec_width: Vector width (1, 2, or 4)
        dtype: Element data type (None for f32, or ir.F32Type, etc.)
        mask: Optional mask for predicated load (i1 type)
        cache_modifier: Cache control flags (0 for default)
        soffset_bytes: Optional scalar offset (in BYTES) added by the buffer instruction (soffset).
                      Use this to fold small constant deltas into the instruction instead of emitting
                      extra VGPR address arithmetic.
        is_scalar: Emit a uniform/SGPR scalar load (llvm.amdgcn.s.buffer.load) instead of the
                      vector buffer load. Use only for wave-uniform addresses to route through the
                      SMEM cache and land the result directly in SGPRs. Restricted to vec_width 1 or 4;
                      dtype is forced to i32 (the result is raw i32 dwords). mask and soffset_bytes
                      are not supported in this mode and raise ValueError if provided.

    Returns:
        Loaded data (scalar or vector depending on vec_width)

    Example:
        >>> # Load 4xf32
        >>> data = buffer_load(rsrc, offset, vec_width=4)
        >>>
        >>> # Load with mask
        >>> data = buffer_load(rsrc, offset, vec_width=4, mask=valid)
    """
    # Scalar (uniform) loads return raw i32 dwords; force the element type so the
    # element->byte offset math below uses 4 and the result type is i32 / v4i32.
    if is_scalar:
        if vec_width not in (1, 4):
            raise ValueError(
                f"buffer_load(is_scalar=True): unsupported vec_width={vec_width}"
            )
        if mask is not None or soffset_bytes is not None:
            raise ValueError(
                "buffer_load(is_scalar=True) does not support mask or soffset_bytes"
            )
        dtype = T.i32()
    # Default dtype to f32
    elif dtype is None:
        dtype = T.f32()
    # Accept DSL Numeric class (e.g. fx.Int32) as dtype: unwrap to ir.Type
    elif hasattr(dtype, "ir_type"):
        dtype = dtype.ir_type

    # Unwrap offset first (accept Python ints and DSL Numeric values).
    if isinstance(offset, int):
        offset = _create_i32_constant(offset)
    elif hasattr(offset, "ir_value"):
        offset = offset.ir_value()
    offset = _unwrap_value(offset)

    # Convert offset to i32 if needed (accepts index/i64/i32)
    offset = _to_i32_offset(offset)

    # IMPORTANT: Buffer load offset is in BYTES, not elements!
    # For vec4xf32, each element is 4 bytes, so multiply offset by 4
    element_bytes = dtype.width // 8
    bytes_const = _create_i32_constant(element_bytes)
    op = std_arith.MulIOp(offset, bytes_const)
    offset = _unwrap_value(op.result)

    # Apply mask by setting invalid offsets to max
    if mask is not None:
        mask = _unwrap_value(mask)
        max_offset = _create_i32_constant(0x7FFFFFFF)
        op = std_arith.SelectOp(mask, offset, max_offset)
        offset = _unwrap_value(op.result)

    # Create vector type
    if vec_width == 1:
        result_type = dtype
    else:
        result_type = ir.VectorType.get([vec_width], dtype)

    # Scalar/uniform load path: emit s.buffer.load with a v4i32 resource and the
    # byte offset computed above. Returns i32 (vec_width 1) or v4i32 (vec_width 4).
    if is_scalar:
        rsrc_v4 = _ptr8_to_v4i32(rsrc)
        cache_policy = _create_i32_constant(cache_modifier)
        suffix = "i32" if vec_width == 1 else "v4i32"
        return llvm.call_intrinsic(
            result_type,
            f"llvm.amdgcn.s.buffer.load.{suffix}",
            [rsrc_v4, offset, cache_policy],
            [],
            [],
        )

    # Create instruction offset and aux flags
    if soffset_bytes is None:
        soffset = _create_i32_constant(0)
    else:
        if isinstance(soffset_bytes, int):
            soffset = _create_i32_constant(soffset_bytes)
        else:
            soffset = _to_i32_offset(_unwrap_value(soffset_bytes))
    aux_attr = (
        ir.IntegerAttr.get(ir.IntegerType.get_signless(32), cache_modifier)
        if cache_modifier
        else None
    )

    # Emit buffer load
    load_op = rocdl.RawPtrBufferLoadOp(
        result_type,
        rsrc,
        offset,
        soffset,
        aux=aux_attr,
    )

    return load_op.result
