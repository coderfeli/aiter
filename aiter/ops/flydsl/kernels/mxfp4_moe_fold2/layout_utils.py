"""Layout helpers for GEMM kernels.

Parses fly layout type strings (e.g. '(4,64):(64,1)') and computes
crd2idx with plain arith ops for static layouts.
Falls back to fly dialect ops for dynamic layouts.
"""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.expr import arith
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T


def _wrap(v):
    """Wrap raw ir.Value in ArithValue for operator overloading compatibility."""
    if isinstance(v, ArithValue):
        return v
    if isinstance(v, ir.Value):
        return ArithValue(v)
    return v


def _parse_dim(tok):
    """Parse a single dimension token: '?' -> None, otherwise int."""
    tok = tok.strip()
    return None if tok == "?" else int(tok)


def _find_layout_groups(s):
    """First '(<g1>):(<g2>)' in s, g1/g2 non-empty and free of ')'; None if absent.

    Same match as the search for the pattern ``\\(([^)]+)\\):\\(([^)]+)\\)``:
    the leftmost '(' whose text up to the next ')' is non-empty and is followed
    by ':(' and a non-empty run up to the next ')'.
    """
    i = s.find("(")
    while i >= 0:
        j = s.find(")", i + 1)
        if j < 0:
            return None
        if j > i + 1 and s.startswith(":(", j + 1):
            k = s.find(")", j + 3)
            if k > j + 3:
                return s[i + 1 : j], s[j + 3 : k]
        i = s.find("(", i + 1)
    return None


def _parse_layout(ly):
    """Parse '(s0,s1,...):(d0,d1,...)' -> (shapes, strides) as lists (None for '?')."""
    ly_str = str(ly.type) if hasattr(ly, "type") else str(ly)
    groups = _find_layout_groups(ly_str)
    if groups is None:
        return None
    shapes = [_parse_dim(s) for s in groups[0].split(",")]
    strides = [_parse_dim(s) for s in groups[1].split(",")]
    return shapes, strides


def _has_dynamic_strides(strides):
    """Check if any stride is dynamic (None)."""
    return any(s is None for s in strides)


def crd2idx(crd, layout):
    """Compute flat index from a coordinate tuple/list.

    For static layouts, computes with plain arith ops.
    For dynamic layouts, falls back to fx.crd2idx with fx.make_coord.
    """
    if not isinstance(crd, (list, tuple)):
        crd = [crd]
    parsed = _parse_layout(layout)

    if parsed is None or _has_dynamic_strides(parsed[1]):
        # fly.make_coord requires i32/i64, not index
        crd_i32 = []
        for c in crd:
            cv = c
            if isinstance(cv, ArithValue):
                cv = cv.ir_value() if hasattr(cv, "ir_value") else cv
            if isinstance(cv, ir.Value) and isinstance(cv.type, ir.IndexType):
                cv = arith.index_cast(T.i32, cv)
            crd_i32.append(cv)
        coord_val = fx.make_coord(*crd_i32)
        scalar = fx.get_scalar(fx.crd2idx(coord_val, layout)).ir_value()
        if not isinstance(scalar.type, ir.IndexType):
            scalar = arith.index_cast(T.index, scalar)
        return _wrap(scalar)

    _, strides = parsed
    result = None
    for coord_v, stride_v in zip(crd, strides):
        if stride_v == 0:
            continue
        term = coord_v if stride_v == 1 else coord_v * arith.index(stride_v)
        result = term if result is None else result + term
    return result if result is not None else arith.index(0)
