"""Global-memory fences and agent-scope fetch-add with explicit ordering and syncscope."""

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm_d
from flydsl.expr import arith


def _to_ptr_global(v):
    """Cast an i64 address to ``!llvm.ptr<1>`` (global address space)."""
    return _llvm_d.IntToPtrOp(
        _llvm_d.PointerType.get(address_space=1), arith.unwrap(v)
    ).result


def fence_acquire(syncscope):
    """Emit an acquire fence for the selected AMDGPU memory scope."""
    _llvm_d.FenceOp(_llvm_d.AtomicOrdering.acquire, syncscope=syncscope)


def fence_release(syncscope):
    """Emit a release fence for the selected AMDGPU memory scope."""
    _llvm_d.FenceOp(_llvm_d.AtomicOrdering.release, syncscope=syncscope)


def atomic_add_global_at(addr_i64, val, syncscope="one-as"):
    """Monotonic global fetch-add with configurable agent/system visibility."""
    ptr = _to_ptr_global(addr_i64)
    kwargs = {} if syncscope is None else {"syncscope": syncscope}
    return _llvm_d.AtomicRMWOp(
        _llvm_d.AtomicBinOp.add,
        ptr,
        arith.unwrap(val),
        _llvm_d.AtomicOrdering.monotonic,
        **kwargs,
    ).res


def atomic_add_agent(addr_i64, val):
    """Agent-scope monotonic global fetch-and-add."""
    return atomic_add_global_at(addr_i64, val, syncscope=fx.rocdl.SyncScope.Agent)
