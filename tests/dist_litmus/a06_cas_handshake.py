# LIT-a06. A two-rank handshake by compare-and-swap at system scope.
# Source: python/triton_dist/kernels/nvidia/common_ops.py:154-168, with its
# docstring (tutorial 05 launches this kernel); TD-API, atomic_cas.
# Table: documents/dist/LitmusTable.md. Front end: Triton-distributed 7908e4e.
# A source for later tickets: not run here, and nothing imports it.
# Launch: ranks 0 and 1, grid 1. x has one and flag two symmetric int32
# words, initially 0; out is local.
import triton.language as tl
import triton_dist
import triton_dist.language as dl
from triton_dist.language.extra.language_extra import atomic_cas, tid


@triton_dist.jit
def cas_handshake(x_ptr, flag_ptr, out_ptr):
    rank = dl.rank()
    peer = 1 - rank
    if tid(0) == 0:
        tl.store(dl.symm_at(x_ptr, peer), 42 + rank)
        while atomic_cas(dl.symm_at(flag_ptr + rank, peer), 0, 1, "sys", "release") != 0:
            pass
        while atomic_cas(flag_ptr + peer, 1, 0, "sys", "acquire") != 1:
            pass
        tl.store(out_ptr, tl.load(x_ptr))
