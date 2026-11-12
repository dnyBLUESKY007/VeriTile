# LIT-a03. Message passing between two ranks at system scope.
# Source: libcu++ "Memory model", "Scope Relationships" and "Example: Message
# Passing" (CU-MM); the store and load of the flag are those of
# python/triton_dist/kernels/nvidia/common_ops.py:177-178.
# Table: documents/dist/LitmusTable.md. Front end: Triton-distributed 7908e4e.
# A source for later tickets: not run here, and nothing imports it.
# Launch: ranks 0 and 1, grid 1. x and flag are symmetric int32 words,
# initially 0; out is local.
import triton.language as tl
import triton_dist
import triton_dist.language as dl
from triton_dist.language.extra.language_extra import ld, st, tid


@triton_dist.jit
def message_passing_system(x_ptr, flag_ptr, out_ptr):
    rank = dl.rank()
    if tid(0) == 0:
        if rank == 0:
            tl.store(dl.symm_at(x_ptr, 1), 42)
            st(dl.symm_at(flag_ptr, 1), 1, scope="sys", semantic="release")
        elif rank == 1:
            while ld(flag_ptr, scope="sys", semantic="acquire") != 1:
                pass
            tl.store(out_ptr, tl.load(x_ptr))
