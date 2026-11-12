# LIT-b13. A store to rank 1 before a barrier over all ranks; ranks 1 and 2
# read it after the barrier.
# Source: NVSHMEM 3.6.5, "Collective Communication" (nvshmem_barrier_all) and
# "Memory Model", "Visibility Guarantees";
# tutorials/01-distributed-notify-wait.py:165-168 (the host form).
# Table: documents/dist/LitmusTable.md. Front end: Triton-distributed 7908e4e.
# A source for later tickets: not run here, and nothing imports it.
# Launch: ranks 0, 1 and 2, grid 1. x is a symmetric word, initially 0; out
# is local.
import triton.language as tl
import triton_dist
import triton_dist.language as dl
from triton_dist.language.extra import libshmem_device
from triton_dist.language.extra.language_extra import tid


@triton_dist.jit
def barrier_all_third_rank(x_ptr, out_ptr):
    rank = dl.rank()
    if rank == 0:
        if tid(0) == 0:
            tl.store(dl.symm_at(x_ptr, 1), 42)
    libshmem_device.barrier_all_block()
    if tid(0) == 0:
        if rank == 1:
            tl.store(out_ptr, tl.load(x_ptr))
        elif rank == 2:
            tl.store(out_ptr, tl.load(dl.symm_at(x_ptr, 1)))
