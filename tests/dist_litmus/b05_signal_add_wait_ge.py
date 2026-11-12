# LIT-b05. Two ranks add to a signal word; the third waits for the sum.
# Source: NVSHMEM 3.6.5, "Signaling Operations" (atomicity, signal operators,
# nvshmemx_signal_op) and "Point-To-Point Synchronization"
# (nvshmem_signal_wait_until).
# Table: documents/dist/LitmusTable.md. Front end: Triton-distributed 7908e4e.
# A source for later tickets: not run here, and nothing imports it.
# Launch: ranks 0, 1 and 2, grid 1. signal is a symmetric uint64 signal word,
# initially 0; out is a local uint64 word.
import triton.language as tl
import triton_dist
import triton_dist.language as dl
from triton_dist.language.extra import libshmem_device
from triton_dist.language.extra.language_extra import tid


@triton_dist.jit
def signal_add_wait_ge(signal_ptr, out_ptr):
    rank = dl.rank()
    if tid(0) == 0:
        if rank == 0:
            v = libshmem_device.signal_wait_until(signal_ptr, libshmem_device.NVSHMEM_CMP_GE, 2)
            tl.store(out_ptr, v)
        else:
            libshmem_device.signal_op(signal_ptr, 1, libshmem_device.NVSHMEM_SIGNAL_ADD, 0)
