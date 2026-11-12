# LIT-c05. A non-blocking put, a fence, then the signal.
# Source: tutorials/04-deepseek-infer-all2all.py:233-276 (TD-T04); NVSHMEM
# 3.6.5, "Memory Ordering" (table and nvshmem_fence) and "Memory Model",
# "Ordering of Operations".
# Table: documents/dist/LitmusTable.md. Front end: Triton-distributed 7908e4e.
# A source for later tickets: not run here, and nothing imports it.
# Launch: ranks 0 and 1, grid 1. Each buffer has NBYTES symmetric bytes (rank
# 0 holds the payload, rank 1 zeros); signal is a symmetric uint64 signal
# word, initially 0; out is local.
import triton.language as tl
import triton_dist
import triton_dist.language as dl
from triton_dist.language.extra import libshmem_device
from triton_dist.language.extra.language_extra import tid


@triton_dist.jit
def put_nbi_fence_signal(buf_ptr, signal_ptr, out_ptr, NBYTES: tl.constexpr):
    rank = dl.rank()
    if rank == 0:
        libshmem_device.putmem_nbi_block(buf_ptr, buf_ptr, NBYTES, 1)
        libshmem_device.fence()
        if tid(0) == 0:
            libshmem_device.signal_op(signal_ptr, 1, libshmem_device.NVSHMEM_SIGNAL_SET, 1)
    elif rank == 1:
        if tid(0) == 0:
            libshmem_device.signal_wait_until(signal_ptr, libshmem_device.NVSHMEM_CMP_EQ, 1)
            tl.store(out_ptr, tl.load(buf_ptr))
