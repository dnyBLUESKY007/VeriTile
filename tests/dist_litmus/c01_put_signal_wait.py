# LIT-c01. A put-with-signal; the target waits on the signal and reads.
# Source: NVSHMEM 3.6.5, "Signaling Operations" (nvshmem_put_signal) and
# "Point-To-Point Synchronization" (nvshmem_signal_wait_until);
# tutorials/03-inter-node-allgather.py:72-95 (TD-T03).
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
def put_signal_wait(buf_ptr, signal_ptr, out_ptr, NBYTES: tl.constexpr):
    rank = dl.rank()
    if rank == 0:
        libshmem_device.putmem_signal_block(
            buf_ptr,
            buf_ptr,
            NBYTES,
            signal_ptr,
            1,
            libshmem_device.NVSHMEM_SIGNAL_SET,
            1,
        )
    elif rank == 1:
        if tid(0) == 0:
            libshmem_device.signal_wait_until(signal_ptr, libshmem_device.NVSHMEM_CMP_EQ, 1)
            tl.store(out_ptr, tl.load(buf_ptr))
