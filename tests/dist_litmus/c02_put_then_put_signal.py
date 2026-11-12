# LIT-c02. A put followed by a put-with-signal; the target reads the data of
# the first put after the signal.
# Source: NVSHMEM 3.6.5, "Signaling Operations", notes of nvshmem_put_signal.
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
def put_then_put_signal(a_ptr, b_ptr, signal_ptr, out_ptr, NBYTES: tl.constexpr):
    rank = dl.rank()
    if rank == 0:
        libshmem_device.putmem_block(a_ptr, a_ptr, NBYTES, 1)
        libshmem_device.putmem_signal_block(
            b_ptr,
            b_ptr,
            NBYTES,
            signal_ptr,
            1,
            libshmem_device.NVSHMEM_SIGNAL_SET,
            1,
        )
    elif rank == 1:
        if tid(0) == 0:
            libshmem_device.signal_wait_until(signal_ptr, libshmem_device.NVSHMEM_CMP_EQ, 1)
            tl.store(out_ptr, tl.load(a_ptr))
