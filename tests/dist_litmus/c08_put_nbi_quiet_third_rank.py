# LIT-c08. Rank 0 puts to rank 1, quiets, and signals rank 2, which then
# loads the destination on rank 1.
# Source: NVSHMEM 3.6.5, "Memory Ordering" (nvshmem_quiet), "Memory Model",
# "Visibility Guarantees", and "Using NVSHMEM", "Data Consistency"; release
# notes of 3.6.5, "Limitations".
# Table: documents/dist/LitmusTable.md. Front end: Triton-distributed 7908e4e.
# A source for later tickets: not run here, and nothing imports it.
# Launch: ranks 0, 1 and 2, grid 1. buf has NBYTES symmetric bytes (rank 0
# holds the payload, the others zeros); signal is a symmetric uint64 signal
# word, initially 0; out is local.
import triton.language as tl
import triton_dist
import triton_dist.language as dl
from triton_dist.language.extra import libshmem_device
from triton_dist.language.extra.language_extra import tid


@triton_dist.jit
def put_nbi_quiet_third_rank(buf_ptr, signal_ptr, out_ptr, NBYTES: tl.constexpr):
    rank = dl.rank()
    if rank == 0:
        libshmem_device.putmem_nbi_block(buf_ptr, buf_ptr, NBYTES, 1)
        libshmem_device.quiet()
        if tid(0) == 0:
            libshmem_device.signal_op(signal_ptr, 1, libshmem_device.NVSHMEM_SIGNAL_SET, 2)
    elif rank == 2:
        if tid(0) == 0:
            libshmem_device.signal_wait_until(signal_ptr, libshmem_device.NVSHMEM_CMP_EQ, 1)
            tl.store(out_ptr, tl.load(dl.symm_at(buf_ptr, 1)))
