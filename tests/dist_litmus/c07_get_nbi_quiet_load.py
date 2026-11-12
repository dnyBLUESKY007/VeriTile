# LIT-c07. A non-blocking get, a quiet, then a load of the local copy.
# Source: NVSHMEM 3.6.5, "Remote Memory Access" (nvshmem_get_nbi).
# Table: documents/dist/LitmusTable.md. Front end: Triton-distributed 7908e4e.
# A source for later tickets: not run here, and nothing imports it.
# Launch: ranks 0 and 1, grid 1. src has NBYTES symmetric bytes (rank 1 holds
# the payload and does not write it during the kernel); local has NBYTES
# bytes on rank 0, initially 0; out is local.
import triton.language as tl
import triton_dist
import triton_dist.language as dl
from triton_dist.language.extra import libshmem_device
from triton_dist.language.extra.language_extra import tid


@triton_dist.jit
def get_nbi_quiet_load(local_ptr, src_ptr, out_ptr, NBYTES: tl.constexpr):
    rank = dl.rank()
    if rank == 0:
        libshmem_device.getmem_nbi_block(local_ptr, src_ptr, NBYTES, 1)
        libshmem_device.quiet()
        if tid(0) == 0:
            tl.store(out_ptr, tl.load(local_ptr))
