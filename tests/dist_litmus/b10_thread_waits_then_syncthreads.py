# LIT-b10. One thread per peer waits for that peer's signal; the block then
# synchronizes.
# Source: tutorials/03-inter-node-allgather.py:65-78 (TD-T03); TD-API,
# __syncthreads. The peers signal with signal_op instead of the tutorial's
# putmem_signal_block.
# Table: documents/dist/LitmusTable.md. Front end: Triton-distributed 7908e4e.
# A source for later tickets: not run here, and nothing imports it.
# Launch: WORLD_SIZE ranks, grid WORLD_SIZE. flag has WORLD_SIZE symmetric
# uint64 signal words, initially 0; done is a local word, initially 0.
import triton.language as tl
import triton_dist
import triton_dist.language as dl
from triton_dist.language.extra import libshmem_device
from triton_dist.language.extra.language_extra import __syncthreads, tid


@triton_dist.jit
def thread_waits_then_syncthreads(symm_flag, done_ptr, WORLD_SIZE: tl.constexpr):
    rank = dl.rank()
    pid = tl.program_id(0)
    thread_idx = tid(0)
    if pid == rank:
        peer = thread_idx
        if peer < WORLD_SIZE and peer != rank:
            libshmem_device.signal_wait_until(symm_flag + peer, libshmem_device.NVSHMEM_CMP_EQ, 1)
        __syncthreads()
        tl.store(done_ptr, 1)
    else:
        peer = pid
        if thread_idx == 0:
            libshmem_device.signal_op(symm_flag + rank, 1, libshmem_device.NVSHMEM_SIGNAL_SET, peer)
