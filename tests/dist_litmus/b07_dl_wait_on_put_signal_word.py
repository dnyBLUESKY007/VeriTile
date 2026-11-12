# LIT-b07. dl.wait on a word that an NVSHMEM put-with-signal writes.
# Source: tutorials/07-overlapping-allgather-gemm.py:181-185 (the wait) and
# 293-301 (the put-with-signal) (TD-T07).
# Table: documents/dist/LitmusTable.md. Front end: Triton-distributed 7908e4e.
# A source for later tickets: not run here, and nothing imports it.
# Launch: ranks 0 and 1, grid 1. buf has NBYTES symmetric bytes (rank 0 holds
# the payload, rank 1 zeros); signal is a symmetric uint64 signal word,
# initially 0; out is local.
import triton.language as tl
import triton_dist
import triton_dist.language as dl
from triton_dist.language.extra import libshmem_device


@triton_dist.jit
def dl_wait_on_put_signal_word(buf_ptr, signal_ptr, out_ptr, NBYTES: tl.constexpr):
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
        token = dl.wait(signal_ptr, 1, "gpu", "acquire", waitValue=1)
        buf_ptr = dl.consume_token(buf_ptr, token)
        tl.store(out_ptr, tl.load(buf_ptr))
