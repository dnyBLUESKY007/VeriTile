# LIT-b06. The receiver polls the flag with an ordinary load.
# Source: NVSHMEM 3.6.5, "Troubleshooting And FAQs" ("Why is the receive
# buffer not updated with remote data even after synchronization with a
# flag?"); "Memory Model", "Atomicity Guarantees"; the sender is the example
# at the head of "Signaling Operations".
# Table: documents/dist/LitmusTable.md. Front end: Triton-distributed 7908e4e.
# A source for later tickets: not run here, and nothing imports it.
# Launch: ranks 0 and 1, grid 1. buf has NBYTES symmetric bytes (rank 0 holds
# the payload, rank 1 zeros); signal is a symmetric uint64 signal word,
# initially 0; out is local.
import triton.language as tl
import triton_dist
import triton_dist.language as dl
from triton_dist.language.extra import libshmem_device
from triton_dist.language.extra.language_extra import tid


@triton_dist.jit
def plain_poll_of_signal_word(buf_ptr, signal_ptr, out_ptr, NBYTES: tl.constexpr):
    rank = dl.rank()
    if rank == 0:
        libshmem_device.putmem_block(buf_ptr, buf_ptr, NBYTES, 1)
        libshmem_device.quiet()
        if tid(0) == 0:
            libshmem_device.signal_op(signal_ptr, 1, libshmem_device.NVSHMEM_SIGNAL_SET, 1)
    elif rank == 1:
        if tid(0) == 0:
            while tl.load(signal_ptr) != 1:
                pass
            tl.store(out_ptr, tl.load(buf_ptr))
