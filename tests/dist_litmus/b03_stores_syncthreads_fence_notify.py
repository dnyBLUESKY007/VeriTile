# LIT-b03. The threads of a block store a tile to the peer, synchronize,
# fence, and one notify follows.
# Source: tutorials/01-distributed-notify-wait.py:100-116 (TD-T01), with the
# wait at system scope so that row LIT-b02 does not apply.
# Table: documents/dist/LitmusTable.md. Front end: Triton-distributed 7908e4e.
# A source for later tickets: not run here, and nothing imports it.
# Launch: ranks 0 and 1, grid 1. queue has BLOCK_SIZE symmetric words and
# signal is a symmetric uint64 signal word, all initially 0.
import triton.language as tl
import triton_dist
import triton_dist.language as dl
from triton_dist.language.extra import libshmem_device
from triton_dist.language.extra.language_extra import __syncthreads


@triton_dist.jit
def stores_syncthreads_fence_notify(input_ptr, queue_ptr, signal_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    rank = dl.rank()
    offs = tl.arange(0, BLOCK_SIZE)
    if rank == 0:
        data = tl.load(input_ptr + offs)
        tl.store(dl.symm_at(queue_ptr, 1) + offs, data)
        __syncthreads()
        libshmem_device.fence()
        dl.notify(signal_ptr, 1, signal=1, sig_op="set", comm_scope="intra_node")
    elif rank == 1:
        token = dl.wait(signal_ptr, 1, "sys", "acquire", waitValue=1)
        queue_ptr = dl.consume_token(queue_ptr, token)
        tl.store(out_ptr + offs, tl.load(queue_ptr + offs))
