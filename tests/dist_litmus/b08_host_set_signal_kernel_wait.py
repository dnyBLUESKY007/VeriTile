# LIT-b08. The host copies to the peer and sets its signal on a stream while
# a kernel of the peer waits on that signal.
# Source: tutorials/07-overlapping-allgather-gemm.py:181-185 (the wait) and
# 325-334 (the copy and set_signal) (TD-T07). The tutorial copies with
# cudart.cudaMemcpyAsync and passes an address and a fourth argument to
# set_signal; this file calls set_signal as
# python/triton_dist/kernels/nvidia/common_ops.py:387 defines it and copies
# as tutorials/05-intra-node-reduce-scatter.py:200 does.
# Table: documents/dist/LitmusTable.md. Front end: Triton-distributed 7908e4e.
# A source for later tickets: not run here, and nothing imports it.
# Launch: rank 0 runs producer_host on stream; rank 1 launches consumer with
# grid 1. peer_buf and peer_signal are rank 0's views of rank 1's symmetric
# buffer and signal word; the signal is initially 0.
import torch
import triton.language as tl
import triton_dist
import triton_dist.language as dl
from triton_dist.kernels.common_ops import set_signal


@triton_dist.jit
def consumer(ready_ptr, a_ptr, out_ptr):
    token = dl.wait(ready_ptr, 1, "gpu", "acquire", waitValue=1)
    a_ptr = dl.consume_token(a_ptr, token)
    tl.store(out_ptr, tl.load(a_ptr))


def producer_host(local_buf: torch.Tensor, peer_buf: torch.Tensor, peer_signal: torch.Tensor,
                  stream: torch.cuda.Stream):
    with torch.cuda.stream(stream):
        peer_buf.copy_(local_buf)
        set_signal(peer_signal, 1, stream)
