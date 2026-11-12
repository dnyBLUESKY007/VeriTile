# LIT-b09. The host copies and sets a signal on a stream while a kernel waits
# on it with T.wait_eq.
# Source: docs/distributed_api_reference.md, "Host Utilities" and "Wait and
# Barrier Operations" (TS-API);
# examples/distributed/allgather_gemm/example_allgather_gemm_overlapped.py:101-104
# (the wait) and 172-184 (the copy and the stream write, which the example
# issues through the CUDA driver binding instead of set_signal).
# Table: documents/dist/LitmusTable.md. Front end: TileScale 71e7145.
# A source for later tickets: not run here, and nothing imports it.
# Launch: one rank runs producer_host on stream and the kernel on another
# stream. buf and signal are allocator-backed; the signal is initially 0.
import torch
import tilelang.language as T
from tilelang.distributed.host import set_signal


def consumer():
    @T.prim_func
    def main(
        buf: T.Tensor((1,), "int32"),
        signal: T.Tensor((1,), "uint32"),
        out: T.Tensor((1,), "int32"),
    ):
        with T.Kernel(1, threads=32) as bx:
            if T.get_thread_binding() == 0:
                T.wait_eq(signal[0], 1)
                out[0] = buf[0]

    return main


def producer_host(peer_buf: torch.Tensor, buf: torch.Tensor, signal: torch.Tensor, stream: torch.cuda.Stream):
    with torch.cuda.stream(stream):
        buf.copy_(peer_buf)
        set_signal(signal, 1, stream)
