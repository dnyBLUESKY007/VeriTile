# LIT-a07. One lane writes a slot of the peer and publishes the tail.
# Source: examples/distributed/deepseek_deepep/deepep.md:190 at TileScale
# 9ba8a605 ("Queue semantics"); docs/distributed_api_reference.md, "Peer
# Memory Operations" and "Wait and Barrier Operations" (TS-API).
# Table: documents/dist/LitmusTable.md. Front end: TileScale 71e7145.
# A source for later tickets: not run here, and nothing imports it.
# The calls also exist with these arguments at 9ba8a605.
# Launch: ranks 0 and 1. data and tail are allocator-backed (peer-visible)
# int32 words, initially 0; out is local.
import tilelang.language as T


def queue_tail_release_acquire():
    @T.prim_func
    def main(
        data: T.Tensor((1,), "int32"),
        tail: T.Tensor((1,), "int32"),
        out: T.Tensor((1,), "int32"),
    ):
        with T.Kernel(1, threads=32) as bx:
            rank = T.get_rank()
            if T.get_thread_binding() == 0:
                if rank == 0:
                    T.st(data[0], 42, dst_pe=1)
                    T.st(tail[0], 1, scope="sys", sem="release", dst_pe=1)
                else:
                    v = T.alloc_var("int32")
                    T.wait_ge(tail[0], 1)
                    T.ld(data[0], v)
                    out[0] = v

    return main
