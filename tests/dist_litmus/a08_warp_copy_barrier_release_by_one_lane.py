# LIT-a08. A warp copies to the peer, the block synchronizes, and a lane of
# another warp publishes the tail; the receiver mirrors it.
# Source: examples/distributed/deepseek_deepep/intranode/dispatch.py:363-418
# and 451-478 at TileScale 9ba8a605 (the form of the sender and receiver
# loops); deepep.md:190 ("Queue semantics");
# docs/programming_guides/instructions.md:191 (T.sync_threads).
# Table: documents/dist/LitmusTable.md. Front end: TileScale 71e7145.
# A source for later tickets: not run here, and nothing imports it.
# The calls also exist with these arguments at 9ba8a605.
# Launch: ranks 0 and 1. buf and tail are allocator-backed; buf and tail are
# initially 0; src holds 32 nonzero int32 values; out is local.
import tilelang.language as T


def warp_copy_barrier_release_by_one_lane():
    @T.prim_func
    def main(
        src: T.Tensor((32,), "int32"),
        buf: T.Tensor((32,), "int32"),
        tail: T.Tensor((1,), "int32"),
        out: T.Tensor((32,), "int32"),
    ):
        with T.Kernel(1, threads=64) as bx:
            rank = T.get_rank()
            tx = T.get_thread_binding()
            if rank == 0:
                if tx < 32:
                    T.put_warp(T.address_of(src[0]), T.address_of(buf[0]), 32, dst_pe=1)
                T.sync_threads()
                if tx == 32:
                    T.st(tail[0], 1, scope="sys", sem="release", dst_pe=1)
            else:
                if tx == 32:
                    T.wait_ge(tail[0], 1)
                T.sync_threads()
                if tx < 32:
                    T.put_warp(T.address_of(buf[0]), T.address_of(out[0]), 32, -1)

    return main
