# LIT-b11. A store to the peer before T.barrier_blocks and a load by the peer
# after it.
# Source: docstrings of barrier_blocks and sync_blocks,
# tilelang/language/distributed/sync.py:155-173 (TS-SYNC).
# Table: documents/dist/LitmusTable.md. Front end: TileScale 71e7145.
# A source for later tickets: not run here, and nothing imports it.
# The calls also exist with these arguments at 9ba8a605.
# Launch: ranks 0 and 1. x and barrier are allocator-backed int32 buffers,
# initially 0; out is local.
import tilelang.language as T


def barrier_blocks_then_load():
    @T.prim_func
    def main(
        x: T.Tensor((1,), "int32"),
        barrier: T.Tensor((2,), "int32"),
        out: T.Tensor((1,), "int32"),
    ):
        with T.Kernel(1, threads=32) as bx:
            rank = T.get_rank()
            tx = T.get_thread_binding()
            if rank == 0:
                if tx == 0:
                    T.st(x[0], 42, dst_pe=1)
            T.barrier_blocks(barrier)
            if rank == 1:
                if tx == 0:
                    out[0] = x[0]

    return main
