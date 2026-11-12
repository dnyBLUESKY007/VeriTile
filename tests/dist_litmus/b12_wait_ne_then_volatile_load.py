# LIT-b12. The sender publishes an offset as -value-1 with a relaxed store;
# the receiver waits until the word is not 0 and reads it again with a
# volatile load.
# Source: examples/distributed/deepseek_deepep/deepep.md:189-190 at TileScale
# 9ba8a605 ("Negative offset encoding", "Queue semantics");
# intranode/dispatch.py:314 and 431-435 there (the form of the calls);
# docs/distributed_api_reference.md, "Wait and Barrier Operations" (TS-API).
# Table: documents/dist/LitmusTable.md. Front end: TileScale 71e7145.
# A source for later tickets: not run here, and nothing imports it.
# The calls also exist with these arguments at 9ba8a605.
# Launch: ranks 0 and 1. start is an allocator-backed int32 word, initially
# 0; out is local.
import tilelang.language as T


def wait_ne_then_volatile_load():
    @T.prim_func
    def main(
        start: T.Tensor((1,), "int32"),
        out: T.Tensor((1,), "int32"),
    ):
        with T.Kernel(1, threads=32) as bx:
            rank = T.get_rank()
            if T.get_thread_binding() == 0:
                if rank == 0:
                    T.st(start[0], -7 - 1, scope="sys", sem="relaxed", dst_pe=1)
                else:
                    v = T.alloc_var("int32")
                    T.wait_ne(start[0], 0)
                    T.ld(start[0], v, sem="volatile")
                    out[0] = -v - 1

    return main
