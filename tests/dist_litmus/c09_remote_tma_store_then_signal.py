# LIT-c09. A TMA store to the peer, the store wait, then a release store of a
# flag; the peer waits on the flag and reads.
# Source: docstring of tma_copy, tilelang/language/copy_op.py:258-269
# (TS-COPY); docstring of tma_store_wait, tilelang/language/builtin.py:382-388;
# docs/distributed_api_reference.md, "Peer Memory Operations" (TS-API).
# Table: documents/dist/LitmusTable.md. Front end: TileScale 71e7145.
# A source for later tickets: not run here, and nothing imports it.
# Launch: ranks 0 and 1. dst and flag are allocator-backed and initially 0;
# src holds the payload; out is local.
import tilelang.language as T


def remote_tma_store_then_signal():
    @T.prim_func
    def main(
        src: T.Tensor((64, 64), "float32"),
        dst: T.Tensor((64, 64), "float32"),
        flag: T.Tensor((1,), "int32"),
        out: T.Tensor((1,), "float32"),
    ):
        with T.Kernel(1, threads=128) as bx:
            rank = T.get_rank()
            tx = T.get_thread_binding()
            shared = T.alloc_shared((64, 64), "float32")
            if rank == 0:
                T.copy(src[0, 0], shared, disable_tma=True)
                T.sync_threads()
                T.tma_copy(shared, dst[0, 0], dst_pe=1)
                T.tma_store_wait(0, False)
                if tx == 0:
                    T.st(flag[0], 1, scope="sys", sem="release", dst_pe=1)
            else:
                if tx == 0:
                    T.wait_eq(flag[0], 1)
                    out[0] = dst[0, 0]

    return main
