# LIT-a09. A load with nc=True of a word that a peer writes during the kernel.
# Source: docstring of ld, tilelang/language/distributed/comm.py:78-85
# (TS-COMM).
# Table: documents/dist/LitmusTable.md. Front end: TileScale 71e7145.
# A source for later tickets: not run here, and nothing imports it.
# Launch: ranks 0 and 1. word is an allocator-backed int32 word, initially 0;
# out is local.
import tilelang.language as T


def nc_load_of_peer_written_word():
    @T.prim_func
    def main(
        word: T.Tensor((1,), "int32"),
        out: T.Tensor((1,), "int32"),
    ):
        with T.Kernel(1, threads=32) as bx:
            rank = T.get_rank()
            if T.get_thread_binding() == 0:
                if rank == 0:
                    T.st(word[0], 42, dst_pe=1)
                else:
                    v = T.alloc_var("int32")
                    T.ld(word[0], v, nc=True)
                    out[0] = v

    return main
