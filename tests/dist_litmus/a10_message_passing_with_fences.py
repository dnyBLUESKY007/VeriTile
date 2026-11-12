# LIT-a10. Message passing through a relaxed flag and two fences.
# Source: libcu++ "cuda::atomic_thread_fence", "Example" (CU-FE); docstring
# of fence_gpu, tilelang/language/builtin.py:1194-1196.
# Table: documents/dist/LitmusTable.md. Front end: TileScale 71e7145.
# A source for later tickets: not run here, and nothing imports it.
# Launch: one rank, 2 blocks. data and flag are int32 words, initially 0.
import tilelang.language as T


def message_passing_with_fences():
    @T.prim_func
    def main(
        data: T.Tensor((1,), "int32"),
        flag: T.Tensor((1,), "int32"),
        out: T.Tensor((1,), "int32"),
    ):
        with T.Kernel(2, threads=32) as bx:
            if T.get_thread_binding() == 0:
                if bx == 0:
                    data[0] = 42
                    T.fence_gpu()
                    T.st(flag[0], 1, scope="gpu", sem="relaxed")
                else:
                    f = T.alloc_var("int32")
                    f = 0
                    while f == 0:
                        T.ld(flag[0], f, scope="gpu", sem="relaxed")
                    T.fence_gpu()
                    out[0] = data[0]

    return main
