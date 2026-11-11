"""Census fixture for the TileLang/TileScale family. Read as text by scripts/census/scan.py; never imported or run."""
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1])
def clamped_matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    floor = 0.0

    @T.macro
    def clamp(buf):
        for i, j in T.Parallel(block_M, block_N):
            buf[i, j] = T.max(buf[i, j], floor)

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local, policy=T.GemmWarpPolicy.FullRow)
            clamp(C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


@tilelang.jit
def row_sums(A, block_N, dtype=T.float32):
    M, N = T.const("M, N")
    A: T.Tensor((M, N), dtype)
    out = T.empty((M,), dtype)
    with T.Kernel(M, threads=128) as bx:
        tile = T.alloc_fragment((block_N,), dtype)
        total = T.alloc_fragment((1,), dtype)
        T.fill(total, 0)
        for start in T.serial(T.ceildiv(N, block_N)):
            for j in T.Parallel(block_N):
                col = start * block_N + j
                tile[j] = T.if_then_else(col < N, A[bx, col], 0)
            T.reduce_sum(tile, total, dim=0, clear=False)
        for _ in range(1):
            out[bx] = total[0]
    return out


@tilelang.jit(compile_once=True)
def gather_rows_kernel(num_ranks, rows_per_rank, width, block_W, dtype=T.float16):
    rows = rows_per_rank * num_ranks

    @T.prim_func
    def main(
        src: T.Tensor((rows_per_rank, width), dtype),
        dst: T.Tensor((rows, width), dtype),
        ready: T.Tensor((num_ranks,), T.uint32),
        row_sum: T.Tensor((rows,), dtype),
    ):
        with T.Kernel(rows_per_rank, num_ranks, threads=128) as (row, peer):
            rank = T.get_rank()
            T.put_block(
                src=T.address_of(src[row, 0]),
                dst=T.address_of(dst[rank * rows_per_rank + row, 0]),
                size=width,
                dst_pe=peer,
            )
            if peer != rank:
                T.wait_eq(ready[peer], 1)
                T.sync_threads()
            tile = T.alloc_shared((block_W,), dtype)
            acc = T.alloc_fragment((1,), dtype)
            T.fill(acc, 0)
            for w in T.Pipelined(T.ceildiv(width, block_W), num_stages=2):
                T.copy(dst[peer * rows_per_rank + row, w * block_W : (w + 1) * block_W], tile)
                T.reduce_sum(tile, acc, dim=0, clear=False)
            row_sum[peer * rows_per_rank + row] = acc[0]

    return main
