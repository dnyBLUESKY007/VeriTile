"""Census fixture for the Triton family. Read as text by scripts/census/scan.py; never imported or run."""
import torch
import triton
import triton.language as tl


@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)


@triton.jit
def row_softmax_kernel(out_ptr, in_ptr, in_stride, out_stride, n_rows, n_cols,
                       BLOCK: tl.constexpr, num_stages: tl.constexpr):
    first_row = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row in tl.range(first_row, n_rows, row_step, num_stages=num_stages):
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(in_ptr + row * in_stride + cols, mask=mask, other=-float('inf'))
        e = tl.exp(x - tl.max(x, axis=0))
        tl.store(out_ptr + row * out_stride + cols, e / tl.sum(e, axis=0), mask=mask)


@triton.jit(do_not_specialize=["K"])
def matvec_kernel(a_ptr, x_ptr, y_ptr, M, K, stride_am, stride_ak,
                  BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, ACTIVATION: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak, mask=a_mask, other=0.0)
        x = tl.load(x_ptr + offs_k, mask=offs_k < K, other=0.0)
        acc += tl.sum(a * x[None, :], axis=1)
    if ACTIVATION == "leaky_relu":
        acc = leaky_relu(acc)
    tl.store(y_ptr + offs_m, acc.to(tl.float16), mask=offs_m < M)


def matvec(a, x):
    # Host code: its range() and its calls must not appear in the census.
    y = torch.empty(a.shape[0], device=a.device, dtype=torch.float16)
    for _ in range(1):
        matvec_kernel[(triton.cdiv(a.shape[0], 32),)](a, x, y, a.shape[0], a.shape[1], a.stride(0), a.stride(1),
                                                      BLOCK_M=32, BLOCK_K=64, ACTIVATION="leaky_relu")
    return y
