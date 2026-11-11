"""Launch harnesses of the first smoke batch.

The `copy` and `half_write` entries use torch only: they exercise the path
from `.tensor` bytes to GPU tensors and back without compiling a kernel.
`add` launches one small Triton kernel; `add_reference` is its torch reference.
"""
import torch
import triton
import triton.language as tl


def copy(inputs, outputs, launch):
    outputs['out'].copy_(inputs['x'])


def half_write(inputs, outputs, launch):
    half = inputs['x'].numel() // 2
    outputs['out'][:half] = inputs['x'][:half]


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def add(inputs, outputs, launch):
    x, y = inputs['x'], inputs['y']
    n, block = x.numel(), launch['block']
    # The output buffer is longer than n; the tail must keep its sentinel.
    add_kernel[(triton.cdiv(n, block),)](x, y, outputs['out'], n, BLOCK=block, num_warps=launch['num_warps'])


def add_reference(inputs, outputs, launch):
    n = inputs['x'].numel()
    outputs['out'][:n] = inputs['x'] + inputs['y']
