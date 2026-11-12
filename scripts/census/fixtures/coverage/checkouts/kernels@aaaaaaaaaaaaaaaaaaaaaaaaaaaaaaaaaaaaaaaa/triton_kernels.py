import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def helper(x):
    return tl.math.exp2(x)


@triton.jit
def kernel(p, n: tl.constexpr):
    x = tl.load(p, mask=None, volatile=True)
    y: tl.constexpr = 1
    for i in tl.range(n, num_stages=2):
        x += helper(tl.cast(x, tl.float32, bitcast=True))
        pass
    d = tl.make_tensor_descriptor(p, shape=[n])
    tl.store(p, d.load([0]).to(tl.float32) + tl.sum(x, axis=0, keep_dims=True) + libdevice.exp(x), mask=None)
