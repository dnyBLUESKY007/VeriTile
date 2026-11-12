import triton
import triton.language as tl


def host(x):
    return len(x)


@triton.jit
def one(p):
    tl.store(p, tl.load(p, other=0))


@triton.jit
def two(p):
    tl.store(p, tl.tanh(tl.load(p)))
