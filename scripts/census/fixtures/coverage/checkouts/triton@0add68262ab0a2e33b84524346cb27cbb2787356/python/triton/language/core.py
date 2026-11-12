class dtype:
    def __init__(self, name):
        self.name = name


float32 = dtype('fp32')
void = dtype('void')


def load(pointer, mask=None, other=None, volatile=False, _semantic=None):
    ...


def store(pointer, value, mask=None, _semantic=None):
    ...


def atomic_max(pointer, val, mask=None):
    ...


def cast(input, dtype, bitcast=False):
    ...


def inline_asm_elementwise(asm, constraints, args, dtype, is_pure, pack):
    ...


def sum(input, axis=None, keep_dims=False):
    ...


def make_tensor_descriptor(base, shape):
    ...


class range:
    def __init__(self, arg1, arg2=None, *, num_stages=None):
        ...
