"""Fixture for test_triton_coverage.py: the shape of `triton.language`, cut down. Never imported."""
from . import math
from .core import (
    atomic_max,
    cast,
    float32,
    inline_asm_elementwise,
    load,
    make_tensor_descriptor,
    range,
    store,
    sum,
    void,
)

__all__ = [
    "atomic_max",
    "cast",
    "float32",
    "inline_asm_elementwise",
    "load",
    "make_tensor_descriptor",
    "math",
    "range",
    "store",
    "sum",
    "void",
]
