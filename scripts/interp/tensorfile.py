"""Reader and writer for `.tensor` files, version 1.

The format is frozen in documents/InterpInterface.md, section 3. A tensor is
kept as raw element bytes, so a round trip never changes a bit; nothing here
interprets floating values. Standard library only: the same file runs on the
development host, the Lean runner's host and the GPU side.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import struct

MAGIC = b'VTTENSR1'
MAX_RANK = 8
# name -> (code, element width in bytes)
DTYPES = {
    'fp32': (1, 4), 'fp16': (2, 2), 'bf16': (3, 2), 'fp64': (4, 8), 'f8e4': (5, 1), 'f8e5': (6, 1),
    'int8': (16, 1), 'int16': (17, 2), 'int32': (18, 4), 'int64': (19, 8),
    'uint8': (24, 1), 'uint16': (25, 2), 'uint32': (26, 4), 'uint64': (27, 8),
    'bool': (32, 1),
}
NAMES = {code: name for name, (code, _) in DTYPES.items()}
DEFAULT_SENTINELS = {
    'fp32': '0x7fc0dead', 'fp16': '0x7ead', 'bf16': '0x7fcd', 'fp64': '0x7ff8dead0000dead',
    'f8e4': '0x7f', 'f8e5': '0x7f',
    'int8': '0x80', 'int16': '0x8000', 'int32': '0x80000000', 'int64': '0x8000000000000000',
    'uint8': '0xad', 'uint16': '0xdead', 'uint32': '0xdeaddead', 'uint64': '0xdeaddeaddeaddead',
}


class TensorFormatError(ValueError):
    """The bytes are not a valid version 1 tensor file."""


def width(dtype):
    if dtype not in DTYPES:
        raise TensorFormatError(f'unknown dtype {dtype!r}')
    return DTYPES[dtype][1]


def count(shape):
    total = 1
    for dim in shape:
        total *= dim
    return total


@dataclass(frozen=True)
class Tensor:
    dtype: str
    shape: tuple
    data: bytes  # elements in row-major order, each little-endian

    def __post_init__(self):
        object.__setattr__(self, 'shape', tuple(self.shape))
        if len(self.shape) > MAX_RANK or any(not isinstance(d, int) or d < 0 or d >= 1 << 64 for d in self.shape):
            raise TensorFormatError(f'unsupported shape {self.shape!r}')
        if len(self.data) != count(self.shape) * width(self.dtype):
            raise TensorFormatError(f'{self.dtype} tensor of shape {self.shape} needs '
                                    f'{count(self.shape) * width(self.dtype)} data bytes, got {len(self.data)}')
        if self.dtype == 'bool' and self.data.strip(b'\x00\x01'):
            raise TensorFormatError('a bool element is neither 0 nor 1')

    def to_bytes(self):
        code, _ = DTYPES[self.dtype]
        header = MAGIC + bytes([code, len(self.shape)]) + bytes(6)
        return header + struct.pack(f'<{len(self.shape)}Q', *self.shape) + self.data

    def sha256(self):
        """SHA-256 of the whole file, the value quoted in requests and results."""
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def element(self, index):
        """Raw bytes of the element at a flat row-major index."""
        w = width(self.dtype)
        return self.data[index * w:(index + 1) * w]


def from_bytes(raw):
    if len(raw) < 16 or raw[:8] != MAGIC:
        raise TensorFormatError('missing VTTENSR1 magic')
    code, rank = raw[8], raw[9]
    if code not in NAMES:
        raise TensorFormatError(f'unknown dtype code {code}')
    if rank > MAX_RANK or any(raw[10:16]):
        raise TensorFormatError('rank above 8 or non-zero reserved bytes')
    if len(raw) < 16 + 8 * rank:
        raise TensorFormatError('truncated dimensions')
    shape = struct.unpack(f'<{rank}Q', raw[16:16 + 8 * rank])
    return Tensor(NAMES[code], shape, bytes(raw[16 + 8 * rank:]))


def read(path, sha256=None):
    """Read a tensor file; with `sha256`, reject a file whose hash differs."""
    raw = Path(path).read_bytes()
    if sha256 is not None and hashlib.sha256(raw).hexdigest() != sha256:
        raise TensorFormatError(f'{path}: sha256 differs from the expected {sha256}')
    return from_bytes(raw)


def write(path, tensor):
    Path(path).write_bytes(tensor.to_bytes())
    return tensor.sha256()


def sentinel_bytes(dtype, sentinel=None):
    """Element bytes of a sentinel given as '0x' plus exactly 2 * width hex digits."""
    text = DEFAULT_SENTINELS.get(dtype) if sentinel is None else sentinel
    w = width(dtype)
    if text is None:
        raise TensorFormatError(f'{dtype} has no sentinel; use a wider shadow buffer')
    if not text.startswith('0x') or len(text) != 2 + 2 * w:
        raise TensorFormatError(f'sentinel {text!r} is not 0x followed by {2 * w} hex digits')
    return int(text, 16).to_bytes(w, 'little')


def filled(dtype, shape, sentinel=None):
    """An output buffer in which every element is the sentinel."""
    return Tensor(dtype, tuple(shape), sentinel_bytes(dtype, sentinel) * count(shape))


def written_mask(tensor, sentinel=None):
    """Per element, whether its bytes differ from the sentinel (the write footprint)."""
    mark = sentinel_bytes(tensor.dtype, sentinel)
    return [tensor.element(i) != mark for i in range(count(tensor.shape))]
