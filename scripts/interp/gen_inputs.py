"""Fixed-seed test inputs with the same bytes on every machine.

`generate(dtype, shape, gen)` returns a tensor for a request's `gen` object,
for example `{"class": "normal", "seed": 1}`. Everything is integer
arithmetic plus exactly rounded conversions, so the result does not depend on
the platform, the Python version or a math library.

Stream: SplitMix64, seeded with the first 8 bytes (little-endian) of the
SHA-256 of `veritile-inputs-v1|<class>|<dtype>|<dims joined by ,>|<seed>`.
Elements are produced in row-major order and each consumes four draws.

Classes:

- `normal`: floats are `(S - 393210) / 65536`, where `S` is the sum of the
  twelve 16-bit limbs of draws 2 to 4: an approximate standard normal with
  values in [-6, 6). Integers are uniform in `[low, high]` (`gen["low"]`,
  `gen["high"]`; default -100..100, or 0..200 unsigned). Booleans are one bit.
- `boundary`: the same values on an independent stream. The class marks tests
  whose sizes do not divide the block size; the request chooses the shape.
- `special`: one element in four is a special value (signed zeros,
  infinities, the canonical quiet NaN, subnormals, extreme finite values; for
  integers the extremes and -1, 0, 1), the others are `normal` values.

No generated tensor contains a default sentinel pattern of
documents/InterpInterface.md, and every NaN is the canonical quiet NaN.
"""
import argparse
import hashlib
import struct

if __package__:
    from .tensorfile import Tensor, count, width, write
else:  # run as a script from scripts/interp
    from tensorfile import Tensor, count, width, write

CLASSES = ('normal', 'boundary', 'special')
FLOATS = {'fp16': (5, 10), 'bf16': (8, 7), 'fp32': (8, 23), 'fp64': (11, 52)}  # exponent, mantissa bits
MASK = (1 << 64) - 1


def stream(seed, cls, dtype, shape):
    key = f"veritile-inputs-v1|{cls}|{dtype}|{','.join(map(str, shape))}|{seed}"
    state = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little')
    while True:
        state = (state + 0x9E3779B97F4A7C15) & MASK
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        yield z ^ (z >> 31)


def encode_float(dtype, x):
    """Bytes of the value nearest to the double `x`, ties to even."""
    if dtype == 'fp64':
        return struct.pack('<d', x)
    if dtype == 'fp32':
        return struct.pack('<f', x)
    if dtype == 'fp16':
        return struct.pack('<e', x)
    # bf16 keeps the upper half of a binary32. Every value produced here is
    # exact in binary32, so rounding its bits once is a single rounding.
    bits = struct.unpack('<I', struct.pack('<f', x))[0]
    upper, lower = bits >> 16, bits & 0xFFFF
    if lower > 0x8000 or (lower == 0x8000 and upper & 1):
        upper += 1
    return upper.to_bytes(2, 'little')


def float_specials(dtype):
    exponent, mantissa = FLOATS[dtype]
    sign = 1 << (exponent + mantissa)
    top = ((1 << exponent) - 1) << mantissa          # +inf
    largest = top - 1                                # largest finite
    patterns = [0, sign, top, sign | top, top | 1 << (mantissa - 1),  # zeros, infinities, quiet NaN
                1, (1 << mantissa) - 1, sign | 1,                     # subnormals
                largest, sign | largest, 1 << mantissa]               # extreme finite, smallest normal
    return [p.to_bytes(width(dtype), 'little') for p in patterns]


def int_range(dtype):
    bits = 8 * width(dtype)
    return (0, (1 << bits) - 1) if dtype.startswith('uint') else (-(1 << (bits - 1)), (1 << (bits - 1)) - 1)


def generate(dtype, shape, gen):
    cls, seed = gen['class'], gen['seed']
    if cls not in CLASSES:
        raise ValueError(f'unknown input class {cls!r}')
    if dtype in ('f8e4', 'f8e5'):
        # shortcut: no FP8 rounding here; add it when a bench kernel takes FP8 inputs.
        raise NotImplementedError(f'no generator for {dtype}')
    draws = stream(seed, cls, dtype, shape)
    w, out = width(dtype), bytearray()
    if dtype in FLOATS:
        specials = float_specials(dtype)
    elif dtype != 'bool':
        lo, hi = int_range(dtype)
        signed = lo < 0
        low = gen.get('low', -100 if signed else 0)
        high = gen.get('high', 100 if signed else 200)
        if not lo <= low <= high <= hi:
            raise ValueError(f'range [{low}, {high}] does not fit {dtype}')
        # The minimum of a signed type is its default sentinel, so it is left out.
        specials = [lo + 1, hi, 0, -1, 1] if signed else [0, hi, 1, hi - 1]
    for _ in range(count(shape)):
        selector, a, b, c = next(draws), next(draws), next(draws), next(draws)
        special = cls == 'special' and selector % 4 == 0
        if dtype == 'bool':
            out.append(a & 1)
        elif dtype in FLOATS:
            if special:
                out += specials[(selector >> 2) % len(specials)]
            else:
                limbs = sum((v >> shift) & 0xFFFF for v in (a, b, c) for shift in (0, 16, 32, 48))
                out += encode_float(dtype, (limbs - 393210) / 65536)
        else:
            value = specials[(selector >> 2) % len(specials)] if special else low + a % (high - low + 1)
            out += value.to_bytes(w, 'little', signed=signed)
    return Tensor(dtype, tuple(shape), bytes(out))


def main():
    parser = argparse.ArgumentParser(description='Generate one input tensor and print its sha256.')
    parser.add_argument('--dtype', required=True)
    parser.add_argument('--shape', type=int, nargs='*', default=[])
    parser.add_argument('--class', dest='cls', choices=CLASSES, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--out', help='also write the tensor file here')
    args = parser.parse_args()
    tensor = generate(args.dtype, args.shape, {'class': args.cls, 'seed': args.seed})
    if args.out:
        write(args.out, tensor)
    print(tensor.sha256())


if __name__ == '__main__':
    main()
