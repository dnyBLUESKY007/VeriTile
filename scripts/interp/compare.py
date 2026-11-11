"""Development-side comparison of a GPU output tensor with a reference tensor.

`compare(gpu, ref, sentinel)` returns, per check, the number of violating
elements (`None` when the check does not apply to the dtype):

- `footprint`: positions written by exactly one side, where "not written"
  means the bytes still equal the sentinel. The value checks below look only
  at positions written by both sides.
- `exact`: integers and booleans that differ.
- `nan_inf`: floats where one side is NaN or infinite and the other is not the
  same kind (NaN with NaN, or the same signed infinity).
- `gate`: finite floats with `|gpu - ref| > 1e-2 + 1e-2 * |ref|`.
- `strict`: fp32 only, `|gpu - ref| > 1e-5 + 1e-4 * |ref|`. Reported, never
  gates.

`passed` is true when the first four counts are zero. Also reported, never
gating: `max_abs`, `max_rel` (over elements whose reference is not zero) and
`max_ulp`, the largest distance between the two bit patterns counted in
representable values. The GPU's reduction order is unspecified, so floats
gate on the tolerance only.
"""
import math
import struct

from .tensorfile import count, width, written_mask

FLOAT_CODES = {'fp16': 'e', 'fp32': 'f', 'fp64': 'd'}
UNSIGNED = {1: 'B', 2: 'H', 4: 'I', 8: 'Q'}
GATE = (1e-2, 1e-2)    # absolute, relative
STRICT = (1e-5, 1e-4)


def decode(tensor):
    """Values and ordinals of the elements.

    The ordinal of a float is its bit pattern read as sign and magnitude, so
    neighbouring floats differ by one and the two zeros coincide. An integer
    is its own ordinal.
    """
    dtype, n, w = tensor.dtype, count(tensor.shape), width(tensor.dtype)
    if dtype in ('f8e4', 'f8e5'):
        # shortcut: no FP8 decoding here; add it when a bench kernel has an FP8 output.
        raise NotImplementedError(f'no comparison for {dtype}')
    bits = struct.unpack(f'<{n}{UNSIGNED[w]}', tensor.data)
    if dtype == 'bf16':  # the upper half of a binary32
        values = struct.unpack(f'<{n}f', struct.pack(f'<{n}I', *(b << 16 for b in bits)))
    elif dtype in FLOAT_CODES:
        values = struct.unpack(f'<{n}{FLOAT_CODES[dtype]}', tensor.data)
    elif dtype.startswith('int'):
        values = struct.unpack(f'<{n}{UNSIGNED[w].lower()}', tensor.data)
        return values, values
    else:
        return bits, bits
    sign = 1 << (8 * w - 1)
    return values, [sign - b if b & sign else b for b in bits]


def compare(gpu, ref, sentinel=None):
    if (gpu.dtype, gpu.shape) != (ref.dtype, ref.shape):
        raise ValueError(f'cannot compare {gpu.dtype}{list(gpu.shape)} with {ref.dtype}{list(ref.shape)}')
    dtype = gpu.dtype
    (gpu_values, gpu_ordinals), (ref_values, ref_ordinals) = decode(gpu), decode(ref)
    floating = dtype == 'bf16' or dtype in FLOAT_CODES
    # A bool has no sentinel (InterpInterface.md, section 3): every element counts as written.
    masks = None if dtype == 'bool' else (written_mask(gpu, sentinel), written_mask(ref, sentinel))
    footprint = exact = nan_inf = gate = strict = 0
    max_abs = max_rel = max_ulp = 0
    for i, (g, r) in enumerate(zip(gpu_values, ref_values)):
        if masks and not (masks[0][i] and masks[1][i]):
            footprint += masks[0][i] != masks[1][i]
            continue
        if floating and not (math.isfinite(g) and math.isfinite(r)):
            nan_inf += not (g == r or (math.isnan(g) and math.isnan(r)))
            continue
        error = abs(g - r)
        max_abs = max(max_abs, error)
        if r:
            max_rel = max(max_rel, error / abs(r))
        max_ulp = max(max_ulp, abs(gpu_ordinals[i] - ref_ordinals[i]))
        if floating:
            gate += error > GATE[0] + GATE[1] * abs(r)
            strict += error > STRICT[0] + STRICT[1] * abs(r)
        else:
            exact += error != 0
    checks = {'footprint': footprint if masks else None, 'exact': None if floating else exact,
              'nan_inf': nan_inf if floating else None, 'gate': gate if floating else None}
    return {**checks, 'strict': strict if dtype == 'fp32' else None,
            'max_abs': max_abs, 'max_rel': max_rel, 'max_ulp': max_ulp, 'passed': not any(checks.values())}
