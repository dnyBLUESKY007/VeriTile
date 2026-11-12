#!/usr/bin/env python3
"""Judge the result bundle of a smoke batch on the development side.

    python3 experiments/interp_gpu/smoke/judge.py <result-bundle-dir> [<batch-id>]

The batch id defaults to `smoke-triton-001`.

The references are computed here from the regenerated inputs: a copy must
return the input bit for bit, the half write must leave the second half as
sentinel, and the sum is the binary32 sum with the tail left as sentinel.
"""
from pathlib import Path
import json
import struct
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
from interp import gen_inputs, report, tensorfile
from interp.tensorfile import Tensor

BATCHES = ROOT / 'experiments/interp_gpu/batches'


def references(batch):
    result = {}
    for test in json.loads((batch / 'batch.json').read_text())['tests']:
        inputs = {i['name']: gen_inputs.generate(i['dtype'], i['shape'], i['gen']) for i in test['inputs']}
        out = test['outputs'][0]
        blank = tensorfile.filled(out['dtype'], out['shape'], out['sentinel'])
        x = inputs['x']
        if test['entry'].endswith('::copy'):
            reference = x
        elif test['entry'].endswith('::half_write'):
            half = len(x.data) // 2
            reference = Tensor(x.dtype, x.shape, x.data[:half] + blank.data[half:])
        else:  # ::add; each sum is exact in a double, so packing rounds once, as binary32 addition does
            n = x.shape[0]
            sums = [a + b for a, b in zip(struct.unpack(f'<{n}f', x.data), struct.unpack(f'<{n}f', inputs['y'].data))]
            reference = Tensor('fp32', blank.shape, struct.pack(f'<{n}f', *sums) + blank.data[4 * n:])
        result[test['id']] = {'out': reference}
    return result


if __name__ == '__main__':
    batch = BATCHES / (sys.argv[2] if len(sys.argv) > 2 else 'smoke-triton-001')
    built = report.build(batch, sys.argv[1], references(batch))
    print(report.render(built))
    raise SystemExit(0 if built['passed'] else 1)
