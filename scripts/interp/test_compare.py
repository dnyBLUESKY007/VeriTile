#!/usr/bin/env python3
"""Self-tests of the comparator and the per-batch report.

Three planted faults must be caught: a wrong boundary mask (by the footprint
check), a missing scale factor that stays inside the 1e-2 gate (by the strict
column only) and a misplaced NaN.
"""
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpurun import run
from interp import gen_inputs, report, tensorfile
from interp.compare import compare
from interp.tensorfile import Tensor

N, LIVE, NAN_AT, SCALE = 12, 10, 3, 1.004
GEN = {'class': 'normal', 'seed': 1}
X = gen_inputs.generate('fp32', [N], GEN)
FAULTS = ('mask', 'scale', 'nan')


def kernel(inputs, outputs, launch):
    """Mock kernel, run by gpurun from this file and called directly below.

    out[i] = SCALE * x[i] for i < LIVE, except a NaN at NAN_AT; the tail keeps
    the sentinel. `launch['fault']` plants one fault: `mask` writes the tail
    too, `scale` drops SCALE, `nan` puts the NaN one element later.
    """
    fault = launch.get('fault')
    out = [struct.pack('<f', v * (1 if fault == 'scale' else SCALE)) for v in struct.unpack(f'<{N}f', inputs['x'].data)]
    out[NAN_AT + (fault == 'nan')] = bytes.fromhex('0000c07f')
    live = N if fault == 'mask' else LIVE
    outputs['out'] = Tensor('fp32', (N,), b''.join(out[:live]) + outputs['out'].data[4 * live:])


def output(fault=None):
    outputs = {'out': tensorfile.filled('fp32', (N,))}
    kernel({'x': X}, outputs, {'fault': fault})
    return outputs['out']


def bits(dtype, *patterns):
    w = tensorfile.width(dtype)
    return Tensor(dtype, (len(patterns),), b''.join(p.to_bytes(w, 'little') for p in patterns))


def fp32(*values):
    return Tensor('fp32', (len(values),), struct.pack(f'<{len(values)}f', *values))


class Compare(unittest.TestCase):
    def test_identical_outputs_pass(self):
        self.assertEqual(compare(output(), output()), {
            'footprint': 0, 'exact': None, 'nan_inf': 0, 'gate': 0, 'strict': 0,
            'max_abs': 0, 'max_rel': 0, 'max_ulp': 0, 'passed': True})

    def test_planted_wrong_boundary_mask_is_caught_by_the_footprint(self):
        result = compare(output('mask'), output())
        self.assertEqual(result['footprint'], N - LIVE)
        self.assertEqual((result['nan_inf'], result['gate'], result['strict']), (0, 0, 0))
        self.assertFalse(result['passed'])
        self.assertEqual(compare(output(), output('mask'))['footprint'], N - LIVE)

    def test_planted_missing_scale_is_caught_by_the_strict_column_only(self):
        result = compare(output('scale'), output())
        self.assertEqual((result['footprint'], result['nan_inf'], result['gate']), (0, 0, 0))
        self.assertTrue(result['passed'])
        self.assertEqual(result['strict'], LIVE - 1)  # every live element but the NaN
        self.assertTrue(3.9e-3 < result['max_rel'] < 4.0e-3, result['max_rel'])

    def test_planted_misplaced_nan_is_caught(self):
        result = compare(output('nan'), output())
        self.assertEqual(result['nan_inf'], 2)
        self.assertEqual((result['footprint'], result['gate']), (0, 0))
        self.assertFalse(result['passed'])

    def test_nan_and_inf_positions_and_inf_sign(self):
        inf, nan = float('inf'), float('nan')
        self.assertEqual(compare(fp32(inf, -inf, nan), fp32(inf, -inf, nan))['nan_inf'], 0)
        self.assertEqual(compare(bits('fp32', 0xffc00001), fp32(nan))['nan_inf'], 0)  # sign and payload are free
        result = compare(fp32(inf, inf, 1.0, nan), fp32(-inf, 3e38, inf, inf))
        self.assertEqual((result['nan_inf'], result['gate'], result['passed']), (4, 0, False))

    def test_gate_strict_and_error_figures(self):
        result = compare(fp32(1.05, 100.5, 0.0, 2.0), fp32(1.0, 100.0, 0.001, 2.0))
        self.assertEqual((result['gate'], result['strict'], result['passed']), (1, 3, False))
        self.assertAlmostEqual(result['max_abs'], 0.5, places=6)
        self.assertAlmostEqual(result['max_rel'], 1.0, places=6)
        self.assertEqual(compare(fp32(1.0), fp32(0.0))['max_rel'], 0)  # a zero reference has no relative error
        fp64 = compare(bits('fp64', 0x3ff0000000000001), bits('fp64', 0x3ff0000000000000))
        self.assertEqual((fp64['gate'], fp64['strict'], fp64['max_ulp'], fp64['max_abs']), (0, None, 1, 2.0 ** -52))

    def test_ulp_distance_is_counted_on_bit_patterns(self):
        result = compare(bits('fp32', 0x80000000, 0x00000001, 0x3f800000), bits('fp32', 0, 0x80000001, 0x3f800003))
        self.assertEqual(result['max_ulp'], 3)
        self.assertEqual(compare(bits('fp32', 1), bits('fp32', 0x80000001))['max_ulp'], 2)  # across zero
        for dtype, one, step in (('bf16', 0x3f80, 2.0 ** -7), ('fp16', 0x3c00, 2.0 ** -10)):
            with self.subTest(dtype):
                result = compare(bits(dtype, one + 1, one | 0x8000), bits(dtype, one, one | 0x8000))
                self.assertEqual((result['max_ulp'], result['max_abs'], result['strict']), (1, step, None))
                self.assertTrue(result['passed'])

    def test_integers_and_booleans_are_exact(self):
        unwritten = 0x80000000  # the default int32 sentinel
        result = compare(bits('int32', 1, 0xfffffffe, unwritten), bits('int32', 1, 3, 7))
        self.assertEqual(result, {'footprint': 1, 'exact': 1, 'nan_inf': None, 'gate': None, 'strict': None,
                                  'max_abs': 5, 'max_rel': 5 / 3, 'max_ulp': 5, 'passed': False})
        self.assertTrue(compare(bits('int32', 1, 2), bits('int32', 1, 2))['passed'])
        self.assertEqual(compare(bits('int16', 5, 0x1234), bits('int16', 5, 6), '0x1234')['footprint'], 1)
        self.assertEqual(compare(bits('uint64', 2 ** 64 - 1), bits('uint64', 1))['max_abs'], 2 ** 64 - 2)
        result = compare(bits('bool', 1, 0, 1), bits('bool', 1, 1, 1))
        self.assertEqual((result['footprint'], result['exact'], result['passed']), (None, 1, False))

    def test_fp8_and_mismatched_tensors_are_refused(self):
        with self.assertRaises(NotImplementedError):
            compare(bits('f8e4', 1), bits('f8e4', 1))
        for other in (fp32(1.0, 2.0), bits('int32', 1)):
            with self.assertRaises(ValueError):
                compare(fp32(1.0), other)


class Report(unittest.TestCase):
    def test_report_over_a_bundle_written_by_gpurun(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        batch_dir, bundle, reference_dir = (Path(temp.name) / name for name in ('batch', 'bundle', 'references'))

        def case(test_id, **extra):
            return {'id': test_id, 'frontend': 'mock', 'entry': 'test_compare.py::kernel', 'launch': {},
                    'inputs': [{'name': 'x', 'dtype': 'fp32', 'shape': [N], 'gen': GEN, 'sha256': X.sha256()}],
                    'outputs': [{'name': 'out', 'dtype': 'fp32', 'shape': [N], 'sentinel': '0x7fc0dead'}],
                    'reference': {'kind': 'none'}, 'repeats': 1, **extra}

        tests = [case('clean', reference={'kind': 'torch', 'entry': 'test_compare.py::kernel'}),
                 *(case(f'fault/{fault}', launch={'fault': fault}) for fault in FAULTS),
                 case('crash', entry='test_compare.py::no_such_kernel'),
                 case('unjudged')]
        batch_dir.mkdir()
        (batch_dir / 'batch.json').write_text(json.dumps(
            {'protocol': 1, 'batch_id': 'self-test', 'mode': 'compare', 'requires': {}, 'timeout_s': 30, 'tests': tests}))
        run.run_batch(batch_dir, bundle, env={'gpus': [], 'versions': {}}, root=Path(__file__).resolve().parent)

        references = {test_id: {'out': output()} for test_id in ('clean', 'crash', *(f'fault/{f}' for f in FAULTS))}
        built = report.build(batch_dir, bundle, references)
        self.assertEqual(json.loads(json.dumps(built)), built)
        self.assertEqual((built['batch_id'], built['passed']), ('self-test', False))
        rows = {(row['test'], row['against']): row for row in built['rows']}
        self.assertEqual(list(rows), [('clean', 'reference'), ('clean', 'torch'), ('fault/mask', 'reference'),
                                      ('fault/scale', 'reference'), ('fault/nan', 'reference'),
                                      ('crash', None), ('unjudged', None)])
        self.assertEqual({key: row['passed'] for key, row in rows.items() if row['passed']},
                         {('clean', 'reference'): True, ('clean', 'torch'): True, ('fault/scale', 'reference'): True})
        self.assertEqual(rows['fault/mask', 'reference']['footprint'], N - LIVE)
        self.assertEqual((rows['fault/scale', 'reference']['gate'], rows['fault/scale', 'reference']['strict']), (0, LIVE - 1))
        self.assertEqual(rows['fault/nan', 'reference']['nan_inf'], 2)
        self.assertEqual((rows['crash', None]['status'], rows['unjudged', None]['status']), ('runtime_error', 'ok'))
        self.assertEqual(rows['unjudged', None]['error'], 'no reference tensor')

        for test_id, tensors in references.items():
            (reference_dir / test_id).mkdir(parents=True)
            tensorfile.write(reference_dir / test_id / 'out.tensor', tensors['out'])
        self.assertEqual(report.build(batch_dir, bundle, reference_dir), built)

        markdown = report.render(built).splitlines()
        self.assertEqual(markdown[0], '## Batch `self-test`: FAIL')
        self.assertIn('| clean | out | ok | torch | 0 | - | 0 | 0 | 0 | 0 | 0 | 0 | pass |', markdown)
        self.assertIn('| fault/mask | out | ok | reference | 2 | - | 0 | 0 | 0 | 0 | 0 | 0 | FAIL |', markdown)
        self.assertIn('| crash | out | runtime_error | - | - | - | - | - | - | - | - | - | FAIL |', markdown)
        self.assertIn('| unjudged | out | ok | - | - | - | - | - | - | - | - | - | FAIL: no reference tensor |', markdown)
        scale = next(line for line in markdown if line.startswith('| fault/scale |'))
        self.assertRegex(scale, r'^\| fault/scale \| out \| ok \| reference \| 0 \| - \| 0 \| 0 \| 9 \| 0\.0\d+ \| 0\.00398 \| \d+ \| pass \|$')

        (bundle / 'DONE').unlink()
        with self.assertRaises(ValueError):
            report.build(batch_dir, bundle, references)


if __name__ == '__main__':
    unittest.main()
