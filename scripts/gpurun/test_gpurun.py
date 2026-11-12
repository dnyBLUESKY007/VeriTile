#!/usr/bin/env python3
"""Self-test of gpurun with a CPU mock launcher: no GPU, torch or kernel needed."""
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
from gpurun import bundle, env as environment, run
from interp import gen_inputs, tensorfile

KERNELS = '''
import struct, time
from interp.tensorfile import Tensor

class CompilationError(Exception):
    pass

def add_all_but_last(inputs, outputs, launch):
    n = inputs['x'].shape[0]
    x = struct.unpack(f'<{n}f', inputs['x'].data)
    y = struct.unpack(f'<{n}f', inputs['y'].data)
    kept = outputs['out'].data[4 * (n - 1):]
    outputs['out'] = Tensor('fp32', (n,), struct.pack(f'<{n - 1}f', *[a + b for a, b in zip(x[:-1], y[:-1])]) + kept)

def boom(inputs, outputs, launch):
    raise RuntimeError('boom in ' + __file__)

def cannot_compile(inputs, outputs, launch):
    raise CompilationError('unsupported construct')

def hang(inputs, outputs, launch):
    time.sleep(60)
'''


def tensor_input(name, seed, shape=(8,)):
    gen = {'class': 'normal', 'seed': seed}
    return {'name': name, 'dtype': 'fp32', 'shape': list(shape), 'gen': gen,
            'sha256': gen_inputs.generate('fp32', shape, gen).sha256()}


def test_case(test_id, function, **extra):
    return {'id': test_id, 'frontend': 'mock', 'entry': f'kernels.py::{function}', 'launch': {'grid': [1]},
            'inputs': [tensor_input('x', 1), tensor_input('y', 2)],
            'outputs': [{'name': 'out', 'dtype': 'fp32', 'shape': [8], 'sentinel': '0x7fc0dead'}],
            'reference': {'kind': 'none'}, 'repeats': 1, **extra}


class Gpurun(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / 'repo'
        self.root.mkdir()
        (self.root / 'kernels.py').write_text(KERNELS)
        self.batch = self.root / 'batch'
        self.batch.mkdir()
        self.out = Path(temp.name) / 'result'

    def write_batch(self, tests, **fields):
        (self.batch / 'batch.json').write_text(json.dumps(
            {'protocol': 1, 'batch_id': 'self-test', 'mode': 'compare', 'supersedes': None,
             'requires': {}, 'timeout_s': 30, 'tests': tests, **fields}))

    def test_batch_records_each_outcome_and_keeps_going(self):
        wrong = test_case('group/input_mismatch', 'add_all_but_last')
        wrong['inputs'][0]['sha256'] = '0' * 64
        self.write_batch([
            test_case('group/ok', 'add_all_but_last', reference={'kind': 'torch', 'entry': 'kernels.py::add_all_but_last'}),
            test_case('group/runtime_error', 'boom'),
            test_case('group/compile_error', 'cannot_compile'),
            wrong,
            test_case('group/timeout', 'hang', timeout_s=1),
            test_case('after_failures', 'add_all_but_last'),
        ])
        self.assertEqual(run.run_batch(self.batch, self.out, env={'gpus': [], 'versions': {}}, root=self.root), 0)
        manifest = json.loads((self.out / 'manifest.json').read_text())
        self.assertEqual(manifest['tests'], {
            'group/ok': 'ok', 'group/runtime_error': 'runtime_error', 'group/compile_error': 'compile_error',
            'group/input_mismatch': 'input_mismatch', 'group/timeout': 'timeout', 'after_failures': 'ok'})
        self.assertTrue((self.out / 'DONE').is_file() and (self.out / 'env.json').is_file())

        result = tensorfile.read(self.out / 'group/ok/out/out.tensor')
        x = struct.unpack('<8f', gen_inputs.generate('fp32', (8,), {'class': 'normal', 'seed': 1}).data)
        y = struct.unpack('<8f', gen_inputs.generate('fp32', (8,), {'class': 'normal', 'seed': 2}).data)
        self.assertEqual(struct.unpack('<7f', result.data[:28]), struct.unpack('<7f', struct.pack('<7f', *[a + b for a, b in zip(x, y)][:7])))
        self.assertEqual(tensorfile.written_mask(result), [True] * 7 + [False])
        meta = json.loads((self.out / 'group/ok/meta.json').read_text())
        self.assertEqual(meta['outputs']['out/out.tensor'], result.sha256())
        self.assertEqual(meta['outputs']['ref/out.tensor'], result.sha256())

        error_log = (self.out / 'group/runtime_error/stderr.log').read_text()
        self.assertIn('RuntimeError: boom in <repo>/kernels.py', error_log)
        self.assertNotIn(str(self.root), error_log)
        self.assertEqual(bundle.scrub('rendezvous at 127.0.0.1:23456', self.root), 'rendezvous at <loopback>:23456')
        self.assertFalse((self.out / 'group/runtime_error/out').exists())
        mismatch = json.loads((self.out / 'group/input_mismatch/meta.json').read_text())
        self.assertFalse(mismatch['inputs']['x']['match'])
        for path in self.out.rglob('*'):
            if path.is_file() and path.suffix != '.tensor':
                self.assertNotRegex(path.read_text(), r'\b(PASS|FAIL)\b', path)

    def test_environment_mismatch_skips_every_test(self):
        self.write_batch([test_case('a', 'add_all_but_last'), test_case('b', 'boom')],
                         requires={'sm': 90, 'gpus': 1, 'env': 'triton'})
        run.run_batch(self.batch, self.out, env={'gpus': [], 'versions': {'triton': '3.7.1'}}, root=self.root)
        manifest = json.loads((self.out / 'manifest.json').read_text())
        self.assertEqual(manifest['tests'], {'a': 'skipped_env', 'b': 'skipped_env'})
        self.assertEqual(len(manifest['env_mismatch']), 2)
        self.assertIn('3.5.1', json.loads((self.out / 'a/meta.json').read_text())['reasons'][1])

    def test_pins_and_hardware_requirements(self):
        h800 = {'gpus': [{'name': 'H800', 'sm': 90}] * 2, 'versions': {'tilelang': '0.0.2+cuda.git71e71457', 'triton': '3.5.1'}}
        self.assertEqual(environment.mismatches({'sm': 90, 'gpus': 2, 'env': 'tilescale'}, h800), [])
        self.assertEqual(environment.mismatches({'sm': 90, 'gpus': 1, 'env': 'triton'}, h800), [])
        self.assertEqual(len(environment.mismatches({'env': 'tilelang'}, h800)), 1)
        # Same module name, different commit: the two TileScale environments never stand in for each other.
        self.assertEqual(len(environment.mismatches({'env': 'tilescale-deepep'}, h800)), 1)
        deepep = {**h800, 'versions': {'tilelang': '0.1.7.post1+cuda.git9ba8a605'}}
        self.assertEqual(environment.mismatches({'sm': 90, 'gpus': 2, 'env': 'tilescale-deepep'}, deepep), [])
        self.assertEqual(len(environment.mismatches({'env': 'tilescale'}, deepep)), 1)
        self.assertEqual(len(environment.mismatches({'env': 'triton-dist'}, h800)), 1)
        self.assertEqual(len(environment.mismatches({'sm': 90, 'gpus': 3}, h800)), 1)
        self.assertEqual(len(environment.mismatches({'sm': 100, 'gpus': 1}, h800)), 1)
        self.assertEqual(len(environment.mismatches({'nodes': 2}, h800)), 1)
        self.assertEqual(len(environment.mismatches({'env': 'cutile'}, h800)), 1)

    def test_command_line(self):
        command = [sys.executable, '-m', 'gpurun']
        env = dict(os.environ, PYTHONPATH=str(SCRIPTS))
        self.write_batch([], requires={'env': 'no-such-environment'})  # refused on every host
        refused = subprocess.run([*command, 'check-env', str(self.batch)], env=env, capture_output=True, text=True)
        self.assertEqual(refused.returncode, 1, refused.stderr)
        self.assertIn('refused:', refused.stderr)
        self.assertIn('"python"', refused.stdout)
        self.write_batch([], mode='litmus')
        unsupported = subprocess.run([*command, 'run', str(self.batch), '--out', str(self.out)], env=env, capture_output=True, text=True)
        self.assertEqual(unsupported.returncode, 2)
        self.assertFalse(self.out.exists())
        self.write_batch([], mode='env-check')
        checked = subprocess.run([*command, 'run', str(self.batch), '--out', str(self.out)], env=env, capture_output=True, text=True)
        self.assertEqual(checked.returncode, 0, checked.stderr)
        self.assertEqual(sorted(p.name for p in self.out.iterdir()), ['DONE', 'env.json', 'manifest.json'])
        again = subprocess.run([*command, 'run', str(self.batch), '--out', str(self.out)], env=env, capture_output=True, text=True)
        self.assertEqual(again.returncode, 2)  # never overwrite an existing bundle

    def test_unsafe_requests_are_refused(self):
        self.write_batch([test_case('../escape', 'boom')])
        with self.assertRaises(ValueError):
            run.load_batch(self.batch)
        self.write_batch([test_case('outside', 'boom', entry='../kernels.py::boom')])
        run.run_batch(self.batch, self.out, env={'gpus': [], 'versions': {}}, root=self.root / 'batch')
        self.assertEqual(json.loads((self.out / 'manifest.json').read_text())['tests'], {'outside': 'runtime_error'})

    def test_logs_are_truncated_in_the_middle(self):
        path = self.root / 'log'
        bundle.write_log(path, b'start' + b'x' * (3 << 20) + b'end', self.root)
        data = path.read_bytes()
        self.assertLessEqual(len(data), bundle.LOG_LIMIT)
        self.assertTrue(data.startswith(b'start') and data.endswith(b'end') and bundle.MARK in data)


if __name__ == '__main__':
    unittest.main()
