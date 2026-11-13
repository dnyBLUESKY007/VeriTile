"""Agreement acceptance: both GPU-side outputs and development-side judgments."""
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
from gpurun import run
from gpurun.test_gpurun import tensor_input
from interp import compare_pair, tensorfile
from interp.tensorfile import Tensor

KERNELS = '''
import struct
from interp.tensorfile import Tensor

def original(inputs, outputs, launch):
    assert launch['grid'] == [1]
    outputs['out'] = inputs['x']
    inputs.clear()
    launch['grid'].append(2)

def candidate(inputs, outputs, launch):
    assert launch['grid'] == [1]
    values = list(struct.unpack('<8f', inputs['x'].data))
    values[0] += launch.get('delta', 0.001)
    outputs['out'] = Tensor('fp32', (8,), struct.pack('<8f', *values))

def boom(inputs, outputs, launch):
    raise RuntimeError('pair stopped')
'''


def case(**extra):
    return {'id': 'pair/agree', 'frontend': 'mock',
            'pair': {'original': 'kernels.py::original', 'candidate': 'kernels.py::candidate'},
            'launch': {'grid': [1]}, 'inputs': [tensor_input('x', 1)],
            'outputs': [{'name': 'out', 'dtype': 'fp32', 'shape': [8]}], **extra}


def fp32(*values):
    return Tensor('fp32', (len(values),), struct.pack(f'<{len(values)}f', *values))


class ComparePair(unittest.TestCase):
    def test_three_way_strict_failure_inside_gate(self):
        checks = compare_pair.compare_pair(fp32(1), fp32(1.001), fp32(1))
        self.assertEqual(list(checks), ['original/candidate', 'original/reference', 'candidate/reference'])
        self.assertTrue(all(result['passed'] for result in checks.values()))
        self.assertEqual([result['gate'] for result in checks.values()], [0, 0, 0])
        self.assertEqual([result['strict'] for result in checks.values()], [1, 0, 1])

    def test_integer_footprint_nan_and_infinity_exact_checks(self):
        integer = lambda value: Tensor('int32', (1,), struct.pack('<i', value))
        cases = [('exact', integer(1), integer(2)),
                 ('footprint', tensorfile.filled('fp32', [1]), fp32(1)),
                 ('nan_inf', fp32(float('nan'), 1), fp32(1, float('nan'))),
                 ('nan_inf', fp32(float('inf')), fp32(float('-inf')))]
        for column, original, candidate in cases:
            with self.subTest(column=column):
                checks = compare_pair.compare_pair(original, candidate, original)
                self.assertGreater(checks['original/candidate'][column], 0)
                self.assertGreater(checks['candidate/reference'][column], 0)
                self.assertFalse(checks['original/candidate']['passed'])
                self.assertTrue(checks['original/reference']['passed'])

    def test_gate_failure_and_optional_reference(self):
        checks = compare_pair.compare_pair(fp32(1), fp32(2))
        self.assertEqual(list(checks), ['original/candidate'])
        self.assertEqual(checks['original/candidate']['gate'], 1)
        self.assertFalse(checks['original/candidate']['passed'])
        with self.assertRaises(ValueError):
            compare_pair.compare_pair(fp32(1), fp32(1, 2))


class Agree(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / 'repo'
        self.root.mkdir()
        (self.root / 'kernels.py').write_text(KERNELS)
        self.batch = self.root / 'batch'
        self.batch.mkdir()
        self.out = Path(temp.name) / 'result'

    def write_batch(self, tests):
        (self.batch / 'batch.json').write_text(json.dumps(
            {'protocol': 1, 'batch_id': 'agree-self-test', 'mode': 'agree',
             'requires': {}, 'timeout_s': 30, 'tests': tests}))

    def run_batch(self, tests, env=None):
        self.write_batch(tests)
        self.assertEqual(run.run_batch(self.batch, self.out, env=env or {'gpus': []}, root=self.root), 0)

    def test_both_outputs_hashes_reference_and_same_inputs(self):
        self.run_batch([case()])
        meta = json.loads((self.out / 'pair/agree/meta.json').read_text())
        self.assertEqual(meta['status'], 'ok')
        tensors = {}
        for name in ('original', 'candidate'):
            tensor = tensorfile.read(self.out / f'pair/agree/{name}/out.tensor')
            tensors[name] = tensor
            self.assertEqual(meta['outputs'][f'{name}/out.tensor'], tensor.sha256())
        self.assertNotEqual(tensors['original'].data, tensors['candidate'].data)
        self.assertEqual(tensors['original'].sha256(), case()['inputs'][0]['sha256'])
        references = {'pair/agree': {'out': tensors['original']}}
        report = compare_pair.build(self.batch, self.out, references)
        self.assertTrue(report['passed'])
        self.assertTrue(report['reference_requested'])
        self.assertEqual([row['strict'] for row in report['rows']], [1, 0, 1])
        self.assertEqual([row['gate'] for row in report['rows']], [0, 0, 0])
        self.assertEqual(len(compare_pair.build(self.batch, self.out)['rows']), 1)
        for path in self.out.rglob('*'):
            if path.is_file() and path.suffix != '.tensor':
                self.assertNotRegex(path.read_text(), r'\b(PASS|FAIL|passed|speedup|verdict)\b')
                self.assertNotIn(str(self.root), path.read_text())

    def test_failed_kernel_environment_and_bad_input_are_not_judged(self):
        bad = case(id='mismatch')
        bad['inputs'][0]['sha256'] = '0' * 64
        self.run_batch([case(id='no_gpu', frontend='triton'), bad,
                        case(id='error', pair={'original': 'kernels.py::original', 'candidate': 'kernels.py::boom'})])
        manifest = json.loads((self.out / 'manifest.json').read_text())
        self.assertEqual(manifest['tests'], {'no_gpu': 'skipped_env', 'mismatch': 'input_mismatch', 'error': 'runtime_error'})
        report = compare_pair.build(self.batch, self.out)
        self.assertFalse(report['passed'])
        self.assertTrue(all(row['against'] is None for row in report['rows']))
        self.assertFalse((self.out / 'error/original').exists())

    def test_missing_frontend_is_skipped_even_with_visible_gpu(self):
        self.run_batch([case(frontend='triton')], env={'gpus': [{'name': 'mock GPU'}], 'versions': {'torch': '2.0'}})
        meta = json.loads((self.out / 'pair/agree/meta.json').read_text())
        self.assertEqual(meta['status'], 'skipped_env')
        self.assertEqual(meta['reasons'], ['needs triton in the environment'])

    def test_missing_reference_output_and_incomplete_or_wrong_bundle_fail(self):
        self.run_batch([case()])
        self.assertFalse(compare_pair.build(self.batch, self.out, {})['passed'])
        self.assertIn('missing requested interpreter reference', compare_pair.build(self.batch, self.out, {})['rows'][0]['error'])
        (self.out / 'pair/agree/candidate/out.tensor').unlink()
        self.assertFalse(compare_pair.build(self.batch, self.out)['passed'])
        (self.out / 'DONE').unlink()
        with self.assertRaises(ValueError):
            compare_pair.build(self.batch, self.out)
        (self.out / 'DONE').touch()
        manifest = json.loads((self.out / 'manifest.json').read_text())
        manifest['batch_id'] = 'another-batch'
        (self.out / 'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            compare_pair.build(self.batch, self.out)

    def test_invalid_requests_are_refused_before_writes(self):
        for extra in ({'pair': {}}, {'repeats': 2}, {'repeats': True}, {'warmup': 0},
                      {'pair': {'original': '/outside.py::run', 'candidate': 'kernels.py::candidate'}},
                      {'outputs': [{'name': '../escape', 'dtype': 'fp32', 'shape': [8]}]}, {'device': 1}):
            with self.subTest(extra=extra):
                self.write_batch([case(**extra)])
                with self.assertRaises(ValueError):
                    run.run_batch(self.batch, self.out, env={'gpus': []}, root=self.root)
                self.assertFalse(self.out.exists())

    def test_public_runner_and_comparator_cli_readback(self):
        for package in ('gpurun', 'interp'):
            shutil.copytree(SCRIPTS / package, self.root / 'scripts' / package,
                            ignore=shutil.ignore_patterns('__pycache__'))
        env = {**run.child_env(), 'PYTHONPATH': str(self.root / 'scripts')}
        self.write_batch([case()])
        commands = [['gpurun', 'check-env', str(self.batch)],
                    ['gpurun', 'run', str(self.batch), '--out', str(self.out)]]
        for command in commands:
            result = subprocess.run([sys.executable, '-m', *command], cwd=self.root, env=env,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        references = self.root / 'references/pair/agree'
        references.mkdir(parents=True)
        shutil.copyfile(self.out / 'pair/agree/original/out.tensor', references / 'out.tensor')
        command = [sys.executable, '-m', 'interp.compare_pair', str(self.batch), str(self.out)]
        for options, count in (([], 1), (['--references', str(self.root / 'references')], 3)):
            result = subprocess.run(command + options, cwd=self.root, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(json.loads(result.stdout)['rows']), count)
        tensorfile.write(self.out / 'pair/agree/candidate/out.tensor', fp32(*([99] * 8)))
        result = subprocess.run(command, cwd=self.root, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertFalse(json.loads(result.stdout)['passed'])
        (self.out / 'DONE').unlink()
        result = subprocess.run(command, cwd=self.root, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('no DONE file', result.stderr)


if __name__ == '__main__':
    unittest.main()
