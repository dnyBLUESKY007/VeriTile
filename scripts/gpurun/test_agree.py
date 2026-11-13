"""Agreement acceptance: both GPU-side outputs and development-side judgments."""
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
from gpurun import agree, run
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
    def test_tolerance_uses_original_and_interpreter_as_references(self):
        original, candidate = fp32(100.0), fp32(101.0152)
        forward = compare_pair.compare_pair(original, candidate, original)
        reverse = compare_pair.compare_pair(candidate, original, candidate)
        self.assertEqual([row['passed'] for row in forward.values()], [False, True, False])
        self.assertEqual([row['passed'] for row in reverse.values()], [True, True, True])
        higher_reference = compare_pair.compare_pair(original, original, candidate)
        self.assertEqual([row['passed'] for row in higher_reference.values()], [True, True, True])

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
        self.assertEqual(meta['reasons'], ['agree needs triton in the environment'])

    def test_failed_test_gets_a_row_even_without_declared_outputs(self):
        self.run_batch([case(), case(id='error',
                       pair={'original': 'kernels.py::boom', 'candidate': 'kernels.py::candidate'})])
        # A legacy/prevalidated batch must still preserve a failing test's status.
        legacy = json.loads((self.batch / 'batch.json').read_text())
        legacy['tests'][1]['outputs'] = []
        with mock.patch.object(compare_pair, 'load_batch', return_value=legacy):
            report = compare_pair.build(self.batch, self.out)
        self.assertFalse(report['passed'])
        self.assertEqual(report['rows'][-1], {'test': 'error', 'output': None,
                         'status': 'runtime_error', 'against': None, 'passed': False})

    def test_custom_sentinel_and_request_dtype_shape_are_respected(self):
        spec = {'name': 'out', 'dtype': 'fp32', 'shape': [1], 'sentinel': '0x3f800000'}
        test = case(outputs=[spec])
        self.run_batch([case()])
        self.write_batch([test])
        original_path = self.out / 'pair/agree/original/out.tensor'
        candidate_path = self.out / 'pair/agree/candidate/out.tensor'
        tensorfile.write(original_path, fp32(1))
        tensorfile.write(candidate_path, fp32(1.00001))
        report = compare_pair.build(self.batch, self.out)
        self.assertEqual(report['rows'][0]['footprint'], 1)
        self.assertFalse(report['passed'])
        for bad in (Tensor('int32', (1,), struct.pack('<i', 1)), fp32(1, 2)):
            with self.subTest(dtype=bad.dtype, shape=bad.shape):
                for path in (original_path, candidate_path):
                    tensorfile.write(path, bad)
                report = compare_pair.build(self.batch, self.out)
                self.assertFalse(report['passed'])
                self.assertEqual(report['rows'][0]['error'], 'pair output dtype or shape does not match the request')

    def test_gpu_pair_uses_one_device_and_fresh_inputs(self):
        events = []
        cuda = types.SimpleNamespace(set_device=lambda device: events.append(('device', device)),
                                     synchronize=lambda: events.append('sync'))
        converted = []
        def convert(value):
            copy = Tensor(value.dtype, value.shape, value.data)
            converted.append(copy)
            events.append('buffer')
            return copy
        work = self.root / 'work'
        work.mkdir()
        test = case(frontend='triton')
        run.write_inputs(test, work)
        (work / 'spec.json').write_text(json.dumps({'test': test, 'root': str(self.root)}))
        with mock.patch.dict(sys.modules, {'torch': types.SimpleNamespace(cuda=cuda)}), \
                mock.patch.object(run, 'to_torch', side_effect=convert), \
                mock.patch.object(run, 'from_torch', side_effect=lambda value, blank: value):
            agree.child(work)
        self.assertEqual(events, [('device', 0), 'buffer', 'buffer', 'sync'] * 2)
        self.assertEqual(converted[1].sha256(), test['inputs'][0]['sha256'])
        self.assertEqual(converted[1], converted[3])
        self.assertIsNot(converted[1], converted[3])
        self.assertEqual(tensorfile.read(work / 'original/out.tensor'), converted[1])
        self.assertNotEqual(tensorfile.read(work / 'candidate/out.tensor'), converted[1])

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
                      {'outputs': [{'name': '../escape', 'dtype': 'fp32', 'shape': [8]}]}, {'device': 1},
                      {'outputs': []}, {'entry': 'kernels.py::original'}):
            with self.subTest(extra=extra):
                self.write_batch([case(**extra)])
                with self.assertRaises(ValueError):
                    run.run_batch(self.batch, self.out, env={'gpus': []}, root=self.root)
                self.assertFalse(self.out.exists())

    def test_refusal_messages_name_agree_and_reject_bad_reference(self):
        for extra in ({'reference': []}, {'reference': None}, {'reference': 'none'},
                      {'reference': {'kind': 'torch'}}, {'frontend': 'unknown'}, {'device': 0}):
            with self.subTest(extra=extra):
                with self.assertRaisesRegex(ValueError, 'agree'):
                    agree.check_request({'tests': [case(**extra)]})

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
        # Pair agreement alone is insufficient when either interpreter comparison fails.
        tensorfile.write(references / 'out.tensor', fp32(*([99] * 8)))
        result = subprocess.run(command + ['--references', str(self.root / 'references')],
                                cwd=self.root, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        report = json.loads(result.stdout)
        self.assertEqual([row['passed'] for row in report['rows']], [True, False, False])
        self.assertFalse(report['passed'])
        tensorfile.write(self.out / 'pair/agree/candidate/out.tensor', fp32(*([99] * 8)))
        result = subprocess.run(command, cwd=self.root, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertFalse(json.loads(result.stdout)['passed'])
        (self.out / 'DONE').unlink()
        result = subprocess.run(command, cwd=self.root, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('no DONE file', result.stderr)
        self.write_batch([case(outputs=[])])
        for module, arguments in (('gpurun', ['run', str(self.batch), '--out', str(self.out / 'invalid')]),
                                  ('interp.compare_pair', [str(self.batch), str(self.out)])):
            result = subprocess.run([sys.executable, '-m', module, *arguments], cwd=self.root, env=env,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('agree requires a non-empty outputs list', result.stderr)
        self.assertFalse((self.out / 'invalid').exists())


if __name__ == '__main__':
    unittest.main()
