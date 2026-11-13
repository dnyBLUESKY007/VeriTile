"""Timing acceptance through mock entries, fake clocks and the public CLI."""
import json
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
from gpurun import run, timing
from gpurun.test_gpurun import tensor_input

KERNELS = '''
from pathlib import Path
import time

class CompilationError(Exception):
    pass

def invoke(name, inputs, outputs, launch):
    assert set(inputs) == {'x'}
    assert launch['grid'] == [1]
    assert set(outputs) == {'out'}
    with Path(__file__).with_name('input-hashes').open('a') as out:
        out.write(name + ' ' + inputs['x'].sha256() + '\\n')
    inputs.clear()
    launch['grid'].append(2)
    calls = Path(__file__).with_name('calls')
    with calls.open('a') as out:
        out.write(name + '\\n')
    if name == 'candidate':
        count = calls.read_text().splitlines().count(name)
        if count == launch.get('stop_on'):
            if launch.get('hang'):
                time.sleep(60)
            if launch.get('compile'):
                raise CompilationError('cannot compile')
            raise RuntimeError('repeat stopped')

def original(inputs, outputs, launch):
    invoke('original', inputs, outputs, launch)

def candidate(inputs, outputs, launch):
    invoke('candidate', inputs, outputs, launch)
'''


def case(**extra):
    return {'id': 'pair/timing', 'frontend': 'mock',
            'pair': {'original': 'kernels.py::original', 'candidate': 'kernels.py::candidate'},
            'warmup': 1, 'repeats': 3, 'launch': {'grid': [1]},
            'inputs': [tensor_input('x', 1)],
            'outputs': [{'name': 'out', 'dtype': 'fp32', 'shape': [8]}], **extra}


class Timing(unittest.TestCase):
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
            {'protocol': 1, 'batch_id': 'timing-self-test', 'mode': 'timing',
             'requires': {}, 'timeout_s': 30, 'tests': tests}))

    def child(self, test, times):
        work = self.root / 'work'
        work.mkdir()
        run.write_inputs(test, work)
        (work / 'spec.json').write_text(json.dumps({'test': test, 'root': str(self.root)}))
        with mock.patch.object(timing.time, 'perf_counter', side_effect=times):
            timing.child(work)
        return json.loads((work / 'timing.json').read_text())

    def test_fake_clock_warmups_and_even_odd_medians(self):
        for repeats, median in ((3, 2), (4, 3)):
            shutil.rmtree(self.root / 'work', ignore_errors=True)
            (self.root / 'calls').unlink(missing_ok=True)
            with self.subTest(repeats=repeats):
                # Neither the odd nor even measured mean equals its median.
                durations = [100, 200, 1, 2, 9, 18, 2, 4, 4, 8][:2 * (1 + repeats)]
                values, clock = [], 0
                for duration in durations:
                    values.extend((clock, clock + duration))
                    clock += duration
                result = self.child(case(repeats=repeats), values)
                for name, factor in (('original', 1), ('candidate', 2)):
                    record = result['timing'][name]
                    self.assertEqual(record['warmup_wall_times_s'], [100 * factor])
                    self.assertEqual(record['wall_times_s'], [x * factor for x in (1, 9, 2, 4)[:repeats]])
                    self.assertEqual(record['median_s'], median * factor)
                self.assertIsNone(result['active'])
                self.assertEqual((self.root / 'calls').read_text().splitlines(),
                                 ['original', 'candidate'] * (1 + repeats))

    def test_gpu_clock_is_bracketed_by_sync_and_buffers_are_outside_it(self):
        events = []
        cuda = types.SimpleNamespace(set_device=lambda device: events.append(('device', device)),
                                     synchronize=lambda device: events.append(('sync', device)))
        torch = types.SimpleNamespace(cuda=cuda)
        def clock():
            events.append('clock')
            return len(events)
        def convert(value):
            events.append('buffer')
            return value
        with mock.patch.dict(sys.modules, {'torch': torch}), \
                mock.patch.object(run, 'to_torch', side_effect=convert), \
                mock.patch.object(timing.time, 'perf_counter', side_effect=clock):
            work = self.root / 'work'
            work.mkdir()
            test = case(frontend='triton', warmup=0, repeats=1)
            run.write_inputs(test, work)
            (work / 'spec.json').write_text(json.dumps({'test': test, 'root': str(self.root)}))
            timing.child(work)
        self.assertEqual(events, [('device', 0)] +
                         [('device', 0), 'buffer', 'buffer', ('sync', 0), 'clock', ('sync', 0), 'clock'] * 2)

    def test_subprocess_failure_preserves_partial_times_and_stops(self):
        test = case(launch={'grid': [1], 'stop_on': 3})
        self.write_batch([test])
        self.assertEqual(run.run_batch(self.batch, self.out, env={'gpus': []}, root=self.root), 0)
        meta = json.loads((self.out / 'pair/timing/meta.json').read_text())
        self.assertEqual(meta['status'], 'runtime_error')
        self.assertEqual(meta['active'], {'phase': 'repeat', 'index': 1, 'kernel': 'candidate'})
        self.assertEqual(len(meta['timing']['original']['wall_times_s']), 2)
        self.assertEqual(len(meta['timing']['candidate']['wall_times_s']), 1)
        self.assertEqual(len((self.root / 'calls').read_text().splitlines()), 6)
        self.assertEqual(meta['launch'], test['launch'])
        self.assertEqual(meta['environment'], '../../env.json')
        for name in timing.ORDER:
            record = meta['timing'][name]
            self.assertEqual(len(record['warmup_wall_times_s']), 1)
            self.assertIsNotNone(record['median_s'])

    def test_timeout_and_compile_error_keep_checkpoint(self):
        tests = [case(id='compile', warmup=0, launch={'grid': [1], 'stop_on': 1, 'compile': True}),
                 case(id='timeout', warmup=0, timeout_s=3,
                      launch={'grid': [1], 'stop_on': 2, 'hang': True})]
        self.write_batch(tests)
        self.assertEqual(run.run_batch(self.batch, self.out, env={'gpus': []}, root=self.root), 0)
        for name, status in (('compile', 'compile_error'), ('timeout', 'timeout')):
            meta = json.loads((self.out / name / 'meta.json').read_text())
            self.assertEqual(meta['status'], status)
            self.assertEqual(len(meta['timing']['original']['wall_times_s']), 1)
            self.assertEqual(meta['timing']['candidate']['wall_times_s'], [])
            self.assertIsNone(meta['timing']['candidate']['median_s'])

    def test_unavailable_gpu_and_bad_input_do_not_launch(self):
        bad = case(id='mismatch')
        bad['inputs'][0]['sha256'] = '0' * 64
        self.write_batch([case(id='no_gpu', frontend='triton'), bad])
        self.assertEqual(run.run_batch(self.batch, self.out, env={'gpus': []}, root=self.root), 0)
        self.assertEqual(json.loads((self.out / 'manifest.json').read_text())['tests'],
                         {'no_gpu': 'skipped_env', 'mismatch': 'input_mismatch'})
        skipped = json.loads((self.out / 'no_gpu/meta.json').read_text())
        self.assertEqual(set(skipped), {'status', 'reasons'})
        mismatch = json.loads((self.out / 'mismatch/meta.json').read_text())
        self.assertNotIn('active', mismatch)
        self.assertNotIn('wall_time_s', mismatch)
        for record in mismatch['timing'].values():
            self.assertEqual(record, {'warmup_wall_times_s': [], 'wall_times_s': [], 'median_s': None})
        self.assertFalse((self.root / 'calls').exists())

    def test_missing_launch_defaults_to_empty_configuration(self):
        test = case()
        del test['launch']
        timing.check_request({'tests': [test]})
        self.assertEqual(timing.metadata(test)['launch'], {})

    def test_missing_torch_or_frontend_is_skipped_with_a_visible_gpu(self):
        for versions in ({'torch': '2.0'}, {'triton': '3.5.1'}):
            with self.subTest(versions=versions):
                self.write_batch([case(frontend='triton')])
                self.assertEqual(run.run_batch(self.batch, self.out,
                                 env={'gpus': [{'name': 'mock GPU'}], 'versions': versions}, root=self.root), 0)
                meta = json.loads((self.out / 'pair/timing/meta.json').read_text())
                self.assertEqual(meta['status'], 'skipped_env')
                missing = 'triton' if 'torch' in versions else 'torch'
                self.assertEqual(meta['reasons'], [f'timing needs {missing} in the environment'])
                self.assertFalse((self.root / 'calls').exists())
                shutil.rmtree(self.out)

    def test_malformed_requests_are_rejected_before_writes(self):
        for extra in ({'warmup': -1}, {'warmup': True}, {'repeats': 0}, {'repeats': 1.5},
                      {'pair': {'original': 'kernels.py::original'}},
                      {'pair': {'original': '../kernels.py::original', 'candidate': 'kernels.py::candidate'}},
                      {'frontend': 'unknown'}, {'device': 1}, {'launch': []},
                      {'inputs': [dict(tensor_input('../escape', 1))]},
                      {'reference': {'kind': 'torch'}}, {'reference': []}, {'reference': None},
                      {'reference': 'none'}, {'entry': 'kernels.py::original'}):
            with self.subTest(extra=extra):
                self.write_batch([case(**extra)])
                with self.assertRaises(ValueError):
                    run.run_batch(self.batch, self.out, env={'gpus': []}, root=self.root)
                self.assertFalse(self.out.exists())

    def test_cli_run_bundle_readback_and_check_env(self):
        # An isolated copy keeps the CLI's real repository-root rule intact.
        shutil.copytree(SCRIPTS / 'gpurun', self.root / 'scripts/gpurun', ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copytree(SCRIPTS / 'interp', self.root / 'scripts/interp', ignore=shutil.ignore_patterns('__pycache__'))
        self.write_batch([case(warmup=0, repeats=1)])
        for arguments in (['check-env', str(self.batch)], ['run', str(self.batch), '--out', str(self.out)]):
            result = subprocess.run([sys.executable, '-m', 'gpurun', *arguments], cwd=self.root,
                                    env={**run.child_env(), 'PYTHONPATH': str(self.root / 'scripts')},
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        meta = json.loads((self.out / 'pair/timing/meta.json').read_text())
        self.assertEqual(meta['status'], 'ok')
        self.assertEqual(set(meta), {'status', 'repeats', 'inputs', 'outputs', 'warmup', 'order',
                                     'launch', 'environment', 'timing', 'wall_time_s', 'active'})
        self.assertEqual(set(meta['timing']), {'original', 'candidate'})
        self.assertEqual(meta['order'], ['original', 'candidate'])
        self.assertEqual(meta['launch'], {'grid': [1]})
        self.assertTrue((self.out / 'pair/timing' / meta['environment']).is_file())
        self.assertTrue((self.out / 'DONE').is_file())
        for name in timing.ORDER:
            self.assertEqual(set(meta['timing'][name]), {'warmup_wall_times_s', 'wall_times_s', 'median_s'})
            self.assertEqual(len(meta['timing'][name]['wall_times_s']), 1)
            self.assertEqual(meta['timing'][name]['warmup_wall_times_s'], [])
        expected_hash = case()['inputs'][0]['sha256']
        self.assertEqual((self.root / 'input-hashes').read_text().splitlines(),
                         [f'original {expected_hash}', f'candidate {expected_hash}'])
        for path in self.out.rglob('*'):
            if path.is_file():
                self.assertNotRegex(path.read_text(), r'\b(PASS|FAIL|speedup|verdict)\b')
                self.assertNotIn(str(self.root), path.read_text())


if __name__ == '__main__':
    unittest.main()
