#!/usr/bin/env python3
"""Self-test of gpurun's dist mode with the CPU mock launcher: no GPU, torch or NVSHMEM needed."""
import importlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
from gpurun import dist, env as environment, run
from interp import tensorfile

KERNELS = '''
import os, struct, subprocess, sys, time
from pathlib import Path
from interp.tensorfile import Tensor

HERE = Path(__file__).parent

class CompilationError(Exception):
    pass

def world(rank, ranks, inputs, outputs, launch):
    outputs['world'] = Tensor('int32', (2,), struct.pack('<2i', rank, ranks))

def topology(rank, ranks, inputs, outputs, launch):
    visible = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    own = visible[int(os.environ['LOCAL_RANK'])]  # the device initialize_distributed would select
    seen = [os.environ[name] for name in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE')]
    outputs['topology'] = Tensor('int32', (6,), struct.pack('<6i', *map(int, seen), int(own), len(visible)))

def rendezvous(rank, ranks, inputs, outputs, launch):
    assert (os.environ['MASTER_ADDR'], os.environ['TILELANG_USE_DISTRIBUTED']) == ('127.0.0.1', '1')
    assert 'TILELANG_USE_NVSHMEM' not in os.environ
    seen = [os.environ[name] for name in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'MASTER_PORT')]
    outputs['rendezvous'] = Tensor('int32', (5,), struct.pack('<5i', *map(int, seen)))

def first_repeat_differs(rank, ranks, inputs, outputs, launch):
    seen = HERE / f'seen{rank}'
    if seen.exists():
        world(rank, ranks, inputs, outputs, launch)
    seen.touch()  # the first repeat leaves the sentinel

def rank1_fails(rank, ranks, inputs, outputs, launch):
    if rank == 1:
        raise RuntimeError('boom in rank 1')
    world(rank, ranks, inputs, outputs, launch)

def rank0_cannot_compile(rank, ranks, inputs, outputs, launch):
    if rank == 0:
        raise CompilationError('unsupported construct')

def hang(rank, ranks, inputs, outputs, launch):
    helper = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    (HERE / f'pids{rank}').write_text(f'{os.getpid()} {helper.pid}')
    time.sleep(60)

def names_hosts(rank, ranks, inputs, outputs, launch):
    print('no route to', os.environ['GPURUN_HOSTS'], file=sys.stderr)
    world(rank, ranks, inputs, outputs, launch)
'''

# The argument and multi-node variable handling of the pinned script, and the
# variables torchrun gives a rank of node `--node_rank` among `--nnodes`.
LAUNCH_SH = '''
for arg in "$@"; do
  case $arg in
    --nproc_per_node=*) procs="${arg#*=}" ;;
    *) rest+=("$arg") ;;
  esac
done
nodes=${ARNOLD_WORKER_NUM:=1}
node=${ARNOLD_ID:=0}
for ((rank = 0; rank < procs; rank++)); do
  RANK=$((node * procs + rank)) LOCAL_RANK=$rank WORLD_SIZE=$((nodes * procs)) LOCAL_WORLD_SIZE=$procs \\
    PYTHON "${rest[@]}" &
done
wait
'''


def test_case(test_id, function, **extra):
    return {'id': test_id, 'frontend': 'mock', 'entry': f'kernels.py::{function}', 'launch': {}, 'inputs': [],
            'outputs': [{'name': 'world', 'dtype': 'int32', 'shape': [2], 'sentinel': '0x80000000'}],
            'reference': {'kind': 'none'}, 'ranks': 2, 'launcher': 'mock', 'repeats': 1, **extra}


def alive(pid):
    try:  # a killed orphan stays a zombie until init reaps it
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except OSError:
        return False


class Dist(unittest.TestCase):
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
            {'protocol': 1, 'batch_id': 'self-test', 'mode': 'dist', 'supersedes': None,
             'requires': {}, 'timeout_s': 30, 'tests': tests, **fields}))

    def run_batch(self, env={'gpus': [], 'versions': {}}):
        self.assertEqual(run.run_batch(self.batch, self.out, env=env, root=self.root), 0)
        return json.loads((self.out / 'manifest.json').read_text())['tests']

    def meta(self, test_id):
        return json.loads((self.out / test_id / 'meta.json').read_text())

    def test_equal_outputs_are_stored_once_with_their_count(self):
        self.write_batch([test_case('same', 'world', ranks=3, repeats=3)])
        self.assertEqual(self.run_batch(), {'same': 'ok'})
        meta = self.meta('same')
        self.assertEqual((meta['ranks'], meta['repeats'], len(meta['runs'])), (3, 3, 3))
        for rank in range(3):
            expected = tensorfile.Tensor('int32', (2,), struct.pack('<2i', rank, 3))
            self.assertEqual(meta['distinct'][f'rank{rank}/world'], {expected.sha256(): 3})
            stored = list((self.out / f'same/out/rank{rank}/world').iterdir())
            self.assertEqual([path.name for path in stored], [expected.sha256() + '.tensor'])
            self.assertEqual(tensorfile.read(stored[0]), expected)
        for record in meta['runs']:
            self.assertEqual((record['status'], record['exit_code'], record['rank_status']), ('ok', 0, ['ok'] * 3))
            self.assertEqual(record['outputs'], {key: next(iter(counts)) for key, counts in meta['distinct'].items()})
        self.assertIn('== repeat 2 ==', (self.out / 'same/stdout.log').read_text())
        for path in self.out.rglob('*'):  # it never writes a verdict
            if path.is_file() and path.suffix != '.tensor':
                self.assertNotRegex(path.read_text(), r'\b(PASS|FAIL)\b', path)

    def test_two_repeats_with_different_outputs_store_two(self):
        self.write_batch([test_case('differs', 'first_repeat_differs', repeats=2)])
        self.assertEqual(self.run_batch(), {'differs': 'ok'})
        meta = self.meta('differs')
        for rank in range(2):
            self.assertEqual(sorted(meta['distinct'][f'rank{rank}/world'].values()), [1, 1])
            stored = sorted(path.stem for path in (self.out / f'differs/out/rank{rank}/world').iterdir())
            self.assertEqual(stored, sorted(meta['distinct'][f'rank{rank}/world']))
            self.assertEqual(stored, sorted(record['outputs'][f'rank{rank}/world'] for record in meta['runs']))
            self.assertEqual(len(stored), 2)

    def test_failed_rank_fails_the_test_and_the_batch_goes_on(self):
        self.write_batch([test_case('fails', 'rank1_fails', repeats=3), test_case('compile', 'rank0_cannot_compile'),
                          test_case('after', 'world')])
        self.assertEqual(self.run_batch(), {'fails': 'runtime_error', 'compile': 'compile_error', 'after': 'ok'})
        meta = self.meta('fails')
        self.assertEqual(len(meta['runs']), 1)  # the first failed repeat ends the test
        self.assertEqual(meta['runs'][0]['rank_status'], ['ok', 'runtime_error'])
        self.assertEqual(self.meta('compile')['runs'][0]['rank_status'], ['compile_error', 'ok'])
        self.assertIn('RuntimeError: boom in rank 1', (self.out / 'fails/stderr.log').read_text())
        self.assertFalse((self.out / 'fails/out').exists())

    def test_timeout_kills_every_rank_and_what_it_started(self):
        self.write_batch([test_case('hangs', 'hang', timeout_s=3), test_case('after', 'world')])
        started = time.monotonic()
        self.assertEqual(self.run_batch(), {'hangs': 'timeout', 'after': 'ok'})
        self.assertLess(time.monotonic() - started, 30)
        self.assertEqual(self.meta('hangs')['runs'][0]['rank_status'], [None, None])
        pids = [int(pid) for rank in range(2) for pid in (self.root / f'pids{rank}').read_text().split()]
        self.assertEqual(len(pids), 4)  # each rank and the helper it started
        deadline = time.monotonic() + 10
        while any(alive(pid) for pid in pids) and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertEqual([pid for pid in pids if alive(pid)], [])

    def test_node_count_comes_from_the_operator(self):
        self.write_batch([test_case('two_nodes', 'world')], requires={'nodes': 2})
        with mock.patch.dict(os.environ):
            os.environ.pop('GPURUN_NODES', None)
            env = environment.collect()
            self.assertEqual(env['nodes'], 1)
            self.assertEqual(len(environment.mismatches({'nodes': 2}, env)), 1)
            self.assertEqual(self.run_batch(env=None), {'two_nodes': 'skipped_env'})
        self.out = self.out.with_name('result2')
        with mock.patch.dict(os.environ, {'GPURUN_NODES': '2'}):
            self.assertEqual(environment.mismatches({'nodes': 2}, environment.collect()), [])
            self.assertEqual(len(environment.mismatches({'nodes': 3}, environment.collect())), 1)
            self.assertEqual(self.run_batch(env=None), {'two_nodes': 'ok'})
        self.assertEqual(json.loads((self.out / 'env.json').read_text())['nodes'], 2)

    def test_real_launcher_is_refused_without_a_gpu_per_rank(self):
        h800 = {'gpus': [{'name': 'H800', 'sm': 90}] * 2, 'versions': {}}
        real = test_case('real', 'world', launcher='triton-dist', ranks=4)
        with mock.patch.object(dist, 'launch_script', return_value=Path('/td/scripts/launch.sh')):
            self.assertEqual(dist.refusals({**real, 'ranks': 2}, h800, 1), [])
            self.assertEqual(len(dist.refusals({**real, 'ranks': 2}, h800, 2)), 1)  # one node only
            self.assertEqual(len(dist.refusals({**real, 'launcher': 'mock', 'frontend': 'triton-dist'}, h800, 1)), 1)
            self.assertEqual(dist.triton_dist({'ranks': 2}, Path('/work')), [
                'bash', '/td/scripts/launch.sh', '--nproc_per_node=2', '-m', 'gpurun', '_rank', '/work'])
            self.write_batch([real, test_case('mock_needs_no_gpu', 'world', ranks=4)])
            self.assertEqual(self.run_batch(env=h800), {'real': 'skipped_env', 'mock_needs_no_gpu': 'ok'})
        self.assertEqual(self.meta('real')['reasons'], ['needs 4 GPUs, one per rank; 2 visible'])
        self.assertEqual(len(dist.refusals({**real, 'ranks': 2}, h800, 1)), 1)  # no triton_dist on this host

    def test_pinned_launcher_is_found_beside_the_module_and_starts_the_ranks(self):
        checkout = self.root / 'Triton-distributed'
        (checkout / 'python/triton_dist').mkdir(parents=True)
        (checkout / 'python/triton_dist/__init__.py').write_text('')
        h800 = {'gpus': [{'name': 'H800', 'sm': 90}] * 2, 'versions': {}}
        with mock.patch.object(sys, 'path', [str(checkout / 'python'), *sys.path]):
            importlib.invalidate_caches()
            self.assertIsNone(dist.launch_script())
            (checkout / 'scripts').mkdir()
            # Stand-in with the argument handling of the pinned script; `torchrun -m` becomes `python -m`.
            (checkout / 'scripts/launch.sh').write_text(LAUNCH_SH.replace('PYTHON', sys.executable))
            self.assertEqual(dist.launch_script(), checkout / 'scripts/launch.sh')
            self.write_batch([test_case('real', 'world', launcher='triton-dist')])
            self.assertEqual(self.run_batch(env=h800), {'real': 'ok'})
        self.assertEqual(sorted(self.meta('real')['distinct']), ['rank0/world', 'rank1/world'])

    def test_logical_nodes_have_their_own_local_ranks_and_devices(self):
        checkout = self.root / 'Triton-distributed'
        (checkout / 'python/triton_dist').mkdir(parents=True)
        (checkout / 'python/triton_dist/__init__.py').write_text('')
        (checkout / 'scripts').mkdir()
        (checkout / 'scripts/launch.sh').write_text(LAUNCH_SH.replace('PYTHON', sys.executable))
        h800 = {'gpus': [{'name': 'H800', 'sm': 90}] * 4, 'versions': {}}
        seen = {'name': 'topology', 'dtype': 'int32', 'shape': [6], 'sentinel': '0x80000000'}
        two_by_one = test_case('two_by_one', 'topology', launcher=dist.LOGICAL, logical_nodes=2, outputs=[seen])
        with mock.patch.object(sys, 'path', [str(checkout / 'python'), *sys.path]), \
                mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '2,3,4,5'}):
            importlib.invalidate_caches()
            self.write_batch([two_by_one, {**two_by_one, 'id': 'two_by_two', 'ranks': 4},
                              test_case('one_node', 'topology', launcher='triton-dist', outputs=[seen])])
            self.assertEqual(self.run_batch(env=h800), {'two_by_one': 'ok', 'two_by_two': 'ok', 'one_node': 'ok'})
            self.assertEqual(len(dist.refusals(two_by_one, h800, 2)), 1)  # logical nodes are not physical nodes
            os.environ['CUDA_VISIBLE_DEVICES'] = '4'
            self.assertEqual(len(dist.refusals(two_by_one, h800, 1)), 1)  # no GPU left for the second node

        def ranks(test_id):  # per rank: RANK, LOCAL_RANK, WORLD_SIZE, LOCAL_WORLD_SIZE, its device, devices it sees
            stored = sorted((self.out / test_id / 'out').glob('rank*/topology/*.tensor'))
            return [struct.unpack('<6i', tensorfile.read(path).data) for path in stored]

        self.assertEqual(ranks('two_by_one'), [(0, 0, 2, 1, 2, 1), (1, 0, 2, 1, 3, 1)])
        self.assertEqual(ranks('two_by_two'), [(0, 0, 4, 2, 2, 2), (1, 1, 4, 2, 3, 2), (2, 0, 4, 2, 4, 2), (3, 1, 4, 2, 5, 2)])
        self.assertEqual(ranks('one_node'), [(0, 0, 2, 2, 2, 4), (1, 1, 2, 2, 3, 4)])  # the same device per rank
        meta = self.meta('two_by_two')
        self.assertEqual((meta['topology'], meta['logical_nodes'], meta['local_ranks']), ('logical-nodes', 2, 2))
        self.assertNotIn('topology', self.meta('one_node'))

    def test_logical_node_command_uses_loopback_and_fails_with_a_node(self):
        script = self.root / 'launch.sh'
        script.write_text('exit $ARNOLD_ID\n')
        with mock.patch.object(dist, 'launch_script', return_value=script), mock.patch.dict(os.environ):
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)  # then rank r is device r, as under `triton-dist`
            command = dist.logical_nodes({'ranks': 2, 'logical_nodes': 2}, Path('/work'))
            self.assertEqual(subprocess.run(command).returncode, 1)  # node 1 failed
            script.write_text('exit 0\n')
            self.assertEqual(subprocess.run(command).returncode, 0)
        self.assertEqual(command[:2] + command[3:], ['bash', '-c', str(script), '-m', 'gpurun', '_rank', '/work'])
        for node in range(2):
            self.assertIn(f'CUDA_VISIBLE_DEVICES={node} ARNOLD_WORKER_NUM=2 ARNOLD_ID={node} '
                          'ARNOLD_WORKER_0_HOST=127.0.0.1 bash "$0" --nproc_per_node=1 "$@" &', command[2])

    def test_deepep_launcher_needs_the_port_beside_the_module_and_starts_one_rank_per_gpu(self):
        checkout = self.root / 'tilescale'
        (checkout / 'tilelang').mkdir(parents=True)
        (checkout / 'tilelang/__init__.py').write_text('')
        h800 = {'gpus': [{'name': 'H800', 'sm': 90}] * 2, 'versions': {}}
        seen = {'name': 'rendezvous', 'dtype': 'int32', 'shape': [5], 'sentinel': '0x80000000'}
        port = test_case('port', 'rendezvous', launcher=dist.DEEPEP, outputs=[seen])
        self.assertEqual(len(dist.refusals(port, h800, 1)), 1)  # no tilelang on this host
        with mock.patch.object(sys, 'path', [str(checkout), *sys.path]), mock.patch.dict(os.environ):
            os.environ.pop('TILELANG_USE_NVSHMEM', None)
            importlib.invalidate_caches()
            self.assertIsNone(dist.deepep_port())  # a wheel: the module without `examples/`
            self.assertEqual(dist.refusals(port, h800, 1), [
                'examples/distributed/deepseek_deepep not found beside an editable install of tilelang'])
            (checkout / 'examples/distributed/deepseek_deepep').mkdir(parents=True)
            (checkout / 'examples/distributed/deepseek_deepep/buffer.py').write_text('')
            self.assertEqual(dist.deepep_port(), checkout / 'examples/distributed/deepseek_deepep')
            self.assertEqual(len(dist.refusals(port, h800, 2)), 1)  # one node only
            self.assertEqual(len(dist.refusals({**port, 'ranks': 4}, h800, 1)), 1)  # a GPU per rank
            self.write_batch([port, test_case('fails', 'rank1_fails', launcher=dist.DEEPEP)])
            self.assertEqual(self.run_batch(env=h800), {'port': 'ok', 'fails': 'runtime_error'})
        stored = sorted((self.out / 'port/out').glob('rank*/rendezvous/*.tensor'))
        ranks = [struct.unpack('<5i', tensorfile.read(path).data) for path in stored]
        self.assertEqual([seen[:4] for seen in ranks], [(0, 0, 2, 2), (1, 1, 2, 2)])
        self.assertEqual(ranks[0][4], ranks[1][4])  # one port for both ranks
        self.assertGreater(ranks[0][4], 0)
        self.assertEqual(self.meta('fails')['runs'][0]['rank_status'], ['ok', 'runtime_error'])

    def test_rank_refuses_a_topology_the_launcher_did_not_give(self):
        logical = test_case('logical', 'world', launcher=dist.LOGICAL, logical_nodes=2)
        (self.root / 'spec.json').write_text(json.dumps({'test': logical, 'root': str(self.root)}))
        one_node = {'RANK': '0', 'LOCAL_RANK': '0', 'WORLD_SIZE': '2', 'LOCAL_WORLD_SIZE': '2'}
        with mock.patch.dict(os.environ, one_node), self.assertRaisesRegex(ValueError, 'LOCAL_WORLD_SIZE=2'):
            dist.rank_body(self.root)

    def test_host_names_stay_out_of_the_bundle(self):
        self.write_batch([test_case('hosts', 'names_hosts')])
        with mock.patch.dict(os.environ, {'GPURUN_HOSTS': 'alpha-node,beta-node'}):
            self.assertEqual(self.run_batch(), {'hosts': 'ok'})
        self.assertIn('no route to <host>,<host>', (self.out / 'hosts/stderr.log').read_text())
        for path in self.out.rglob('*'):
            if path.is_file():
                self.assertNotIn(b'-node', path.read_bytes(), path)

    def test_malformed_requests_are_refused(self):
        for change in ({'launcher': 'ssh'}, {'ranks': 0}, {'repeats': True}, {'reference': {'kind': 'torch'}},
                       {'logical_nodes': 2}, {'launcher': dist.LOGICAL}, {'launcher': dist.LOGICAL, 'logical_nodes': 1},
                       {'launcher': dist.LOGICAL, 'logical_nodes': 2, 'ranks': 3},
                       {'outputs': [{'name': '../world', 'dtype': 'int32', 'shape': [2]}]}):
            self.write_batch([test_case('bad', 'world', **change)])
            with self.assertRaises(ValueError, msg=change):
                run.load_batch(self.batch)

    def test_nvshmem_library_and_bindings_are_reported_apart(self):
        installed = {'nvidia-nvshmem-cu12': '3.3.20', 'nvshmem4py-cu12': '0.3.0'}

        def version(name):
            if name not in installed:
                raise importlib.metadata.PackageNotFoundError(name)
            return installed[name]

        with mock.patch.object(importlib.metadata, 'version', version):
            versions = environment.collect()['versions']
        self.assertEqual((versions['nvshmem_library'], versions['nvshmem_bindings']), ('3.3.20', '0.3.0'))
        self.assertNotIn('nvshmem', versions)


if __name__ == '__main__':
    unittest.main()
