#!/usr/bin/env python3
"""Self-test of the logical-node tutorial requests: the judge, the two batches and the harness's calls. No GPU, torch or triton_dist needed.

    python3 -B -m unittest discover -s experiments/interp_gpu/internode_logical -p 'test_*.py'
"""
import contextlib
import inspect
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness
import judge
from judge import ROOT
from gpurun import run
from interp import gen_inputs, tensorfile
from interp.tensorfile import Tensor

AG03, RS06 = 'internode-logical-ag03-001', 'internode-logical-rs06-001'
TESTS = {name: json.loads((judge.BATCHES / name / 'batch.json').read_text())['tests'] for name in (AG03, RS06)}
ONE_NODE = {'ranks': 2, 'launcher': 'triton-dist'}
TWO_NODES = {'ranks': 2, 'launcher': judge.LOGICAL, 'logical_nodes': 2}


class Expected(unittest.TestCase):
    def test_all_gather_is_the_segments_in_rank_order(self):
        self.assertEqual(judge.all_gather([[1, 2], [3, 4], [5, 6]]), [1, 2, 3, 4, 5, 6])

    def test_reduce_scatter_by_hand(self):
        x = [[[1, 2], [3, 4], [5, 6], [7, 8]], [[10, 20], [30, 40], [50, 60], [70, -8]]]
        self.assertEqual(judge.reduce_scatter(x), [[[11, 22], [33, 44]], [[55, 66], [77, 0]]])

    def test_topology_is_what_the_environment_batch_returned(self):
        # The decoded values of `internode-logical-env-002`, recorded on issue #353.
        self.assertEqual([judge.topology(ONE_NODE, rank) for rank in (0, 1)], [[0, 2, 2, 1, 0, 0], [1, 2, 2, 1, 1, 1]])
        self.assertEqual([judge.topology(TWO_NODES, rank) for rank in (0, 1)], [[0, 2, 1, 2, 0, 0], [1, 2, 1, 2, 0, 0]])

    def test_a_sum_bfloat16_cannot_hold_is_refused(self):
        self.assertEqual(judge.encode('bf16', -200), bytes.fromhex('48c3'))
        with self.assertRaisesRegex(ValueError, 'not exact in bf16'):
            judge.encode('bf16', 257)  # nine significant bits: the reference would need a rounding rule


class Batches(unittest.TestCase):
    def test_gpurun_loads_them_and_the_inputs_are_the_pinned_ones(self):
        for name, tests in TESTS.items():
            batch = run.load_batch(judge.BATCHES / name)
            self.assertEqual((batch['batch_id'], batch['mode'], batch['requires']),
                             (name, 'dist', {'sm': 90, 'gpus': 2, 'env': 'triton-dist'}))
            for test in tests:
                path, _, function = test['entry'].partition('::')
                self.assertEqual(ROOT / path, Path(harness.__file__).resolve())
                self.assertTrue(callable(getattr(harness, function)))
                for spec in test['inputs']:
                    self.assertEqual(gen_inputs.generate(spec['dtype'], spec['shape'], spec['gen']).sha256(), spec['sha256'])

    def test_every_logical_node_test_has_its_control(self):
        for tests in TESTS.values():
            sides = {}
            for test in tests:
                side, _, what = test['id'].partition('/')
                self.assertEqual({key: test.get(key) for key in TWO_NODES},
                                 {'one_node': {**ONE_NODE, 'logical_nodes': None}, 'logical_nodes': TWO_NODES}[side])
                sides.setdefault(side, {})[what] = {key: test[key] for key in ('entry', 'launch', 'inputs', 'outputs')}
            self.assertEqual(sides['one_node'], sides['logical_nodes'])
        self.assertEqual([test['launch'] for test in TESTS[AG03][:2]], [{'kernel': 'push_1d'}, {'kernel': 'push_2d'}])

    def test_shapes_stay_where_the_tutorials_are_read(self):
        for test in TESTS[AG03]:
            (given,), gathered = test['inputs'], test['outputs'][1]
            self.assertEqual((given['shape'], given['shape'][0] % test['ranks']), (gathered['shape'], 0))
        for test in TESTS[RS06]:
            (given,), reduced = test['inputs'], test['outputs'][2]
            ranks, M, N = given['shape']
            self.assertEqual((ranks, reduced['shape']), (test['ranks'], [M // ranks, N]))
            # Every tile of `kernel_ring_reduce_tma` (256 rows by 64 or 128 columns) lies inside its tensors, as with
            # the tutorial's own 8192 by 16384: the request does not depend on how a tile behaves across a border.
            self.assertEqual(((M // ranks) % 256, N % 128), (0, 0))

    def test_the_one_predicted_failure_is_last_and_bounded(self):
        self.assertFalse(any(judge.predicted_failure(test) for test in TESTS[AG03]))
        *others, last = TESTS[RS06]
        self.assertEqual([judge.predicted_failure(test) for test in TESTS[RS06]], [False, False, False, True])
        self.assertEqual((last['id'], last['timeout_s'], last['repeats']), ('logical_nodes/tutorial_context', 240, 1))
        self.assertFalse(any('timeout_s' in test for test in others))


# What a rank of a correct run returns, as an entry of gpurun's CPU mock; the predicted failure fails as predicted.
ENTRY = '''
import json, sys
sys.path.insert(0, {here!r})
import judge

def run(rank, ranks, inputs, outputs, launch):
    batch, name = launch['of']
    test, = [test for test in json.loads((judge.BATCHES / batch / 'batch.json').read_text())['tests'] if test['id'] == name]
    if judge.predicted_failure(test) and rank >= ranks // test['logical_nodes']:
        raise IndexError('list index out of range')
    outputs.update(judge.references(test)[rank])
'''


class Judge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """The bundles gpurun itself writes for the two batches when every rank returns the reference."""
        temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temp.cleanup)
        cls.written = Path(temp.name)
        (cls.written / 'entry.py').write_text(ENTRY.format(here=str(Path(__file__).resolve().parent)))
        for name in TESTS:
            batch = json.loads((judge.BATCHES / name / 'batch.json').read_text())
            batch['requires'] = {}
            for test in batch['tests']:
                test.update(frontend='mock', launcher='mock', entry='entry.py::run', launch={'of': [name, test['id']]})
                test.pop('logical_nodes', None)  # the mock launcher has no nodes
            (cls.written / name).mkdir()
            (cls.written / name / 'batch.json').write_text(json.dumps(batch))
            run.run_batch(cls.written / name, cls.written / 'bundles' / name, env={'gpus': [], 'versions': {}},
                          root=cls.written)

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.bundles = shutil.copytree(self.written / 'bundles', Path(temp.name) / 'bundles')

    def edit(self, batch, name, change):
        text = json.loads((self.bundles / batch / name).read_text())
        change(text)
        (self.bundles / batch / name).write_text(json.dumps(text))

    def plant(self, batch, test, key, change):
        """Replace the stored value of one output by `change(reference bytes)` in both repeats."""
        (sha256,) = json.loads((self.bundles / batch / test / 'meta.json').read_text())['distinct'][key]
        right = tensorfile.read(self.bundles / batch / test / 'out' / key / (sha256 + '.tensor'))
        wrong = Tensor(right.dtype, right.shape, change(right.data))
        tensorfile.write(self.bundles / batch / test / 'out' / key / (wrong.sha256() + '.tensor'), wrong)
        self.edit(batch, test + '/meta.json', lambda meta: meta['distinct'].update({key: {wrong.sha256(): 2}}))

    def judged(self, batch):
        return {entry['test']: entry for entry in judge.judge(judge.BATCHES / batch, self.bundles / batch)}

    def verdicts(self, batch):
        return {test: entry['verdict'] for test, entry in self.judged(batch).items()}

    def different(self, batch, test):
        """The outputs of a test that differ, with their differences."""
        return {key: found for key, found in self.judged(batch)[test]['outputs'].items() if found}

    def test_reference_bundles_are_equal_and_the_predicted_failure_is_seen(self):
        self.assertEqual(self.verdicts(AG03), {test['id']: 'equal' for test in TESTS[AG03]})
        self.assertEqual(self.verdicts(RS06), {
            'one_node/tutorial_context': 'equal', 'one_node/harness_context': 'equal',
            'logical_nodes/harness_context': 'equal', 'logical_nodes/tutorial_context': 'failed as predicted'})
        entry = self.judged(RS06)['logical_nodes/harness_context']
        self.assertEqual(entry['outputs'], {f'rank{rank}/{name}': [] for rank in (0, 1)
                                            for name in ('topology', 'barrier', 'reduced')})
        text = judge.render(self.judged(RS06).values())
        self.assertIn('  rank1/reduced: equal\n', text)
        self.assertIn("logical_nodes/tutorial_context: failed as predicted (GPU status runtime_error, "
                      "rank status ['ok', 'runtime_error'])", text)

    def test_planted_all_gather_without_the_peers_segment_is_flagged(self):
        # Rank 0 after a kernel whose peer never sent: its own 32 bytes, the sentinel in segment 1.
        self.plant(AG03, 'logical_nodes/push_2d', 'rank0/gathered', lambda data: data[:32] + b'\x80' * 32)
        self.assertEqual(self.verdicts(AG03), {**{test['id']: 'equal' for test in TESTS[AG03]},
                                               'logical_nodes/push_2d': 'different'})
        (key, (found,)), = self.different(AG03, 'logical_nodes/push_2d').items()
        self.assertEqual((key, found['repeats'], found['footprint'], found['exact']), ('rank0/gathered', 2, 32, 0))
        self.assertIn('  rank0/gathered: repeats 2, footprint 32,', judge.render(self.judged(AG03).values()))

    def test_planted_all_gather_with_segments_exchanged_is_flagged(self):
        self.plant(AG03, 'one_node/push_1d', 'rank1/gathered', lambda data: data[32:] + data[:32])
        (key, (found,)), = self.different(AG03, 'one_node/push_1d').items()
        self.assertEqual((key, found['footprint']), ('rank1/gathered', 0))
        self.assertGreater(found['exact'], 0)

    def test_planted_reduce_scatter_that_did_not_add_the_peer_is_flagged(self):
        # Rank 1 returns its own rows of its own input: the sum without what rank 0 sent across the nodes.
        spec, = TESTS[RS06][2]['inputs']
        own = judge.nested(gen_inputs.generate(spec['dtype'], spec['shape'], spec['gen']))[1][256:]
        unreduced = b''.join(judge.encode('bf16', value) for value in judge.flat(own))
        self.plant(RS06, 'logical_nodes/harness_context', 'rank1/reduced', lambda data: unreduced)
        self.assertEqual(self.verdicts(RS06)['logical_nodes/harness_context'], 'different')
        (key, (found,)), = self.different(RS06, 'logical_nodes/harness_context').items()
        self.assertEqual((key, found['footprint']), ('rank1/reduced', 0))
        self.assertGreater(found['gate'], 0)

    def test_one_wrong_bit_of_a_sum_is_flagged(self):
        self.plant(RS06, 'one_node/tutorial_context', 'rank0/reduced', lambda data: bytes([data[0] ^ 1]) + data[1:])
        (key, (found,)), = self.different(RS06, 'one_node/tutorial_context').items()
        self.assertEqual((key, found['gate'], found['max_ulp']), ('rank0/reduced', 0, 1))  # inside every tolerance

    def test_logical_nodes_that_came_up_as_one_node_are_flagged(self):
        one_node = judge.references(TESTS[RS06][1])
        for rank in (0, 1):
            self.plant(RS06, 'logical_nodes/harness_context', f'rank{rank}/topology',
                       lambda data, rank=rank: one_node[rank]['topology'].data)
        self.assertEqual(set(self.different(RS06, 'logical_nodes/harness_context')), {'rank0/topology', 'rank1/topology'})

    def test_a_pair_without_native_atomics_is_flagged(self):
        self.plant(RS06, 'one_node/harness_context', 'rank0/barrier', lambda data: data[:8] + bytes(4))
        self.assertEqual(set(self.different(RS06, 'one_node/harness_context')), {'rank0/barrier'})

    def test_predicted_failure_that_did_not_happen_or_failed_otherwise_is_not_met(self):
        last = 'logical_nodes/tutorial_context'
        (self.bundles / RS06 / last / 'stderr.log').write_text('RuntimeError: CUDA error\n')
        self.assertEqual(self.verdicts(RS06)[last], 'prediction not met')
        self.edit(RS06, 'manifest.json', lambda manifest: manifest['tests'].update({last: 'ok'}))
        self.assertEqual(self.verdicts(RS06)[last], 'prediction not met')

    def test_predicted_failure_with_rank_0_killed_at_the_timeout_is_met(self):
        last = 'logical_nodes/tutorial_context'
        self.edit(RS06, 'manifest.json', lambda manifest: manifest['tests'].update({last: 'timeout'}))
        self.edit(RS06, last + '/meta.json', lambda meta: meta['runs'][0].update(rank_status=[None, 'runtime_error']))
        self.assertEqual(self.verdicts(RS06)[last], 'failed as predicted')
        self.edit(RS06, last + '/meta.json', lambda meta: meta['runs'][0].update(rank_status=['runtime_error', None]))
        self.assertEqual(self.verdicts(RS06)[last], 'prediction not met')  # the rank on node 1 must be the one that raised

    def test_missing_output_and_failed_test_are_not_equal(self):
        self.edit(AG03, 'one_node/push_2d/meta.json', lambda meta: meta['distinct'].pop('rank1/gathered'))
        self.edit(AG03, 'manifest.json', lambda manifest: manifest['tests'].update({'logical_nodes/push_1d': 'timeout'}))
        self.assertEqual(self.verdicts(AG03), {'one_node/push_1d': 'equal', 'one_node/push_2d': 'different',
                                               'logical_nodes/push_1d': 'not judged', 'logical_nodes/push_2d': 'equal'})


# A file shaped like tutorial 03: definitions, then a program with no `__main__` guard.
PROGRAM_AT_IMPORT = '''"""Tutorial."""
import os
from dataclasses import dataclass


def jit(function):
    return function


@dataclass
class Context:
    rank: int
    signal_value: int = 15


@jit
def kernel(a, b):
    return a + b  # the kernel text


RANK = os.environ.setdefault('TUTORIAL_PROGRAM_RAN', '1')
raise SystemExit('the program of the tutorial ran ✅')
'''


class Loaders(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / '03-program-at-import.py'
        names = set(sys.modules)
        self.addCleanup(lambda: [sys.modules.pop(name) for name in set(sys.modules) - names])

    def test_definitions_runs_no_program_and_keeps_the_source_text(self):
        self.path.write_text(PROGRAM_AT_IMPORT, encoding='utf-8')
        module = harness.definitions(self.path)
        self.assertNotIn('TUTORIAL_PROGRAM_RAN', os.environ)
        self.assertEqual(sorted(name for name in vars(module) if not name.startswith('__')),
                         ['Context', 'dataclass', 'jit', 'kernel', 'os'])
        self.assertEqual((module.kernel(1, 2), module.Context(rank=1).signal_value), (3, 15))
        # What Triton's JIT reads: the text of the file, at its lines.
        self.assertEqual(inspect.getsourcelines(module.kernel),
                         (['@jit\n', 'def kernel(a, b):\n', '    return a + b  # the kernel text\n'], 16))
        self.assertEqual(module.kernel.__code__.co_filename, str(self.path))

    def test_imported_runs_what_import_runs(self):
        self.path.write_text(PROGRAM_AT_IMPORT.replace("raise SystemExit", "if __name__ == '__main__':\n    raise SystemExit"),
                             encoding='utf-8')
        with mock.patch.dict(os.environ):
            module = harness.imported(self.path)
            self.assertEqual((module.RANK, module.kernel(1, 2)), ('1', 3))


class Harness(unittest.TestCase):
    """Every line of the harnesses runs here against stand-ins; what the pinned code does with the calls is not tested."""

    def call(self, function, launch, inputs, tutorial):
        torch, utils, common_ops = mock.MagicMock(), mock.MagicMock(), mock.MagicMock()
        group = utils.initialize_distributed.return_value
        group.rank.return_value, group.size.return_value = 1, 2
        utils.nvshmem_create_tensors.side_effect = lambda *args: [mock.MagicMock()]  # one tensor per rank of the node
        modules = {'torch': torch, 'triton_dist': mock.MagicMock(), 'triton_dist.utils': utils,
                   'triton_dist.kernels': mock.MagicMock(), 'triton_dist.kernels.nvidia': mock.MagicMock(),
                   'triton_dist.kernels.nvidia.common_ops': common_ops}
        outputs = mock.MagicMock()
        variables = {'RANK': '1', 'WORLD_SIZE': '2', 'LOCAL_WORLD_SIZE': '1', 'LOCAL_RANK': '0'}  # rank 1 of two nodes of one
        with mock.patch.dict(sys.modules, modules), mock.patch.dict(os.environ, variables), \
                contextlib.redirect_stdout(io.StringIO()), \
                mock.patch.object(harness, 'tutorial_file', lambda name: Path('/checkout/tutorials') / name), \
                mock.patch.object(harness, 'definitions', return_value=tutorial) as definitions, \
                mock.patch.object(harness, 'imported', return_value=tutorial) as imported:
            function(1, 2, inputs, outputs, launch)
        return torch, utils, common_ops, outputs, definitions, imported

    def test_all_gather_calls_the_chosen_kernel_of_tutorial_03(self):
        tutorial, x = mock.MagicMock(), mock.MagicMock()
        x.numel.return_value = 64
        torch, utils, _, outputs, definitions, imported = self.call(
            harness.all_gather, TESTS[AG03][3]['launch'], {'x': x}, tutorial)
        definitions.assert_called_once_with(Path('/checkout/tutorials/03-inter-node-allgather.py'))
        imported.assert_not_called()  # importing tutorial 03 would run its program
        buffer = utils.nvshmem_create_tensor.return_value
        self.assertEqual(utils.nvshmem_create_tensor.call_args_list, [
            mock.call((64, ), torch.int8), *[mock.call((2, ), utils.NVSHMEM_SIGNAL_DTYPE)] * 2])  # a signal word per rank
        buffer.__getitem__.assert_called_once_with(slice(32, 64))  # rank 1 writes segment 1 only
        tutorial.AllGatherContext.assert_called_once_with(rank=1, node=1, num_ranks=2, num_nodes=2,
                                                          symm_signals=[buffer, buffer], signal_value=10)
        tutorial.all_gather_push_2d.assert_called_once_with(tutorial.AllGatherContext.return_value, buffer)
        tutorial.all_gather_push_1d.assert_not_called()
        self.assertEqual(utils.nvshmem_barrier_all_on_stream.call_count, 2)
        self.assertEqual(utils.nvshmem_free_tensor_sync.call_count, 3)
        utils.finalize_distributed.assert_called_once_with()
        self.assertEqual([call.args[0] for call in outputs.__getitem__.call_args_list],
                         ['topology', 'gathered', 'gathered'])

    def reduce_scatter(self, launch):
        tutorial, given = mock.MagicMock(), mock.MagicMock()
        x = given.__getitem__.return_value.to.return_value
        x.shape = (512, 128)
        torch, utils, common_ops, outputs, definitions, imported = self.call(
            harness.reduce_scatter, launch, {'x': given}, tutorial)
        imported.assert_called_once_with(Path('/checkout/tutorials/06-inter-node-reduce-scatter.py'))
        definitions.assert_not_called()
        given.__getitem__.assert_called_once_with(1)  # this rank's matrix
        given.__getitem__.return_value.to.assert_called_once_with(torch.bfloat16)
        self.assertEqual([call.args[0] for call in outputs.__getitem__.call_args_list], ['topology', 'barrier', 'reduced'])
        outputs.__getitem__.return_value.copy_.assert_called_with(tutorial.reduce_scatter_2d_op.return_value)
        utils.finalize_distributed.assert_called_once_with()
        return tutorial, x, torch, utils, common_ops

    def test_reduce_scatter_with_the_tutorials_context(self):
        tutorial, x, _, utils, _ = self.reduce_scatter(TESTS[RS06][3]['launch'])
        tutorial.create_reduce_scater_2d_ctx.assert_called_once_with(512, 128, 1, 2, 1, x.dtype, overlap_with_gemm=False)
        tutorial.ReduceScatter2DContext.assert_not_called()
        utils.nvshmem_create_tensors.assert_not_called()
        context = tutorial.create_reduce_scater_2d_ctx.return_value
        tutorial.reduce_scatter_2d_op.assert_called_once_with(x, context)
        context.finalize.assert_called_once_with()

    def test_reduce_scatter_with_the_harness_context(self):
        # The stand-in returns one tensor, as the pinned code does for a node of one rank: `signal[1]`, the
        # tutorial's index for rank 1, would raise here as it is predicted to on the GPU.
        tutorial, x, torch, utils, common_ops = self.reduce_scatter(TESTS[RS06][2]['launch'])
        tutorial.create_reduce_scater_2d_ctx.assert_not_called()
        self.assertEqual(utils.nvshmem_create_tensors.call_args_list, [
            *[mock.call([512, 128], x.dtype, 1, 1)] * 3, mock.call([4], utils.NVSHMEM_SIGNAL_DTYPE, 1, 1)])
        (), fields = tutorial.ReduceScatter2DContext.call_args
        (signal,) = fields['symm_signal_bufs']
        signal.zero_.assert_called_once_with()
        self.assertEqual({name: fields[name] for name in ('max_M', 'N', 'rank', 'world_size', 'local_world_size',
                                                          'dtype', 'overlap_with_gemm', 'barrier', 'num_sync_sms',
                                                          'num_p2p_sms', 'num_reduction_sms')},
                         {'max_M': 512, 'N': 128, 'rank': 1, 'world_size': 2, 'local_world_size': 1, 'dtype': x.dtype,
                          'overlap_with_gemm': False, 'barrier': common_ops.BarrierAllContext.return_value,
                          'num_sync_sms': 0, 'num_p2p_sms': 1, 'num_reduction_sms': 15})
        common_ops.BarrierAllContext.assert_called_once_with(True)
        self.assertEqual(len({id(fields[name]) for name in ('symm_scatter_bufs', 'symm_rs_per_node_bufs',
                                                             'symm_p2p_bufs', 'symm_signal_bufs')}), 4)
        context = tutorial.ReduceScatter2DContext.return_value
        tutorial.reduce_scatter_2d_op.assert_called_once_with(x, context)
        context.finalize.assert_called_once_with()
        context.barrier.finalize.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
