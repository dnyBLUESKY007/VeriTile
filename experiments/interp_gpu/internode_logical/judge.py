#!/usr/bin/env python3
"""Judge the result bundle of a logical-node tutorial batch on the development side.

    python3 experiments/interp_gpu/internode_logical/judge.py <result-bundle-dir> <batch-id>

for `internode-logical-ag03-001` and `internode-logical-rs06-001`. Per test,
rank and output it prints `equal` or `different`: `equal` when the output has
the bytes of the reference in every repeat. The references are computed here
from the regenerated inputs, with the standard library only:

- `gathered` (tutorial 03): the segments of all ranks in rank order, the same
  in every rank (`all_gather`). Rank r contributes segment r of the input.
- `reduced` (tutorial 06): rank r holds rows `r * M / ranks` onwards of the
  element-wise sum of the inputs of all ranks (`reduce_scatter`). The inputs
  are integers of magnitude at most 100, so every sum of two is an integer of
  magnitude at most 200, which bfloat16 holds exactly: no order of additions
  and no rounding mode can change a bit, and the comparison has no tolerance.
- `topology`: what the launcher of the test must give rank r (`topology`).
- `barrier` (tutorial 06): local rank r of a node of all `ranks` ranks, native
  atomics 1, under both launchers. NVSHMEM puts the ranks whose host name
  hashes agree into one node team (`src/host/team/team_internal.cpp:1106-1130`
  at tag `v3.6.5-0`), so the tutorial's intra-node barrier spans both logical
  nodes. `supports_p2p_native_atomic()` returns true without asking the
  driver when the process sees one GPU (`python/triton_dist/utils.py:534-547`
  at the pin), as under logical nodes; under one node it asks for the pair. A
  0 there means the pair has no native atomics and the logical-node test used
  them all the same.

One test is expected to fail (`predicted_failure`): tutorial 06 with its own
context constructor under logical nodes. Its verdict is `failed as predicted`
when the test did not end `ok`, every rank outside node 0 ended with
`runtime_error`, and the log holds the `IndexError`.

The exit code is 0 when every test is `equal` or `failed as predicted`.
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
from interp import gen_inputs, tensorfile
from interp.compare import compare, decode
from interp.tensorfile import Tensor

BATCHES = ROOT / 'experiments/interp_gpu/batches'
CHECKS = ('footprint', 'exact', 'nan_inf', 'gate', 'max_abs', 'max_ulp')
LOGICAL = 'triton-dist-logical-nodes'
PREDICTED = 'IndexError: list index out of range'


def all_gather(segments):
    """What every rank holds after an all-gather of `segments[rank]`."""
    return [value for segment in segments for value in segment]


def reduce_scatter(x):
    """Per rank, its share of the element-wise sum of the matrices `x[rank]`: equal blocks of rows in rank order."""
    total = [[sum(column) for column in zip(*rows)] for rows in zip(*x)]
    share = len(total) // len(x)
    return [total[rank * share:(rank + 1) * share] for rank in range(len(x))]


def topology(test, rank):
    """rank, world size, LOCAL_WORLD_SIZE, nodes, LOCAL_RANK, CUDA device index: a node sees only its own GPUs."""
    nodes = test.get('logical_nodes', 1)
    local = test['ranks'] // nodes
    return [rank, test['ranks'], local, nodes, rank % local, rank % local]


def predicted_failure(test):
    """Tutorial 06's own constructor raises in every rank outside node 0 (`T06:202`; see harness.py)."""
    return test['launch'].get('context') == 'tutorial' and test['launcher'] == LOGICAL


def nested(tensor):
    """The values of a tensor as nested lists."""
    values = list(decode(tensor)[0])
    for dim in reversed(tensor.shape[1:]):
        values = [values[i:i + dim] for i in range(0, len(values), dim)]
    return values


def encode(dtype, value):
    """Element bytes of `value`, which the dtype must hold exactly."""
    if dtype in gen_inputs.FLOATS:
        data = gen_inputs.encode_float(dtype, value)
        if decode(Tensor(dtype, (1,), data))[0][0] != value:
            raise ValueError(f'{value!r} is not exact in {dtype}: the reference would need a rounding rule')
        return data
    return value.to_bytes(tensorfile.width(dtype), 'little', signed=dtype.startswith('int'))


def flat(values):
    return [leaf for value in values for leaf in flat(value)] if isinstance(values, list) else [values]


def references(test):
    """`{rank: {output name: Tensor}}` for one test of either batch."""
    spec, = test['inputs']
    x = nested(gen_inputs.generate(spec['dtype'], spec['shape'], spec['gen']))
    ranks = range(test['ranks'])
    if 'kernel' in test['launch']:
        segment = len(x) // test['ranks']
        gathered = all_gather([x[rank * segment:(rank + 1) * segment] for rank in ranks])
        values = [{'gathered': gathered} for _ in ranks]
    else:
        values = [{'reduced': reduced, 'barrier': [rank, test['ranks'], 1]}
                  for rank, reduced in enumerate(reduce_scatter(x))]
    result = {}
    for rank in ranks:
        values[rank]['topology'] = topology(test, rank)
        result[rank] = {out['name']: Tensor(out['dtype'], out['shape'], b''.join(
            encode(out['dtype'], value) for value in flat(values[rank][out['name']]))) for out in test['outputs']}
    return result


def judge(batch_dir, bundle_dir):
    """One entry per test: its id, the GPU side's status, the verdict, and `equal` or the differences per output."""
    batch, bundle = json.loads((Path(batch_dir) / 'batch.json').read_text()), Path(bundle_dir)
    if not (bundle / 'DONE').is_file():
        raise ValueError('the result bundle is incomplete: no DONE file')
    manifest = json.loads((bundle / 'manifest.json').read_text())
    if manifest['batch_id'] != batch['batch_id']:
        raise ValueError(f"the bundle is for batch {manifest['batch_id']!r}, not {batch['batch_id']!r}")
    results = []
    for test in batch['tests']:
        status = manifest['tests'].get(test['id'], 'not_run')
        entry = {'test': test['id'], 'status': status, 'verdict': 'not judged', 'outputs': {}}
        results.append(entry)
        if predicted_failure(test):
            meta = json.loads((bundle / test['id'] / 'meta.json').read_text()) if status != 'not_run' else {}
            ranks = (meta.get('runs') or [{}])[-1].get('rank_status', [])
            local = test['ranks'] // test['logical_nodes']
            log = bundle / test['id'] / 'stderr.log'
            met = (status in ('runtime_error', 'timeout') and len(ranks) == test['ranks']
                   and all(rank == 'runtime_error' for rank in ranks[local:])
                   and log.is_file() and PREDICTED in log.read_text(errors='replace'))
            entry['verdict'] = 'failed as predicted' if met else 'prediction not met'
            entry['rank_status'] = ranks
            continue
        if status != 'ok':
            continue
        distinct = json.loads((bundle / test['id'] / 'meta.json').read_text())['distinct']
        for rank, outputs in references(test).items():
            for spec in test['outputs']:
                key = f"rank{rank}/{spec['name']}"
                differences = entry['outputs'][key] = []
                if not distinct.get(key):
                    differences.append({'error': 'no output tensor'})
                for sha256, repeats in distinct.get(key, {}).items():
                    try:
                        found = tensorfile.read(bundle / test['id'] / 'out' / key / (sha256 + '.tensor'), sha256)
                        if found != outputs[spec['name']]:
                            checks = compare(found, outputs[spec['name']], spec.get('sentinel'))
                            differences.append({'repeats': repeats, **{name: checks[name] for name in CHECKS}})
                    except (OSError, ValueError) as error:
                        differences.append({'repeats': repeats, 'error': str(error)})
        entry['verdict'] = 'different' if any(entry['outputs'].values()) else 'equal'
    return results


def render(results):
    lines = []
    for entry in results:
        lines.append(f"{entry['test']}: {entry['verdict']} (GPU status {entry['status']}"
                     + (f", rank status {entry['rank_status']})" if 'rank_status' in entry else ')'))
        for key, differences in entry['outputs'].items():
            lines.append(f'  {key}: ' + ('; '.join(', '.join(f'{name} {value}' for name, value in difference.items())
                                                    for difference in differences) or 'equal'))
    return '\n'.join(lines) + ('\nIn a difference, footprint, exact, nan_inf and gate count elements '
                               '(interp/compare.py); None is a check that does not apply to the dtype.\n')


if __name__ == '__main__':
    judged = judge(BATCHES / sys.argv[2], sys.argv[1])
    print(render(judged))
    raise SystemExit(0 if judged and all(entry['verdict'] in ('equal', 'failed as predicted') for entry in judged) else 1)
