#!/usr/bin/env python3
"""Judge the result bundle of a DeepEP version 1 batch on the development side.

    python3 experiments/interp_gpu/deepep_v1/judge.py <result-bundle-dir> [<batch-id>]

The batch id defaults to `deepep-v1-intranode-001`. Per test it prints `equal`
or `different`: `equal` when every output of every rank, in every repeat, has
the bytes of the reference. The reference is computed here from the
regenerated inputs (`expected`), with the standard library only.

What the reference says, for `ranks` ranks that own `experts / ranks`
consecutive experts each (DeepEPInventory.md, section 3.2):

- Layout of rank s: token t is in rank d when one of its selections is an
  expert of d; the counts per rank count tokens, the counts per expert count
  selections.
- Dispatch to rank d: one row per token that is in d, ordered by source rank
  and then by token index. A row holds the token's values, its index, its
  selections as experts local to d (-1 elsewhere) and their weights (0
  elsewhere). The remaining rows keep the sentinel.
- Combine on rank s: token t is the sum of the rows returned for it. The
  harness lets rank d return a row times `scale[d]`, so the sum is
  `x[t] * scale[d]` over the ranks d the token is in. The weights are summed
  the same way, unscaled.

Order and tolerance of the sum. The port starts from zero and adds the returned
rows in increasing rank in float32, then stores bf16
(`intranode/combine.py:306-325` of the port). The comparison is exact, with no
tolerance: `encode` refuses a reference value that bf16 (or fp32, for the
weights) cannot hold, and the batch uses factors (1, 1) and (1, -0.5), for
which every partial sum is x, -x/2, x/2, 2x or 0. Such sums are exact in
float32 and in bf16, so neither the order of the additions nor a rounding mode
can change a bit. With two ranks a sum has at most two terms anyway.
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


def fold(terms):
    """Left fold from zero: the order of the port's combine."""
    total = 0.0
    for term in terms:
        total += term
    return total


def expected(x, topk_idx, topk_weights, experts, scale):
    """Per rank, what every output holds; all arguments and results are nested lists `[rank][token]...`."""
    ranks, tokens = range(len(x)), range(len(x[0]))
    local = experts // len(x)
    # dests[s][t]: the ranks token t of rank s is sent to, increasing; -1 // local is -1, no rank.
    dests = [[sorted({e // local for e in topk_idx[s][t]} - {-1}) for t in tokens] for s in ranks]
    result = []
    for r in ranks:
        rows = [(s, t) for s in ranks for t in tokens if r in dests[s][t]]
        result.append({
            'is_token_in_rank': [[int(d in dests[r][t]) for d in ranks] for t in tokens],
            'num_tokens_per_rank': [sum(d in dests[r][t] for t in tokens) for d in ranks],
            'num_tokens_per_expert': [sum(row.count(e) for row in topk_idx[r]) for e in range(experts)],
            'recv_x': [x[s][t] for s, t in rows],
            'recv_src_idx': [t for _, t in rows],
            'recv_topk_idx': [[e - r * local if e // local == r else -1 for e in topk_idx[s][t]] for s, t in rows],
            'recv_topk_weights': [[w if e // local == r else 0.0 for e, w in zip(topk_idx[s][t], topk_weights[s][t])]
                                  for s, t in rows],
            'combined_x': [[fold(scale[d] * value for d in dests[r][t]) for value in x[r][t]] for t in tokens],
            'combined_topk_weights': [[fold(w if e // local == d else 0.0 for d in dests[r][t])
                                       for e, w in zip(topk_idx[r][t], topk_weights[r][t])] for t in tokens],
        })
    return result


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


def fill(blank, values):
    """`blank` with the nested `values` written from its start; the rows that hold nothing keep the sentinel."""
    data = b''.join(encode(blank.dtype, value) for value in flat(values))
    if len(data) > len(blank.data):
        raise ValueError(f'an output of shape {list(blank.shape)} is too small for the reference')
    return Tensor(blank.dtype, blank.shape, data + blank.data[len(data):])


def references(test):
    """`{rank: {output name: Tensor}}` for one test of the batch."""
    inputs = {i['name']: nested(gen_inputs.generate(i['dtype'], i['shape'], i['gen'])) for i in test['inputs']}
    per_rank = expected(inputs['x'], inputs['topk_idx'], inputs['topk_weights'],
                        test['launch']['num_experts'], test['launch']['scale'])
    return {rank: {spec['name']: fill(tensorfile.filled(spec['dtype'], spec['shape'], spec.get('sentinel')),
                                      values[spec['name']]) for spec in test['outputs']}
            for rank, values in enumerate(per_rank)}


def judge(batch_dir, bundle_dir):
    """One entry per test: its id, the status the GPU side reported, the verdict and every difference."""
    batch, bundle = json.loads((Path(batch_dir) / 'batch.json').read_text()), Path(bundle_dir)
    if not (bundle / 'DONE').is_file():
        raise ValueError('the result bundle is incomplete: no DONE file')
    manifest = json.loads((bundle / 'manifest.json').read_text())
    if manifest['batch_id'] != batch['batch_id']:
        raise ValueError(f"the bundle is for batch {manifest['batch_id']!r}, not {batch['batch_id']!r}")
    results = []
    for test in batch['tests']:
        status = manifest['tests'].get(test['id'], 'not_run')
        entry = {'test': test['id'], 'status': status, 'verdict': 'not judged', 'differences': []}
        results.append(entry)
        if status != 'ok':
            continue
        distinct = json.loads((bundle / test['id'] / 'meta.json').read_text())['distinct']
        for rank, outputs in references(test).items():
            for spec in test['outputs']:
                key = f"rank{rank}/{spec['name']}"
                if not distinct.get(key):
                    entry['differences'].append({'output': key, 'error': 'no output tensor'})
                for sha256, repeats in distinct.get(key, {}).items():
                    path = bundle / test['id'] / 'out' / key / (sha256 + '.tensor')
                    try:
                        found = tensorfile.read(path, sha256)
                        if found != outputs[spec['name']]:
                            checks = compare(found, outputs[spec['name']], spec.get('sentinel'))
                            entry['differences'].append({'output': key, 'repeats': repeats,
                                                         **{name: checks[name] for name in CHECKS}})
                    except (OSError, ValueError) as error:
                        entry['differences'].append({'output': key, 'repeats': repeats, 'error': str(error)})
        entry['verdict'] = 'different' if entry['differences'] else 'equal'
    return results


def render(results):
    lines = []
    for entry in results:
        lines.append(f"{entry['test']}: {entry['verdict']} (GPU status {entry['status']})")
        for difference in entry['differences']:
            lines.append(f"  {difference['output']}: " + ', '.join(
                f'{name} {value}' for name, value in difference.items() if name != 'output'))
    return '\n'.join(lines) + ('\nIn a difference, footprint, exact, nan_inf and gate count elements '
                               '(interp/compare.py); None is a check that does not apply to the dtype.\n')


if __name__ == '__main__':
    judged = judge(BATCHES / (sys.argv[2] if len(sys.argv) > 2 else 'deepep-v1-intranode-001'), sys.argv[1])
    print(render(judged))
    raise SystemExit(0 if judged and all(entry['verdict'] == 'equal' for entry in judged) else 1)
