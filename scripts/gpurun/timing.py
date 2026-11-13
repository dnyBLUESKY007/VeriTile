"""Time an original/candidate pair in one process, without comparing it."""
from copy import deepcopy
import json
from pathlib import Path
from statistics import median
import time

from interp import tensorfile
from . import dist, run

ORDER = ('original', 'candidate')


def check_request(batch):
    """Validate timing fields before generating files or starting a process."""
    for test in batch.get('tests', []):
        for key, minimum in (('warmup', 0), ('repeats', 1)):
            if type(test.get(key)) is not int or test[key] < minimum:
                raise ValueError(f'{key} must be an integer >= {minimum}')
        pair = test.get('pair')
        if not isinstance(pair, dict) or set(pair) != set(ORDER):
            raise ValueError('pair must name original and candidate')
        for entry in pair.values():
            if not isinstance(entry, str):
                raise ValueError('pair entries must be repository-relative file::function strings')
            path, sep, name = entry.partition('::')
            if not sep or not name.isidentifier() or not path or Path(path).is_absolute() or '..' in Path(path).parts:
                raise ValueError('pair entries must be repository-relative file::function strings')
        if test.get('frontend') not in (*run.TORCH_FRONTENDS, 'mock'):
            raise ValueError('unknown timing frontend')
        if test.get('reference', {}).get('kind', 'none') != 'none':
            raise ValueError('timing has no reference')
        if any(key in test for key in ('device', 'host', 'hostname', 'ranks', 'launcher')):
            raise ValueError('timing uses one operator-selected GPU, with no request device or launcher')
        if not isinstance(test.get('launch', {}), dict):
            raise ValueError('launch must be an object')
        for field in ('inputs', 'outputs'):
            names = [spec['name'] for spec in test.get(field, [])]
            if any(not isinstance(name, str) or not dist.OUTPUT_NAME.fullmatch(name) for name in names):
                raise ValueError(f'{field} names must be safe file names')
            if len(names) != len(set(names)):
                raise ValueError(f'{field} names must be unique')


def refusals(test, env):
    if test['frontend'] == 'mock':
        return []
    module = {'triton-dist': 'triton_dist', 'tilescale': 'tilelang'}.get(test['frontend'], test['frontend'])
    reasons = []
    if not env.get('gpus'):
        reasons.append('timing needs one visible GPU')
    for name in ('torch', module):
        if not env.get('versions', {}).get(name):
            reasons.append(f'timing needs {name} in the environment')
    return reasons


def metadata(test):
    return {'warmup': test['warmup'], 'repeats': test['repeats'], 'order': list(ORDER),
            'launch': deepcopy(test.get('launch', {})),
            'environment': '../' * len(test['id'].split('/')) + 'env.json',
            'timing': {name: {'warmup_wall_times_s': [], 'wall_times_s': [], 'median_s': None}
                       for name in ORDER}}


def child(work):
    """Prepare buffers outside the interval; time each harness and device completion."""
    work = Path(work)
    test, root, inputs = run.read_spec(work)
    records = {'timing': metadata(test)['timing'], 'active': None}

    def checkpoint():
        # A timeout must leave the last complete record, never half a JSON file.
        pending = work / 'timing.pending'
        pending.write_text(json.dumps(records))
        pending.replace(work / 'timing.json')

    checkpoint()
    functions = {name: run.load_entry(test['pair'][name], root) for name in ORDER}
    blank = {spec['name']: tensorfile.filled(spec['dtype'], spec['shape'], spec.get('sentinel'))
             for spec in test['outputs']}
    gpu = test['frontend'] in run.TORCH_FRONTENDS
    if gpu:
        import torch
        torch.cuda.set_device(0)  # CUDA_VISIBLE_DEVICES is the operator's selection.
    for phase, count, field in (('warmup', test['warmup'], 'warmup_wall_times_s'),
                                ('repeat', test['repeats'], 'wall_times_s')):
        for index in range(count):
            for name in ORDER:
                records['active'] = {'phase': phase, 'index': index, 'kernel': name}
                checkpoint()
                if gpu:
                    torch.cuda.set_device(0)
                    call_inputs = {key: run.to_torch(value) for key, value in inputs.items()}
                    outputs = {key: run.to_torch(value) for key, value in blank.items()}
                else:
                    call_inputs, outputs = dict(inputs), dict(blank)
                launch = deepcopy(test.get('launch', {}))
                if gpu:
                    torch.cuda.synchronize(0)
                start = time.perf_counter()
                functions[name](call_inputs, outputs, launch)
                if gpu:
                    torch.cuda.synchronize(0)
                elapsed = time.perf_counter() - start
                record = records['timing'][name]
                record[field].append(elapsed)
                if phase == 'repeat':
                    record['median_s'] = median(record['wall_times_s'])
                records['active'] = None
                checkpoint()
