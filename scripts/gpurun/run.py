"""Run one request directory and write its result bundle."""
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback

from interp import gen_inputs, tensorfile
from . import agree, bundle, dist, env as environment, timing

ROOT = Path(__file__).resolve().parents[2]
MODES = ('env-check', 'compare', 'dist', 'timing', 'agree')
TORCH_FRONTENDS = ('triton', 'tilelang', 'triton-dist', 'tilescale')
BATCH_WALL_S = 6 * 3600
COMPILE_ERROR_EXIT = 3
TEST_ID = re.compile(r'[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)*')


def now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def load_batch(batch_dir):
    batch = json.loads((Path(batch_dir) / 'batch.json').read_text())
    if batch.get('protocol') != 1:
        raise ValueError(f"unsupported protocol {batch.get('protocol')!r}")
    if batch.get('mode') not in MODES:
        raise ValueError(f"mode {batch.get('mode')!r} is not supported by this gpurun; supported: {MODES}")
    for test in batch.get('tests', []):
        if not TEST_ID.fullmatch(test['id']) or '..' in test['id'].split('/'):
            raise ValueError(f"test id {test['id']!r} is not a safe relative path")
    if batch['mode'] == 'dist':
        dist.check_request(batch)
    if batch['mode'] == 'timing':
        timing.check_request(batch)
    if batch['mode'] == 'agree':
        agree.check_request(batch)
    return batch


def request_commit(root):
    try:
        return subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_inputs(test, work):
    """Generate the inputs of a test into `work`; return the sha256 check of each."""
    checks = {}
    for spec in test.get('inputs', []):
        tensor = gen_inputs.generate(spec['dtype'], spec['shape'], spec['gen'])
        match = tensor.sha256() == spec['sha256']
        checks[spec['name']] = {'expected_sha256': spec['sha256'], 'sha256': tensor.sha256(), 'match': match}
        (work / 'in').mkdir(exist_ok=True)
        tensorfile.write(work / 'in' / (spec['name'] + '.tensor'), tensor)
    return checks


def child_env():
    """The environment of a child process: this one, with `gpurun` importable."""
    return dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(Path(__file__).resolve().parents[1]), os.environ.get('PYTHONPATH', '')]).rstrip(os.pathsep))


def run_test(test, timeout_s, test_dir, root, mode='compare'):
    """Run one test in its own process; return its meta.json content."""
    meta = {'repeats': 1, 'inputs': {}, 'outputs': {}}
    if mode == 'timing':
        meta.update(timing.metadata(test))
    with tempfile.TemporaryDirectory(prefix='gpurun-') as work:
        work = Path(work)
        meta['inputs'] = write_inputs(test, work)
        if not all(entry['match'] for entry in meta['inputs'].values()):
            return {**meta, 'status': 'input_mismatch'}
        (work / 'spec.json').write_text(json.dumps({'test': test, 'root': str(root), 'mode': mode}))
        started = time.monotonic()
        with (work / 'stdout').open('wb') as out, (work / 'stderr').open('wb') as err:
            # Its own session, so a timeout also kills compilers and launchers it started.
            process = subprocess.Popen([sys.executable, '-m', 'gpurun', '_child', str(work)], cwd=root,
                                       env=child_env(), stdout=out, stderr=err, start_new_session=True)
            try:
                code = process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                code = None
        meta['wall_time_s'] = round(time.monotonic() - started, 3)
        bundle.write_log(test_dir / 'stdout.log', (work / 'stdout').read_bytes(), root)
        bundle.write_log(test_dir / 'stderr.log', (work / 'stderr').read_bytes(), root)
        if mode == 'timing' and (work / 'timing.json').exists():
            meta.update(json.loads((work / 'timing.json').read_text()))
        if code is None:
            return {**meta, 'status': 'timeout'}
        if code != 0:
            return {**meta, 'status': 'compile_error' if code == COMPILE_ERROR_EXIT else 'runtime_error',
                    'exit_code': code}
        for kind in ('original', 'candidate') if mode == 'agree' else ('out', 'ref'):
            for path in sorted((work / kind).glob('*.tensor')):
                (test_dir / kind).mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, test_dir / kind / path.name)
                meta['outputs'][f'{kind}/{path.name}'] = tensorfile.read(path).sha256()
    return {**meta, 'status': 'ok'}


def run_batch(batch_dir, out_dir, env=None, root=ROOT):
    """Write the result bundle; return 0, or 3 when the bundle exceeds the size limits."""
    batch, out_dir, root = load_batch(batch_dir), Path(out_dir), Path(root)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError(f'{out_dir} is not empty')
    out_dir.mkdir(parents=True, exist_ok=True)
    env = environment.collect() if env is None else env
    bundle.write_json(out_dir / 'env.json', env)
    requires = batch.get('requires', {})
    reasons = environment.mismatches(requires, env)
    manifest = {'batch_id': batch['batch_id'], 'protocol': 1, 'mode': batch['mode'],
                'request_commit': request_commit(root), 'started': now(), 'env_mismatch': reasons, 'tests': {}}
    deadline = time.monotonic() + BATCH_WALL_S
    for test in batch.get('tests', []) if batch['mode'] != 'env-check' else []:
        test_dir = out_dir / test['id']
        if time.monotonic() > deadline:
            (out_dir / 'TIMEOUT').write_text('batch wall time exceeded; remaining tests were not run\n')
            break
        if reasons:
            meta = {'status': 'skipped_env', 'reasons': reasons}
        else:
            timeout_s = test.get('timeout_s', batch.get('timeout_s', 600))
            try:
                if batch['mode'] == 'dist':
                    meta = dist.run_test(test, timeout_s, test_dir, root, env, requires.get('nodes', 1))
                elif batch['mode'] == 'timing' and (missing := timing.refusals(test, env)):
                    meta = {'status': 'skipped_env', 'reasons': missing}
                elif batch['mode'] == 'agree' and (missing := agree.refusals(test, env)):
                    meta = {'status': 'skipped_env', 'reasons': missing}
                else:
                    meta = run_test(test, timeout_s, test_dir, root, batch['mode'])
            except Exception:  # one broken test must not stop the batch
                meta = {'status': 'runtime_error', 'runner_error': bundle.scrub(traceback.format_exc(), root)}
        bundle.write_json(test_dir / 'meta.json', meta)
        manifest['tests'][test['id']] = meta['status']
    manifest['finished'] = now()
    manifest['limits_exceeded'] = bundle.over_limits(out_dir)
    bundle.write_json(out_dir / 'manifest.json', manifest)
    (out_dir / 'DONE').write_text('')
    return 3 if manifest['limits_exceeded'] else 0


# ---- child process: one test, isolated so that a crash or a hang stays local ----

def load_entry(entry, root):
    relative, _, name = entry.partition('::')
    path = (root / relative).resolve()
    if not name or not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f'entry {entry!r} is not <file inside the repository>::<function>')
    spec = importlib.util.spec_from_file_location('gpurun_entry_' + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, name)


# The torch adapter moves bit patterns through an integer view of the same
# width, so NaN payloads and bfloat16 survive. It has not run on this host,
# which has no torch; the first env-check and smoke batch exercise it.
TORCH_NAMES = {'fp32': 'float32', 'fp16': 'float16', 'bf16': 'bfloat16', 'fp64': 'float64', 'bool': 'bool',
               **{n: n for n in ('int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64')}}
CARRIERS = {1: 'uint8', 2: 'int16', 4: 'int32', 8: 'int64'}


def to_torch(tensor):
    import torch
    if tensor.dtype not in TORCH_NAMES:
        raise ValueError(f'no torch dtype for {tensor.dtype}')
    dtype = getattr(torch, TORCH_NAMES[tensor.dtype])
    if not tensor.data:
        return torch.empty(tensor.shape, dtype=dtype).cuda()
    carrier = getattr(torch, CARRIERS[tensorfile.width(tensor.dtype)])
    return torch.frombuffer(bytearray(tensor.data), dtype=carrier).view(dtype).reshape(tensor.shape).cuda()


def from_torch(value, like):
    import torch
    carrier = getattr(torch, CARRIERS[tensorfile.width(like.dtype)])
    if tuple(value.shape) != like.shape:
        raise ValueError(f'output shape changed from {like.shape} to {tuple(value.shape)}')
    return tensorfile.Tensor(like.dtype, like.shape, value.detach().cpu().contiguous().view(carrier).numpy().tobytes())


def read_spec(work):
    """The test, the repository root and the input tensors handed to a child process."""
    spec = json.loads((work / 'spec.json').read_text())
    test = spec['test']
    assert sys.byteorder == 'little', 'tensor files are little-endian'
    if test['frontend'] not in TORCH_FRONTENDS and test['frontend'] != 'mock':
        raise ValueError(f"unknown frontend {test['frontend']!r}")
    inputs = {s['name']: tensorfile.read(work / 'in' / (s['name'] + '.tensor')) for s in test.get('inputs', [])}
    return test, Path(spec['root']), inputs


def call_entry(test, root, entry, inputs, out_dir, *leading):
    """Call one entry on sentinel-filled outputs and write them to `out_dir`.

    `leading` goes before `(inputs, outputs, launch)`: nothing in `compare`,
    `(rank, ranks)` in `dist`.
    """
    blank = {s['name']: tensorfile.filled(s['dtype'], s['shape'], s.get('sentinel')) for s in test['outputs']}
    function = load_entry(entry, root)
    if test['frontend'] in TORCH_FRONTENDS:
        import torch
        outputs = {name: to_torch(t) for name, t in blank.items()}
        function(*leading, {name: to_torch(t) for name, t in inputs.items()}, outputs, test.get('launch', {}))
        torch.cuda.synchronize()
        outputs = {name: from_torch(outputs[name], blank[name]) for name in blank}
    else:  # CPU mock used by the self-test: the entry replaces entries of `outputs`
        outputs = dict(blank)
        function(*leading, dict(inputs), outputs, test.get('launch', {}))
    out_dir.mkdir(parents=True)
    for name, tensor in outputs.items():
        if (tensor.dtype, tensor.shape) != (blank[name].dtype, blank[name].shape):
            raise ValueError(f'output {name} changed dtype or shape')
        tensorfile.write(out_dir / (name + '.tensor'), tensor)


def child(work):
    """Run the kernel entry and, when asked, the torch reference of one test."""
    work = Path(work)
    if json.loads((work / 'spec.json').read_text()).get('mode') == 'timing':
        return timing.child(work)
    if json.loads((work / 'spec.json').read_text()).get('mode') == 'agree':
        return agree.child(work)
    test, root, inputs = read_spec(work)
    calls = [('out', test['entry'])]
    if test.get('reference', {}).get('kind') == 'torch':
        calls.append(('ref', test['reference']['entry']))
    for kind, entry in calls:
        call_entry(test, root, entry, inputs, work / kind)


def child_main(work, body=child):
    try:
        body(work)
    except BaseException as error:
        traceback.print_exc()
        kind = type(error)
        return COMPILE_ERROR_EXIT if 'compil' in (kind.__module__ + kind.__name__).lower() else 1
    return 0
