"""`dist` mode: a test is `ranks` processes, one GPU each, launched `repeats` times.

A test of a `dist` batch has the fields of a `compare` test without a
reference, and three more:

    "ranks": 2, "launcher": "triton-dist", "repeats": 5

Its `entry` is a per-rank harness, called once in every rank process:

    def run(rank, ranks, inputs, outputs, launch): ...

`rank` (0 .. ranks-1) and `ranks` are what the launcher gave this process
(`RANK`, `WORLD_SIZE`). `inputs`, `outputs` and `launch` are as in `compare`:
`torch` tensors on this rank's GPU (visible GPU number `LOCAL_RANK`), every
output filled with its sentinel and written in place. Every rank gets the same
inputs and its own copy of every output. The harness starts and stops the
distributed runtime itself (Triton-distributed: `initialize_distributed()` ...
`finalize_distributed()`); gpurun reads the outputs after it returns.

A launcher is an entry of `LAUNCHERS`: a function `(test, work)` that returns
the one command which starts `python -m gpurun _rank <work>` once per rank
with `RANK`, `LOCAL_RANK` and `WORLD_SIZE` set, and exits non-zero when a rank
failed. Every repeat is a fresh launch. `timeout_s` bounds the whole test; when
it passes, every process below the launcher is killed. The first repeat that
fails ends the test with that repeat's status.

The launcher `triton-dist-logical-nodes` takes one more field, `"logical_nodes": 2`.
It starts the ranks on this one host as that many logical nodes of
`ranks / logical_nodes` local ranks each, so a rank sees `LOCAL_WORLD_SIZE`
= `ranks / logical_nodes` and only its own node's GPUs, and still runs on the
GPU the launcher `triton-dist` gives that rank. Such a test is not a run on
several physical nodes: `requires.nodes` stays 1, and its meta.json says
`"topology": "logical-nodes"` with `logical_nodes` and `local_ranks`.

Result of a test:

    meta.json                              status, ranks, launcher, repeats, inputs, wall_time_s,
                                           topology, logical_nodes, local_ranks (logical nodes only),
                                           runs[i]: status, exit_code, wall_time_s, rank_status,
                                                    outputs {"rank<r>/<name>": sha256}
                                           distinct {"rank<r>/<name>": {sha256: number of repeats}}
    out/rank<r>/<name>/<sha256>.tensor     one copy of every distinct value
    stdout.log, stderr.log                 all repeats, each after a line `== repeat <i> ==`

Host names never come from a request. A launcher that needs them reads
`hosts()` (the operator's `GPURUN_HOSTS`), and they are removed from the logs.
"""
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from interp import tensorfile
from . import bundle, run  # run imports this module too: use nothing of it at import time

OUTPUT_NAME = re.compile(r'[A-Za-z0-9_][A-Za-z0-9_.-]*')  # becomes a directory name


def hosts():
    """Host names of the nodes the operator lets the runner use, this node first."""
    return [name for name in os.environ.get('GPURUN_HOSTS', '').split(',') if name]


# ---- launchers ----

def launch_script():
    """`scripts/launch.sh` of the checkout the installed `triton_dist` was built from, or None.

    Found through the editable install that the pinned build instructions
    prescribe, so the launcher is of the same commit as the module env.json pins.
    """
    spec = importlib.util.find_spec('triton_dist')
    script = Path(spec.origin).parents[2] / 'scripts' / 'launch.sh' if spec and spec.origin else None
    return script if script and script.is_file() else None


def triton_dist(test, work):
    """The pinned launcher: NVSHMEM settings, then torchrun with one process per visible GPU."""
    # `--nproc_per_node=` with `=`: without it the script counts physical GPUs
    # with nvidia-smi and ignores CUDA_VISIBLE_DEVICES.
    return ['bash', str(launch_script()), f"--nproc_per_node={test['ranks']}", '-m', 'gpurun', '_rank', str(work)]


LOGICAL = 'triton-dist-logical-nodes'
LOOPBACK = '127.0.0.1'  # torchrun rendezvous of the logical nodes: this host, whatever the environment names


def devices(ranks):
    """The operator's GPUs in rank order: the entries of `CUDA_VISIBLE_DEVICES`, or devices 0 .. ranks-1 without it."""
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    return [str(index) for index in range(ranks)] if visible is None else [name for name in visible.split(',') if name]


def logical_nodes(test, work):
    """The pinned launcher once per logical node of this host, driven through its own multi-node variables.

    Node n is one `launch.sh` with torchrun node rank n of `logical_nodes`, the
    loopback rendezvous, and only its own slice of the operator's GPUs visible.
    The NVSHMEM settings stay those of the pinned script.
    """
    nodes = test['logical_nodes']
    local = test['ranks'] // nodes
    own = devices(test['ranks'])
    script = [f'CUDA_VISIBLE_DEVICES={shlex.quote(",".join(own[node * local:(node + 1) * local]))} '
              f'ARNOLD_WORKER_NUM={nodes} ARNOLD_ID={node} ARNOLD_WORKER_0_HOST={LOOPBACK} '
              f'bash "$0" --nproc_per_node={local} "$@" & pids+=($!)' for node in range(nodes)]
    # shortcut: when one node's launch fails the others run on until the test's timeout; kill them here if that wait ever matters.
    script += ['code=0', 'for pid in "${pids[@]}"; do wait "$pid" || code=1; done', 'exit $code']
    return ['bash', '-c', '\n'.join(script), str(launch_script()), '-m', 'gpurun', '_rank', str(work)]


def mock(test, work):
    """CPU launcher of the self-test."""
    return [sys.executable, '-m', 'gpurun', '_mock_torchrun', str(test['ranks']), str(work)]


LAUNCHERS = {'triton-dist': triton_dist, LOGICAL: logical_nodes, 'mock': mock}


def mock_torchrun(ranks, work):
    """What torchrun does, as far as gpurun relies on it: each rank in its own session, its variables, 1 if a rank failed."""
    processes = [subprocess.Popen(
        [sys.executable, '-m', 'gpurun', '_rank', str(work)], start_new_session=True,
        env=dict(os.environ, RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(ranks), LOCAL_WORLD_SIZE=str(ranks)))
        for rank in range(ranks)]
    return int(any([process.wait() for process in processes]))


# ---- request and machine checks ----

def check_request(batch):
    """Refuse a malformed `dist` request before anything runs."""
    for test in batch.get('tests', []):
        where = f"test {test['id']!r}: "
        if test.get('launcher') not in LAUNCHERS:
            raise ValueError(where + f"launcher {test.get('launcher')!r} is not one of {sorted(LAUNCHERS)}")
        for field in ('ranks', 'repeats'):
            if type(test.get(field)) is not int or test[field] < 1:
                raise ValueError(where + f'{field} must be a positive integer')
        logical = test.get('logical_nodes')
        if test['launcher'] != LOGICAL and logical is not None:
            raise ValueError(where + f'logical_nodes needs the launcher {LOGICAL!r}')
        if test['launcher'] == LOGICAL and (type(logical) is not int or logical < 2 or test['ranks'] % logical):
            raise ValueError(where + 'logical_nodes must be an integer of at least 2 that divides ranks')
        if test.get('reference', {}).get('kind', 'none') != 'none':
            raise ValueError(where + 'a dist test has no reference')
        for spec in test['outputs']:
            if not OUTPUT_NAME.fullmatch(spec['name']):
                raise ValueError(where + f"output name {spec['name']!r} is not a safe file name")


def refusals(test, env, nodes):
    """Reasons why this machine must not launch this test."""
    reasons = []
    visible = len(env.get('gpus', []))
    # Only the self-test's pair of mock launcher and mock frontend touches no GPU.
    if (test['launcher'], test['frontend']) != ('mock', 'mock') and test['ranks'] > visible:
        reasons.append(f"needs {test['ranks']} GPUs, one per rank; {visible} visible")
    if test['launcher'] in ('triton-dist', LOGICAL):
        if nodes > 1:  # physical nodes: logical nodes never count as them
            reasons.append(f"launcher {test['launcher']!r} starts ranks on one node; the batch requires {nodes}")
        if launch_script() is None:
            reasons.append('scripts/launch.sh not found beside an editable install of triton_dist')
    if test['launcher'] == LOGICAL and len(devices(test['ranks'])) < test['ranks']:
        reasons.append(f"CUDA_VISIBLE_DEVICES lists {len(devices(test['ranks']))} GPUs; "
                       f"the logical nodes need {test['ranks']}, one per rank")
    return reasons


# ---- runner ----

def descendants(pid):
    """Every process below `pid`, in any session: torchrun starts each rank in its own."""
    parents = {}
    for stat in Path('/proc').glob('[0-9]*/stat'):
        try:  # `pid (comm) state ppid ...`; comm may contain anything, so split after it
            parents[int(stat.parent.name)] = int(stat.read_text().rsplit(')', 1)[1].split()[1])
        except (OSError, ValueError, IndexError):  # ended meanwhile
            continue
    found = [pid]
    for parent in found:  # grows while iterating
        found.extend(child for child, above in parents.items() if above == parent)
    return found[1:]


def kill_tree(pid):
    """Kill the launcher's process group and everything it started."""
    # shortcut: a process forked between the listing and the kill survives; stop the tree first if ranks are ever seen to outlive a timeout.
    for kill, target in [(os.killpg, pid), *((os.kill, child) for child in descendants(pid))]:
        try:
            kill(target, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_test(test, timeout_s, test_dir, root, env, nodes):
    """Launch one test `repeats` times; return its meta.json content."""
    reasons = refusals(test, env, nodes)
    if reasons:
        return {'status': 'skipped_env', 'reasons': reasons}
    meta = {'ranks': test['ranks'], 'launcher': test['launcher'], 'repeats': test['repeats'],
            'runs': [], 'distinct': {}, 'status': 'ok'}
    if test['launcher'] == LOGICAL:  # the label that keeps this result apart from a run on physical nodes
        meta.update(topology='logical-nodes', logical_nodes=test['logical_nodes'],
                    local_ranks=test['ranks'] // test['logical_nodes'])
    deadline = time.monotonic() + timeout_s
    with tempfile.TemporaryDirectory(prefix='gpurun-') as work:
        work = Path(work)
        meta['inputs'] = run.write_inputs(test, work)
        if not all(entry['match'] for entry in meta['inputs'].values()):
            return {**meta, 'status': 'input_mismatch'}
        (work / 'spec.json').write_text(json.dumps({'test': test, 'root': str(root)}))
        command = LAUNCHERS[test['launcher']](test, work)
        with (work / 'stdout').open('wb') as out, (work / 'stderr').open('wb') as err:
            for repeat in range(test['repeats']):
                if time.monotonic() >= deadline:
                    meta['status'] = 'timeout'
                    break
                for stream in (out, err):
                    stream.write(f'== repeat {repeat} ==\n'.encode())
                    stream.flush()
                (work / 'run').mkdir()
                started = time.monotonic()
                process = subprocess.Popen(command, cwd=root, env=run.child_env(), stdout=out, stderr=err,
                                           start_new_session=True)
                try:
                    code = process.wait(timeout=deadline - started)
                except subprocess.TimeoutExpired:
                    kill_tree(process.pid)
                    process.wait()
                    code = None
                record = one_run(test, work / 'run', code)
                record['wall_time_s'] = round(time.monotonic() - started, 3)
                meta['runs'].append(record)
                if record['status'] != 'ok':
                    meta['status'] = record['status']
                    break
                for key, sha256 in record['outputs'].items():
                    counts = meta['distinct'].setdefault(key, {})
                    if sha256 not in counts:
                        (test_dir / 'out' / key).mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(work / 'run' / (key + '.tensor'), test_dir / 'out' / key / (sha256 + '.tensor'))
                    counts[sha256] = counts.get(sha256, 0) + 1
                shutil.rmtree(work / 'run')
        for name in ('stdout', 'stderr'):
            raw = (work / name).read_bytes()
            for host in hosts():
                raw = raw.replace(host.encode(), b'<host>')
            bundle.write_log(test_dir / (name + '.log'), raw, root)
    meta['wall_time_s'] = round(sum(record['wall_time_s'] for record in meta['runs']), 3)
    return meta


def one_run(test, run_dir, code):
    """Status of one repeat, what each rank left, and the sha256 of every output."""
    ranks = range(test['ranks'])
    names = {0: 'ok', run.COMPILE_ERROR_EXIT: 'compile_error'}
    left = [path.read_text() if path.is_file() else '' for path in (run_dir / f'rank{rank}.exit' for rank in ranks)]
    # None: the rank wrote no exit code (killed, crashed or never started).
    rank_status = [names.get(int(text), 'runtime_error') if text else None for text in left]
    record = {'exit_code': code, 'rank_status': rank_status, 'outputs': {}}
    if code is None:
        return {**record, 'status': 'timeout'}
    if 'compile_error' in rank_status:
        return {**record, 'status': 'compile_error'}
    if code != 0 or rank_status != ['ok'] * len(ranks):
        return {**record, 'status': 'runtime_error'}
    for rank in ranks:
        for spec in test['outputs']:
            key = f"rank{rank}/{spec['name']}"
            record['outputs'][key] = tensorfile.read(run_dir / (key + '.tensor')).sha256()
    return {**record, 'status': 'ok'}


# ---- rank process ----

def rank_body(work):
    """Call the per-rank entry of one test in this rank."""
    work = Path(work)
    test, root, inputs = run.read_spec(work)
    rank, ranks = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    if ranks != test['ranks']:
        raise ValueError(f"the launcher started {ranks} ranks; the test asks for {test['ranks']}")
    if test['launcher'] == LOGICAL and int(os.environ['LOCAL_WORLD_SIZE']) * test['logical_nodes'] != ranks:
        raise ValueError(f"the launcher gave LOCAL_WORLD_SIZE={os.environ['LOCAL_WORLD_SIZE']}; "
                         f"{test['logical_nodes']} logical nodes of {ranks} ranks have {ranks // test['logical_nodes']}")
    if test['frontend'] in run.TORCH_FRONTENDS:
        import torch
        # The mapping of the pinned runtime (initialize_distributed): rank to visible GPU LOCAL_RANK.
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    run.call_entry(test, root, test['entry'], inputs, work / 'run' / f'rank{rank}', rank, ranks)


def rank_main(work):
    code = run.child_main(work, rank_body)
    if 'RANK' in os.environ:  # the launcher's own exit code hides which rank failed and how
        (Path(work) / 'run' / f"rank{os.environ['RANK']}.exit").write_text(str(code))
    return code
