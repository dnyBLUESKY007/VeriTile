"""Run a pair on identical inputs; only the development side compares outputs."""
from copy import deepcopy
from pathlib import Path

from . import run, timing


def check_request(batch):
    for test in batch.get('tests', []):
        if type(test.get('repeats', 1)) is not int or test.get('repeats', 1) != 1 or 'warmup' in test:
            raise ValueError('agree runs each kernel once, without warm-up')
        if not isinstance(test.get('outputs'), list) or not test['outputs']:
            raise ValueError('agree requires a non-empty outputs list')
        # The same pair, tensor names and launch interface as a single timing iteration.
        timing.check_request({'tests': [{**test, 'warmup': 0, 'repeats': 1}]}, mode='agree')


def refusals(test, env):
    return timing.refusals(test, env, mode='agree')


def child(work):
    work = Path(work)
    test, root, inputs = run.read_spec(work)
    if test['frontend'] in run.TORCH_FRONTENDS:
        import torch
    for name in timing.ORDER:
        if test['frontend'] in run.TORCH_FRONTENDS:
            torch.cuda.set_device(0)  # The operator selects the visible device.
        run.call_entry(deepcopy(test), root, test['pair'][name], inputs, work / name)
