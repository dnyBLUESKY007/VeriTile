"""Run a pair on identical inputs; only the development side compares outputs."""
from copy import deepcopy
from pathlib import Path

from . import run, timing


def check_request(batch):
    for test in batch.get('tests', []):
        if type(test.get('repeats', 1)) is not int or test.get('repeats', 1) != 1 or 'warmup' in test:
            raise ValueError('agree runs each kernel once, without warm-up')
        # The same pair, tensor names and launch interface as a single timing iteration.
        timing.check_request({'tests': [{**test, 'warmup': 0, 'repeats': 1}]})


def refusals(test, env):
    return [reason.removeprefix('timing ') for reason in timing.refusals(test, env)]


def child(work):
    work = Path(work)
    test, root, inputs = run.read_spec(work)
    if test['frontend'] in run.TORCH_FRONTENDS:
        import torch
    for name in timing.ORDER:
        if test['frontend'] in run.TORCH_FRONTENDS:
            torch.cuda.set_device(0)  # The operator selects the visible device.
        run.call_entry(deepcopy(test), root, test['pair'][name], inputs, work / name)
