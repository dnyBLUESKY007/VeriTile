"""Per-batch comparison report, computed on the development side only.

`build(request_dir, bundle_dir, references)` judges a `compare` batch of
documents/GpuExchangeProtocol.md: `request_dir` holds `batch.json`,
`bundle_dir` is the result bundle written by gpurun, and `references` holds
the development-side reference outputs, either as a directory with
`<test-id>/<name>.tensor` files or as a mapping `{test id: {name: Tensor}}`.

The result is JSON-serialisable: `batch_id`, `passed` and `rows`. A test whose
status is `ok` gets one row per output and reference it is compared with:
`reference`, and `torch` when the bundle holds `ref/<name>.tensor`. Any other
status, a missing tensor or no reference at all gives one row that does not
pass. Nothing from the bundle is taken as a verdict. `render` turns the result
into Markdown.
"""
import json
import os
from pathlib import Path

from . import tensorfile
from .compare import compare

COLUMNS = ('footprint', 'exact', 'nan_inf', 'gate', 'strict', 'max_abs', 'max_rel', 'max_ulp')
HEADER = ('Test', 'Output', 'GPU status', 'Against', 'Footprint', 'Exact', 'NaN / Inf', 'Gate 1e-2',
          'Strict fp32', 'Max abs', 'Max rel', 'Max ULP', 'Result')


def read_if_present(path):
    return tensorfile.read(path) if path.is_file() else None


def build(request_dir, bundle_dir, references):
    batch = json.loads((Path(request_dir) / 'batch.json').read_text())
    bundle = Path(bundle_dir)
    if batch.get('mode') != 'compare':
        raise ValueError(f"no report for mode {batch.get('mode')!r}")
    if not (bundle / 'DONE').is_file():
        raise ValueError('the result bundle is incomplete: no DONE file')
    manifest = json.loads((bundle / 'manifest.json').read_text())
    if manifest['batch_id'] != batch['batch_id']:
        raise ValueError(f"the bundle is for batch {manifest['batch_id']!r}, not {batch['batch_id']!r}")
    rows = []
    for test in batch['tests']:
        status = manifest['tests'].get(test['id'], 'not_run')
        for spec in test['outputs']:
            file = spec['name'] + '.tensor'
            row = {'test': test['id'], 'output': spec['name'], 'status': status, 'against': None, 'passed': False}
            if status != 'ok':
                rows.append(row)
                continue
            try:
                out = read_if_present(bundle / test['id'] / 'out' / file)
                if out is None:
                    raise ValueError('no output tensor')
                if isinstance(references, (str, os.PathLike)):
                    reference = read_if_present(Path(references) / test['id'] / file)
                else:
                    reference = references.get(test['id'], {}).get(spec['name'])
                found = {'reference': reference, 'torch': read_if_present(bundle / test['id'] / 'ref' / file)}
                found = {against: tensor for against, tensor in found.items() if tensor is not None}
                if not found:
                    raise ValueError('no reference tensor')
                rows += [{**row, 'against': against, **compare(out, tensor, spec.get('sentinel'))}
                         for against, tensor in found.items()]
            except (ValueError, NotImplementedError) as error:
                rows.append({**row, 'error': str(error)})
    return {'batch_id': batch['batch_id'], 'passed': bool(rows) and all(row['passed'] for row in rows), 'rows': rows}


def cell(value):
    if value is None:
        return '-'
    return f'{value:.3g}' if isinstance(value, float) else str(value)


def render(report):
    verdict = {True: 'pass', False: 'FAIL'}
    lines = [f"## Batch `{report['batch_id']}`: {verdict[report['passed']]}", '',
             '| ' + ' | '.join(HEADER) + ' |', '|' + ' --- |' * len(HEADER)]
    for row in report['rows']:
        result = verdict[row['passed']] + (f": {row['error']}" if 'error' in row else '')
        cells = [row['test'], row['output'], row['status'], row['against'] or '-',
                 *(cell(row.get(column)) for column in COLUMNS), result]
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', 'The check columns count violating elements. Footprint, Exact, NaN / Inf and Gate decide the '
              'result; Strict fp32 and the error figures are reported only.']
    return '\n'.join(lines) + '\n'
