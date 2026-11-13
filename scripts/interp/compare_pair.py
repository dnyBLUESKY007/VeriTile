"""Development-side judgment of an agree bundle, optionally against interpreter outputs.

Run `PYTHONPATH=scripts python -m interp.compare_pair REQUEST BUNDLE
--references DIRECTORY`. Reference files are `<test-id>/<output>.tensor`.
Omit --references to compare only the original with the candidate.

The original/candidate row compares candidate (under test) against original
(reference). Original/reference and candidate/reference each compare the GPU
output against the interpreter reference. Tolerance scales with the reference.
"""
import argparse
import json
import os
from pathlib import Path
import sys

from gpurun.run import load_batch
from .compare import compare
from .report import read_if_present


def compare_pair(original, candidate, reference=None, sentinel=None):
    """Compare candidate to original, then each GPU output to the interpreter."""
    pairs = [('original/candidate', candidate, original)]
    if reference is not None:
        pairs.extend((('original/reference', original, reference),
                      ('candidate/reference', candidate, reference)))
    return {name: compare(left, right, sentinel) for name, left, right in pairs}


def build(request_dir, bundle_dir, references=None):
    """Judge tensors, never a GPU-side verdict; missing requested evidence fails."""
    batch, bundle = load_batch(request_dir), Path(bundle_dir)
    if batch['mode'] != 'agree':
        raise ValueError('pair comparison requires an agree batch')
    if not (bundle / 'DONE').is_file():
        raise ValueError('the result bundle is incomplete: no DONE file')
    manifest = json.loads((bundle / 'manifest.json').read_text())
    if (manifest.get('batch_id'), manifest.get('mode'), manifest.get('protocol')) != (batch['batch_id'], 'agree', 1):
        raise ValueError('the result manifest does not match the agree request')
    rows = []
    for test in batch['tests']:
        status = manifest['tests'].get(test['id'], 'not_run')
        if status != 'ok':
            rows.append({'test': test['id'], 'output': None, 'status': status, 'against': None, 'passed': False})
            continue
        for spec in test['outputs']:
            row = {'test': test['id'], 'output': spec['name'], 'status': status,
                   'against': None, 'passed': False}
            try:
                file = spec['name'] + '.tensor'
                original = read_if_present(bundle / test['id'] / 'original' / file)
                candidate = read_if_present(bundle / test['id'] / 'candidate' / file)
                if original is None or candidate is None:
                    raise ValueError('missing original or candidate output tensor')
                if any((tensor.dtype, list(tensor.shape)) != (spec['dtype'], spec['shape'])
                       for tensor in (original, candidate)):
                    raise ValueError('pair output dtype or shape does not match the request')
                if isinstance(references, (str, os.PathLike)):
                    reference = read_if_present(Path(references) / test['id'] / file)
                else:
                    reference = references.get(test['id'], {}).get(spec['name']) if references is not None else None
                if references is not None and reference is None:
                    raise ValueError('missing requested interpreter reference tensor')
                rows.extend({**row, 'against': name, **result} for name, result in
                            compare_pair(original, candidate, reference, spec.get('sentinel')).items())
            except (OSError, ValueError, NotImplementedError) as error:
                rows.append({**row, 'error': str(error)})
    return {'batch_id': batch['batch_id'], 'reference_requested': references is not None,
            'passed': bool(rows) and all(row['passed'] for row in rows), 'rows': rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request', type=Path)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('--references', type=Path)
    args = parser.parse_args(argv)
    try:
        result = build(args.request, args.bundle, args.references)
    except (OSError, ValueError, KeyError) as error:
        print(f'compare_pair: {error}', file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
