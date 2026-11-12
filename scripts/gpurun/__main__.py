"""Command line of gpurun: check-env and run."""
import argparse
import json
from pathlib import Path
import sys

from . import dist, env as environment, run


def main():
    parser = argparse.ArgumentParser(prog='gpurun', description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    check = commands.add_parser('check-env', help='print env.json; exit 1 if the batch must not run here')
    check.add_argument('batch', type=Path)
    go = commands.add_parser('run', help='run a request directory and write the result bundle')
    go.add_argument('batch', type=Path)
    go.add_argument('--out', type=Path, required=True)
    commands.add_parser('_child').add_argument('work', type=Path)
    commands.add_parser('_rank').add_argument('work', type=Path)
    mock = commands.add_parser('_mock_torchrun')
    mock.add_argument('ranks', type=int)
    mock.add_argument('work', type=Path)
    args = parser.parse_args()
    if args.command == '_child':
        return run.child_main(args.work)
    if args.command == '_rank':
        return dist.rank_main(args.work)
    if args.command == '_mock_torchrun':
        return dist.mock_torchrun(args.ranks, args.work)
    try:
        if args.command == 'check-env':
            batch, env = run.load_batch(args.batch), environment.collect()
            print(json.dumps(env, indent=2, sort_keys=True))
            reasons = environment.mismatches(batch.get('requires', {}), env)
            for reason in reasons:
                print('refused: ' + reason, file=sys.stderr)
            return 1 if reasons else 0
        return run.run_batch(args.batch, args.out)
    except (OSError, ValueError, KeyError) as error:
        print(f'gpurun: {type(error).__name__}: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
