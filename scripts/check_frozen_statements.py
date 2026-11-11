#!/usr/bin/env python3
"""Check that signed statements still have their recorded fingerprints.

documents/dist/FrozenStatements.tsv lists one signed declaration per row. Its
fingerprint is the SHA-256 of the elaborated statement (the declaration's
type: every constant, universe level, implicit argument and coercion), up to
the names of bound variables. It does not cover the definitions the statement
mentions, and it is the same in staging and after the proof lands, as long as
the declaration keeps its name.

`--fill` computes the fingerprint of rows whose fingerprint is `-`; it never
replaces a recorded one.
"""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
TSV = ROOT / 'documents/dist/FrozenStatements.tsv'
HEADER = ['statement', 'module', 'sha256', 'signed']

SERIALIZE = r'''
open Lean Elab Command

private partial def frozenSer : Expr → String
  | .bvar i => s!"#{i}"
  | .sort l => s!"(S {l})"
  | .const n ls => s!"(C {n} {ls})"
  | .app f a => s!"(A {frozenSer f} {frozenSer a})"
  | .lam _ t b _ => s!"(L {frozenSer t} {frozenSer b})"
  | .forallE _ t b _ => s!"(P {frozenSer t} {frozenSer b})"
  | .letE _ t v b _ => s!"(Z {frozenSer t} {frozenSer v} {frozenSer b})"
  | .lit (.natVal n) => s!"(N {n})"
  | .lit (.strVal s) => s!"(T {s.quote})"
  | .mdata _ e => frozenSer e
  | .proj s i e => s!"(J {s} {i} {frozenSer e})"
  | e => s!"(? {e})"

elab "#frozenStatements " "[" names:str,* "]" : command => do
  let env ← getEnv
  for name in names.getElems do
    match env.find? name.getString.toName with
    | some info => liftIO <| IO.println s!"{name.getString}\t{frozenSer info.type}"
    | none => liftIO <| IO.println s!"{name.getString}"
'''


def read(tsv):
    lines = tsv.read_text().splitlines()
    rows = {}
    for index, line in enumerate(lines):
        if not line.strip() or line.startswith('#'):
            continue
        fields = line.split('\t')
        if fields == HEADER:
            continue
        if len(fields) != len(HEADER) or not all(fields):
            raise ValueError(f'{tsv.name}:{index + 1}: expected {len(HEADER)} nonempty tab-separated fields')
        if any(fields[0] == row[0] for row in rows.values()):
            raise ValueError(f'{tsv.name}:{index + 1}: duplicate statement {fields[0]}')
        rows[index] = fields
    return lines, rows


def fingerprints(rows):
    """Elaborate the listed modules and hash each listed statement."""
    modules = sorted({row[1] for row in rows})
    if subprocess.run(['lake', 'build', *modules], cwd=ROOT).returncode:
        raise ValueError('modules holding frozen statements do not build')
    program = 'import Lean\n' + ''.join(f'import {m}\n' for m in modules) + SERIALIZE
    program += '#frozenStatements [' + ', '.join('"' + row[0] + '"' for row in rows) + ']\n'
    with tempfile.TemporaryDirectory(prefix='veritile-frozen-') as temp:
        path = Path(temp) / 'Frozen.lean'
        path.write_text(program)
        checked = subprocess.run(['lake', 'env', 'lean', str(path)], cwd=ROOT,
                                 text=True, capture_output=True)
    if checked.returncode:
        raise ValueError('fingerprint program did not elaborate:\n' + checked.stdout + checked.stderr)
    found = dict(line.split('\t', 1) for line in checked.stdout.splitlines() if '\t' in line)
    return {name: hashlib.sha256(text.encode()).hexdigest() for name, text in found.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tsv', type=Path, default=TSV)
    parser.add_argument('--fill', action='store_true', help="record fingerprints of rows marked '-'")
    args = parser.parse_args()
    lines, rows = read(args.tsv)
    if not rows:
        print('[ok] no frozen statement is recorded yet')
        return
    actual = fingerprints(list(rows.values()))
    errors = []
    for index, (name, module, recorded, signed) in rows.items():
        if name not in actual:
            errors.append(f'{name}: not declared in {module}')
        elif recorded == '-' and args.fill:
            lines[index] = '\t'.join([name, module, actual[name], signed])
        elif recorded == '-':
            errors.append(f'{name}: no fingerprint recorded; run with --fill at sign-off')
        elif recorded != actual[name]:
            errors.append(f'{name}: statement changed since sign-off '
                          f'(recorded {recorded}, now {actual[name]})')
    if errors:
        raise ValueError('frozen statements differ from their sign-off:\n' + '\n'.join(errors))
    if args.fill:
        args.tsv.write_text('\n'.join(lines) + '\n')
    print(f'[ok] {len(rows)} frozen statement(s) match their recorded fingerprints')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError) as error:
        print(f'[FAIL] {error}', file=sys.stderr)
        raise SystemExit(1)
