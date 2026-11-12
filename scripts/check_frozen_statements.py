#!/usr/bin/env python3
"""Check that signed statements still say what they said at sign-off.

Two tables under documents/dist/ record two fingerprints.

FrozenStatements.tsv has one row per signed declaration. Its fingerprint is
the SHA-256 of the elaborated statement (the declaration's type: every
constant, universe level, implicit argument and coercion), up to the names of
bound variables. It is the same in staging and after the proof lands, as long
as the declaration keeps its name.

FrozenDefinitions.tsv has one row per project constant that a signed statement
depends on: every constant the statement mentions, and every constant those
mention in their types, definition bodies and constructors, as far as they are
declared in this repository (Lean core and Mathlib are pinned by the
toolchain and the manifest). Its fingerprint covers the constant's kind, type
and, for a definition, its body. Proofs are not followed.

A changed statement always fails. A changed, new or no longer needed
definition fails too, and names the signed statements it affects, because it
changes what those statements say without touching their text.

`--fill` records statements whose fingerprint is `-`, and definitions that are
not recorded yet. It never replaces a recorded fingerprint.
`--accept-definitions` rewrites FrozenDefinitions.tsv to the current state:
the reviewed, deliberate way to accept a definition that changed or was
filled in after the freeze.
Re-signing a statement whose definitions changed with it needs both flags in
one run: each alone refuses while the other table still differs.
"""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
STATEMENTS = ROOT / 'documents/dist/FrozenStatements.tsv'
DEFINITIONS = ROOT / 'documents/dist/FrozenDefinitions.tsv'
STATEMENT_HEADER = ['statement', 'module', 'sha256', 'signed']
DEFINITION_HEADER = ['definition', 'sha256']

LEAN = r'''
open Lean Elab Command

private partial def frozenSer : Expr → String
  | .bvar i => s!"#{i}"
  | .sort l => s!"(S {l})"
  | .const n ls => s!"(C {n} {ls})"
  -- A `sorry` carries a tag made from its source position; the tag is not part of the meaning.
  | e@(.app f a) =>
    if e.getAppFn.isConstOf ``sorryAx then "(sorry)" else s!"(A {frozenSer f} {frozenSer a})"
  | .lam _ t b _ => s!"(L {frozenSer t} {frozenSer b})"
  | .forallE _ t b _ => s!"(P {frozenSer t} {frozenSer b})"
  | .letE _ t v b _ => s!"(Z {frozenSer t} {frozenSer v} {frozenSer b})"
  | .lit (.natVal n) => s!"(N {n})"
  | .lit (.strVal s) => s!"(T {s.quote})"
  | .mdata _ e => frozenSer e
  | .proj s i e => s!"(J {s} {i} {frozenSer e})"
  | e => s!"(? {e})"

/-- Declared in this repository, not in the toolchain or a dependency. -/
private def frozenProject (env : Environment) (n : Name) : Bool :=
  match env.getModuleIdxFor? n with
  | some idx => #[`VeriTile, `bench, `tests].contains env.header.moduleNames[idx.toNat]!.getRoot
  | none => Bool.true  -- the Triton DSL makes `true` a keyword

/-- The text that fixes a constant's meaning, and the constants that text mentions. -/
private def frozenEntry (info : ConstantInfo) : String × Array Name :=
  let (kind, value, more) : String × Option Expr × Array Name := match info with
    | .defnInfo d => ("def", some d.value, #[])
    | .inductInfo i => ("inductive", none, i.ctors.toArray)
    | .opaqueInfo _ => ("opaque", none, #[])
    | .ctorInfo _ => ("constructor", none, #[])
    | .axiomInfo _ => ("axiom", none, #[])
    | .thmInfo _ => ("theorem", none, #[])
    | _ => ("other", none, #[])
  (s!"{kind} : {frozenSer info.type}" ++ (match value with | some v => s!" := {frozenSer v}" | none => ""),
   info.type.getUsedConstants ++ (match value with | some v => v.getUsedConstants | none => #[]) ++ more)

private partial def frozenClosure (env : Environment) (work : List Name) (seen : NameSet) : NameSet :=
  match work with
  | [] => seen
  | n :: rest =>
    if seen.contains n || !frozenProject env n then frozenClosure env rest seen else
    match env.find? n with
    | none => frozenClosure env rest seen
    | some info => frozenClosure env ((frozenEntry info).2.toList ++ rest) (seen.insert n)

elab "#frozenStatements " "[" names:str,* "]" : command => do
  let env ← getEnv
  let mut all : NameSet := {}
  for name in names.getElems do
    let some info := env.find? name.getString.toName | continue
    liftIO <| IO.println s!"S\t{name.getString}\t{frozenSer info.type}"
    let closure := frozenClosure env info.type.getUsedConstants.toList {}
    for constant in closure.toList do
      liftIO <| IO.println s!"M\t{name.getString}\t{constant}"
      all := all.insert constant
  for constant in all.toList do
    let some info := env.find? constant | continue
    liftIO <| IO.println s!"D\t{constant}\t{(frozenEntry info).1}"
'''


def read(path, header):
    """Lines of a table and its data rows, keyed by line index."""
    lines = path.read_text().splitlines() if path.exists() else ['\t'.join(header)]
    rows = {}
    for index, line in enumerate(lines):
        if not line.strip() or line.startswith('#'):
            continue
        fields = line.split('\t')
        if fields == header:
            continue
        if len(fields) != len(header) or not all(fields):
            raise ValueError(f'{path.name}:{index + 1}: expected {len(header)} nonempty tab-separated fields')
        if any(fields[0] == row[0] for row in rows.values()):
            raise ValueError(f'{path.name}:{index + 1}: duplicate {fields[0]}')
        rows[index] = fields
    return lines, rows


def elaborate(rows):
    """Hash each listed statement and every project constant under it."""
    modules = sorted({row[1] for row in rows})
    if subprocess.run(['lake', 'build', *modules], cwd=ROOT).returncode:
        raise ValueError('modules holding frozen statements do not build')
    program = 'import Lean\n' + ''.join(f'import {m}\n' for m in modules) + LEAN
    program += '#frozenStatements [' + ', '.join('"' + row[0] + '"' for row in rows) + ']\n'
    with tempfile.TemporaryDirectory(prefix='veritile-frozen-') as temp:
        path = Path(temp) / 'Frozen.lean'
        path.write_text(program)
        checked = subprocess.run(['lake', 'env', 'lean', str(path)], cwd=ROOT, text=True, capture_output=True)
    if checked.returncode:
        raise ValueError('fingerprint program did not elaborate:\n' + checked.stdout[-2000:] + checked.stderr[-2000:])
    statements, definitions, users = {}, {}, {}
    for line in checked.stdout.splitlines():
        tag, name, text = line.split('\t', 2)
        if tag == 'S':
            statements[name] = hashlib.sha256(text.encode()).hexdigest()
        elif tag == 'D':
            definitions[name] = hashlib.sha256(text.encode()).hexdigest()
        else:
            users.setdefault(text, []).append(name)
    return statements, definitions, users


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tsv', type=Path, default=STATEMENTS)
    parser.add_argument('--definitions', type=Path, default=DEFINITIONS)
    parser.add_argument('--fill', action='store_true', help="record what is marked '-' or not recorded yet")
    parser.add_argument('--accept-definitions', action='store_true',
                        help='rewrite the definitions table to the current state')
    args = parser.parse_args()
    lines, rows = read(args.tsv, STATEMENT_HEADER)
    definition_lines, definition_rows = read(args.definitions, DEFINITION_HEADER)
    recorded = {name: digest for name, digest in definition_rows.values()}
    if not rows:
        if recorded:
            raise ValueError(f'{args.definitions.name} has rows but no statement is signed')
        print('[ok] no frozen statement is recorded yet')
        return
    statements, definitions, users = elaborate(list(rows.values()))
    errors = []
    for index, (name, module, digest, signed) in rows.items():
        if name not in statements:
            errors.append(f'{name}: not declared in {module}')
        elif digest == '-' and args.fill:
            lines[index] = '\t'.join([name, module, statements[name], signed])
        elif digest == '-':
            errors.append(f'{name}: no fingerprint recorded; run with --fill at sign-off')
        elif digest != statements[name]:
            errors.append(f'{name}: statement changed since sign-off (recorded {digest}, now {statements[name]})')
    if args.accept_definitions:
        recorded = dict(definitions)
    elif args.fill:
        recorded.update({name: digest for name, digest in definitions.items() if name not in recorded})
    for name in sorted(set(recorded) | set(definitions)):
        affected = ', '.join(sorted(users.get(name, [])))
        if name not in definitions:
            errors.append(f'{name}: recorded, but no signed statement depends on it any more')
        elif name not in recorded:
            errors.append(f'{name}: not recorded, and now under {affected}')
        elif recorded[name] != definitions[name]:
            errors.append(f'{name}: definition changed since it was recorded; affects {affected}')
    if errors:
        raise ValueError('frozen statements differ from their sign-off:\n' + '\n'.join(errors) +
                         '\nA changed definition changes what the signed statements say. After review, '
                         'record the new state with --accept-definitions.')
    if args.fill:
        args.tsv.write_text('\n'.join(lines) + '\n')
    if args.fill or args.accept_definitions:
        comments = [line for line in definition_lines if line.startswith('#')]
        table = ['\t'.join(DEFINITION_HEADER)] + [f'{name}\t{recorded[name]}' for name in sorted(recorded)]
        args.definitions.write_text('\n'.join(comments + table) + '\n')
    print(f'[ok] {len(rows)} frozen statement(s) and {len(definitions)} definition(s) under them '
          'match their recorded fingerprints')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError) as error:
        print(f'[FAIL] {error}', file=sys.stderr)
        raise SystemExit(1)
