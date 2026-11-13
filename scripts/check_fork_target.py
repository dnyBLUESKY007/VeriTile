#!/usr/bin/env python3
"""Gate one fork build target: Dist, TileLang, Interp, Equiv or Staging.

A main target must build, contain no `sorry`, import no staging module and
declare no axiom outside scripts/artifact-axiom-whitelist.txt. The staging
target only has to build. `--replay` also runs the official comparator and
the statement-audit trust gates on every file of a main target.
"""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
TARGETS = {
    'VeriTileDist': 'VeriTile/Dist',
    'VeriTileTileLang': 'VeriTile/TileLang',
    'VeriTileInterp': 'VeriTile/Interp',
    'VeriTileEquiv': 'VeriTile/Equiv',
    'VeriTileStaging': 'VeriTile/Staging',
}

# Read the compiled environment, as scripts/source_axioms.py does: a `sorry`
# always leaves a direct `sorryAx` reference in a declaration of its own module.
AUDIT = r'''
open Lean Elab Command in
elab "#forkTargetAudit " "[" modules:ident,* "]" : command => do
  let own := modules.getElems.map Syntax.getId
  let env ← getEnv
  let mut rows : Array String := #[]
  for module in env.header.moduleNames do
    if (`VeriTile.Staging).isPrefixOf module then rows := rows.push s!"STAGING\t{module}"
  for (name, info) in env.constants.toList do
    let some idx := env.getModuleIdxFor? name | continue
    let origin := env.header.moduleNames[idx.toNat]!
    unless own.contains origin do continue
    let userName := ((privateToUserName? name).getD name).eraseMacroScopes
    if let .axiomInfo _ := info then
      let file := String.intercalate "/" (origin.components.map Name.toString) ++ ".lean"
      rows := rows.push s!"AXIOM\t{file}:{userName}"
    let used := info.type.getUsedConstants ++
      ((info.value? (allowOpaque := Bool.true)).map (·.getUsedConstants)).getD #[]
    if used.contains ``sorryAx then rows := rows.push s!"SORRY\t{origin}\t{userName}"
  for row in rows.qsort (· < ·) do
    liftIO <| IO.println row
'''


def sources(directory):
    root = ROOT / (directory + '.lean')
    return ([root] if root.is_file() else []) + sorted((ROOT / directory).rglob('*.lean'))


def audit(files):
    """Return the violations of a built main target, one line each."""
    modules = ['.'.join(f.relative_to(ROOT).with_suffix('').parts) for f in files]
    program = 'import Lean\n' + ''.join(f'import {m}\n' for m in modules)
    program += AUDIT + '\n#forkTargetAudit [' + ', '.join(modules) + ']\n'
    with tempfile.TemporaryDirectory(prefix='veritile-fork-target-') as temp:
        path = Path(temp) / 'Audit.lean'
        path.write_text(program)
        checked = subprocess.run(['lake', 'env', 'lean', str(path)], cwd=ROOT,
                                 text=True, capture_output=True)
    if checked.returncode:
        raise ValueError('target audit did not elaborate:\n' + checked.stdout + checked.stderr)
    whitelist = {line.strip() for line in
                 (ROOT / 'scripts/artifact-axiom-whitelist.txt').read_text().splitlines()
                 if line.strip() and not line.lstrip().startswith('#')}
    return [row for row in checked.stdout.splitlines()
            if not (row.startswith('AXIOM\t') and row.split('\t', 1)[1] in whitelist)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('library', choices=sorted(TARGETS))
    parser.add_argument('--replay', action='store_true',
                        help='also replay every file with the official comparator (Linux judge host)')
    args = parser.parse_args()
    library, directory = args.library, TARGETS[args.library]
    if subprocess.run(['lake', 'build', library], cwd=ROOT).returncode:
        raise ValueError(f'lake build {library} failed')
    files = sources(directory)
    if library == 'VeriTileStaging':
        print(f'[ok] {library}: {len(files)} module(s) build; sorry is allowed here')
        return
    if not files:
        raise ValueError(f'{library} has no source module')
    violations = audit(files)
    if violations:
        raise ValueError(f'{library} violates the main-target rules '
                         '(SORRY, STAGING import, or AXIOM outside the whitelist):\n'
                         + '\n'.join(violations))
    print(f'[ok] {library}: {len(files)} module(s), no sorry, no staging import, no new axiom')
    if args.replay:
        with tempfile.TemporaryDirectory(prefix='veritile-fork-replay-') as workspace:
            for f in files:
                subprocess.run([sys.executable, str(ROOT / 'scripts/check_comparator.py'),
                                '--file', str(f), '--trust', '--workspace', workspace],
                               cwd=ROOT, check=True)
        print(f'[ok] {library}: comparator and trust gates accept {len(files)} file(s)')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f'[FAIL] {error}', file=sys.stderr)
        raise SystemExit(1)
