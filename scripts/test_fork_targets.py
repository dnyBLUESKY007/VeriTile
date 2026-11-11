#!/usr/bin/env python3
"""Negative examples for the fork target gates.

Writes throwaway probe modules into VeriTile/Dist and VeriTile/Staging and
removes them again; do not run it while another build uses this checkout.
"""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SORRY = 'theorem veritileGateProbe (n : Nat) : n + 0 = n := sorry\n'


def run(script, *args):
    return subprocess.run([sys.executable, str(ROOT / 'scripts' / script), *args],
                          cwd=ROOT, text=True, capture_output=True)


class ForkTargetGates(unittest.TestCase):
    def probe(self, relative, text):
        path = ROOT / relative
        self.assertFalse(path.exists(), f'{relative} already exists')
        self.addCleanup(path.unlink)
        path.write_text(text)
        return path

    def test_sorry_fails_a_main_target_and_not_staging(self):
        self.probe('VeriTile/Dist/GateProbe.lean', SORRY)
        rejected = run('check_fork_target.py', 'VeriTileDist')
        self.assertNotEqual(rejected.returncode, 0, rejected.stdout)
        self.assertIn('SORRY\tVeriTile.Dist.GateProbe\tveritileGateProbe', rejected.stderr)
        self.probe('VeriTile/Staging/GateProbe.lean', SORRY.replace('Probe', 'StagingProbe'))
        accepted = run('check_fork_target.py', 'VeriTileStaging')
        self.assertEqual(accepted.returncode, 0, accepted.stderr)

    def test_main_target_cannot_import_staging(self):
        self.probe('VeriTile/Staging/GateProbe.lean', 'def veritileGateStagingProbe := 0\n')
        self.probe('VeriTile/Dist/GateProbe.lean', 'import VeriTile.Staging.GateProbe\n')
        rejected = run('check_fork_target.py', 'VeriTileDist')
        self.assertNotEqual(rejected.returncode, 0, rejected.stdout)
        self.assertIn('STAGING\tVeriTile.Staging.GateProbe', rejected.stderr)

    def test_main_target_cannot_declare_an_axiom(self):
        self.probe('VeriTile/Dist/GateProbe.lean', 'axiom veritileGateProbe : 0 = 0\n')
        rejected = run('check_fork_target.py', 'VeriTileDist')
        self.assertNotEqual(rejected.returncode, 0, rejected.stdout)
        self.assertIn('AXIOM\tVeriTile/Dist/GateProbe.lean:veritileGateProbe', rejected.stderr)

    def test_edited_frozen_statement_fails(self):
        probe = self.probe('VeriTile/Staging/GateProbe.lean', SORRY)
        with tempfile.TemporaryDirectory() as temp:
            tsv = Path(temp) / 'FrozenStatements.tsv'
            tsv.write_text('statement\tmodule\tsha256\tsigned\n'
                           'veritileGateProbe\tVeriTile.Staging.GateProbe\t-\ttest\n')
            unfilled = run('check_frozen_statements.py', '--tsv', str(tsv))
            self.assertNotEqual(unfilled.returncode, 0, unfilled.stdout)
            filled = run('check_frozen_statements.py', '--tsv', str(tsv), '--fill')
            self.assertEqual(filled.returncode, 0, filled.stderr)
            # Renaming a bound variable or finishing the proof keeps the statement.
            probe.write_text('theorem veritileGateProbe (m : Nat) : m + 0 = m := rfl\n')
            same = run('check_frozen_statements.py', '--tsv', str(tsv))
            self.assertEqual(same.returncode, 0, same.stderr)
            probe.write_text(SORRY.replace('n + 0 = n', '0 + n = n'))
            changed = run('check_frozen_statements.py', '--tsv', str(tsv))
            self.assertNotEqual(changed.returncode, 0, changed.stdout)
            self.assertIn('veritileGateProbe: statement changed since sign-off', changed.stderr)


if __name__ == '__main__':
    unittest.main()
