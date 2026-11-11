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

    def test_frozen_statement_and_its_definitions(self):
        source = ('namespace VeriTileGateProbe\n'
                  'def limit : Nat := 4\n'
                  'def Small (n : Nat) : Prop := n < limit\n'
                  'theorem bound (n : Nat) : Small n → n < 10 := sorry\n'
                  'end VeriTileGateProbe\n')
        probe = self.probe('VeriTile/Staging/GateProbe.lean', source)
        with tempfile.TemporaryDirectory() as temp:
            tsv, definitions = Path(temp) / 'FrozenStatements.tsv', Path(temp) / 'FrozenDefinitions.tsv'
            tsv.write_text('statement\tmodule\tsha256\tsigned\n'
                           'VeriTileGateProbe.bound\tVeriTile.Staging.GateProbe\t-\ttest\n')

            def check(*flags):
                return run('check_frozen_statements.py', '--tsv', str(tsv), '--definitions', str(definitions), *flags)
            self.assertNotEqual(check().returncode, 0)  # nothing recorded yet
            self.assertEqual(check('--fill').returncode, 0)
            self.assertEqual(sorted(line.split('\t')[0] for line in definitions.read_text().splitlines()[1:]),
                             ['VeriTileGateProbe.Small', 'VeriTileGateProbe.limit'])
            # Renaming a bound variable or finishing the proof changes neither fingerprint.
            probe.write_text(source.replace('(n : Nat) : Small n → n < 10 := sorry',
                                            '(m : Nat) : Small m → m < 10 := fun h => Nat.lt_trans h (by decide)'))
            same = check()
            self.assertEqual(same.returncode, 0, same.stderr)
            probe.write_text(source.replace('n < 10', 'n < 11'))
            edited = check()
            self.assertNotEqual(edited.returncode, 0, edited.stdout)
            self.assertIn('VeriTileGateProbe.bound: statement changed since sign-off', edited.stderr)
            # A definition used only inside another definition's body still changes the meaning.
            probe.write_text(source.replace('limit : Nat := 4', 'limit : Nat := 400'))
            redefined = check()
            self.assertNotEqual(redefined.returncode, 0, redefined.stdout)
            self.assertIn('VeriTileGateProbe.limit: definition changed since it was recorded; '
                          'affects VeriTileGateProbe.bound', redefined.stderr)
            self.assertNotIn('statement changed', redefined.stderr)
            self.assertNotEqual(check('--fill').returncode, 0)  # --fill never replaces a fingerprint
            before = definitions.read_text().splitlines()
            self.assertEqual(check('--accept-definitions').returncode, 0)
            after = definitions.read_text().splitlines()
            self.assertEqual([a == b for a, b in zip(before, after)], [True, True, False])
            self.assertEqual(check().returncode, 0)


if __name__ == '__main__':
    unittest.main()
