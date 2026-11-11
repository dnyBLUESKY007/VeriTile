#!/usr/bin/env python3
"""Self-tests of the minimal-syntax report."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from census import report

HERE = Path(__file__).resolve().parent
A, B = 'a' * 40, 'b' * 40
KERNELS = (
    'import triton\n'
    'import triton.language as tl\n'
    'def helper(x):\n'
    '    while x:\n'
    '        x = tl.abs(x)\n'
    '    return x\n'
    '@triton.jit\n'
    'def first(p):\n'
    '    tl.store(p, helper(tl.load(p, mask=None)))\n'
    '@triton.jit\n'
    'def second(p):\n'
    '    for i in range(2):\n'
    '        tl.store(p, tl.load(p, other=0))\n')
ROWS = [
    f'K-T\tTriton\thttps://example.org/o/repo.git\t{A}\tk.py\tfirst family',
    f'K-M\tTriton\thttps://example.org/o/repo.git\t{A}\tk.py\tsecond family, same file',
    f'K-D\tTileScale\thttps://example.org/o/scale\t{B}\t\tsource undecided: a | b',
    'K-N\tTriton and TileLang\t\t\t\tno kernel exists',
]
EXPECTED = f'''## Sources

| repository | commit | files scanned |
| --- | --- | --- |
| https://example.org/o/repo.git | `{A}` | 1 |
| https://example.org/o/scale | `{B}` | 0 |

## Rows without a file

| family | front end | source | reason |
| --- | --- | --- | --- |
| K-D | TileScale | `scale@bbbbbbbbbbbb` | source undecided: a \\| b |
| K-N | Triton and TileLang | none | no kernel exists |

## Triton

2 kernels in 1 files.

### Triton: kernels

| # | family | source | file | kernel | line |
| --- | --- | --- | --- | --- | --- |
| 1 | K-M, K-T | `repo@aaaaaaaaaaaa` | `k.py` | `first` | 8 |
| 2 | K-M, K-T | `repo@aaaaaaaaaaaa` | `k.py` | `second` | 11 |

### Triton: calls

| call | kernels | keywords seen | used by |
| --- | --- | --- | --- |
| `triton.language.load` | 2 | `mask` `other` | 1-2 |
| `triton.language.store` | 2 |  | 1-2 |
| `(local function: undecorated)` | 1 |  | 1 |
| `range` | 1 |  | 2 |
| `triton.language.abs` | 1 |  | 1 |

### Triton: statements

| statement | kernels | used by |
| --- | --- | --- |
| `Expr` | 2 | 1-2 |
| `Assign` | 1 | 1 |
| `For` | 1 | 2 |
| `Return` | 1 | 1 |
| `While` | 1 | 1 |
'''


class Report(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / f'repo@{A}').mkdir()
        (self.root / f'repo@{A}' / 'k.py').write_text(KERNELS, encoding='utf-8')

    def run_report(self, rows):
        return report.report(report.read_manifest('# comment\n\n' + '\n'.join(rows) + '\n'), self.root)

    def test_helper_constructs_count_for_the_kernel_that_reaches_them(self):
        self.assertEqual(self.run_report(ROWS), EXPECTED)

    def test_row_order_does_not_change_the_report(self):
        self.assertEqual(self.run_report(ROWS[::-1]), EXPECTED)

    def test_a_listed_file_that_is_missing_stops_the_run(self):
        with self.assertRaisesRegex(SystemExit, 'repo@a+/gone.py'):
            self.run_report([f'K-T\tTriton\thttps://example.org/o/repo\t{A}\tgone.py\t'])

    def test_malformed_rows_stop_the_run(self):
        for row, message in [('K-T\tTriton\tu\t' + A, '4 columns'),
                             ('K-T\tTriton\tu\t7908e4e\tk.py\t', 'full hash'),
                             ('K-T\tTriton\t\t\tk.py\t', 'needs a repository'),
                             ('K-T\tTriton\t\t\t\t', 'a path or a reason')]:
            with self.assertRaisesRegex(SystemExit, message):
                report.read_manifest(row)

    def test_update_keeps_the_text_above_the_marker(self):
        manifest, document = self.root / 'targets.tsv', self.root / 'doc.md'
        manifest.write_text('\n'.join(ROWS) + '\n', encoding='utf-8')
        document.write_text(f'# By hand\n\n{report.MARKER}\nstale\n', encoding='utf-8')
        for _ in range(2):  # a second run changes nothing
            report.main([str(manifest), str(self.root), '--update', str(document)])
            self.assertEqual(document.read_text(encoding='utf-8'), f'# By hand\n\n{report.MARKER}\n\n{EXPECTED}')
        document.write_text('no marker\n', encoding='utf-8')
        with self.assertRaisesRegex(SystemExit, 'marker'):
            report.main([str(manifest), str(self.root), '--update', str(document)])

    def test_ranges(self):
        self.assertEqual(report.ranges({7, 1, 2, 3, 10, 9}), '1-3, 7, 9-10')


class CheckedInManifest(unittest.TestCase):
    def test_the_manifest_is_well_formed_and_names_every_family(self):
        rows = report.read_manifest((HERE / 'targets.tsv').read_text(encoding='utf-8'))
        self.assertEqual({row['family'] for row in rows}, {'K-T', 'K-D', 'K-M', 'K-N', 'K-L', 'K-S', 'V3'})
        tutorials = sorted(row['path'][10:12] for row in rows if row['path'].startswith('tutorials/') and row['family'] == 'K-T')
        self.assertEqual(tutorials, ['01', '02', '03', '04', '05', '06', '07', '08'])


if __name__ == '__main__':
    unittest.main()
