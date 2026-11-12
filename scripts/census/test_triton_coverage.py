#!/usr/bin/env python3
"""Self-tests of the Triton coverage matrix, on the cut-down inputs under fixtures/coverage/."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from census import report, triton_coverage as coverage

FIXTURES = Path(__file__).resolve().parent / 'fixtures' / 'coverage'
MANIFEST, CHECKOUTS, ROOT = FIXTURES / 'targets.tsv', FIXTURES / 'checkouts', FIXTURES / 'repo'
READ = {
    '`tl.load(mask=)`': ('rule', 'Expansion/Main.lean', 4, '| "mask"', ''),
    '`tl.load(other=)`': ('rule', 'Expansion/Main.lean', 4, '| "other"', 'the text is on line 5'),
    '`tl.inline_asm_elementwise(...)`': (
        'never (PTX boundary)', 'Syntax.lean', 18, 'tl.inline_asm_elementwise(', 'out of scope'),
    '`With`': (None, 'Syntax.lean', 6, 'no such text', 'a note that keeps the mechanical status'),
}


def matrix(read=READ):
    return coverage.matrix(report.read_manifest(MANIFEST.read_text(encoding='utf-8')), CHECKOUTS, ROOT, read)


class Lean(unittest.TestCase):
    def test_comments_go_and_lines_and_strings_stay(self):
        text = 'a -- c "x"\n/- b /- n -/ "\n-/ c "-- s" \'"\' d\n'
        self.assertEqual(coverage.lean_code(text), 'a \n\n c "-- s" \'"\' d\n')

    def test_declarations_become_forms(self):
        declarations, code = coverage.rules(ROOT)
        self.assertEqual(sorted(code), ['Expansion/Main.lean', 'Syntax.lean'])
        for expected in [
                ('tritonDType', 'tl.float32', 'Syntax.lean', 8),
                ('tritonExpr', 'tl.load(<tritonExpr>«,<tritonMemKwarg>»*)', 'Syntax.lean', 11),
                ('tritonExpr', '<tritonExpr>.to(<tritonDType>)', 'Syntax.lean', 12),
                ('tritonStmt', 'tl.store(<tritonExpr>,<tritonExpr>«,<tritonMemKwarg>»*)', 'Syntax.lean', 20),
                ('tritonStmt', 'for<ident>in range(<tritonExpr>){<tritonStmt>*}', 'Syntax.lean', 25),
                ('controlSourceStmt', 'pass', 'Syntax.lean', 26)]:
            self.assertIn(expected, declarations)
        self.assertEqual(len(declarations), 18)  # the two `tl.void` declarations are in comments


class Api(unittest.TestCase):
    def test_names_kinds_and_optional_parameters(self):
        names = list(coverage.api(CHECKOUTS / report.checkout(coverage.TRITON)))
        self.assertEqual(names, [
            ('atomic_max', 'function', ['mask']),
            ('cast', 'function', ['bitcast']),
            ('float32', 'value', []),
            ('inline_asm_elementwise', 'function', []),
            ('load', 'function', ['mask', 'other', 'volatile']),  # `_semantic` is private
            ('make_tensor_descriptor', 'function', []),
            ('math', 'module', []),
            ('range', 'class', ['arg2', 'num_stages']),
            ('store', 'function', ['mask']),
            ('sum', 'function', ['axis', 'keep_dims']),
            ('void', 'value', []),
        ])


class Matrix(unittest.TestCase):
    def test_section_matches_the_expected_file(self):
        self.assertEqual(matrix() + '\n', (FIXTURES / 'expected.md').read_text(encoding='utf-8'))

    def test_a_reading_applies_only_while_its_line_holds(self):
        def rows(text):
            return {line.split(' | ')[0][2:]: line.split(' | ')[1:3] for line in text.splitlines() if line[:3] == '| `'}
        position = 'a keyword position exists (`Syntax.lean:9`), so the expander decides; '
        read = rows(matrix())
        self.assertEqual(read['`tl.load(mask=)`'], ['rule', position + 'by reading `Expansion/Main.lean:4`'])
        self.assertEqual(read['`tl.load(other=)`'], [
            'not determined', position + 'the line read for this row, `Expansion/Main.lean:4`, has changed'])
        self.assertEqual(rows(matrix(read={}))['`tl.load(mask=)`'], [
            'not determined', position + 'it was not read for this keyword'])

    def test_without_the_benchmark_checkout_the_column_says_so(self):
        with mock.patch.dict(coverage.BENCH, commit='0' * 40):
            text = matrix()
        self.assertIn('| Triton (2) | Triton-distributed (1) | bench | ticket |', text)
        self.assertIn('| `tl.store(...)` | rule | `tritonStmt` `Syntax.lean:20` | 1 | 0 | not counted |  |', text)
        self.assertIn('- Benchmark kernels: not counted', text)


class Update(unittest.TestCase):
    def run_update(self, document):
        coverage.main([str(MANIFEST), str(CHECKOUTS), '--root', str(ROOT), '--update', str(document)])
        return document.read_text(encoding='utf-8')

    def test_section_is_appended_then_rewritten_in_place(self):
        with tempfile.TemporaryDirectory() as tmp:
            document = Path(tmp) / 'doc.md'
            document.write_text('# Title\n\nHand-written.\n', encoding='utf-8')
            first = self.run_update(document)
            self.assertTrue(first.startswith(f'# Title\n\nHand-written.\n\n{coverage.BEGIN}\n'))
            self.assertTrue(first.endswith(f'{coverage.END}\n'))
            self.assertEqual(self.run_update(document), first)
            document.write_text(first.replace('44 rows.', 'stale').replace(coverage.END, coverage.END + '\n\nAfter.'),
                                encoding='utf-8')
            self.assertEqual(self.run_update(document), first + '\nAfter.\n')

    def test_a_begin_marker_without_an_end_marker_stops_the_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            document = Path(tmp) / 'doc.md'
            document.write_text(f'head\n{coverage.BEGIN}\nold\n', encoding='utf-8')
            with self.assertRaises(SystemExit):
                self.run_update(document)
            self.assertEqual(document.read_text(encoding='utf-8'), f'head\n{coverage.BEGIN}\nold\n')


if __name__ == '__main__':
    unittest.main()
