#!/usr/bin/env python3
"""Self-tests of the census scanner.

`fixtures/expected.json` is the scanner's output for the three fixture files.
After an intended change, regenerate it from `scripts/census/fixtures`:

    python3 ../scan.py triton_kernels.py triton_dist_kernels.py tilelang_kernels.py > expected.json
"""
import json
from pathlib import Path
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from census import scan

FIXTURES = Path(__file__).resolve().parent / 'fixtures'
NAMES = ['triton_kernels.py', 'triton_dist_kernels.py', 'tilelang_kernels.py']


def census(source):
    return {r['function']: r for r in scan.scan_source(source, 'case.py')}


def names(record):
    return {c['name'] for c in record['calls']}


class Fixtures(unittest.TestCase):
    def test_command_line_output_matches_the_checked_in_census(self):
        run = subprocess.run([sys.executable, '-B', str(FIXTURES.parent / 'scan.py'), *NAMES],
                             cwd=FIXTURES, capture_output=True, text=True, check=True)
        self.assertEqual(run.stdout, (FIXTURES / 'expected.json').read_text(encoding='utf-8'))

    def test_every_family_reports_kernels_and_a_helper_edge(self):
        records = json.loads((FIXTURES / 'expected.json').read_text(encoding='utf-8'))
        for name in NAMES:
            mine = [r for r in records if r['file'] == name]
            self.assertTrue(any(r['kernel'] for r in mine), name)
            self.assertTrue(any(r['helpers'] for r in mine), name)
        self.assertNotIn('matvec', {r['function'] for r in records})  # undecorated host code


class Resolution(unittest.TestCase):
    def test_tl_range_is_separate_from_builtin_range(self):
        record = census(
            'import triton\n'
            'import triton.language as tl\n'
            '@triton.jit\n'
            'def k(n, S: tl.constexpr):\n'
            '    for i in tl.range(0, n, num_stages=S):\n'
            '        pass\n'
            '    for j in range(n):\n'
            '        pass\n')['k']
        self.assertEqual(record['calls'], [
            {'name': 'range', 'args': 1, 'kwargs': [], 'count': 1},
            {'name': 'triton.language.range', 'args': 2, 'kwargs': ['num_stages'], 'count': 1},
        ])
        self.assertEqual(record['statements'], {'For': 2, 'Pass': 2})

    def test_range_imported_from_triton_is_not_the_builtin(self):
        record = census(
            'import triton\n'
            'from triton.language import range\n'
            '@triton.jit\n'
            'def k(n):\n'
            '    for i in range(n):\n'
            '        pass\n')['k']
        self.assertEqual(names(record), {'triton.language.range'})

    def test_alias_follows_the_import_not_its_spelling(self):
        records = census(
            'import tilelang as tl\n'
            'from tilelang import language as T\n'
            '@tl.jit\n'
            'def eager(A):\n'
            '    with T.Kernel(4) as bx:\n'
            '        T.clear(A)\n'
            '@tl.jit\n'
            'def build(n):\n'
            '    @T.prim_func(private=True)\n'
            '    def main(A: T.Tensor((n,), "float32")):\n'
            '        T.clear(A)\n'
            '    return main\n')
        self.assertEqual(sorted(records), ['build.main', 'eager'])  # `build` only builds `main`
        self.assertEqual(records['eager']['decorators'], ['tilelang.jit'])
        self.assertEqual(records['build.main']['decorators'], ['tilelang.language.prim_func'])
        self.assertEqual(names(records['build.main']), {'tilelang.language.Tensor', 'tilelang.language.clear'})

    def test_local_names_shadow_imports(self):
        record = census(
            'import triton\n'
            'import tilelang.language as T\n'
            '@triton.jit\n'
            'def k(x, T, fn):\n'
            '    y = fn(x)\n'
            '    return T.to(y)[0].sum()(x)\n')['k']
        self.assertEqual(names(record), {'fn', '.to', '.sum', '<expr>'})

    def test_helpers_are_followed_transitively_and_scoped(self):
        records = census(
            'import triton\n'
            'def unused(x):\n'
            '    return x\n'
            'def inner(x):\n'
            '    return abs(x)\n'
            'def outer(x):\n'
            '    return inner(x)\n'
            'class Box:\n'
            '    def inner(self, x):\n'
            '        return x\n'
            '    @triton.jit\n'
            '    def k(x):\n'
            '        def local(v):\n'
            '            return outer(v)\n'
            '        return local(inner(x))\n')
        self.assertEqual(sorted(records), ['Box.k', 'Box.k.local', 'inner', 'outer'])
        self.assertEqual(records['Box.k']['helpers'], ['Box.k.local', 'inner'])  # not the method Box.inner
        self.assertEqual(records['Box.k']['statements'], {'FunctionDef': 1, 'Return': 1})
        self.assertEqual(records['Box.k.local']['helpers'], ['outer'])
        self.assertEqual(records['outer']['helpers'], ['inner'])
        self.assertEqual([records[q]['kernel'] for q in sorted(records)], [True, False, False, False])

    def test_both_definitions_of_a_repeated_name_are_reported(self):
        records = scan.scan_source(
            'import tilelang.language as T\n'
            'def build(decode):\n'
            '    if decode:\n'
            '        @T.prim_func\n'
            '        def main(A: T.Tensor((4,), "float32")):\n'
            '            T.clear(A)\n'
            '    else:\n'
            '        @T.prim_func\n'
            '        def main(A: T.Tensor((4,), "float32")):\n'
            '            T.fill(A, 0)\n'
            '    return main\n', 'case.py')
        self.assertEqual([(r['function'], r['line']) for r in records], [('build.main', 5), ('build.main', 9)])
        self.assertEqual([sorted(names(r)) for r in records], [
            ['tilelang.language.Tensor', 'tilelang.language.clear'],
            ['tilelang.language.Tensor', 'tilelang.language.fill'],
        ])

    def test_star_and_double_star_arguments(self):
        record = census(
            'import triton\n'
            '@triton.jit\n'
            'def k(a, kw):\n'
            '    f(*a, 1, b=2, **kw)\n')['k']
        self.assertEqual(record['calls'], [{'name': 'f', 'args': 2, 'kwargs': ['**', 'b'], 'count': 1}])


if __name__ == '__main__':
    unittest.main()
