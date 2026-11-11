#!/usr/bin/env python3
"""AST census of GPU kernel sources: what each kernel calls and which statements it uses.

    python3 scripts/census/scan.py FILE... > census.json

The files are parsed with `ast` and never imported or run, so Triton,
Triton-distributed, TileLang and TileScale need not be installed. All four
front ends share one schema. The front end is not guessed: TileLang and
TileScale use the same package name, and Triton-distributed files also use
`@triton.jit`. The caller knows which repository a file came from.

Output: a JSON list with one record per function, in source order per file.

    {
      "file": "softmax.py",             path as given on the command line
      "function": "softmax_kernel",     qualified name; a nested def is "outer.main"
      "line": 12,                       line of the `def`
      "kernel": true,                   false: a helper, reported because a kernel reaches it
      "decorators": ["triton.jit"],     resolved names in source order, arguments dropped
      "statements": {"Assign": 6, "For": 1},
                                        `ast` statement class name -> count
      "calls": [                        one entry per distinct (name, args, kwargs)
        {"name": "triton.language.load", "args": 1, "kwargs": ["mask", "other"], "count": 2},
        {"name": "range", "args": 2, "kwargs": [], "count": 1}
      ],
      "helpers": ["leaky_relu"]         functions of this file that it calls directly
    }

A function is a kernel when it is decorated with `triton.jit`,
`triton_dist.jit` or a `prim_func`, or with `tilelang.jit` and no `prim_func`
nested in it (eager TileLang, where the decorated function is the kernel body;
otherwise it only builds the nested `prim_func`). Every function of the file
that a kernel reaches through calls also gets a record, so the records and
their `helpers` lists are the helper-call graph. A name defined twice in one
file (`if decode: def main ... else: def main`) gets one record per
definition, told apart by `line`; a `helpers` entry then means all of them.

`statements`, `calls` and `helpers` cover the function's parameters and body
without its nested defs (a nested def counts as one `FunctionDef` statement).
`args` counts positional arguments, a `*sequence` as one; `kwargs` are the
sorted keyword names, `**` for a `**mapping` argument.

Names are resolved through the file's imports, so `tl.range` with
`import triton.language as tl` is `triton.language.range` while the builtin
stays `range`, and `tl.jit` with `import tilelang as tl` is `tilelang.jit`.
The other forms of a call name:

    "leaky_relu"     a function of this file, by qualified name; it is also in `helpers`
    ".to"            a method of a local value (`x.to(...)`, `a[i].to(...)`)
    "f"              anything else: a builtin, a star-imported name or a callable argument
    "<expr>"         the callee is itself a call or subscript (`f(x)(y)`)

Not reported: attribute uses that are not calls (`tl.constexpr`,
`T.GemmWarpPolicy.FullRow`), operators and other expression kinds, and
helpers that are imported from another file, reached through `self.` or
passed as values. A file the running Python cannot parse raises `SyntaxError`
and nothing is printed.
"""
import argparse
import ast
from collections import Counter
import json

DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _functions(node, prefix=''):
    """Every def under `node` as (qualified name, node), outer before inner."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, DEFS):
            if not isinstance(child, ast.ClassDef):
                yield prefix + child.name, child
            yield from _functions(child, prefix + child.name + '.')
        else:
            yield from _functions(child, prefix)


def _own(fn):
    """Nodes of the parameters and body of `fn`, not entering nested defs."""
    stack = [fn.args, *fn.body]
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, DEFS):
            stack.extend(ast.iter_child_nodes(node))


def scan_source(source, filename):
    """Census records of one source text; `filename` is copied into each record."""
    tree = ast.parse(source, filename)
    # shortcut: one import table per file, so an import inside one function counts
    # everywhere; make it per scope if two functions bind one alias differently.
    imports = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                top = a.name.split('.')[0]  # `import a.b` binds `a`
                imports[a.asname or top] = a.name if a.asname else top
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.name != '*':
                    imports[a.asname or a.name] = '.' * node.level + '.'.join(filter(None, [node.module, a.name]))
    # A file may define one name twice (`if decode: def main ... else: def main`), so
    # definitions are a list; a name stands for every definition that carries it.
    defs = list(_functions(tree))
    bound = {}  # function -> its parameters and assigned names, which shadow imports
    for qual, fn in defs:
        names = bound.setdefault(qual, set())
        for n in _own(fn):
            if isinstance(n, ast.arg):
                names.add(n.arg)
            elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                names.add(n.id)

    def resolve(node, scope):
        """(reported name, called function of this file or None) of an expression in function `scope`."""
        attrs = []
        while isinstance(node, ast.Attribute):
            attrs.insert(0, node.attr)
            node = node.value
        if not isinstance(node, ast.Name):
            return '.' + attrs[-1] if attrs else '<expr>', None
        while True:  # innermost scope first, as Python looks names up
            if not scope or scope in bound:  # a class body is not visible from its methods
                target = (scope + '.' if scope else '') + node.id
                if target in bound:
                    return '.'.join([target] + attrs), None if attrs else target
                if node.id in bound.get(scope, ()):
                    return '.' + attrs[-1] if attrs else node.id, None
            if not scope:
                return '.'.join([imports.get(node.id, node.id)] + attrs), None
            scope = scope.rpartition('.')[0]

    decorators = [[resolve(d.func if isinstance(d, ast.Call) else d, qual.rpartition('.')[0])[0]
                   for d in fn.decorator_list] for qual, fn in defs]
    prim_funcs = {qual for (qual, _), decs in zip(defs, decorators)
                  if any(d.split('.')[-1] == 'prim_func' for d in decs)}
    records = []
    for (qual, fn), decs in zip(defs, decorators):
        nodes = list(_own(fn))
        calls, helpers = Counter(), set()
        for node in nodes:
            if isinstance(node, ast.Call):
                name, target = resolve(node.func, qual)
                if target:
                    helpers.add(target)
                calls[name, len(node.args), tuple(sorted(k.arg or '**' for k in node.keywords))] += 1
        records.append({
            'file': filename,
            'function': qual,
            'line': fn.lineno,
            'kernel': bool({'triton.jit', 'triton_dist.jit'} & set(decs) or qual in prim_funcs
                           or 'tilelang.jit' in decs and not any(q.startswith(qual + '.') for q in prim_funcs)),
            'decorators': decs,
            'statements': dict(sorted(Counter(type(n).__name__ for n in nodes if isinstance(n, ast.stmt)).items())),
            'calls': [{'name': name, 'args': args, 'kwargs': list(kwargs), 'count': count}
                      for (name, args, kwargs), count in sorted(calls.items())],
            'helpers': sorted(helpers),
        })
    reached, grow = set(), {r['function'] for r in records if r['kernel']}
    while grow:
        reached |= grow
        grow = {h for r in records if r['function'] in grow for h in r['helpers']} - reached
    return [r for r in records if r['function'] in reached]


def main():
    parser = argparse.ArgumentParser(description='Print the kernel census of Python source files as JSON.')
    parser.add_argument('files', nargs='+')
    records = []
    for path in parser.parse_args().files:
        with open(path, encoding='utf-8') as f:
            records += scan_source(f.read(), path)
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
