# Staging

`VeriTile/Staging/` holds statements that are not proved yet. It is the lake
target `VeriTileStaging`, the only fork target in which `sorry` is allowed.

## Rules

- A freeze ticket puts its interface and statements here, under
  `VeriTile/Staging/{Dist,TileLang,Interp,Equiv}/`, with `sorry` in place of the proofs.
- Declare each statement, and each definition it mentions, in its final
  namespace and under its final name. Fingerprints are computed from names and
  elaborated terms, so a stub and its later replacement are recognised as the
  same declaration only if the name is kept.
- A statement that the user has not signed yet says so in its docstring
  (`UNSIGNED`). A signed statement gets one row in
  [`documents/dist/FrozenStatements.tsv`](../../documents/dist/FrozenStatements.tsv).
- The ticket that proves a statement moves it to its main target and deletes
  the stub in the same change, then updates the `module` column of its row.
  The fingerprint stays as recorded.
- No module of `VeriTileDist`, `VeriTileTileLang`, `VeriTileInterp` or
  `VeriTileEquiv` imports a module from here, directly or through another module.

## Targets and gates

| Target | Sources | `sorry` | CI shard |
| --- | --- | --- | --- |
| `VeriTileDist` | `VeriTile/Dist.lean`, `VeriTile/Dist/` | rejected | `.github/workflows/dist.yml` |
| `VeriTileTileLang` | `VeriTile/TileLang.lean`, `VeriTile/TileLang/` | rejected | `.github/workflows/tilelang.yml` |
| `VeriTileInterp` | `VeriTile/Interp.lean`, `VeriTile/Interp/` | rejected | `.github/workflows/interp.yml` |
| `VeriTileEquiv` | `VeriTile/Equiv.lean`, `VeriTile/Equiv/` | rejected | `.github/workflows/equiv.yml` |
| `VeriTileStaging` | `VeriTile/Staging/` | allowed | `.github/workflows/staging.yml` |

Run the same gates locally from the repository root:

```sh
python3 scripts/check_fork_target.py VeriTileDist            # build; no sorry, staging import or new axiom
python3 scripts/check_fork_target.py VeriTileDist --replay   # also comparator and trust gates (Linux judge host)
python3 scripts/check_fork_target.py VeriTileStaging         # build only
python3 scripts/check_frozen_statements.py                   # signed statements and the definitions under them are unchanged
python3 scripts/test_fork_targets.py                         # the gates reject deliberate violations
```

## Signing a statement

Add a row `statement<TAB>module<TAB>-<TAB>date, ticket and who signed` to
`documents/dist/FrozenStatements.tsv`, then run
`python3 scripts/check_frozen_statements.py --fill`. It records two things and
never replaces a fingerprint that is already recorded:

- in `FrozenStatements.tsv`, the fingerprint of the elaborated statement, up to
  the names of bound variables;
- in `documents/dist/FrozenDefinitions.tsv`, one fingerprint for every constant
  of this repository that the statement depends on: its kind, its type and,
  for a definition, its body. The search follows types, definition bodies and
  constructors, not proofs, and stops at Lean core and Mathlib.

## When the check fails

- **A statement changed.** A signed statement is edited only with a new
  signature: restore it, or change its row by hand after the user signs again.
- **A definition changed, appeared or is no longer needed.** The message names
  the definition and the signed statements it affects. The statements read the
  same but may no longer say the same thing. This is expected when a stub of a
  freeze ticket gets its real definition and when an upstream merge changes
  the semantics; it is a defect when a definition was bent to make a proof go
  through. After review, run
  `python3 scripts/check_frozen_statements.py --accept-definitions`; the diff
  of `FrozenDefinitions.tsv` then lists exactly the definitions that were
  accepted.
