# Staging

`VeriTile/Staging/` holds statements that are not proved yet. It is the lake
target `VeriTileStaging`, the only fork target in which `sorry` is allowed.

## Rules

- A freeze ticket puts its interface and statements here, under
  `VeriTile/Staging/{Dist,TileLang,Interp,Equiv}/`, with `sorry` in place of the proofs.
- Declare each statement in its final namespace and under its final name. The
  fingerprint of a statement depends on the declaration's name and its
  elaborated type, so a stub and its later proof agree only if both match.
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
python3 scripts/check_frozen_statements.py                   # signed statements keep their fingerprints
python3 scripts/test_fork_targets.py                         # the gates reject deliberate violations
```

## Signing a statement

Add a row `statement<TAB>module<TAB>-<TAB>date and ticket` to
`FrozenStatements.tsv`, then run `python3 scripts/check_frozen_statements.py --fill`.
`--fill` records fingerprints only for rows marked `-`. A recorded fingerprint
changes only by editing the file by hand, which a reviewer sees in the diff.

The fingerprint covers the elaborated statement up to the names of bound
variables. It does not cover the bodies of the definitions the statement
mentions; review those with `#auditStmt` from `VeriTile.Meta.StatementAudit`.
