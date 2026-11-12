# VeriTile/Interp

Sources of the lake target `VeriTileInterp`: the executable interpreter and its bridge to the proof semantics.
`VeriTile/Interp.lean` is its root module.

`Adequacy/` holds the lemmas that connect the executable definitions to `FP.Structural`; the header of `Adequacy/State.lean` lists each state operation and helper twin with its upstream original and the lemma that states their agreement.

No `sorry`, no import from `VeriTile/Staging/`, no new axiom; see
[the staging rules and gates](../Staging/README.md).
