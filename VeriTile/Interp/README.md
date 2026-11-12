# VeriTile/Interp

Sources of the lake target `VeriTileInterp`: the executable interpreter and its bridge to the proof semantics.
`VeriTile/Interp.lean` is its root module.

`Bridge/` holds the bridge to the proof semantics: `RealAlgebra.lean` defines the algebra and the erasure, and `EvalOp.lean` proves that expressions evaluate alike on both sides; its header lists the named preconditions and every constructor it covers.

`Adequacy/` holds the lemmas that connect the executable definitions to `FP.Structural`; the header of `Adequacy/State.lean` lists each state operation and helper twin with its upstream original and the lemma that states their agreement.

No `sorry`, no import from `VeriTile/Staging/`, no new axiom; see
[the staging rules and gates](../Staging/README.md).

`Reflect.lean` provides `reflect_kernel k`: from a kernel `k : … → ComputeKernel` it adds the rational mirror
`k.mirror : … → QKernel` and the theorem `k.toReal_mirror : ∀ …, (k.mirror …).toReal = k …`
(checked in `tests/lean/InterpReflectSmoke.lean`).
