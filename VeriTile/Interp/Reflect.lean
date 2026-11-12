/-
VeriTile.Interp.Reflect

`reflect_kernel k` reads the definition of a kernel `k : … → ComputeKernel`
and adds, for the same parameters,

* `k.mirror : … → QKernel`, the kernel with every real literal replaced by a
  rational one, and
* `k.toReal_mirror : ∀ …, (k.mirror …).toReal = k …`.

The body of `k` is unfolded until only constructors of the kernel AST remain;
a `let` is replaced by its value. Each constructor is then renamed to its
mirror (`VeriTile.Interp.Mirror`), and the argument of `Op.const` and
`Op.constFloat` is rewritten over `ℚ`. A literal may be built from numerals,
scientific literals, a cast of a rational, `-`, `+`, `*` and `/`.

The command fails with

* `reflect_kernel: non-rational literal r`, at the first literal that has
  another form (`√2`, `π`, a real parameter of the kernel), and
* `reflect_kernel: unsupported constructor c`, at the first constant `c` that
  yields or consumes kernel AST and has no definition to unfold (a recursor,
  an opaque constant), which includes kernels computed by a `match` on AST.

A constant that is polymorphic in the AST type (`ite`, `++`, `List.map`) is
kept as it is. The theorem then holds only if both sides compute to the same
constructors: an `if` on a closed condition does; an `if` on a parameter is
rejected by the kernel with a type mismatch.

The theorem is checked by the kernel. Its proof is one `push_cast` equation
`(q : ℝ) = r` per literal and a congruence over them; that `toReal` of the
mirror unfolds to the kernel with the literals `(q : ℝ)` is left to the
kernel's definitional equality, which unfolds `toReal` once per AST node.
-/
import Mathlib.Data.Rat.Cast.CharZero
import Mathlib.Data.Rat.Cast.Lemmas
import VeriTile.Interp.Mirror

namespace VeriTile.Interp.Reflect

open Lean Meta Elab Command
open VeriTile.Triton

/-- The upstream AST types and their mirrors, as checked by `#assert_mirror`. -/
private def mirrors : List (Name × Name) :=
  [(``Op, ``QOp), (``MemAccess, ``QMemAccess), (``MaskOpt, ``QMaskOpt), (``Stmt, ``QAlgStmt),
   (``ComputeOp, ``QComputeOp), (``ComputeExpr, ``QExpr), (``ComputeStmt, ``QStmt),
   (``ComputeKernel, ``QKernel)]

/-- The mirror of an upstream AST type or of one of its constructors. -/
private def mirrorName? (env : Environment) (n : Name) : Option Name :=
  let type := match env.find? n with
    | some (.ctorInfo c) => c.induct
    | _ => n
  (mirrors.lookup type).map (n.replacePrefix type ·)

/-- The rational expression with the shape of the real expression `r`, a part of the
literal `lit`. -/
private partial def toRat (lit r : Expr) : MetaM Expr := do
  let r := r.consumeMData
  let rat := mkConst ``Rat
  let nonRational : MetaM Expr := throwError "reflect_kernel: non-rational literal {lit}"
  match r.getAppFn.constName?, r.getAppArgs with
  | some ``Rat.cast, #[_, _, q] => return q
  | some ``OfNat.ofNat, #[_, n, _] => mkAppOptM ``OfNat.ofNat #[rat, n, none]
  | some ``OfScientific.ofScientific, #[_, _, m, s, e] =>
      mkAppOptM ``OfScientific.ofScientific #[rat, none, m, s, e]
  | some ``Neg.neg, #[_, _, a] => mkAppM ``Neg.neg #[← toRat lit a]
  | some op, #[_, _, _, _, a, b] =>
      if [``HAdd.hAdd, ``HSub.hSub, ``HMul.hMul, ``HDiv.hDiv].contains op then
        mkAppM op #[← toRat lit a, ← toRat lit b]
      else nonRational
  | _, _ => nonRational

/-- `reflect_kernel k` adds the rational mirror `k.mirror` of the kernel `k` and the theorem
`k.toReal_mirror : ∀ …, (k.mirror …).toReal = k …`. -/
elab "reflect_kernel " k:ident : command => liftTermElabM do
  let kName ← realizeGlobalConstNoOverloadWithInfo k
  let info ← getConstInfo kName
  forallTelescope info.type fun xs type => do
  unless type.isConstOf ``ComputeKernel && info.levelParams.isEmpty do
    throwError "reflect_kernel: {kName} is not a function into ComputeKernel"
  let env ← getEnv
  let isAst (e : Expr) : Bool :=
    (e.find? fun | .const n _ => (mirrors.lookup n).isSome | _ => false).isSome
  -- One entry per literal: the variable that stands for it, its rational form `q`, and `r`.
  let literals ← IO.mkRef (#[] : Array (Expr × Expr × Expr))
  let literal (e : Expr) : MetaM TransformStep := do
    let r := e.appArg!
    if r.hasLooseBVars then
      throwError "reflect_kernel: non-rational literal (it depends on a bound variable)"
    let x := mkFVar (← mkFreshFVarId)
    literals.modify (·.push (x, ← toRat r r, r))
    return .done (mkApp e.appFn! x)
  -- The kernel as a term of AST constructors, with a variable in place of each literal.
  let body ← Core.transform (mkAppN (mkConst kName) xs) (pre := fun e => do
    if let .letE _ _ v b _ := e then return .visit (b.instantiate1 v)
    let .const c _ := e.getAppFn | return .continue
    if e.isAppOfArity ``Op.const 1 || e.isAppOfArity ``Op.constFloat 2 then return ← literal e
    if (mirrorName? env c).isSome && c != ``Op.const && c != ``Op.constFloat then
      return .continue
    unless isAst (← getConstInfo c).type do return .continue
    let some e ← delta? e | throwError "reflect_kernel: unsupported constructor {c}"
    return .visit e)
  let literals ← literals.get
  let vars := literals.map (·.1)
  let mirror := (body.replace fun
    | .const n us => (mirrorName? env n).map (mkConst · us)
    | _ => none).replaceFVars vars (literals.map (·.2.1))
  let mirrorName := kName ++ `mirror
  let value ← mkLambdaFVars xs mirror
  addAndCompile <| .defnDecl {
    name := mirrorName, levelParams := [], type := ← mkForallFVars xs (mkConst ``QKernel), value,
    hints := .regular (getMaxHeight env value + 1), safety := .safe }
  -- `context (q₁ : ℝ) … = context r₁ …`; the kernel checks that this is the statement.
  -- shortcut: one congruence over all literals costs time quadratic in their number (about
  -- 5 s at 800); build it constructor by constructor if a kernel ever has that many.
  let context := literals.foldr (fun _ b => .lam `x (mkConst ``Real) b .default) (body.abstract vars)
  let mut proof ← mkEqRefl context
  for (_, q, r) in literals do
    let h ← Term.elabTermEnsuringType (← `(by push_cast; with_reducible rfl))
      (← mkEq (← mkAppOptM ``Rat.cast #[mkConst ``Real, none, q]) r)
    Term.synthesizeSyntheticMVarsNoPostponing
    proof ← mkCongr proof (← instantiateMVars h)
  let statement ← mkEq (← mkAppM ``QKernel.toReal #[mkAppN (mkConst mirrorName) xs])
    (mkAppN (mkConst kName) xs)
  addDecl <| .thmDecl {
    name := kName ++ `toReal_mirror, levelParams := [], type := ← mkForallFVars xs statement,
    value := ← mkLambdaFVars xs proof }

end VeriTile.Interp.Reflect
