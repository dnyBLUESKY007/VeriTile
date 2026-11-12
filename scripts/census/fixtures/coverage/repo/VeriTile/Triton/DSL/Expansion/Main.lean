def expandLoad (name : String) : MacroM Unit := do
  let quote := '"'  -- a character, not the start of a string
  match name with
  | "mask" => pure ()
  | "other" => pure ()
  | unknown => Macro.throwError ("tl.load: unknown kwarg -- " ++ unknown)

def expandAtomicMax : MacroM Syntax :=
  `(ComputeStmt.effectMarker "tl.atomic_max")
