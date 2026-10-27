import Lean

open Lean Elab

/-- Only direct tactic-head tokens are eligible; tactic-sequence wrappers are excluded. -/
def directHead? (stx : Syntax) : Option Syntax :=
  let head := match stx with
    | .node _ _ args => args[0]?
    | .atom .. | .ident .. => some stx
    | .missing => none
  head.bind fun leaf => match leaf with
    | .atom .. | .ident .. => some leaf
    | _ => none

partial def collect (tree : InfoTree) (lower upper : Nat)
    (ctx? : Option ContextInfo := none) : IO (Array Json) := do
  match tree with
  | .context info child => collect child lower upper (info.mergeIntoOuter? ctx?)
  | .node info children =>
    let mut result := #[]
    if let (.ofTacticInfo ti, some ctx) := (info, ctx?) then
      if let some leaf := directHead? ti.stx then
        if let (some start, some stop) :=
            (leaf.getPos? (canonicalOnly := true), leaf.getTailPos? (canonicalOnly := true)) then
          if lower ≤ start.byteIdx && stop.byteIdx ≤ upper then
            let before ← ({ ctx with mctx := ti.mctxBefore } : ContextInfo).ppGoals ti.goalsBefore
            let after ← ({ ctx with mctx := ti.mctxAfter } : ContextInfo).ppGoals ti.goalsAfter
            result := result.push <| Json.mkObj [
              ("start", toJson start.byteIdx), ("end", toJson stop.byteIdx),
              ("syntax_start", toJson (ti.stx.getPos?.getD start).byteIdx),
              ("syntax_end", toJson (ti.stx.getTailPos?.getD stop).byteIdx),
              ("kind", toJson ti.stx.getKind.toString),
              ("has_missing", toJson ti.stx.hasMissing),
              ("head", toJson (if leaf.isAtom then leaf.getAtomVal else leaf.getId.toString)),
              ("before", toJson before.pretty), ("after", toJson after.pretty)]
    for child in children do
      result := result ++ (← collect child lower upper ctx?)
    return result
  | .hole _ => return #[]

def emitResult (tactics : Array Json) (messages : MessageLog)
    (ictx : Parser.InputContext) : IO Unit := do
  let mut diagnostics := #[]
  for message in messages.toList do
    let j ← message.toJson
    let j := j.setObjVal! "start_byte" <| toJson (ictx.fileMap.ofPosition message.pos).byteIdx
    let j := j.setObjVal! "end_byte" <| toJson (message.endPos.map fun p => (ictx.fileMap.ofPosition p).byteIdx)
    diagnostics := diagnostics.push j
  IO.println <| "__TACTIC_PROBE__" ++ (Json.mkObj [
    ("compiled", toJson (!messages.hasErrors)),
    ("tactics", .arr tactics), ("messages", .arr diagnostics)]).compress

unsafe def main (args : List String) : IO UInt32 := do
  if args.isEmpty then
    IO.eprintln "usage: tactic_probe.lean FILE [START_BYTE END_BYTE]"
    return 1
  enableInitializersExecution
  initSearchPath (← findSysroot)
  let file := args.head!
  let input ← IO.FS.readFile file
  let lower := args[1]?.bind String.toNat? |>.getD 0
  let upper := args[2]?.bind String.toNat? |>.getD input.utf8ByteSize
  let ictx := Parser.mkInputContext input file
  let opts := ({} : Options).setBool `Elab.async false
  let (header, pstate, msgs) ← Parser.parseHeader ictx
  let (env, msgs) ← processHeader header opts msgs ictx (mainModule := `Mutation)
  -- Header failures must survive independently of command snapshots. In
  -- particular, an unbuilt import must not become a downstream parse error.
  if msgs.hasErrors then
    emitResult #[] msgs ictx
    return 0
  let cstate := { Command.mkState env msgs opts with infoState.enabled := true }
  let state ← Elab.IO.processCommands ictx pstate cstate
  let mut tactics := #[]
  for tree in state.commandState.infoState.trees do
    tactics := tactics ++ (← collect tree lower upper)
  emitResult tactics (msgs ++ state.commandState.messages) ictx
  return 0
