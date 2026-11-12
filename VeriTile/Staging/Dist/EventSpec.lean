/-
VeriTile.Staging.Dist.EventSpec

Draft of the event interface of the Dist line for the freeze ticket S2.1
(#37). Everything here is UNSIGNED: no declaration has a row in
`documents/dist/FrozenStatements.tsv`. The user signs after deciding the
choices listed below.

An event is what one agent does at one synchronization point. The file gives
the `Event` type, the interface of the three patterns and the statements of
the pattern lemmas:

- (a) a memory access, an atomic or a fence, with a scope and an order
  (S2.3, #39);
- (b) a write to a monotone counter, or a blocking wait on one (S2.4, #40);
- (c) an asynchronous transfer: an initiation event and a completion event
  that is bound to a counter (S2.5, #41).

Each theorem is a statement with `sorry`. The ticket named in a docstring moves
the declaration to `VeriTile/Dist/` under the same name and type, proves it and
deletes it here in the same change. `PrimitiveSources.md` below is
`documents/dist/PrimitiveSources.md` (S2.2); `D5`, `W1`, `A2`, `E-NV12` and the
like are its decisions, rules, assumptions and evidence items.

Open choices. Each definition marked PROVISIONAL takes the first option of the
choice it names, so that the statements elaborate. A different decision
changes that definition before anything is signed.

 1. Relation to upstream `VeriTile.Triton.Concurrent.Event`: here a separate
    type over `Addr`, `Agent` and `Scope`; nothing upstream is imported. The
    embedding of upstream events is left to the bridge of S4.4.
 2. Scope and order on an access: here one `Mode` per access with three
    cases: `plain` (no atomicity, no scope), `atomic scope order` (CUDA C++
    model) and `shmem order` (NVSHMEM signalling and ordering routines).
 3. One atomic family or two: here two. Accesses of different families to one
    cell conflict (E-NV12), and only an edge of the NVSHMEM family is subject
    to choice 4.
 4. D2, how far NVSHMEM visibility reaches: here a parameter `ShmemReach` of
    `SyncEdge`, not a fixed reading.
 5. Monotonicity of a counter: here an obligation on the trace
    (`Counter.SetsMonotone`), a hypothesis of `Counter.mono`; not a property
    of the type.
 6. Identity of a counter: here a type `Counter` with four kinds: a memory
    word, a barrier object of a CTA, a rendezvous of several ranks, and the
    delivered transfers of an agent.
 7. Binding of a completion to its counter: here the transfer names the
    counter write that its completion performs (`Transfer.completion`), and
    the completion event performs the copy and that write in one step.
    A transfer without one never completes (W5).
 8. Ranks: here `Rank` (a natural number), through `Addr`, `Agent` and `Cta`;
    no `Fin numRanks` and no rank-count parameter on `Event`.
 9. What an event carries: here evaluated addresses and the values written
    to counters; not source expressions, and not the value that a load, an
    update or a wait reads.
10. Where races are stated: here only `Conflict` and the conditions without
    which no synchronization edge exists (`SyncEdge`). Happens-before, the
    data race and the trace conditions of an edge (which write a read reads,
    fences) belong to S4.1 (#47). The lemma of S2.5 "a read without a
    completion edge races with the transfer" is therefore stated here in two
    parts that do not need happens-before (`inFlight_of_initiate`,
    `conflict_complete`) and one that bounds the edges
    (`SyncEdge.of_complete`).

Smaller provisional points: a trace entry is a pair of an agent and an event;
the completion of a transfer is paired with the agent that initiated it and is
not in the program order of that agent; a transfer is identified by its
fields, not by a number; counter values are natural numbers, so the
wrap-around of a 64-bit word is not modelled; a `dl.wait` under D6 is proved
for both comparisons `eq` and `ge`.

What each pattern has to carry (read from `documents/census/MinimalSyntax.md`,
kernels 106–119, from `PrimitiveSources.md` and from
`documents/dist/DeepEPInventory.md`, section 4.2; no kernel source was read
for this file):

- (a) Tutorial 01: stores and loads through `dl.symm_at`, `libshmem_device.fence`.
  Tutorial 04: `fence`. Tutorial 05: `atomic_cas` in
  `barrier_all_intra_node_atomic_cas_block`. Tutorial 08: `tl.atomic_add` with
  `release` and `gpu`. Helper kernels: `ld`, `st`, `atomic_add` with or without
  a semantic. DeepEP port: `T.st` to a peer, `T.ld` (also with `nc=True`),
  `T.put_warp`; the reference publishes the tail with a release store and reads
  it with an acquire load, both at system scope.
- (b) Tutorial 01: `dl.notify`, `dl.wait`, `__syncthreads`. Tutorials 03 and
  04: `signal_wait_until`, `signal_op`, `__syncthreads`. Tutorial 06: host
  `wait_eq`. Tutorial 07: `dl.wait`, host `set_signal`. Tutorial 08:
  `dl.notify`, the counter of `tl.atomic_add`. DeepEP: `T.wait_ge` and
  `T.wait_ne` on head and tail words, `T.sync_blocks`, `T.barrier_blocks`,
  `T.sync_threads`.
- (c) Tutorials 02, 03 and 07: `putmem_signal_block`. Tutorials 03 and 04:
  `putmem_signal_nbi_block`. Tutorial 04: `putmem_nbi_block`. Tutorial 06:
  `putmem_block`. DeepEP version 1 normal mode: none (its copies are
  `T.put_warp`, pattern a).

Hazards of S2.2 and how they are stated:

- Tutorial 07, host `set_signal` (I6): a `bump` in mode `plain`. It conflicts
  with the `wait` of the kernel on the same word (`conflict_of_plain`; the
  example at the end of the file).
- Tutorial 01, `fence` after `__syncthreads` (D7): the fence is an event of
  the one thread that calls it. The barrier before it is a counter written
  and waited on at scope `cta`, so the only edge from the store of another
  thread to that thread has scope `cta`. Whether a chain through such an edge
  reaches the consumer is the scope check of S5.2.
- D2: the parameter `ShmemReach`. `SyncEdge.shmem_ownRank` is what the weaker
  reading gives.

Not stated here: the step relation, enabledness, happens-before and the data
race (S4.1); the map from each primitive to its events (S2.6); that a counter
word is written only by counter writes (E-NV11), which needs the state; how a
plain access inside a sequential segment (S3.1) meets `Conflict`.
-/
import VeriTile.Dist.SymmHeap

namespace VeriTile.Dist

/-! ## Scope and order of an access (pattern a) -/

/-- UNSIGNED. Owner: S2.3 (#39). The memory order of an atomic operation, as
in the CUDA C++ memory model. -/
inductive MemOrder where
  | relaxed
  | acquire
  | release
  | acqRel
  deriving DecidableEq, Repr

/-- UNSIGNED. Owner: S2.3 (#39). The order has a release side. -/
def MemOrder.releases : MemOrder → Bool
  | .release | .acqRel => true
  | .relaxed | .acquire => false

/-- UNSIGNED. Owner: S2.3 (#39). The order has an acquire side. -/
def MemOrder.acquires : MemOrder → Bool
  | .acquire | .acqRel => true
  | .relaxed | .release => false

/-- UNSIGNED, PROVISIONAL (choices 2 and 3). Owner: S2.3 (#39). How an access
synchronizes. `plain` is the weakest reading (W1, W2): an ordinary, unordered,
non-atomic access. `atomic scope order` is an atomic operation of the CUDA C++
memory model (D5; A1 and A2 where the documentation is silent). `shmem order`
is an NVSHMEM signalling or ordering routine, atomic only with respect to
other such routines (E-NV12). -/
inductive Mode where
  | plain
  | atomic (scope : Scope) (order : MemOrder)
  | shmem (order : MemOrder)
  deriving DecidableEq, Repr

/-- UNSIGNED. Owner: S2.3 (#39). What an access does to its cell. `update` is
a read-modify-write. -/
inductive AccessKind where
  | load
  | store
  | update
  deriving DecidableEq, Repr

/-- UNSIGNED. Owner: S2.3 (#39). The access reads its cell. -/
def AccessKind.reads : AccessKind → Bool
  | .load | .update => true
  | .store => false

/-- UNSIGNED. Owner: S2.3 (#39). The access writes its cell. -/
def AccessKind.writes : AccessKind → Bool
  | .store | .update => true
  | .load => false

/-! ## Counters (pattern b) -/

/-- UNSIGNED. Owner: S2.4 (#40). How a counter is written: `set` replaces its
value, `add` adds to it. -/
inductive SignalOp where
  | set
  | add
  deriving DecidableEq, Repr

/-- UNSIGNED. Owner: S2.4 (#40). The new value of a counter that holds `old`
and is written with `v`. -/
def SignalOp.apply : SignalOp → Nat → Nat → Nat
  | .set, _, v => v
  | .add, old, v => old + v

/-- UNSIGNED. Owner: S2.4 (#40). The memory access that a counter write is. -/
def SignalOp.kind : SignalOp → AccessKind
  | .set => .store
  | .add => .update

/-- UNSIGNED. Owner: S2.4 (#40). The comparison of a wait. -/
inductive Cmp where
  | eq
  | ne
  | ge
  | le
  | gt
  | lt
  deriving DecidableEq, Repr

/-- UNSIGNED. Owner: S2.4 (#40). `cmp.holds v n`: a counter with value `v`
satisfies the wait condition `cmp n`. -/
def Cmp.holds : Cmp → Nat → Nat → Prop
  | .eq, v, n => v = n
  | .ne, v, n => v ≠ n
  | .ge, v, n => n ≤ v
  | .le, v, n => v ≤ n
  | .gt, v, n => n < v
  | .lt, v, n => v < n

/-- UNSIGNED, PROVISIONAL (choice 6). Owner: S2.4 (#40). An object that is
written and waited on. `word p` is the memory word at `p`: a signal, a flag, a
head or a tail; every memory word can serve. `phase c id` is barrier object
`id` of CTA `c` (`__syncthreads`, a named barrier). `rendezvous id` is a
barrier of several ranks (`barrier_all`, `T.barrier_blocks`). `delivered a`
counts the transfers of agent `a` that have completed (`quiet`). -/
inductive Counter where
  | word (p : Addr)
  | phase (c : Cta) (id : Nat)
  | rendezvous (id : Nat)
  | delivered (a : Agent)
  deriving DecidableEq

/-- UNSIGNED. Owner: S2.4 (#40). One write to a counter: which counter, how,
with which value, and how the write synchronizes. -/
structure Bump where
  counter : Counter
  op : SignalOp
  amount : Nat
  mode : Mode
  deriving DecidableEq

/-! ## Transfers (pattern c) and the event type -/

/-- UNSIGNED, PROVISIONAL (choice 7). Owner: S2.5 (#41). An asynchronous copy
of the cells `src` to the cells `dst`. With `nbi = false` the source is read
when the transfer is initiated (a blocking put, E-NV4); with `nbi = true` it
is read when the transfer completes (E-NV5). `completion` is the counter write
that the completion performs; a transfer with `none` never completes (W5). -/
structure Transfer where
  src : List Addr
  dst : List Addr
  nbi : Bool
  completion : Option Bump
  deriving DecidableEq

/-- UNSIGNED, PROVISIONAL (choices 1, 8 and 9). Owner: S2.3 (#39). What one
agent does at one synchronization point.

- `access kind p mode`: a load, store or update of the cell `p`.
- `fence mode`: a fence. `fence .plain` orders nothing (W2).
- `bump b`: a write to a counter.
- `wait c cmp n mode`: blocks until counter `c` satisfies `cmp n`.
- `initiate t`: starts the transfer `t`.
- `complete t`: the transfer `t` completes: the copy and the counter write
  `t.completion`, in one step. No kernel statement is this event. -/
inductive Event where
  | access (kind : AccessKind) (p : Addr) (mode : Mode)
  | fence (mode : Mode)
  | bump (b : Bump)
  | wait (c : Counter) (cmp : Cmp) (n : Nat) (mode : Mode)
  | initiate (t : Transfer)
  | complete (t : Transfer)
  deriving DecidableEq

/-- UNSIGNED, PROVISIONAL. Owner: S2.3 (#39). The events of a run in the order
in which they happen, each with the agent that performs it. The completion of
a transfer is listed under the agent that initiated the transfer. -/
abbrev Trace := List (Agent × Event)

/-- UNSIGNED. Owner: S2.4 (#40). The memory access that a counter write is;
none for a counter that is not a memory word. -/
def Bump.accesses (b : Bump) : List (AccessKind × Addr × Mode) :=
  match b.counter with
  | .word p => [(b.op.kind, p, b.mode)]
  | _ => []

/-- UNSIGNED. Owner: S2.3 (#39). The memory accesses of an event: kind, cell
and mode. A completion stores to every destination cell as a plain access. -/
def Event.accesses : Event → List (AccessKind × Addr × Mode)
  | .access kind p mode => [(kind, p, mode)]
  | .fence _ => []
  | .bump b => b.accesses
  | .wait (.word p) _ _ mode => [(.load, p, mode)]
  | .wait _ _ _ _ => []
  | .initiate t => if t.nbi then [] else t.src.map fun p => (.load, p, .plain)
  | .complete t =>
      t.dst.map (fun p => (.store, p, .plain)) ++
        (if t.nbi then t.src.map fun p => (.load, p, .plain) else []) ++
        (t.completion.map Bump.accesses).getD []

/-- UNSIGNED. Owner: S2.3 (#39). Agent `a` may perform every memory access of
the event (`AccessibleAt` of the location model). -/
def Event.AccessibleBy (symm : VeriTile.Triton.RegionName → Bool) (a : Agent) (e : Event) :
    Prop :=
  ∀ kind p mode, (kind, p, mode) ∈ e.accesses → AccessibleAt symm a p

/-- UNSIGNED. Owner: S2.3 (#39). The side of an event that can start a
synchronization edge: the object it writes and the mode of that write. -/
def Event.releaseOf? : Event → Option (Counter × Mode)
  | .access kind p mode => if kind.writes then some (.word p, mode) else none
  | .bump b => some (b.counter, b.mode)
  | .complete t => t.completion.map fun b => (b.counter, b.mode)
  | _ => none

/-- UNSIGNED. Owner: S2.3 (#39). The side of an event that can end a
synchronization edge: the object it reads and the mode of that read. -/
def Event.acquireOf? : Event → Option (Counter × Mode)
  | .access kind p mode => if kind.reads then some (.word p, mode) else none
  | .bump b => if b.op = .add then some (b.counter, b.mode) else none
  | .wait c _ _ mode => some (c, mode)
  | _ => none

/-! ## Pattern (a): conflicts and synchronization edges (S2.3) -/

/-- UNSIGNED, PROVISIONAL (choice 3). Owner: S2.3 (#39). Two accesses by the
agents `a` and `b` are atomic with respect to each other: both are CUDA
atomics whose scopes each contain both agents (E-CU2), or both are NVSHMEM
routines (E-NV12). -/
def Mode.AtomicFor (a b : Agent) : Mode → Mode → Prop
  | .atomic s _, .atomic s' _ => s.covers a b ∧ s'.covers a b
  | .shmem _, .shmem _ => True
  | _, _ => False

/-- UNSIGNED. Owner: S2.3 (#39). Two events conflict: they access one cell, at
least one of them writes it, and the two accesses are not atomic with respect
to each other. A conflict is a data race when happens-before does not order
the two events (S4.1). -/
def Conflict (x y : Agent × Event) : Prop :=
  ∃ kind kind' p mode mode',
    (kind, p, mode) ∈ x.2.accesses ∧ (kind', p, mode') ∈ y.2.accesses ∧
      (kind.writes = true ∨ kind'.writes = true) ∧ ¬ Mode.AtomicFor x.1 y.1 mode mode'

/-- UNSIGNED, PROVISIONAL (choice 4). Owner: S2.3 (#39). Which cells an edge
of the NVSHMEM family makes visible to the acquiring agent: every cell
(`global`, the reading D2 takes for a system with NVLink alone), or only the
device memory of the acquiring agent's rank (`ownRank`, E-NV15 and E-NV26). -/
inductive ShmemReach where
  | global
  | ownRank
  deriving DecidableEq, Repr

/-- UNSIGNED, PROVISIONAL (choices 3 and 4). Owner: S2.3 (#39). A release
side in mode `m` by agent `a` and an acquire side in mode `m'` by agent `b`
can form an edge that orders accesses to the cell `p`. -/
def Mode.Pairs (reach : ShmemReach) (p : Addr) (a b : Agent) : Mode → Mode → Prop
  | .atomic s o, .atomic s' o' =>
      o.releases = true ∧ o'.acquires = true ∧ s.covers a b ∧ s'.covers a b
  | .shmem o, .shmem o' =>
      o.releases = true ∧ o'.acquires = true ∧
        (reach = .global ∨ p.space = .global b.rankOf)
  | _, _ => False

/-- UNSIGNED, PROVISIONAL (choices 4 and 10). Owner: S2.3 (#39). The
conditions without which the event `x` has no synchronization edge to the
event `y`, as far as the cell `p` is concerned: `x` writes and `y` reads the
same object, and their modes pair. The conditions on the trace (that `y` reads
what `x` wrote, and the fence rules) are added by S4.1; every edge of
happens-before satisfies `SyncEdge`. -/
def SyncEdge (reach : ShmemReach) (p : Addr) (x y : Agent × Event) : Prop :=
  ∃ c m m', x.2.releaseOf? = some (c, m) ∧ y.2.acquireOf? = some (c, m') ∧
    Mode.Pairs reach p x.1 y.1 m m'

/-- UNSIGNED. Proved by S2.3 (#39). An edge between two CUDA atomics exists
only when the scope of each side contains both agents. -/
theorem SyncEdge.covers {reach : ShmemReach} {p : Addr} {x y : Agent × Event}
    {c c' : Counter} {s s' : Scope} {o o' : MemOrder} (h : SyncEdge reach p x y)
    (hx : x.2.releaseOf? = some (c, .atomic s o))
    (hy : y.2.acquireOf? = some (c', .atomic s' o')) :
    s.covers x.1 y.1 ∧ s'.covers x.1 y.1 := sorry

/-- UNSIGNED. Proved by S2.3 (#39). An acquire at `gpu` scope has no edge from
a writer on another rank. -/
theorem not_syncEdge_of_acquire_gpu {reach : ShmemReach} {p : Addr} {x y : Agent × Event}
    {c : Counter} {o : MemOrder} (hy : y.2.acquireOf? = some (c, .atomic .gpu o))
    (hr : x.1.rankOf ≠ y.1.rankOf) : ¬ SyncEdge reach p x y := sorry

/-- UNSIGNED. Proved by S2.3 (#39). A plain write starts no edge and a plain
read ends none (W1, W2). -/
theorem not_syncEdge_of_plain {reach : ShmemReach} {p : Addr} {x y : Agent × Event}
    {c : Counter}
    (h : x.2.releaseOf? = some (c, .plain) ∨ y.2.acquireOf? = some (c, .plain)) :
    ¬ SyncEdge reach p x y := sorry

/-- UNSIGNED. Proved by S2.3 (#39). Under the reach `ownRank`, an edge that
ends in an NVSHMEM routine orders only accesses to the device memory of the
acquiring agent's rank. -/
theorem SyncEdge.shmem_ownRank {p : Addr} {x y : Agent × Event} {c : Counter} {o : MemOrder}
    (h : SyncEdge .ownRank p x y) (hy : y.2.acquireOf? = some (c, .shmem o)) :
    p.space = .global y.1.rankOf := sorry

/-- UNSIGNED. Proved by S2.3 (#39). A plain access conflicts with every access
to the same cell when one of the two writes (inference I1). -/
theorem conflict_of_plain {x y : Agent × Event} {kind kind' : AccessKind} {p : Addr}
    {mode : Mode} (hx : (kind, p, .plain) ∈ x.2.accesses)
    (hy : (kind', p, mode) ∈ y.2.accesses) (hw : kind.writes = true ∨ kind'.writes = true) :
    Conflict x y := sorry

/-- UNSIGNED. Proved by S2.3 (#39). A CUDA atomic and an NVSHMEM routine on
the same cell conflict when one of the two writes (E-NV12). -/
theorem conflict_of_atomic_of_shmem {x y : Agent × Event} {kind kind' : AccessKind}
    {p : Addr} {s : Scope} {o o' : MemOrder} (hx : (kind, p, .atomic s o) ∈ x.2.accesses)
    (hy : (kind', p, .shmem o') ∈ y.2.accesses)
    (hw : kind.writes = true ∨ kind'.writes = true) : Conflict x y := sorry

/-! ## Pattern (b): monotone counter plus wait (S2.4) -/

/-- UNSIGNED. Owner: S2.4 (#40). The counter write that an event performs: its
own for `bump`, the bound one for a completion. -/
def Event.bump? : Event → Option Bump
  | .bump b => some b
  | .complete t => t.completion
  | _ => none

/-- UNSIGNED. Owner: S2.4 (#40). The value of counter `c` after the trace
`tr`, starting from `init c`. -/
def Counter.value (init : Counter → Nat) (tr : Trace) (c : Counter) : Nat :=
  tr.foldl
    (fun v x =>
      match x.2.bump? with
      | some b => if b.counter = c then b.op.apply v b.amount else v
      | none => v)
    (init c)

/-- UNSIGNED, PROVISIONAL (choice 5). Owner: S2.4 (#40). The obligation under
which a counter with `set` writes is monotone: every `set` of `c` in `tr`
writes a value that is at least the value `c` has just before it. -/
def Counter.SetsMonotone (init : Counter → Nat) (tr : Trace) (c : Counter) : Prop :=
  ∀ (pre : Trace) (x : Agent × Event) (post : Trace) (b : Bump),
    tr = pre ++ x :: post → x.2.bump? = some b → b.counter = c → b.op = .set →
      Counter.value init pre c ≤ b.amount

/-- UNSIGNED. Proved by S2.4 (#40). Under the obligation, the value of a
counter never decreases along a trace. -/
theorem Counter.mono {init : Counter → Nat} {tr pre : Trace} {c : Counter}
    (h : Counter.SetsMonotone init tr c) (hpre : pre <+: tr) :
    Counter.value init pre c ≤ Counter.value init tr c := sorry

/-- UNSIGNED. Proved by S2.4 (#40). Under the obligation, a wait for `c ≥ n`
whose condition holds keeps holding. -/
theorem Counter.ge_stable {init : Counter → Nat} {tr pre : Trace} {c : Counter} {n : Nat}
    (h : Counter.SetsMonotone init tr c) (hpre : pre <+: tr)
    (hn : Cmp.ge.holds (Counter.value init pre c) n) :
    Cmp.ge.holds (Counter.value init tr c) n := sorry

/-! ## Pattern (c): asynchronous transfer (S2.5) -/

/-- UNSIGNED, PROVISIONAL. Owner: S2.5 (#41). How often the transfer `t` has
been initiated in `tr` and has not completed yet. -/
def Transfer.pending (t : Transfer) (tr : Trace) : Nat :=
  tr.countP (fun x => x.2 = .initiate t) - tr.countP (fun x => x.2 = .complete t)

/-- UNSIGNED. Owner: S2.5 (#41). After `tr` the cell `p` is the destination of
a transfer that has not completed: it holds unspecified data, and the
execution semantics rejects an access to it. -/
def InFlight (tr : Trace) (p : Addr) : Prop :=
  ∃ t : Transfer, 0 < t.pending tr ∧ p ∈ t.dst

/-- UNSIGNED. Proved by S2.5 (#41). The destination of a transfer that was
initiated and has not completed is in flight. This covers a transfer that
never completes (W5) and a non-blocking put without `quiet`. -/
theorem inFlight_of_initiate {tr : Trace} {a : Agent} {t : Transfer} {p : Addr}
    (hi : (a, Event.initiate t) ∈ tr) (hc : ∀ x ∈ tr, x.2 ≠ Event.complete t)
    (hp : p ∈ t.dst) : InFlight tr p := sorry

/-- UNSIGNED. Proved by S2.5 (#41). The completion of a transfer conflicts
with every access to a destination cell, whatever the mode of that access. -/
theorem conflict_complete {a : Agent} {t : Transfer} {y : Agent × Event} {kind : AccessKind}
    {p : Addr} {mode : Mode} (hp : p ∈ t.dst) (hy : (kind, p, mode) ∈ y.2.accesses) :
    Conflict (a, Event.complete t) y := sorry

/-- UNSIGNED. Proved by S2.5 (#41). The only edge out of a completion goes
through the counter that the transfer is bound to. -/
theorem SyncEdge.of_complete {reach : ShmemReach} {p : Addr} {a : Agent} {t : Transfer}
    {y : Agent × Event} (h : SyncEdge reach p (a, Event.complete t) y) :
    ∃ b m, t.completion = some b ∧ y.2.acquireOf? = some (b.counter, m) := sorry

/-- Tutorial 07 as written (inference I6): the plain store of the host's
`set_signal` conflicts with the `dl.wait` of the kernel on the same word. -/
example (p : Addr) (host waiter : Agent) :
    Conflict (host, .bump ⟨.word p, .set, 1, .plain⟩)
      (waiter, .wait (.word p) .ge 1 (.atomic .gpu .acquire)) :=
  ⟨.store, .load, p, .plain, .atomic .gpu .acquire,
    by simp [Event.accesses, Bump.accesses, SignalOp.kind], by simp [Event.accesses],
    Or.inl rfl, by simp [Mode.AtomicFor]⟩

end VeriTile.Dist
