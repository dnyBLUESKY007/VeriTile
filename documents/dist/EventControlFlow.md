# Control flow and agents around synchronization events

Status: delivered for ticket S3.5 (#406) on 2026-11-12, not yet reviewed by
the user. Nothing here is a proved result, and nothing was executed: every
source was read as text at the commits of section 2.

The page lists, for every kernel of the two proof-target sets, each
synchronization or communication event, the control flow that encloses it and
the sequential agents that execute it. It is the evidence for two open choices
of `VeriTile/Staging/Dist/SegmentSpec.lean`: who owns a segment (choice 5) and
what slicing returns (choices 6 and 7). It also says which `Event`
constructors of `VeriTile/Staging/Dist/EventSpec.lean` each kernel needs. It
states facts and no recommendation.

Each claim carries one of three marks, as in `DeepEPInventory.md`:

- **R** (read): stated by the source at the file and line given. A statement
  with a citation and no mark is R.
- **I** (inferred): a conclusion drawn from lines that were read.
- **U** (unchecked): not established. Section 9 lists every such item.

Vocabulary: `PrimitiveSources.md` for the primitives, their patterns (a, b, c)
and the rules W1 to W6, A1, A2, D4 to D7; `VeriTile/Dist/Agent.lean` for the
agents (rank, CTA, group, thread).

## 1. Summary

The two sets hold 28 rows: 21 for Triton-distributed tutorials 01 to 08 (14
functions in the tutorial files, 6 kernels and 1 helper under
`python/triton_dist/` that the tutorials reach) and 7 for the DeepEP version 1
port. Every row was read; none is "not read". K1 to K28 are the row numbers
of section 4.

### 1.1 Which kernels a flat list covers

"Flat list" means: after resolving the branches on rank, program id and
compile-time constants and unrolling the counted loops, the body of one CTA
is a list of statements and events whose length is fixed by the launch
arguments (scalars, grid, tensor shapes).

| Group | Rows | Kernels |
| --- | --- | --- |
| No event; the body is one segment | 9 | K5 `ceil_div`, K7 `calc_gather_index._kernel`, K8 `_quant_kernel`, K9 `_dequant_kernel`, K10 `kernel_ring_reduce`, K13 `kernel_ring_reduce_tma`, K14 `kernel_ring_reduce_non_tma`, K21 `add_continuous_kernel`, K22 `get_dispatch_layout_main` |
| Events; a flat list per CTA exists | 8 | K1 `producer_consumer_kernel`, K2 `nvshmem_device_producer_all_gather_2d_put_block_kernel`, K6 `all_to_all_kernel`, K12 and K20 `kernel_inter_node_p2p_for_same_local_rank` (two copies), K17 `kernel_consumer_gemm_persistent`, K18 `nvshmem_device_producer_p2p_put_block_kernel`, K24 `cached_notify_dispatch_main` |
| Events; a flat list exists per thread, and per CTA only after an order is fixed among events that the source gives to different threads | 4 | K3 `all_gather_push_1d_kernel`, K4 `all_gather_push_2d_kernel`, K23 `notify_dispatch_main`, K27 `cached_notify_combine_main` |
| No flat list fixed before execution (section 1.2) | 7 | K11, K15, K16, K19, K25, K26, K28 |

So 17 rows are covered with the CTA as the only agent, 4 more are covered if
either the owner is a thread or an order among thread events is chosen, and 7
are not covered.

Three qualifications:

- **K22** has no event, but its threads write rows of a CTA-shared array and
  then read all rows with no barrier call in the source (`DL:88-108`,
  `DL:119-152`). One segment for the CTA presupposes an order among the
  threads that the source does not state (U, as in `DeepEPInventory.md`,
  section 4.2).
- **In the group of 8**, the length of the list is fixed by launch arguments
  that are run-time scalars, not compile-time constants, in K1 (`num_inputs`,
  `T01:69`, `T01:85`), K17 (`M`, `N`, `K`, `T07:82-84`, `T07:148`) and K18
  (`world_size`, `local_world_size`, `T07:277-287`). K6 has events whose
  sizes and destination addresses are loaded from memory (`T04:148-150`,
  `T04:178`, `T04:233-258`): the list of events is fixed, the evaluated
  arguments of the events are not (section 1.4).
- **In the group of 4**, the events that different threads own are: waits on
  distinct words in K3 and K4 (a wait writes no memory, I); plain stores to
  distinct peers in K23 (thread `tx` stores on rank `tx`, `DD:65-68`); plain
  loads of local memory in K27.

### 1.2 Which kernels need a per-agent continuation, and why

| Why | Rows | Kernels | Where |
| --- | --- | --- | --- |
| A branch on a value that an atomic update returns, on a word the other CTAs of the kernel write. The event under it is fired or not; no loop | 1 | K19 `kernel_gemm_rs_producer_persistent` | `T08:274-284` |
| A spin loop with an empty body: the number of rounds depends on the schedule, and a failing round does nothing but read the word again | 3 | K11 `barrier_all_intra_node_atomic_cas_block`, K15 `barrier_all_intra_node_non_atomic_block`, K16 its helper | `CO:162-167`, `CO:178-179` |
| A loop with events and copies in its body whose number of rounds depends on input data (senders) and on the schedule (receivers) | 3 | K25 `dispatch_main`, K26 `cached_dispatch_main`, K28 `combine_main` | `DD:332-418`, `DD:451-514`, `DD:601-660`, `DD:693-743`, `DC:171-224`, `DC:242-265`, `DC:276-345` |

Facts behind the three lines:

- **K19.** The number of `tl.atomic_add` events of a CTA is fixed by the
  launch arguments (`T08:209`, `T08:266`). After each one the CTA fires
  `dl.notify` exactly when the returned value equals the tile count of the
  rank minus one (`T08:280-284`). Which CTA sees that value depends on the
  schedule (I). The list of a CTA is therefore fixed up to one optional event
  after each atomic add.
- **K11, K15, K16.** Each of the first `local_world_size` threads runs
  `while atomic_cas(...) != 0: pass` and `while atomic_cas(...) != 1: pass`
  (`CO:162-167`), or `st(...)` followed by `while ld(...) != target_value:
  pass` (`CO:177-179`). Each failing round is one more atomic access. Read as
  one blocking event (wait until the word holds the expected value, and for
  `atomic_cas` swap it in the same step), each thread has a fixed list of
  three events (I). The same spin form occurs inside K25, K26 and K28
  (`DD:452-458`, `DD:694-700`, `DC:284-287`).
- **K25, K26, K28, senders.** The number of rounds of the sender loop is a
  function of input data alone: the column of `is_token_in_rank` for the
  destination rank (`DD:343-354`, `DD:612-623`), or entries of the two prefix
  matrices (`DC:152-173`). No value loaded from memory that a peer writes
  enters the sender's control flow: `T.wait_ge` returns no value
  (`DD:334-338`, `DC:175-179`) (I). A flat list exists for each content of
  that input; it is a function of the memory at kernel start, not of the
  launch arguments.
- **K25, K26, receivers.** The number of tokens to receive is loaded from
  words that the peer's sender writes (`DD:431-436`), and each round takes
  `tail - head` tokens, where `tail` is what one thread loaded at that moment
  (`DD:453-466`). The number of rounds depends on how far the peer had
  published when the load happened (I).
- **K28, receiver.** The lanes of warp 0 loop until every other warp has set
  its `warp_retired` flag (`DC:242-249`); the number of rounds is not bounded
  by any input (I). Each other warp runs a counted loop over tokens
  (`DC:276`) with a spin loop per token (`DC:284-287`) and inner loops whose
  bound comes from `send_head` (`DC:296-301`, `DC:310`).

### 1.3 Which kernels have several sequential agents inside one CTA

11 rows. "Agent" is the unit to which the source gives its own events and its
own local variables.

| Rows | Agents inside one CTA | What they share | Where |
| --- | --- | --- | --- |
| K3, K4 | In the program that waits: one thread per peer rank, each with one wait on its own word | Nothing written. All threads join at `__syncthreads` | `T03:65-78`, `T03:149-155` |
| K11, K15, K16 | The first `local_world_size` threads of the single CTA, one per rank of the node | Thread `t` writes word `local_rank` of the flag array on rank `t` and reads or swaps word `t` of its own rank: distinct threads, distinct words (I). All threads join at `__syncthreads` | `CO:158-168`, `CO:173-180` |
| K22 | 256 threads | The CTA-shared arrays `tokens_per_expert_per_thread` and `tokens_per_rank_per_thread`: thread `tx` writes row `tx`, then a few threads read every row. No barrier call | `DL:84-108`, `DL:119-152` |
| K23 | Block 0: threads `tx < num_ranks`, each storing to rank `tx`. Other blocks: warps, each counting its channels | Block 0: the local `per_rank_buffer` and `per_expert_buffer`, one column per thread (I), between `T.barrier_blocks` and `T.sync_threads()`. Other blocks: `channel_prefix_matrix`, one cell per warp and channel, then a prefix sum by thread 0 after `T.sync_threads()` | `DD:59-99`, `DD:100-118` |
| K25, K26 | `num_ranks` groups of `768 // num_ranks` threads, one group per peer rank (`responsible_rank`); inside a group, warps (own wait, own copies) and elected lanes (own stores) | Sender groups: the inputs, read only (I), and `send_head`, one column per group. Receiver groups: the CTA-shared `shared_channel_tail_idx`, one cell per group; the outputs, at rows that start at `total_offset` (disjoint between groups: I, not checked). Groups meet only at their own barrier `T.sync_threads(responsible_rank, num_threads_per_rank)` | `DD:300-304`, `DD:410`, `DD:445-462`, `DD:509`, `DD:571-574`, `DD:652`, `DD:687-704`, `DD:738` |
| K27 | Blocks other than block 0: one warp per rank (`rank_id = tx // 32`) | `send_head`, one column per warp. No barrier | `DC:40-69` |
| K28 | Sender: `num_ranks` groups of warps, group of warp `w` is `(responsible_channel + w) % num_ranks`. Receiver: the lanes `lane_id < num_ranks` of warp 0, one per source rank, and each of the other 23 warps | Sender groups: inputs, read only (I); they meet at `T.sync_threads(send_rank_id, threads_per_rank)`. Receiver: three CTA-shared arrays: `shared_channel_tail_idx` (stored by warp 0 with `release` at scope `cta`, loaded by the other warps with `acquire` at scope `cta`), `warp_channel_head_idx` and `warp_retired` (written by the other warps and read by warp 0 with ordinary accesses). One CTA barrier at the start; none after it | `DC:147-149`, `DC:216`, `DC:228-237`, `DC:239-265`, `DC:266-350` |

Two kernels have a single distinguished thread and no second agent: K6
(thread 0 issues the signal and the wait, `T04:262-276`) and one branch of K4
(`T03:161-166`). K1 has no thread index in its text; a comment says that
`notify` "is also single thread scope" (`T01:108`).

Facts that bear on the owner of a segment:

- Each agent above holds local variables of its own. In K25 each thread has
  `cached_channel_tail_idx`, `token_idx` and `cached_channel_head_idx`
  (`DD:328-331`, `DD:447-450`), and the groups of one CTA hold different
  values, because each group follows the queue of a different peer (I). In
  K11 each thread has its own `remote_ptr` (`CO:161`).
- In K25, K26 and K28 the groups of one CTA run loops with different numbers
  of rounds, and the only barrier inside the loop is the barrier of the group
  (`DD:410`, `DD:461`, `DD:509`, `DC:216`). The groups of one CTA are
  therefore not at the same program point (I).
- That the two-argument `T.sync_threads(id, count)` is a barrier of the
  `count` threads that pass the same `id` is an inference from the arguments
  and from the comment "here all warps should share the same new tail"
  (`DD:409`). The docstring at the pin says "Synchronize all threads in a
  block" for every form (`TB:702-703`); section 8, item 1.

### 1.4 Event arguments that depend on memory content

Outside the three questions, but it bears on choice 6 of the segment draft
(the body carries "the event fired there") together with choice 9 of the
event draft (an event carries evaluated addresses).

| Rows | Argument | Depends on | Where |
| --- | --- | --- | --- |
| K6 | Size of each `putmem`; in combine mode the destination address | `send_splits_cumsum` and `recv_offset`, loaded in the segment before the event | `T04:148-150`, `T04:178-182`, `T04:233-258` |
| K25, K26 | Expected value of `T.wait_ge`; destination slot of each copy; published tail | `is_token_in_rank` | `DD:334-338`, `DD:358-365`, `DD:412-418` |
| K25, K26 | Source slots and destination rows of the receiver's copies; published head | Tail and offsets loaded at run time from words a peer writes | `DD:431-442`, `DD:466-511` |
| K28 | Expected value of `T.wait_ge`; source ranks and slots of the receiver's loads | Prefix matrices; `send_head` | `DC:152-179`, `DC:296-316` |

In K1 to K4, K11, K12, K15 to K20, K23, K24 and K27 every address and every
counter value of an event is computed from launch arguments, loop counters
and thread or program ids (I, from the lines of section 5).

### 1.5 `Event` constructors the kernels need

Read off the pattern column of `PrimitiveSources.md` and the constructor
docstrings of the event draft (I; ticket S2.6 fixes the map from primitives to
events).

| Constructor or parameter | Used by | Not used in the two sets |
| --- | --- | --- |
| `access` | K11, K15, K16 (`atomic_cas`, `ld`, `st`), K19 (`tl.atomic_add`), K23, K25 to K28 (`T.st`, `T.ld`, `T.put_warp`) | |
| `fence` | K1, K6 (`libshmem_device.fence`) | |
| `bump` | K1, K19 (`dl.notify`), K6 (`signal_op`); the barrier calls of K1, K3, K4, K11, K15, K16 and K23 to K28 | |
| `wait` | K1, K17 (`dl.wait`), K3, K4, K6 (`signal_wait_until`), K25, K26, K28 (`T.wait_ge`, `T.wait_ne`) | |
| `initiate`, `complete` | K2, K3, K4, K18 (`putmem_signal_block`), K4, K6 (`putmem_signal_nbi_block`), K6 (`putmem_nbi_block`), K12, K20 (`putmem_block`) | |
| `Cmp` | `eq` (`NVSHMEM_CMP_EQ`, `T03:74`, `T04:274`), `ge` (`T.wait_ge`, `DD:334`), `ne` (`T.wait_ne`, `DD:431`); `dl.wait` under D6 | `le`, `gt`, `lt` |
| `Counter` | `word`; `phase` (`__syncthreads`, `T.sync_threads`); `rendezvous` (`T.sync_blocks`, `T.barrier_blocks`) | `delivered`: no kernel of the two sets calls `quiet` |
| `Mode` | `plain`; `atomic` at scope `sys` and `gpu`; `shmem` | `atomic` at scope `cta`: the string occurs only in K28 (`T.st` and `T.ld` with `scope="cta"`, `DC:255`, `DC:284`, `DC:286`), where A1 assumes nothing for it, so these accesses are `plain` (W1) |

`dl.wait` of K17 waits on `rank_end - rank_beg + 1` words in one call
(`T07:178-185`); K1 passes 1 (`T01:91`, `T01:129`).

## 2. Sources

| ID | Repository and commit | Path |
| --- | --- | --- |
| T01 | Triton-distributed `7908e4ea9010bb2238f9f05b28c904fd4b71c60a` | `tutorials/01-distributed-notify-wait.py` |
| T02 | same | `tutorials/02-intra-node-allgather.py` |
| T03 | same | `tutorials/03-inter-node-allgather.py` |
| T04 | same | `tutorials/04-deepseek-infer-all2all.py` |
| T05 | same | `tutorials/05-intra-node-reduce-scatter.py` |
| T06 | same | `tutorials/06-inter-node-reduce-scatter.py` |
| T07 | same | `tutorials/07-overlapping-allgather-gemm.py` |
| T08 | same | `tutorials/08-overlapping-gemm-reduce-scatter.py` |
| CO | same | `python/triton_dist/kernels/nvidia/common_ops.py` |
| KC | same | `python/triton_dist/kernels/common_ops.py` |
| RS | same | `python/triton_dist/kernels/nvidia/reduce_scatter.py` |
| AG | same | `python/triton_dist/kernels/nvidia/allgather_gemm.py` |
| UT | same | `python/triton_dist/utils.py` |
| SI | same | `python/triton_dist/language/simt_ops.py` |
| DL | TileScale `9ba8a605d347f600a95d0275e2639cb00122c084` (branch `wt/deepep-internode`) | `examples/distributed/deepseek_deepep/intranode/get_dispatch_layout.py` |
| DD | same | `examples/distributed/deepseek_deepep/intranode/dispatch.py` |
| DC | same | `examples/distributed/deepseek_deepep/intranode/combine.py` |
| DB | same | `examples/distributed/deepseek_deepep/buffer.py` |
| DU | same | `examples/distributed/deepseek_deepep/deepep_utils.py` |
| TB | same | `tilelang/language/builtin.py` |
| TC | same | `tilelang/language/distributed/common.py` |

A citation is `ID:line` or `ID:first-last`. T01 to T08, CO, KC, RS, DL, DD
and DC were read whole. Of AG lines 25 to 173 and 445 to 553 were read; of
UT, SI, DB, DU, TB and TC only the lines cited. Tutorials 03a, 09 and 10 are
out of scope by the map and were not opened. The files were read on
2026-11-12 from checkouts whose `HEAD` is the commit above.

**How the Triton-distributed rows were chosen.** Every function decorated
with `triton.jit` or `triton_dist.jit` in T01 to T08 has a row (14, the same
14 as kernels 106 to 119 of `documents/census/MinimalSyntax.md`). The
functions under `python/triton_dist/` were followed by reading from each call
in a tutorial:

| Tutorial | Calls into `python/triton_dist/kernels/` | Kernels reached |
| --- | --- | --- |
| 05 | `barrier_all_intra_node_atomic_cas_block[(1, )]` (`T05:216`) | K11 |
| 06 | `barrier_all_on_stream` (`T06:344`) with an intra-node context (`T06:223`), which launches K11 if `supports_p2p_native_atomic()` and K15 otherwise (`CO:254-262`); `ring_reduce` (`T06:349`, `T06:406`), which launches K13 if `has_tma()` and K14 otherwise (`RS:787-790`); host `wait_eq` (`T06:282`) | K11, K15 (with K16), K13, K14 |
| 07 | `create_ag_gemm_context` (`T07:495`), host code that launches no kernel (`AG:481-503`, `AG:511-551`; its callee `get_auto_all_gather_method`, `python/triton_dist/kernels/nvidia/allgather.py:57-72`, returns an enum value); host `set_signal`, `wait_eq` | none |
| 08 | `create_reduce_scater_2d_ctx` (`T08:110`), host code (`RS:148-178`); `reduce_scatter_2d_op` (`T08:423`), which reaches `reduce_scatter_for_each_node` (K13 or K14, K20) or, if `has_fullmesh_nvlink()` is false, `reduce_scatter_for_each_node_ring` (K21) (`RS:807-810`, `RS:583-635`, `RS:493-548`, `RS:243-281`, `RS:206-224`) | K13, K14, K20, K21 |

Tutorials 01 to 04 call no kernel outside their own file. Section 7 lists the
kernels of CO and RS that no tutorial of the set reaches.

**The DeepEP rows** are the seven `T.prim_func` functions of DL, DD and DC.
`internode/get_dispatch_layout.py` lies outside the directory of the ticket
and has no row.

## 3. Classes of enclosing control flow

The ticket names five classes. Two more were needed; they are marked "added".

| Class | Meaning |
| --- | --- |
| none | The event is in the straight-line body |
| B-id | A branch on rank, number of ranks, program id or a compile-time constant (`tl.constexpr`, a parameter of the `tilelang.jit` factory) |
| B-thread (added) | A branch on the thread index: `tid`, `tx`, a lane or warp index, `T.shuffle_elect`. It decides which agent inside the CTA fires the event |
| L-count | A counted loop whose bound is computed from launch arguments (scalars, grid size, tensor shapes) or compile-time constants |
| L-data | A loop whose number of rounds depends on the content of memory that no agent writes during the kernel (input data) |
| L-sched | A loop whose number of rounds depends on a value read at run time from memory that another agent writes during the kernel |
| B-sched (added) | A branch on such a value. The event under it is fired at most once per evaluation |

A branch on a loop counter inside an L-count loop (`if ki == 0`, `T07:150`)
is resolved by unrolling and is not listed.

## 4. Kernel table

Columns: **Events** gives the classes of section 3 that enclose at least one
event, with the lines of the enclosing statements; section 5 has every event.
**Agents** says who fires events: `CTA`, or agents below it. **Flat list**
answers question 3 of the ticket.

### 4.1 Triton-distributed tutorials 01 to 08

| Row | Kernel (definition) | Launch | Events | Agents | Flat list fixed before execution |
| --- | --- | --- | --- | --- | --- |
| K1 | `producer_consumer_kernel` (`T01:64`) | grid 20, 4 warps (`T01:191-204`) | B-id (`T01:79`, `T01:117`); L-count (`T01:85`, `T01:123`), bound `num_inputs` | CTA; no thread index in the text | Yes. Per round: wait, store to the peer, barrier, fence, notify (producer) or wait, load, barrier, fence, notify (consumer) |
| K2 | `nvshmem_device_producer_all_gather_2d_put_block_kernel` (`T02:108`) | grid `num_ranks` (`T02:196-205`) | B-id (`T02:120`) | CTA | Yes: one transfer |
| K3 | `all_gather_push_1d_kernel` (`T03:62`) | grid `num_ranks` (`T03:100`) | B-id (`T03:67`); B-thread (`T03:69`) | Program `pid == rank`: one thread per peer rank (`T03:68-76`). Other programs: CTA | Per thread: yes. Per CTA: the waiting program has `WORLD_SIZE - 1` waits owned by different threads, then one barrier; the others have one transfer |
| K4 | `all_gather_push_2d_kernel` (`T03:112`) | grid `num_ranks`, 32 warps (`T03:182-191`) | B-id (`T03:134`, `T03:135`, `T03:160`); B-thread (`T03:149`, `T03:161`) | As K3 in the waiting program (`T03:149-154`); thread 0 alone at `T03:161-166`; otherwise CTA | Per thread: yes. Per CTA: as K3 for the waiting program; the others have at most one wait, one barrier and one transfer |
| K5 | `ceil_div` (`T04:78`) | called at `T04:198`, `T04:223` | no event | — | Yes: no event |
| K6 | `all_to_all_kernel` (`T04:94`) | grid `WORLD_SIZE`, 16 warps (`T04:89-92`, `T04:537`) | B-id on `MODE` and `ONLINE_QUANT_FP8` (`T04:240`, `T04:249`, `T04:264`); B-thread (`T04:262`). The L-data loops (`T04:198`, `T04:223`) hold no event | CTA; thread 0 alone fires the signal and the wait | Yes: one to three transfers, a fence, at most one signal, one wait. Event sizes and one address depend on loaded data (section 1.4) |
| K7 | `calc_gather_index._kernel` (`T04:296`) | `T04:321-331` | no event | CTA | Yes: no event |
| K8 | `_quant_kernel` (`T04:336`) | `T04:413-419` | no event | CTA | Yes: no event |
| K9 | `_dequant_kernel` (`T04:375`) | `T04:425-427` | no event | CTA | Yes: no event |
| K10 | `kernel_ring_reduce` (`T05:64`) | `T05:137-166` | no event. It loads symmetric memory that peers filled by host copies (`T05:200`, `T05:219`) | CTA | Yes: no event |
| K11 | `barrier_all_intra_node_atomic_cas_block` (`CO:154`) | grid 1 (`T05:216`, `CO:258`) | B-thread (`CO:160`, `CO:165`); L-sched, empty body (`CO:162-163`, `CO:166-167`) | One thread per rank of the node, inside `dl.simt_exec_region` (`CO:158`) | No: the number of `atomic_cas` events of a thread depends on the schedule. With each spin loop read as one blocking event: three events per thread (I) |
| K12 | `kernel_inter_node_p2p_for_same_local_rank` (`T06:234`) | grid 1, 16 warps (`T06:211`, `T06:365-374`) | none | CTA | Yes: one transfer |
| K13 | `kernel_ring_reduce_tma` (`RS:666`) | `RS:749-775` | no event | CTA | Yes: no event |
| K14 | `kernel_ring_reduce_non_tma` (`RS:639`) | `RS:721-729` | no event | CTA | Yes: no event |
| K15 | `barrier_all_intra_node_non_atomic_block` (`CO:184`) | grid 1 (`CO:260`) | Calls K16 twice (`CO:194`, `CO:196`) | As K16 | No, as K16; six events per thread under the same reading (I) |
| K16 | `_barrier_all_intra_node_non_atomic_once_block` (`CO:172`), helper of K15 | — | B-thread (`CO:174`); L-sched, empty body (`CO:178-179`) | One thread per rank of the node (`CO:173-174`) | No: the number of `ld` events of a thread depends on the schedule. With the spin loop read as one blocking event: store, wait, barrier (I) |
| K17 | `kernel_consumer_gemm_persistent` (`T07:78`) | grid `min(num_gemm_sms, tiles)`, 8 warps (`T07:410-435`) | L-count (`T07:148`), bound from `M`, `N`, `K`, block sizes, `NUM_SMS`, program id (`T07:100-103`, `T07:129-131`) | CTA | Yes: `tiles_per_SM` times one wait and one segment |
| K18 | `nvshmem_device_producer_p2p_put_block_kernel` (`T07:270`) | grid `n_nodes - 1`, 32 warps (`T07:233-244`) | L-count (`T07:287`) | CTA | Yes. With that grid each CTA runs one round (I) |
| K19 | `kernel_gemm_rs_producer_persistent` (`T08:134`) | grid `min(num_gemm_sms, tiles)`, 8 warps (`T08:320-346`) | L-count (`T08:209`, `T08:266`); B-sched (`T08:280`) | CTA | No for the `dl.notify`: fired or not by the value an atomic add returned. Yes for the atomic adds |
| K20 | `kernel_inter_node_p2p_for_same_local_rank` (`RS:472`) | grid 1, 16 warps (`RS:172`, `RS:621-630`) | none | CTA | Yes: one transfer |
| K21 | `add_continuous_kernel` (`RS:182`) | grid 16 (`RS:210-223`) | no event | CTA | Yes: no event |

### 4.2 DeepEP version 1 normal mode, TileScale port

Six of the seven kernels read the thread index with `T.get_thread_binding()`
(`DL:85`, `DD:56`, `DD:301`, `DD:572`, `DC:33`, `DC:142`); K24 does not.

| Row | Kernel (definition) | Launch | Events | Agents | Flat list fixed before execution |
| --- | --- | --- | --- | --- | --- |
| K22 | `get_dispatch_layout_main` (`DL:78`) | `ceil(E/4) + ceil(R/8)` blocks, 256 threads (`DL:69-72`, `DL:84`) | no event. Branches on block and thread (`DL:95`, `DL:103`, `DL:118`, `DL:147`); L-count loops (`DL:96`, `DL:106`, `DL:127`, `DL:150`) | Threads that share CTA memory with no barrier call (`DL:88-108`, `DL:119-152`) | Yes per thread: no event. One segment per CTA presupposes an order among threads (U) |
| K23 | `notify_dispatch_main` (`DD:37`) | `num_ranks + 1` blocks, 128 threads (`DD:30`, `DD:55`) | B-id (`DD:59`); B-thread (`DD:65`); L-count (`DD:67`), compile-time bound | Block 0: threads `tx < num_ranks` own the stores to peers; the barrier calls are not under a thread branch. Other blocks: warps | Per thread: yes. Per CTA: block 0 has three rendezvous and, between the first two, `num_ranks * (1 + num_local_experts)` stores owned by `num_ranks` threads; the other blocks have one barrier |
| K24 | `cached_notify_dispatch_main` (`DD:192`) | 1 block, 128 threads (`DD:200`) | none | CTA | Yes: rendezvous, four clears, rendezvous |
| K25 | `dispatch_main` (`DD:270`) | `num_sms` blocks, 768 threads (`DD:255`, `DD:300`) | B-id (`DD:306`); B-thread (many); sender: L-data (`DD:332`, `DD:343`); receiver: L-sched (`DD:451`, `DD:452`) and loops bounded by loaded values (`DD:467`, `DD:481`, `DD:493`) | Groups per peer rank; warps and elected lanes inside a group (section 1.3) | No. Sender group: fixed once `is_token_in_rank` is fixed. Receiver group: depends on the schedule |
| K26 | `cached_dispatch_main` (`DD:547`) | as K25 (`DD:534`, `DD:571`) | as K25: `DD:576`; `DD:601`, `DD:612`; `DD:693`, `DD:694`, `DD:709`, `DD:722` | as K25 | No, as K25 |
| K27 | `cached_notify_combine_main` (`DC:25`) | `num_sms // 2 + 1` blocks, `max(128, 32 * num_ranks)` threads (`DC:19-20`, `DC:32`) | B-id (`DC:35`); in the other blocks L-count (`DC:52`) and a branch on the thread and the counter (`DC:55`) | Block 0: CTA. Other blocks: one warp per rank; a warp whose index is `num_ranks` or more returns (`DC:42-45`) | Per thread: yes. Block 0: rendezvous, two clears, rendezvous. Other blocks: one plain load per lane and round |
| K28 | `combine_main` (`DC:120`) | `num_sms` blocks, 768 threads (`DC:109`, `DC:141`) | B-id (`DC:147`); B-thread (many); sender: L-data (`DC:171`), L-count inside a round (`DC:183`); receiver: L-sched (`DC:242`, `DC:285`), L-count over tokens (`DC:276`), L-data (`DC:310`, `DC:331`) | Sender: groups of warps per destination rank. Receiver: lanes of warp 0 and 23 reduction warps (section 1.3) | No. Sender group: fixed once the prefix matrices are fixed. Receiver: depends on the schedule |

## 5. Events of each kernel

Columns: **Pattern** is the pattern of the primitive's row in
`PrimitiveSources.md`; "plain access" is an ordinary load or store, with no
primitive, of memory on another rank or of memory that a peer writes; "no
row" is a base TileLang call that the table does not list. **By** is the
agent that executes the line. **Under** lists the enclosing control flow
from the outside in.

Kernels without an event (K5, K7 to K10, K13, K14, K21, K22) have no table.

### K1 `producer_consumer_kernel`

| Lines | Call | Pattern | By | Under |
| --- | --- | --- | --- | --- |
| `T01:88-96` | `dl.wait` on a word of the peer rank (`dl.symm_at`), `"gpu"`, `"acquire"` | b | CTA | B-id `T01:79`; L-count `T01:85` |
| `T01:102-104` | `tl.store` through `dl.symm_at(queue_ptr, peer_rank)` | plain access | CTA | same |
| `T01:106` | `__syncthreads` | b | CTA | same |
| `T01:107` | `libshmem_device.fence` | a | CTA | same |
| `T01:109-116` | `dl.notify` to the peer, `sig_op="set"`, `comm_scope="intra_node"` | b | CTA | same |
| `T01:126-133` | `dl.wait` on a local word, `"gpu"`, `"acquire"` | b | CTA | B-id `T01:117`; L-count `T01:123` |
| `T01:135` | `tl.load` of the local queue, which the peer's producer wrote | plain access | CTA | same |
| `T01:137` | `__syncthreads` | b | CTA | same |
| `T01:138` | `libshmem_device.fence` | a | CTA | same |
| `T01:139-146` | `dl.notify` to the own rank, `sig_op="set"`, `comm_scope="intra_node"` | b | CTA | same |

### K2 `nvshmem_device_producer_all_gather_2d_put_block_kernel`

| Lines | Call | Pattern | By | Under |
| --- | --- | --- | --- | --- |
| `T02:123-131` | `libshmem_device.putmem_signal_block`, `NVSHMEM_SIGNAL_SET` | c | CTA | B-id `T02:120` |

### K3 `all_gather_push_1d_kernel`

| Lines | Call | Pattern | By | Under |
| --- | --- | --- | --- | --- |
| `T03:72-76` | `libshmem_device.signal_wait_until`, `NVSHMEM_CMP_EQ`, word `symm_flag + peer` with `peer = thread_idx` | b | thread | B-id `T03:67`; B-thread `T03:69` |
| `T03:78` | `__syncthreads` | b | CTA | B-id `T03:67` |
| `T03:85-95` | `libshmem_device.putmem_signal_block`, `NVSHMEM_SIGNAL_SET` | c | CTA | B-id `T03:79` |

### K4 `all_gather_push_2d_kernel`

| Lines | Call | Pattern | By | Under |
| --- | --- | --- | --- | --- |
| `T03:139-147` | `libshmem_device.putmem_signal_nbi_block` | c | CTA | B-id `T03:134`, `T03:135` |
| `T03:150-154` | `libshmem_device.signal_wait_until`, `NVSHMEM_CMP_EQ`, word `symm_flag + thread_idx` | b | thread | B-id `T03:134`, `T03:148`; B-thread `T03:149` |
| `T03:155` | `__syncthreads` | b | CTA | B-id `T03:134`, `T03:148` |
| `T03:162-166` | `libshmem_device.signal_wait_until`, `NVSHMEM_CMP_EQ` | b | thread 0 | B-id `T03:156`, `T03:160`; B-thread `T03:161` |
| `T03:167` | `__syncthreads` | b | CTA | B-id `T03:156`, `T03:160` |
| `T03:169-177` | `libshmem_device.putmem_signal_block` | c | CTA | B-id `T03:156` |

### K6 `all_to_all_kernel`

| Lines | Call | Pattern | By | Under |
| --- | --- | --- | --- | --- |
| `T04:233-239` | `libshmem_device.putmem_nbi_block` (data) | c | CTA | none |
| `T04:242-247` | `libshmem_device.putmem_nbi_block` (splits) | c | CTA | B-id `T04:240` (`MODE == 0`) |
| `T04:250-258` | `libshmem_device.putmem_signal_nbi_block` (scales), `NVSHMEM_SIGNAL_SET` | c | CTA | B-id `T04:249` (`ONLINE_QUANT_FP8`) |
| `T04:261` | `libshmem_device.fence` | a | CTA (the call is outside the thread branch) | none |
| `T04:265-270` | `libshmem_device.signal_op`, `NVSHMEM_SIGNAL_SET` | b | thread 0 | B-thread `T04:262`; B-id `T04:264` |
| `T04:272-276` | `libshmem_device.signal_wait_until`, `NVSHMEM_CMP_EQ` | b | thread 0 | B-thread `T04:262` |

The stores before the first event go to local symmetric buffers
(`T04:164-166`, `T04:208-210`, `T04:226-227`), inside loops of class L-data
(`T04:198`, `T04:223`: the bound is loaded at `T04:148-150`).

### K11 `barrier_all_intra_node_atomic_cas_block`

| Lines | Call | Pattern | By | Under |
| --- | --- | --- | --- | --- |
| `CO:162` | `atomic_cas(remote_ptr, 0, 1, "sys", "release")` on a word of rank `thread_idx` (`dl.symm_at`, `CO:161`) | a | thread | B-thread `CO:160`; L-sched `CO:162-163` (the call is the loop condition) |
| `CO:166` | `atomic_cas(symm_flag_ptr + thread_idx, 1, 0, "sys", "acquire")` on a local word | a | thread | B-thread `CO:165`; L-sched `CO:166-167` |
| `CO:168` | `__syncthreads` | b | CTA | none |

The whole body is inside `with dl.simt_exec_region() as (thread_idx,
block_size)` (`CO:158`), a "SIMT (per-thread) execution region" by its
docstring (`SI:257-266`).

### K12 and K20 `kernel_inter_node_p2p_for_same_local_rank`

| Lines | Call | Pattern | By | Under |
| --- | --- | --- | --- | --- |
| `T06:256-261` | `libshmem_device.putmem_block` | c | CTA | none |
| `RS:485-490` | `libshmem_device.putmem_block` | c | CTA | none |

The kernel does not read the program id (`T06:246-261`, `RS:475-490`); both
launches use a grid of one (`T06:211`, `T06:365`; `RS:172`, `RS:621`).

### K15 and K16 `barrier_all_intra_node_non_atomic_block` and its helper

| Lines | Call | Pattern | By | Under |
| --- | --- | --- | --- | --- |
| `CO:177` | `st(remote_ptr, target_value, scope="sys", semantic="release")` on a word of rank `thread_idx` (`CO:176`) | a | thread | B-thread `CO:174` |
| `CO:178` | `ld(symm_flags + thread_idx, scope="sys", semantic="acquire")` on a local word | a | thread | B-thread `CO:174`; L-sched `CO:178-179` (the call is the loop condition) |
| `CO:180` | `__syncthreads` | b | CTA | none |

K15 runs these three lines twice, the second time on the words `num_ranks`
further (`CO:194-196`).

### K17 `kernel_consumer_gemm_persistent`

| Lines | Call | Pattern | By | Under |
| --- | --- | --- | --- | --- |
| `T07:181-185` | `dl.wait` on `rank_end - rank_beg + 1` local words, `"gpu"`, `"acquire"` | b | CTA | L-count `T07:148`; counter branch `T07:150` |
| `T07:190` | `a_desc.load` of the symmetric workspace that peers fill | plain access | CTA | L-count `T07:148` |

The function is decorated with `triton.jit`, not `triton_dist.jit`
(`T07:77`).

### K18 `nvshmem_device_producer_p2p_put_block_kernel`

| Lines | Call | Pattern | By | Under |
| --- | --- | --- | --- | --- |
| `T07:293-301` | `libshmem_device.putmem_signal_block`, `NVSHMEM_SIGNAL_SET` | c | CTA | L-count `T07:287` |

### K19 `kernel_gemm_rs_producer_persistent`

| Lines | Call | Pattern | By | Under |
| --- | --- | --- | --- | --- |
| `T08:274-277` | `tl.atomic_add(counter_ptr + counter_id, 1, sem="release", scope="gpu")`; the result is used | a | CTA | L-count `T08:209`; counter branch `T08:245`; L-count `T08:266` |
| `T08:281-284` | `dl.notify` to the own rank, `signal=1`, `comm_scope="gpu"` | b | CTA | the same, then B-sched `T08:280` |

The counter is a local tensor of the launch (`T08:394-396`), written by the
CTAs of this kernel only (I).

### K23 `notify_dispatch_main`

| Lines | Call | Pattern | By | Under |
| --- | --- | --- | --- | --- |
| `DD:61` | `T.sync_blocks(barrier_signal)` | b | block 0 | B-id `DD:59` |
| `DD:66` | `T.st(per_rank_buffer[rank, tx], ..., dst_pe=tx)`, default scope and semantic | a | thread | B-id `DD:59`; B-thread `DD:65` |
| `DD:68` | `T.st(per_expert_buffer[rank, i], ..., dst_pe=tx)`, default scope and semantic | a | thread | the same, then L-count `DD:67` |
| `DD:70` | `T.barrier_blocks(barrier_signal)` | b | block 0 | B-id `DD:59` |
| `DD:75` | read and write of `per_rank_buffer`, which the peers filled | plain access | thread | B-id `DD:59`; B-thread `DD:73`; L-count `DD:74` |
| `DD:84` | read of `per_expert_buffer`, which the peers filled | plain access | thread | B-id `DD:59`; B-thread `DD:80`; L-count `DD:83` |
| `DD:87` | `T.sync_threads()` | no row | block 0 | B-id `DD:59` |
| `DD:99` | `T.barrier_blocks(barrier_signal)` | b | block 0 | B-id `DD:59` |
| `DD:114` | `T.sync_threads()` | no row | block | B-id `DD:100` |

Thread `rank` of block 0 writes the counter that the host polls (`DD:76-77`;
section 6), and threads `tx < num_local_experts` write the per-expert
counters (`DD:80-86`).

### K24 `cached_notify_dispatch_main`

| Lines | Call | Pattern | By | Under |
| --- | --- | --- | --- | --- |
| `DD:201` | `T.sync_blocks(barrier_signal)` | b | CTA | none |
| `DD:208` | `T.barrier_blocks(barrier_signal)` | b | CTA | none |

### K25 `dispatch_main`

Sender blocks (`bx % 2 == 0`, `DD:306`). A "group" is the threads with one
`responsible_rank` (`DD:303`); "warp" is one warp of the group; "elected
lane" is the lane that `T.shuffle_elect(32)` picks in a warp.

| Lines | Call | Pattern | By | Under |
| --- | --- | --- | --- | --- |
| `DD:314` | `T.st(channel_start_offset[...], scope="sys", sem="relaxed", dst_pe=responsible_rank)` | a | elected lane of warp 0 of the group | B-thread `DD:311` |
| `DD:316` | `T.st(channel_end_offset[...], scope="sys", sem="relaxed", dst_pe=responsible_rank)` | a | the same | B-thread `DD:311` |
| `DD:317` | `T.sync_warp()` | no row | warp | none |
| `DD:334-338` | `T.wait_ge` on the peer's head word (`peer = responsible_rank`) | b | elected lane of each warp | L-data `DD:332`; B-thread `DD:333` |
| `DD:339` | `T.sync_warp()` | no row | warp | L-data `DD:332` |
| `DD:363-370` | `T.put_warp(..., dst_pe=responsible_rank)` | a | the warp whose turn it is | L-data `DD:332`, `DD:343`; data branch `DD:352`; B-thread `DD:360` |
| `DD:374` | `T.st(channel_src_idx_buffers[...], dst_pe=responsible_rank)`, default scope and semantic | a | elected lane of that warp | the same, then B-thread `DD:373` |
| `DD:383` | `T.ld(topk_idx[...], nc=True)` of local input | a | lanes below `num_topk` | the same, then B-thread `DD:377` |
| `DD:387-391` | `T.st(channel_topk_idx_buffers[...], dst_pe=responsible_rank)` | a | the same | the same |
| `DD:395` | `T.ld(topk_weights[...], nc=True)` of local input | a | the same | the same |
| `DD:397-401` | `T.st(channel_topk_weights_buffers[...], dst_pe=responsible_rank)` | a | the same | the same |
| `DD:410` | `T.sync_threads(responsible_rank, num_threads_per_rank)` | no row | group | L-data `DD:332` |
| `DD:412-418` | `T.st(channel_tail_idx[...], scope="sys", sem="release", dst_pe=responsible_rank)` | a | elected lane of warp 0 of the group | L-data `DD:332`; B-thread `DD:411` |

Receiver blocks (`DD:420`). The group is the threads with one source rank
`responsible_rank`.

| Lines | Call | Pattern | By | Under |
| --- | --- | --- | --- | --- |
| `DD:431` | `T.wait_ne(channel_start_offset[...], 0)` on a local word | b | elected lane of each warp | B-thread `DD:430` |
| `DD:432` | `T.ld(channel_start_offset[...], sem="volatile")` | a | the same | B-thread `DD:430` |
| `DD:433` | `T.wait_ne(channel_end_offset[...], 0)` | b | the same | B-thread `DD:430` |
| `DD:434` | `T.ld(channel_end_offset[...], sem="volatile")` | a | the same | B-thread `DD:430` |
| `DD:453` | `T.ld(channel_tail_idx[...], sem="acquire", scope="sys")` | a | thread 0 of the group | L-sched `DD:451`; L-sched `DD:452` (left at `DD:456-458` when the tail differs from the head) |
| `DD:461` | `T.sync_threads(responsible_rank, num_threads_per_rank)` | no row | group | L-sched `DD:451` |
| `DD:471-478` | `T.put_warp` from the local ring buffer, which the peer fills, to `recv_x` (`dst_pe` is -1) | a | warp | L-sched `DD:451`; loop `DD:467`, bound `tail - head` |
| `DD:485-489` | `T.ld(channel_src_idx_buffers[...], nc=True)` | a | thread | L-sched `DD:451`; loop `DD:481` |
| `DD:497-502` | reads of `channel_topk_idx_buffers` and `channel_topk_weights_buffers`, which the peer fills | plain access | thread | L-sched `DD:451`; loop `DD:493` |
| `DD:509` | `T.sync_threads(responsible_rank, num_threads_per_rank)` | no row | group | L-sched `DD:451` |
| `DD:511` | `T.st(channel_head_idx[...], scope="sys", sem="relaxed")` on a local word | a | elected lane of the last warp of the group | L-sched `DD:451`; B-thread `DD:510` |

In the inner sender loop the round ends early with `continue` for a token
that is not selected (`DD:352-354`).

### K26 `cached_dispatch_main`

The text is that of K25 without the top-k copies. Same columns, by reference:

| Lines | Call | Corresponds to K25 line |
| --- | --- | --- |
| `DD:584`, `DD:586` | `T.st`, `sys`, `relaxed`, to the peer | `DD:314`, `DD:316` |
| `DD:587`, `DD:608` | `T.sync_warp()` | `DD:317`, `DD:339` |
| `DD:603-607` | `T.wait_ge` on the peer's head word, under L-data `DD:601` | `DD:334-338` |
| `DD:632-639` | `T.put_warp` to the peer, under L-data `DD:601`, `DD:612` | `DD:363-370` |
| `DD:643` | `T.st` of the source index to the peer | `DD:374` |
| `DD:652` | `T.sync_threads(responsible_rank, num_threads_per_rank)` | `DD:410` |
| `DD:654-660` | `T.st` of the tail, `sys`, `release`, under `T.shuffle_elect(96)` (`DD:653`) | `DD:412-418` |
| `DD:673-676` | `T.wait_ne`, `T.ld` volatile, twice | `DD:431-434` |
| `DD:695` | `T.ld` of the tail, `acquire`, `sys`, under L-sched `DD:693`, `DD:694` | `DD:453` |
| `DD:703`, `DD:738` | `T.sync_threads(responsible_rank, num_threads_per_rank)` | `DD:461`, `DD:509` |
| `DD:712-719` | `T.put_warp` from the ring buffer to `recv_x` | `DD:471-478` |
| `DD:726-730` | `T.ld(..., nc=True)` of the source index | `DD:485-489` |
| `DD:740` | `T.st` of the head, `sys`, `relaxed`, under `T.shuffle_elect(96)` (`DD:739`) | `DD:511` |

### K27 `cached_notify_combine_main`

| Lines | Call | Pattern | By | Under |
| --- | --- | --- | --- | --- |
| `DC:36` | `T.sync_blocks(barrier_signal)` | b | block 0 | B-id `DC:35` |
| `DC:39` | `T.barrier_blocks(barrier_signal)` | b | block 0 | B-id `DC:35` |
| `DC:56` | `T.ld(send_head[token_idx, rank_id], nc=True)` of local memory | a | lane | B-id `DC:40`; L-count `DC:52`; branch `DC:55` |

### K28 `combine_main`

Sender blocks (`DC:147`). The group of a warp is `send_rank_id` (`DC:148`).

| Lines | Call | Pattern | By | Under |
| --- | --- | --- | --- | --- |
| `DC:175-179` | `T.wait_ge` on the peer's head word (`peer=send_rank_id`) | b | elected lane of each warp | L-data `DC:171`; B-thread `DC:174` |
| `DC:180` | `T.sync_warp()` | no row | warp | L-data `DC:171` |
| `DC:189-196` | `T.put_warp(..., dst_pe=send_rank_id)` | a | warp | L-data `DC:171`; L-count `DC:183`, bound `num_round_tokens` (`DC:173`) |
| `DC:201` | `T.ld(src_idx[...], nc=True)` of local input | a | elected lane | the same, then B-thread `DC:200` |
| `DC:202` | `T.st(channel_src_idx_buffers[...], dst_pe=send_rank_id)` | a | elected lane | the same |
| `DC:207` | `T.ld(topk_weights[...], nc=True)` of local input | a | lanes below `num_topk` | the same loops, then B-thread `DC:205` |
| `DC:208-210` | `T.st(channel_topk_weights_buffers[...], dst_pe=send_rank_id)` | a | the same | the same |
| `DC:216` | `T.sync_threads(send_rank_id, threads_per_rank)` | no row | group | L-data `DC:171` |
| `DC:218-224` | `T.st(channel_tail_idx[...], scope="sys", sem="release", dst_pe=send_rank_id)` | a | the lane that `T.shuffle_elect(96)` picks | L-data `DC:171`; B-thread `DC:217` |

Receiver blocks (`DC:226`).

| Lines | Call | Pattern | By | Under |
| --- | --- | --- | --- | --- |
| `DC:237` | `T.sync_threads()` | no row | CTA | none |
| `DC:253` | `T.ld(channel_tail_idx[responsible_channel, lane_id], sem="acquire", scope="sys")` | a | lane `lane_id < num_ranks` of warp 0 | B-thread `DC:239`; L-sched `DC:242` (left at `DC:248-249` when every other warp has retired) |
| `DC:255` | `T.st(shared_channel_tail_idx[lane_id], sem="release", scope="cta")` to CTA-shared memory | a | the same | the same |
| `DC:265` | `T.st(channel_head_idx[...], sem="relaxed", scope="sys")` on a local word | a | the same | the same, then a branch on values read from CTA-shared memory `DC:263` |
| `DC:281` | `T.ld(send_head[token_idx, lane_id], nc=True)` of local input | a | lanes below `num_ranks` of a reduction warp | B-thread `DC:266`; L-count `DC:276`; B-thread `DC:280` |
| `DC:284` | `T.ld(shared_channel_tail_idx[lane_id], sem="acquire", scope="cta")` | a | lane of a reduction warp | B-thread `DC:266`; L-count `DC:276` |
| `DC:286` | the same load again | a | the same | the same, then L-sched `DC:285` (`T.warp_any`: the warp leaves together) |
| `DC:289` | `T.sync_warp()` | no row | warp | B-thread `DC:266`; L-count `DC:276` |
| `DC:312-316` | `T.ld(channel_x_buffers[...], nc=True)` of the ring buffer, which the peers fill | a | lane | L-count `DC:276`, `DC:308`; L-data `DC:310` |
| `DC:333-337` | `T.ld(channel_topk_weights_buffers[...], nc=True)` | a | lanes below `num_topk` | L-count `DC:276`; B-thread `DC:328`; L-data `DC:331` |
| `DC:348` | `T.sync_warp()` | no row | warp | B-thread `DC:266` |

Ordinary accesses to CTA-shared memory by different agents, with no barrier
after `DC:237`: `warp_retired` is read at `DC:247` and `DC:261` and written
at `DC:350`; `warp_channel_head_idx` is read at `DC:262` and written at
`DC:343-345`.

## 6. Host-side calls

Python outside a kernel that writes, waits on or orders memory which a
kernel of the same row also uses. These are not kernel events.

| Source | Host call | Where |
| --- | --- | --- |
| Tutorial 01 | `signal.fill_(0)`; `nvshmem_barrier_all_on_stream` before each launch | `T01:164`, `T01:168`, `T01:188-189` |
| Tutorial 02 | Copy-engine variant, no kernel: `dst.copy_(src)` from the peer's buffer, then `cuda.cuStreamWriteValue32` on the own signal word, in a host loop over ranks | `T02:83-100` |
| Tutorial 02 | Buffer and signal reset; `nvshmem_barrier_all_on_stream` before and after the launch of K2 | `T02:188-195`, `T02:207` |
| Tutorial 03 | `nvshmem_barrier_all_on_stream` once before the timed launches; the host changes `signal_value` and the signal buffer per launch | `T03:99-103`, `T03:181-185`, `T03:240` |
| Tutorial 04 | No host synchronization call between `initialize_distributed()` and the launches; the host reads the receive buffers after the dispatch launch | `T04:458`, `T04:552-599` |
| Tutorial 05 | `remote_buf.copy_(local_buf)` into each peer's buffer in a host loop; `nvshmem_barrier_all_on_stream`; then K11, then K10 | `T05:181-200`, `T05:213-220`, `T05:267`, `T05:273` |
| Tutorial 06 | `wait_eq` on a signal word, only under `overlap_with_gemm`, which the tutorial's main sets to `False` | `T06:280-287`, `T06:466` |
| Tutorial 06 | `remote_buf.copy_`; `barrier_all_on_stream` (K11 or K15); stream waits; `nvshmem_barrier_all_on_stream`; `reset_barriers` | `T06:294`, `T06:344-346`, `T06:376`, `T06:390`, `T06:398`, `T06:424-432`, `T06:131` |
| Tutorial 07 | `set_signal` on the own ready word after the local copy | `T07:507-509` |
| Tutorial 07 | `cudart.cudaMemcpyAsync` to each peer of the node, then `set_signal` on that peer's word, in host loops; `wait_eq` on the own word for data from another node | `T07:316-360` |
| Tutorial 07 | `nvshmem_barrier_all_on_stream` after the reset of the ready words; stream waits | `T07:504-505`, `T07:262`, `T07:398`, `T07:437-438` |
| Tutorial 08 | `nvshmem_barrier_all_on_stream`; stream waits around the reduce-scatter | `T08:391`, `T08:424`, `T08:505`, `T08:514` |
| Tutorial 08, through RS | `_wait_eq_cuda` on the word that K19 notifies, before each `cudaMemcpyAsync` (`overlap_with_gemm` keeps its default `True`) | `RS:551-552`, `RS:566-580`, `RS:604` |
| Tutorial 08, through RS | `nvshmem_barrier_all_on_stream`; stream waits; `reset_barriers` | `RS:608-609`, `RS:632`, `RS:803`, `RS:815`, `RS:828-830`, `RS:105-106` |
| Tutorial 08, through RS, without full-mesh NVLink | `_wait_eq_cuda`, `dst.copy_`, `_set_signal_cuda` in a host loop over stages; `pynvshmem.putmem_on_stream`; `nvshmem_barrier_all_on_stream` | `RS:262-280`, `RS:538-544` |
| Tutorials 06 to 08 | `wait_eq` and `set_signal` are `_wait_eq_cuda` and `_set_signal_cuda` for the CUDA back end: `cuStreamWaitValue32` or `64`, `cuStreamWriteValue32` or `64` | `KC:136-138`, `CO:364-406` |
| DeepEP, K23 | `ep_ext.wait_for_counters_ready`: a host loop that reads the mapped counters through `volatile` pointers until all are non-negative; the counters are reset to -1 before the launch | `DD:164-165`, `DD:184`, `DU:293-316` |
| DeepEP, all kernels | `kernel.initialize(allocator=..., stream=...)`; launches under `torch.cuda.stream(comm_stream)`, except K23, which is launched outside such a block | `DD:158`, `DD:167`, `DD:228-229`, `DD:855-856`, `DD:892-893`, `DC:85`, `DC:379`, `DC:395-396` |
| DeepEP, K28 | `compute_stream.wait_stream(comm_stream)` after the combine | `DC:413-414` |
| DeepEP, buffer | `barrier_signal` zeroed once at allocation; `torch.cuda.synchronize()` and `self.group.barrier()` at construction | `DB:81-82`, `DB:93` |

Treatment of the host signals is in `PrimitiveSources.md` (sections 5.6 and
5.7, inference I6).

## 7. Kernels of the files read that no tutorial of the set reaches

They are outside the two sets and have no row. They are listed because they
were read and show the same forms.

| Function | Where | Form |
| --- | --- | --- |
| `unsafe_barrier_on_this_grid`, `cooperative_barrier_on_this_grid`, `barrier_on_this_grid` | `CO:61-142` | Thread 0 of each CTA: `atomic_add` with `release` at `gpu`, then a spin loop on `ld_acquire` (`CO:76-83`, `CO:125-132`), between two `__syncthreads` |
| `barrier_all_intra_node_non_atomic` | `CO:200-224` | K16 under a branch on the program id, and `barrier_on_this_grid` |
| `copy_continuous_kernel` | `RS:195-203` | No event |
| `reduce_scatter_ring_push_1d_intra_node_kernel` | `RS:285-339` | L-count over stages; thread 0 spins on `ld(..., "acquire", "sys")` (`RS:312-314`, `RS:321-323`); `atomic_add` whose result decides a `st` with `release` at `sys` (`RS:331-335`), the form of K19; `barrier_all_block` under a branch on the program id (`RS:338-339`) |
| `reduce_scatter_ring_push_1d_intra_node_rma_kernel` | `RS:380-443` | The same spin loops (`RS:420-429`); `putmem_signal_nbi_block` under a branch on stage and program id (`RS:436-440`) |
| `bisect_*` kernels | `CO:275-361` | No event |
| `copy_kernel`, `copy_and_barrier_all_intra_node_kernel`, `barrier_all_intra_node_kernel` | `AG:47-137` | Launched by `local_copy_and_barrier_all` (`AG:140-172`), which a text search of `kernels/nvidia/` finds called only at line 604 of AG, not from tutorial 07; `st` to the barrier words by threads below `num_ranks` (`AG:107-109`, `AG:134-136`) |

Checked by reading the call sites: the host functions that launch the two
`reduce_scatter_ring_push_1d_intra_node_*` kernels (`RS:342`, `RS:446`) are
not called from `reduce_scatter_2d_op` or below (`RS:793-831`).

## 8. Observations made on the way

1. **The group barrier is not documented.** `T.sync_threads(barrier_id,
   arrive_count)` has the docstring "Synchronize all threads in a block" at
   `9ba8a605` (`TB:702-709`). The programming guide of TileScale `71e7145`
   lists `T.sync_threads([barrier_id, arrive_count])` as "Block-wide barrier"
   (`docs/programming_guides/instructions.md:191`). K25, K26 and K28 pass a
   rank index and the thread count of one group (`DD:410`, `DD:461`,
   `DD:509`, `DD:652`, `DD:703`, `DD:738`, `DC:216`). No document read here
   states that this form waits for `arrive_count` threads only.
2. **The waits of the port carry no scope or order.** At `9ba8a605`
   `wait_ne` and `wait_ge` take `(value, expected, peer=-1)` (`TC:135-142`).
   `PrimitiveSources.md` (E-TS6) cites `71e7145`, where they also take
   `scope="sys"` and `semantics="acquire"`.
3. **`T.shuffle_elect(96)`** elects "exactly one lane" in each group of 96
   consecutive threads by its docstring (`TB:438-445`). K26 and K28 use it
   to publish tail and head (`DD:653`, `DD:739`, `DC:217`); K25 uses
   `T.shuffle_elect(32)` with a warp test (`DD:411`, `DD:510`).
   `DeepEPInventory.md`, section 4.2, has the consequence for fewer than 8
   ranks. In the sender of K28 a group is not 96 consecutive threads: warps
   are assigned to ranks round-robin (`DC:148`).
4. **Ordinary accesses to CTA-shared memory in the receiver of K28**
   (`DC:247`, `DC:261-262`, `DC:343-345`, `DC:350`) are made by different
   warps with no barrier between them. Under W1 and I1 of
   `PrimitiveSources.md` such a pair conflicts (I).
5. **Tutorial 03, signal buffer.** Each signal tensor is created with shape
   `(1, )` (`T03:276-278`) and one of them is passed as `symm_flag`
   (`T03:103`, `T03:185`). The kernels index `symm_flag + peer`, `symm_flag +
   segment` and `symm_flag + thread_idx` with values up to `WORLD_SIZE - 1`
   (`T03:73`, `T03:91`, `T03:143`, `T03:151`, `T03:163`, `T03:173`). For
   more than one rank these words lie outside the tensor (I; not run).
6. **Tutorial 07, context call.** The tutorial calls
   `create_ag_gemm_context(A, B, rank, WORLD_SIZE, max_M=M, BLOCK_M=...,
   BLOCK_N=..., BLOCK_K=..., stages=...)` (`T07:495-503`); the definition is
   `create_ag_gemm_context(max_M, N, K, dtype, rank, num_ranks,
   num_local_ranks=8, ag_intranode_stream=None, ag_internode_stream=None)`
   (`AG:511-521`). The call does not fit the definition (I; not run). This
   adds to E-TD17 of `PrimitiveSources.md`.
7. **Tutorial 08 has two host paths**, chosen by `has_fullmesh_nvlink()`
   (`RS:807-810`); only one of them launches K13 or K14 and K20.

## 9. Not checked

- Nothing was compiled or run. Every "I" above is a reading of the text.
- What the compilers emit: whether a per-thread branch of a Triton kernel
  outside `dl.simt_exec_region` (K3, K4, K6) is executed per thread as
  written; whether TileLang inserts a barrier in K22; what `T.put_warp`,
  `T.ld(nc=True)` and the two-argument `T.sync_threads` become.
- Whether `T.sync_blocks`, `T.barrier_blocks`, `T.copy` and `T.clear`, which
  K23, K24 and K27 call outside any thread branch, are one event of the
  block or one per thread.
- `T.thread_return` (`DC:45`) has no Python definition in the tree
  (`DeepEPInventory.md`, section 9).
- The bodies of `has_fullmesh_nvlink` (`python/triton_dist/nv_utils.py:298`),
  `has_tma`, `supports_p2p_native_atomic` and `nvshmem_create_tensor(s)`, and
  the rest of UT; the `nvshmem.core`, `pynvshmem`, `cuda` and `cudart` host
  bindings.
- Disjointness claims marked I: rows of the outputs written by different
  receiver groups of K25 and K26, columns of `send_head`, slots written by
  the warps of one sender group.
- That exactly one CTA of K19 sees the last counter value per rank.
- The functions of AG outside lines 25 to 173 and 445 to 553, among them the
  second `kernel_consumer_gemm_persistent` (`AG:201`), which tutorial 07
  does not call.
- The DeepEP reference (`deepseek-ai/DeepEP` at `b57e5e21`) was not opened
  for this page; only the port was read.
