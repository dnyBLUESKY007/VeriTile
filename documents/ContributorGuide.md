# Contributor guide

For everyone who works on a ticket of `dnyBLUESKY007/VeriTile` outside the coordinating session: the people a ticket is assigned to, and agents that take a ticket from the open pool. A ticket, this guide and the files the ticket names are meant to be all you need. If something you need is missing from them, say so in a comment on the ticket; that is a defect of the ticket.

## Roles

- **Coordinator.** One session, acting through the account `dnyBLUESKY007`. It plans, writes the tickets, reviews every pull request, runs the gate of record, merges, and is the only one that talks to the GPU side.
- **Contributor.** You. You deliver one ticket at a time as a pull request and a report.
- **User.** The owner of the project. Design choices, sign-offs of frozen statements and changes of scope are theirs.

## How tickets are given out

**Assigned tickets.** A ticket with a section `## Assignment` is assigned. Who it belongs to is in the assignee field, or, until that person has accepted, in the coordinator's assignment comment on the ticket; ticket descriptions name no person. The coordinator assigns under these rules:

1. Only tickets whose dependencies are merged, that are specified to the level of files, deliverable and acceptance, and that reserve no decision for the user.
2. Evenly: the sizes of the open tickets of each contributor (the `Size` line, in agent-days) are kept as equal as the ticket sizes allow. A new ticket goes to whoever has the least open work.
3. One area per contributor at a time, so that the files of two contributors do not meet. The tickets of one contributor are listed in the order to take them.
4. Statement freezes, sign-offs, integration across lines and GPU requests stay with the coordinator.

Accept an assigned ticket by commenting `Accept: <login>` on it within two working days. (GitHub can record you as assignee only after you have commented.) To give it back, comment `Release: <login>` with the reason and the state of the work. A ticket with no pull request (a draft counts) and no progress comment five working days after it was accepted is asked about once and may then go to someone else.

**Who works on a ticket** is a label of its own, independent of whether the ticket is ready (`parallel-ready`) or taken (`claimed`):

| Label | Worked on by |
| --- | --- |
| `owner:coordinator` | the coordinating session and its lanes |
| `owner:codex` | the user's Codex sessions |
| `owner:contributor` | a human contributor, named in the assignee field |

A ticket without an `owner:` label has not been given out. The coordinator sets and changes these labels; when a ticket moves from one party to another, only the label and the assignee change. To list one party's open tickets: `gh api 'repos/dnyBLUESKY007/VeriTile/issues?labels=owner:codex&state=open&per_page=100' -q '.[]|"#\(.number) \(.title)"'`.

**Open pool.** An open ticket with both labels `parallel-ready` and `parallel-easy` and no assignee may be taken by anyone:

```sh
gh api 'repos/dnyBLUESKY007/VeriTile/issues?labels=parallel-ready,parallel-easy&state=open&per_page=50' \
  -q '.[]|select(.assignees|length==0)|"#\(.number) \(.title)"'
```

Claim it with a comment whose first line is `Claim: <claimant-id>`, where the ID names you alone (`<agent or login>-<session start in UTC as YYYYMMDDTHHMMZ>`), then read the comments again: the ticket belongs to the earliest `Claim:` whose author has posted no later `Release:`. If that is not yours, comment `Release: <claimant-id> lost the claim` and take another. If your account may set labels, also add `claimed` and remove `parallel-ready`.

Read a ticket with `gh api repos/dnyBLUESKY007/VeriTile/issues/<n> -q .body`; `gh issue view` fails on this repository. The map, issue #1, states the goal, the pins of every external repository and the index of ticket IDs.

## Where to work

Fork `dnyBLUESKY007/VeriTile` (this repository, not `VeriInfra-Lab/VeriTile`, which is its upstream). Work on a branch named after the ticket, for example `i2.4-step-exec`, and open the pull request against `main` of `dnyBLUESKY007/VeriTile`. Never push to `VeriInfra-Lab/VeriTile` and leave its issues alone.

## Rules for the work

These hold for every ticket. A ticket may add to them, never relax them.

1. **Files.** Change only the files the ticket owns, plus the import line of a target's root file (`VeriTile/Dist.lean`, `VeriTile/Interp.lean`, …) when you add a module. If another file must change, say so in your report instead of changing it.
2. **Upstream code is read-only.** Never edit anything under `VeriTile/Triton/`, nor `.github/`. In `lakefile.toml` you may append a line to a list; existing lines stay as they are. After any change to `lakefile.toml`, apply the same change to `library/lakefile.toml`, upstream's library-only view of it (`library/README.md`); the gate compares the two. New code lives in `VeriTile/Dist/`, `VeriTile/TileLang/`, `VeriTile/Interp/`, `VeriTile/Equiv/` or the directory the ticket names.
3. **Nothing at or below PTX.** The lowering from user code to executable code is trust base: it is neither cited nor modelled. The meaning of user code comes only from the front-end documentation (Triton, TileLang, Triton-distributed, TileScale), the CUDA C++ memory model (`cuda::thread_scope_*`) and the NVSHMEM API documentation. The behaviour of an implementation, a test or a GPU run is evidence, never a definition.
4. **Weakest semantics where documentation is silent.** A guarantee that documentation does not state does not hold. An undocumented helper is an ordinary, unordered, non-atomic element-wise access. An extra assumption is never used silently: propose it in your report with its statement, source and the kernels it affects.
5. **Main targets are complete.** No `sorry`, no new axiom, no `native_decide`, and no import of `VeriTile/Staging/` in `VeriTileDist`, `VeriTileTileLang`, `VeriTileInterp`, `VeriTileEquiv`. An unproved statement lives only under `VeriTile/Staging/`.
6. **Frozen statements.** `documents/dist/FrozenStatements.tsv` lists the statements the user signed. Never change the name or the text of one. When you prove one, move it to its main-target module under the same fully qualified name and change only its `module` column. Never edit a fingerprint and never run `check_frozen_statements.py` with `--fill` or `--accept-definitions`; if the check fails, report its output.
7. **Negative examples.** Each main theorem comes with an example that must fail without a hypothesis or on a wrong program, in the place the neighbouring modules keep theirs.
8. **Upstream kernels stay visible.** A transcribed kernel keeps the statements, casts, precision, masks and control flow of its source; a proof is about that program, not a simplified one.
9. **Python** uses the standard library only and `unittest`; the suites pass on Python 3.10 to 3.13.
10. **Documents** are in English. A document with a `.zh.md` twin gets both updated with the same structure and facts. Keep apart what you read in a source, what you infer and what you did not check. Cite every source: URL with access date, or repository, pinned commit, path and line. External repositories are read at the commits the map pins, in a scratch directory outside the checkout; read them, do not run them, and copy nothing from them unless the ticket says so.
11. **Public repository.** Nothing you commit or post contains a secret, token, host name, user name, home-directory path, network address or e-mail address.
12. **Smallest change.** Deliver what the ticket asks. No extra options, abstractions or files; comments say only what the code cannot.
13. **Decisions.** When the work needs a choice the ticket does not make, do not decide it silently. If it is a design choice (a definition others will build on, a statement's shape, scope), comment on the ticket with the background, the options, what each costs and the one you recommend, and continue with what does not depend on it. If it is small and local, take it and list it in your report.

## Setup and gate

Lean and Lake come from `lean-toolchain` through `elan`. The comparator is built once with `scripts/setup-comparator.sh <dir>` and put on `PATH` (`scripts/README.md`). The first build fetches the Mathlib cache with `lake exe cache get`. Never run `lake clean` without a package name; it deletes the Mathlib build.

Run the gate before you hand back, and report each command with its exit code. Judge by exit code, not by the look of the output.

```sh
bash scripts/local_gate.sh            # everything: documents, Python suites, Lean targets, comparator replay, fingerprints
bash scripts/local_gate.sh --no-lean  # a change of documents or Python only
```

If you cannot build the comparator on your machine, run the commands of the script that you can and say which you skipped. The gate of record is the same script run by the coordinator on the development server at the head commit of your pull request; its result is posted on the commit as the status `local-gate`. Hosted workflow results are advisory.

## GPU

You need no GPU and you never contact the GPU side. Kernels run only on one host with H800 GPUs (sm_90), at most two GPUs per request, through `documents/GpuExchangeProtocol.md`. A ticket that needs a run delivers the batch directory and the judge; the coordinator announces the batch and judges the result. Do not open `[gpu]` issues and do not set `gpu:*` labels. A proof never depends on a GPU generation or on a run.

## Hand back

1. Commit on your branch in focused commits with messages that state the purpose. If an AI agent wrote part of the change, say so in the pull-request body.
2. Open the pull request against `main` of `dnyBLUESKY007/VeriTile`. Its body has three sections: `## Summary` (the change as a small tree, table or diff sketch), `## Evidence` (the gate commands with their exit codes; before and after where that applies), `## Merge Danger` (`**Door:**` one-way or two-way; `**Blast Radius:**` one word). Name the ticket as `Refs #<n>`, not `Closes`.
3. Comment on the ticket: the pull request; what is complete; what is not; every decision you took that the ticket did not dictate; what you could not verify.
4. Do not merge and do not close the ticket.

A report that says plainly what failed or was skipped is worth more than one that reads as finished.

## Review

The coordinator reviews every pull request before it merges:

1. the diff stays inside the files the ticket owns and inside the rules above;
2. each sentence of the ticket's Deliverable and Acceptance is met, partly met or not met;
3. for documents, the citations are checked against the sources; for Lean, the statements proved are the ones asked, and the negative examples fail for the stated reason;
4. the gate of record passes at the head commit.

The outcome is one of: merged; changes requested on the pull request; or merged with the gaps written as a follow-up ticket assigned to the author. A wrong fact that a later decision would rest on, a broken rule or an acceptance that is mostly unmet is never merged.
