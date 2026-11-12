# Taking a `parallel-easy` ticket

Give this document to an agent that helps from outside the coordinating session. It tells the agent which tickets it may take, how to claim one so that no other agent takes it too, and what to hand back.

## Which tickets

Take only an open issue of `dnyBLUESKY007/VeriTile` that carries both labels `parallel-ready` and `parallel-easy` and has no assignee:

```sh
gh api 'repos/dnyBLUESKY007/VeriTile/issues?labels=parallel-ready,parallel-easy&state=open&per_page=50' \
  -q '.[]|select(.assignees|length==0)|"#\(.number) \(.title)"'
```

`parallel-ready` means the ticket is specified and nothing blocks it. `parallel-easy` adds that it is self-contained: documents or Python only, no Lean build, no decision reserved for the user. Read a ticket with `gh api repos/dnyBLUESKY007/VeriTile/issues/<n> -q .body`; `gh issue view` fails on this repository. Every ticket states its deliverable, its acceptance and the files it owns. The map, issue #1, states the goal, the hard constraints and the pinned versions of every external repository.

## Claim before you start

Several agents work at once, so a claim has three marks, set together:

1. Assign the issue to the GitHub account you act through.
2. Add the label `claimed` and remove `parallel-ready`.
3. Comment with the first line `Claim: <claimant-id>`, where the claimant ID names you alone: `<agent>-<session start in UTC as YYYYMMDDTHHMMZ>`, for example `codex-20261112T0930Z`.

Then read the comments again. The ticket belongs to the earliest `Claim:` comment whose author has not posted a later `Release:` comment. If that is not yours, comment `Release: <claimant-id> lost the claim`, leave the labels and the assignee alone, and take another ticket. Before claiming, also check that no other claimed open issue owns one of the same files.

To give a ticket back unfinished, comment `Release: <claimant-id>` with the reason and the state of the work, remove the assignee and `claimed`, and add `parallel-ready` again.

## Rules for the work

- Change only the files the ticket owns. If another file must change, say so in your report instead of changing it.
- Never edit anything under `VeriTile/Triton/`, `lakefile.toml` or `.github/`.
- Python uses the standard library only and is tested with `unittest`; the suites must pass on Python 3.10 through 3.13.
- Documents are in English. Keep what you read in a source, what you infer and what you did not check apart, and cite every source with its URL and access date, or with its file path and the pinned commit.
- Nothing at or below PTX is cited or modelled. A guarantee that documentation does not state does not hold.
- External repositories are read at the commits the map pins, in a scratch directory outside the checkout. Read them; do not run their code. Do not copy their files into this repository unless the ticket says so.
- No secrets, host names, user names or home-directory paths in anything you commit or post.

## Hand back

1. Run the checks the ticket names, and always `python3 scripts/check_docs.py` when a Markdown file changed. Report each command with its exit code.
2. Commit on a branch named after the ticket, for example `k-m.1-moe-inventory`. Push it to `dnyBLUESKY007/VeriTile` if your account may, otherwise to your own fork of it. Never push to `main` and never to `VeriInfra-Lab/VeriTile`.
3. Open a pull request against `main` of `dnyBLUESKY007/VeriTile`. Its body has three sections: `## Summary` (what the change is, as a small diagram, table or tree), `## Evidence` (before and after, with the exact commands and results), `## Merge Danger` (`**Door:**` one-way or two-way; `**Blast Radius:**` one word).
4. Comment on the issue with the pull request, what is complete, what is not, every decision you took that the ticket did not dictate, and what you could not verify.
5. Do not merge the pull request and do not close the issue. The coordinating session reviews, merges, closes the ticket and labels what it unblocks.

A report that says plainly what failed or was skipped is worth more than one that reads as finished.
