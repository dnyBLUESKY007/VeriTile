# LIT-a05. A counter of finished tiles; the last one fires the signal.
# Source: tutorials/08-overlapping-gemm-reduce-scatter.py:274-284 (TD-T08);
# docstring of triton.language.atomic_add (TR-CORE).
# Table: documents/dist/LitmusTable.md. Front end: Triton-distributed 7908e4e.
# A source for later tickets: not run here, and nothing imports it.
# Launch: one rank, grid 2. counter is an int32 word and barrier a uint64
# signal word, both initially 0; val has one int32 word per program.
import triton.language as tl
import triton_dist
import triton_dist.language as dl


@triton_dist.jit
def atomic_counter_last_tile(counter_ptr, barrier_ptr, val_ptr):
    pid = tl.program_id(0)
    rank = dl.rank()
    val = tl.atomic_add(counter_ptr, 1, sem="release", scope="gpu")
    tl.store(val_ptr + pid, val)
    if val == 1:
        dl.notify(barrier_ptr, rank, signal=1, comm_scope="gpu")
