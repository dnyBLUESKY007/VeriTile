# LIT-b04. A load after a wait whose token is not consumed.
# Source: docs/python-api/triton-dist.semantics.rst, "Token-Based Data
# Dependency" (TD-SEM).
# Table: documents/dist/LitmusTable.md. Front end: Triton-distributed 7908e4e.
# A source for later tickets: not run here, and nothing imports it.
# Launch: ranks 0 and 1, grid 1. queue is a symmetric word and signal a
# symmetric uint64 signal word, both initially 0; out is local.
import triton.language as tl
import triton_dist
import triton_dist.language as dl


@triton_dist.jit
def wait_without_consume_token(queue_ptr, signal_ptr, out_ptr):
    rank = dl.rank()
    if rank == 0:
        tl.store(dl.symm_at(queue_ptr, 1), 42)
        dl.notify(signal_ptr, 1, signal=1, sig_op="set", comm_scope="intra_node")
    elif rank == 1:
        token = dl.wait(signal_ptr, 1, "sys", "acquire", waitValue=1)
        tl.store(out_ptr, tl.load(queue_ptr))
