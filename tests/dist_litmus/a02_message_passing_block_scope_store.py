# LIT-a02. Message passing where the store of the flag has block scope.
# Source: libcu++ "Memory model", second listing of "Example: Message
# Passing" ("UB: data race") (CU-MM).
# Table: documents/dist/LitmusTable.md. Front end: Triton-distributed 7908e4e.
# A source for later tickets: not run here, and nothing imports it.
# Launch: one rank, grid 2. x, flag and out are int32 words, initially 0.
import triton.language as tl
import triton_dist
from triton_dist.language.extra.language_extra import ld, st, tid


@triton_dist.jit
def message_passing_block_scope_store(x_ptr, flag_ptr, out_ptr):
    pid = tl.program_id(0)
    if tid(0) == 0:
        if pid == 0:
            tl.store(x_ptr, 42)
            st(flag_ptr, 1, scope="cta", semantic="release")
        elif pid == 1:
            while ld(flag_ptr, scope="gpu", semantic="acquire") != 1:
                pass
            tl.store(out_ptr, tl.load(x_ptr))
