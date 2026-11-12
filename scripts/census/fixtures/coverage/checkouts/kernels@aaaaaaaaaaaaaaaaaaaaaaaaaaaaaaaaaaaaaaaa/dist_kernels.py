import triton
import triton.language as tl
import triton_dist.language as dl


@triton.jit
def dist(p):
    with dl.simt_exec_region() as (thread, size):
        tl.atomic_max(p, dl.rank(), mask=None)
    while thread:
        break
    tl.inline_asm_elementwise("nop", "", [], dtype=tl.void, is_pure=True, pack=1)
