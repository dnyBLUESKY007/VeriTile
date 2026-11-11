"""Census fixture for the Triton-distributed family. Read as text by scripts/census/scan.py; never imported or run."""
import triton
import triton.language as tl
import triton_dist
import triton_dist.language as dl
from triton_dist.language.extra import libshmem_device
from triton_dist.language.extra.language_extra import __syncthreads, tid


@triton.jit
def copy_block(dst_ptr, src_ptr, offs, mask):
    tl.store(dst_ptr + offs, tl.load(src_ptr + offs, mask=mask), mask=mask)


@triton_dist.jit
def ring_send_kernel(rank: tl.constexpr, num_ranks: tl.constexpr, in_ptr, buf_ptr, signal_ptr, n_blocks,
                     BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    peer = (rank + 1) % num_ranks
    offs = tl.arange(0, BLOCK)
    for i in range(pid, n_blocks, tl.num_programs(0)):
        copy_block(dl.symm_at(buf_ptr, peer) + i * BLOCK, in_ptr + i * BLOCK, offs, offs < BLOCK)
        __syncthreads()
        libshmem_device.fence()
        dl.notify(signal_ptr + i, peer, signal=1, sig_op="set", comm_scope="intra_node")


@triton_dist.jit(do_not_specialize=["rank"])
def ring_receive_kernel(rank, buf_ptr, out_ptr, signal_ptr, n_blocks, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    for i in range(pid, n_blocks, tl.num_programs(0)):
        token = dl.wait(signal_ptr + i, 1, "gpu", "acquire", waitValue=1)
        buf_ptr = dl.consume_token(buf_ptr, token)
        copy_block(out_ptr + i * BLOCK, buf_ptr + i * BLOCK, offs, offs < BLOCK)


@triton_dist.jit
def push_segment_kernel(remote_ptr, flag_ptr, elems_per_rank, elem_bytes, flag_value, NUM_PUSH_BLOCKS: tl.constexpr):
    pid = tl.program_id(axis=0)
    me = dl.rank()
    if pid < NUM_PUSH_BLOCKS:
        peer = (me + pid + 1) % dl.num_ranks()
        libshmem_device.putmem_signal_block(
            remote_ptr + me * elems_per_rank,
            remote_ptr + me * elems_per_rank,
            elems_per_rank * elem_bytes,
            flag_ptr + me,
            flag_value,
            libshmem_device.NVSHMEM_SIGNAL_SET,
            peer,
        )
    else:
        peer = tid(0)
        if peer < dl.num_ranks() and peer != me:
            libshmem_device.signal_wait_until(flag_ptr + peer, libshmem_device.NVSHMEM_CMP_EQ, flag_value)
        __syncthreads()
