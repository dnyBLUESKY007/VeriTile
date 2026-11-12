"""Per-rank harnesses of the batches `internode-logical-ag03-001` and `internode-logical-rs06-001`.

They call the functions of tutorials 03 and 06 of Triton-distributed at
`7908e4ea` on small fixed inputs and return what every rank holds afterwards.
Nothing of a tutorial is copied here and no tutorial is edited: its file is
read from the checkout that the environment `triton-dist` was built from
(found as `gpurun.dist.launch_script` finds the pinned launcher). The
harnesses have not run on the development host, which has no GPU.
`T03:n` and `T06:n` below are lines of the two tutorial files at the pin.

Both report, as `dist_logical_nodes.py` of the environment batch does:

    topology   int32 [6]   rank, world size, LOCAL_WORLD_SIZE, nodes, LOCAL_RANK, CUDA device index

`all_gather` (tutorial 03), `launch["kernel"]` is `push_1d` or `push_2d`:

    x          int8 [bytes]   the same in every rank; rank r owns segment r of `ranks` equal segments
    gathered   int8 [bytes]   the rank's buffer after the kernel. Before it, it holds the rank's own
                              segment and the sentinel elsewhere, so a segment that no peer sent stays
                              sentinel. `judge.py` expects `x` in every rank.

The tutorial has no `__main__` guard: importing the file runs its whole
program (`T03:256-299`: runtime start, random data, a timing loop, runtime
stop). The harness therefore executes only the tutorial's top-level imports,
classes and functions (`definitions`) and is itself the program around them.
What it does differently from that program:

- one buffer of `bytes` and one kernel call per test, where the tutorial has a
  double buffer and calls each kernel again in a timing loop;
- each signal tensor has `ranks` words. The tutorial creates them with shape
  `(1, )` (`T03:276-278`) while the kernels address word `segment`, `peer` or
  `thread_idx`, up to `WORLD_SIZE - 1` (`T03:73`, `T03:91`, `T03:143`,
  `T03:151`): with two ranks, rank 1 signals word 1 on rank 0 and rank 0 waits
  on it, one word past the tensor. The harness does not reproduce that;
- the signals are cleared and a barrier is passed before the kernel, so that no
  signal is set before its receiver has cleared it.

`reduce_scatter` (tutorial 06), `launch["context"]` is `tutorial` or `harness`:

    x          int8 [ranks, M, N]   rank r reduces `x[r]` converted to bfloat16, the tutorial's dtype
    barrier    int32 [3]            local rank and node size of the tutorial's intra-node barrier (the
                                    NVSHMEM node team), and whether `supports_p2p_native_atomic()`
    reduced    bf16 [M / ranks, N]  what `reduce_scatter_2d_op` returns; `judge.py` expects rows
                                    r * M / ranks onwards of the sum of all `x[s]`

The file has a `__main__` guard and is imported whole. With context `tutorial`
the context comes from the tutorial's `create_reduce_scater_2d_ctx`, called as
its `__main__` calls it (`T06:460-466`). That function indexes the tensors of
the rank's node with the global rank (`signal_bufs[rank]`, `T06:202`; the list
has `local_world_size` entries, `nvshmem_create_tensors` in
`python/triton_dist/utils.py:257-272`), so it raises `IndexError` in every rank
that is not on node 0. With context `harness` the same context is built here
with that one index taken within the node; everything after it is the
tutorial's.
"""
import ast
import importlib.util
import os
import sys
import types


def tutorial_file(name):
    from gpurun import dist
    return dist.launch_script().parents[1] / 'tutorials' / name


def definitions(path):
    """A module of what the top-level imports, classes and functions of `path` define; nothing else of it runs.

    The syntax tree keeps the file name and the line numbers, so
    `inspect.getsource`, with which Triton reads a kernel, returns the
    unchanged text of the file.
    """
    tree = ast.parse(path.read_bytes(), str(path))  # bytes: the compiler's decoding, whatever the locale
    tree.body = [node for node in tree.body
                 if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef))]
    module = types.ModuleType('tutorial_definitions_' + ''.join(c if c.isalnum() else '_' for c in path.stem))
    module.__file__ = str(path)
    sys.modules[module.__name__] = module  # `inspect` and `dataclasses` look a definition's module up by name
    exec(compile(tree, str(path), 'exec'), module.__dict__)
    return module


def imported(path):
    """The module of a tutorial whose file name is not a module name: what `import` would execute."""
    spec = importlib.util.spec_from_file_location('tutorial_' + ''.join(c if c.isalnum() else '_' for c in path.stem),
                                                  path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def report_topology(torch, group, outputs):
    """What the launcher and the runtime report, not what gpurun passed in: the dev side compares the two."""
    world, local = int(os.environ['WORLD_SIZE']), int(os.environ['LOCAL_WORLD_SIZE'])
    device = torch.cuda.current_device()
    seen = [group.rank(), group.size(), local, world // local, int(os.environ['LOCAL_RANK']), device]
    outputs['topology'].copy_(torch.tensor(seen, dtype=torch.int32))
    # The log survives a later hang, the outputs do not.
    print(f'rank {seen[0]} of {seen[1]}: LOCAL_WORLD_SIZE={local} nodes={seen[3]} LOCAL_RANK={seen[4]} '
          f'device={device}', flush=True)
    return world, local


def all_gather(rank, ranks, inputs, outputs, launch):
    import torch
    from triton_dist.utils import (NVSHMEM_SIGNAL_DTYPE, finalize_distributed, initialize_distributed,
                                   nvshmem_barrier_all_on_stream, nvshmem_create_tensor, nvshmem_free_tensor_sync)
    group = initialize_distributed()  # selects GPU LOCAL_RANK; the torch process group over all ranks, then NVSHMEM
    world, local = report_topology(torch, group, outputs)
    tutorial = definitions(tutorial_file('03-inter-node-allgather.py'))

    x = inputs['x']
    segment = x.numel() // world
    own = slice(rank * segment, (rank + 1) * segment)
    buffer = nvshmem_create_tensor((x.numel(), ), torch.int8)
    signals = [nvshmem_create_tensor((world, ), NVSHMEM_SIGNAL_DTYPE) for _ in range(2)]  # the tutorial: (1, )
    buffer.copy_(outputs['gathered'])  # the sentinel
    buffer[own].copy_(x[own])
    for signal in signals:
        signal.fill_(0)
    stream = torch.cuda.current_stream()
    nvshmem_barrier_all_on_stream(stream)  # every rank has cleared its signals and its buffer before a put can land

    context = tutorial.AllGatherContext(rank=group.rank(), node=rank // local, num_ranks=world,
                                        num_nodes=world // local, symm_signals=signals, signal_value=10)  # T03:271-280
    getattr(tutorial, 'all_gather_' + launch['kernel'])(context, buffer)
    nvshmem_barrier_all_on_stream(stream)  # the cross-node put of the 2D kernel does not block (T03:139)
    torch.cuda.synchronize()
    outputs['gathered'].copy_(buffer)
    print(f"rank {rank}: all_gather_{launch['kernel']} returned", flush=True)

    for tensor in (buffer, *signals):
        nvshmem_free_tensor_sync(tensor)
    finalize_distributed()


def reduce_scatter(rank, ranks, inputs, outputs, launch):
    import torch
    from triton_dist.kernels.nvidia.common_ops import BarrierAllContext
    from triton_dist.utils import (NVSHMEM_SIGNAL_DTYPE, finalize_distributed, initialize_distributed,
                                   nvshmem_barrier_all_on_stream, nvshmem_create_tensors,
                                   supports_p2p_native_atomic)
    group = initialize_distributed()
    world, local = report_topology(torch, group, outputs)
    tutorial = imported(tutorial_file('06-inter-node-reduce-scatter.py'))

    x = inputs['x'][rank].to(torch.bfloat16)  # T06:458; the batch keeps every value and every sum exact in bf16
    M, N = x.shape
    if launch['context'] == 'tutorial':
        context = tutorial.create_reduce_scater_2d_ctx(M, N, rank, world, local, x.dtype, overlap_with_gemm=False)
    else:  # T06:188-229 with `signal_bufs[rank]` of T06:202 taken within the node
        scatter, per_node, p2p, signal = (nvshmem_create_tensors(shape, dtype, rank, local) for shape, dtype in (
            ([M, N], x.dtype), ([M // local, N], x.dtype), ([M // local, N], x.dtype),
            ([2 * world], NVSHMEM_SIGNAL_DTYPE)))
        signal[rank % local].zero_()
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
        context = tutorial.ReduceScatter2DContext(
            max_M=M, N=N, rank=rank, world_size=world, local_world_size=local, dtype=x.dtype, overlap_with_gemm=False,
            symm_scatter_bufs=scatter, symm_rs_per_node_bufs=per_node, symm_p2p_bufs=p2p, symm_signal_bufs=signal,
            barrier=BarrierAllContext(True), reduction_stream=torch.cuda.Stream(priority=-1),
            p2p_stream=torch.cuda.Stream(priority=-1), num_sync_sms=0, num_p2p_sms=1, num_reduction_sms=15)
    # The tutorial's barrier asks NVSHMEM, not the launcher, which ranks share its node.
    team = [context.barrier.local_rank, context.barrier.num_local_ranks, int(supports_p2p_native_atomic())]
    outputs['barrier'].copy_(torch.tensor(team, dtype=torch.int32))
    print(f'rank {rank}: context {launch["context"]}, barrier local rank {team[0]} of {team[1]}, '
          f'native atomics {team[2]}', flush=True)

    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())  # T06:474
    reduced = tutorial.reduce_scatter_2d_op(x, context)
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())  # T06:479
    torch.cuda.synchronize()
    outputs['reduced'].copy_(reduced)
    print(f'rank {rank}: reduce_scatter_2d_op returned', flush=True)

    context.finalize()
    context.barrier.finalize()  # the tutorial leaves this tensor allocated (T06:490-492)
    finalize_distributed()
