[English](GpuInterNodeRuns.md) | [中文](GpuInterNodeRuns.zh.md)

# 跨节点程序在单机上的覆盖范围

Issue：[#353](https://github.com/dnyBLUESKY007/VeriTile/issues/353)。检查日期：2026-11-10；逻辑节点 launcher 于 2026-11-12 加入。状态：源码检查完成；在这一台主机上启动两个逻辑节点的 launcher 及其首个请求 `internode-logical-env-001` 已写好，尚未在 GPU 上运行。Issue 允许的两种结果目前都不成立：没有为 H800 主机安排第二个 sm_90 节点；“跨节点代码路径在该主机上运行”的检查记录仍需等待 GPU 端。本文说明这种运行覆盖什么、不覆盖什么，并列出相关批次。

## 硬件与范围

所检查的主机可见六张 H800 PCIe，计算能力均为 9.0。NVLink 卡对为 0–1、2–3、4–5。每个分布式请求计划只使用两张卡；本次环境检查使用当时空闲的 4–5 卡对。设备选择不代表资源预留，每次执行前都必须重新检查占用。

下述源码观察均针对 [Triton-distributed `7908e4ea9010bb2238f9f05b28c904fd4b71c60a`](https://github.com/ByteDance-Seed/Triton-distributed/tree/7908e4ea9010bb2238f9f05b28c904fd4b71c60a)。它们说明控制流和配置约束，不代表运行成功或数值正确。

普通单机双进程启动时，`WORLD_SIZE=2`、`LOCAL_WORLD_SIZE=2`、`NNODES=1`。名称带有 `inter_node` 的程序此时可能只执行退化的单节点路径，跳过它原本需要验证的跨节点通信。

## 源码发现

| 程序 | 单机两卡的源码评估 | 未覆盖部分或阻碍 |
| --- | --- | --- |
| [Tutorial 03](https://github.com/ByteDance-Seed/Triton-distributed/blob/7908e4ea9010bb2238f9f05b28c904fd4b71c60a/tutorials/03-inter-node-allgather.py) | 1D all-gather 通过 NVSHMEM 向另一 rank 发送。2D all-gather 接受 `NNODES=1`，执行节点内 peer 分支。尚未实测运行。 | `NNODES=1` 时不覆盖 2D 的相同 local rank、不同节点发送及其接收／转发顺序。 |
| [Tutorial 03a](https://github.com/ByteDance-Seed/Triton-distributed/blob/7908e4ea9010bb2238f9f05b28c904fd4b71c60a/tutorials/03a-inter-node-allgather.py) | 此固定版本不是 NVIDIA 教程：导入 `pyrocshmem`，使用 ROCSHMEM 设备上下文和信号操作。 | 无法原样在 H800/CUDA 环境执行。需要 AMD/ROCm 主机或单独审阅的 NVIDIA 移植。本次未移植。 |
| [Tutorial 06](https://github.com/ByteDance-Seed/Triton-distributed/blob/7908e4ea9010bb2238f9f05b28c904fd4b71c60a/tutorials/06-inter-node-reduce-scatter.py) | 代码明确支持 `nnodes == 1`，此时返回本地 reduce-scatter 缓冲区。尚未实测运行。 | 只有 `nnodes > 1` 才启动 `kernel_inter_node_p2p_for_same_local_rank`；普通单机双 rank 不会执行该 kernel。 |
| [跨节点 all-to-all kernel](https://github.com/ByteDance-Seed/Triton-distributed/blob/7908e4ea9010bb2238f9f05b28c904fd4b71c60a/python/triton_dist/kernels/nvidia/all_to_all_vdev_2d_offset_inter_node.py) | 核心上下文可以描述 `world_size=2, local_world_size=2`。现有测试调用 `create_context` 时未覆盖默认的 `local_world_size=8`。 | 现有测试直接使用双 rank 会违反 `world_size % local_world_size == 0`。开发端适配器必须显式传入 2。只有一个逻辑节点时跳过异节点接收等待分支。本次未修改测试或 kernel。 |
| [跨节点序列并行 all-gather attention kernel](https://github.com/ByteDance-Seed/Triton-distributed/blob/7908e4ea9010bb2238f9f05b28c904fd4b71c60a/python/triton_dist/kernels/nvidia/sp_ag_attention_inter_node.py) | 现有测试由 `WORLD_SIZE / LOCAL_WORLD_SIZE` 得到 `NNODES`，并将二者传入上下文。单机双 rank 配置执行节点内 all-gather 路径。尚未实测运行。 | `NNODES=1` 时不覆盖异节点 put 及跨节点接收／转发同步。 |

## 启动约束

固定版本的 `scripts/launch.sh` 识别 `--nproc_per_node=2`（必须带 `=`）。默认分支通过 `nvidia-smi` 统计物理设备，因此在这台六卡机器上仅设置 `CUDA_VISIBLE_DEVICES` 不够。后续开发端请求必须明确选择两张可见卡和两个进程。

不能只修改 `LOCAL_WORLD_SIZE` 来伪造拓扑：初始化会将 `LOCAL_RANK` 映射到 CUDA 设备，对称 peer 缓冲区的构造也依赖节点内分组。在一台物理主机上运行两个逻辑节点，需要明确的 launcher 与 batch 描述；下一节给出二者。即使它执行了异节点源码分支，也不能证明跨主机网卡、RDMA/IBGDA、隔离、延迟或故障行为。

## 单机上的逻辑节点

2026-11-12 加入。本节内容均未在 GPU 上运行。launcher 在没有 GPU 的开发机上用固定脚本的替身做了测试。凡涉及 torchrun、NVSHMEM 或 NCCL 的陈述，都是从所引源码读到的，或由其推断的，并逐条注明是哪一种。

### Launcher

`gpurun` 增加了第二个 `dist` launcher：`triton-dist-logical-nodes`，以及测试字段 `logical_nodes`（见 [GpuExchangeProtocol.md](GpuExchangeProtocol.md) 第 5、6、9 节）。它在同一台主机上为每个逻辑节点启动一次固定版本的 `scripts/launch.sh`，并通过该脚本本来就读取的多节点变量来驱动它。以 `ranks` 为 2、`logical_nodes` 为 2 为例：

| 传给逻辑节点 n（0 或 1）的设置 | 固定脚本如何使用（读自固定版本的 `scripts/launch.sh`） |
| --- | --- |
| `ARNOLD_WORKER_NUM=2`、`ARNOLD_ID=n` | 向 torchrun 传入 `--nnodes=2 --node_rank=n`。 |
| `--nproc_per_node=1` | 每个节点一个 rank。不带该参数时脚本用 `nvidia-smi` 统计物理 GPU。 |
| `ARNOLD_WORKER_0_HOST=127.0.0.1` | 传入 `--rdzv_endpoint=127.0.0.1:<port>`。该地址是 launcher 中的常量。端口沿用脚本的取值：`23456`，或操作员设置的 `ARNOLD_WORKER_0_PORT`。 |
| `CUDA_VISIBLE_DEVICES=` 操作员列表中的第 n 项 | 脚本不读取；由 torchrun 和 rank 进程继承。 |

其余设置保持脚本原样，包括 NVSHMEM 的设置：`NVSHMEM_BOOTSTRAP=UID`、bootstrap 网络接口、对称堆大小、`NVSHMEM_DISABLE_CUDA_VMM=1`。launcher 不需要接口名：两个逻辑节点都在同一台主机上，操作员已为 `dist` 批次选定了这台主机的接口。请求中不出现主机名、接口名或地址。

按 GPU 端报告的 PyTorch 版本（2.8.0）的 torchrun 源码（`torch/distributed/elastic/agent/server/local_elastic_agent.py`，以及 `api.py` 中的 `_assign_worker_ranks`），每个 rank 预期看到：`WORLD_SIZE=2`、`LOCAL_WORLD_SIZE=1`、`LOCAL_RANK=0`；节点 0 上 `RANK` 为 0，节点 1 上为 1。固定版本的 `initialize_distributed` 随后在两个 rank 中都调用 `torch.cuda.set_device(0)`，而设备 0 在两个 rank 中是不同的物理 GPU：即该 rank 在 launcher `triton-dist` 下使用的那张卡。程序由此得到 `nnodes = WORLD_SIZE // LOCAL_WORLD_SIZE = 2`。如果某个 rank 的 `LOCAL_WORLD_SIZE` 不等于 `ranks / logical_nodes`，`gpurun` 在调用 harness 之前就终止该 rank。

设计选择。另一种方案是自己写 launcher：自行启动每个 rank，并设置 `RANK`、`LOCAL_RANK`、`WORLD_SIZE`、`LOCAL_WORLD_SIZE`、torch 的 rendezvous 变量和 NVSHMEM 设置。最终选择为每个逻辑节点启动一次固定脚本，因为这样保留固定的 NVSHMEM 设置和 torchrun，新增代码约二十行。它依赖：

1. 脚本的多节点变量保持在固定版本中读到的含义；
2. torchrun 的静态 rendezvous 接受同一主机上的两个 agent：node rank 为 0 的 agent 在该端点上提供 store，另一个 agent 连接它（读自 `torch/distributed/elastic/rendezvous/static_tcp_rendezvous.py`；未运行）；
3. torchrun 把自身的环境变量（包括 `CUDA_VISIBLE_DEVICES`）传给每个 rank（读自 `torch/distributed/elastic/multiprocessing/subprocess_handler/subprocess_handler.py`）；
4. NCCL 和 NVSHMEM 能够访问本进程不可见的 GPU（见下一小节）。

每个逻辑节点只能看到自己的 GPU，与真实集群中的节点一致。如果 GPU 运行表明第 4 项不成立，退路是让每个逻辑节点都能看到全部 GPU，并把本节点的 GPU 排在最前：这样 NVSHMEM 和 NCCL 看到的 peer 与 `triton-dist` 下相同，但统计可见设备数的程序会看到比真实节点更多的设备。该退路尚未实现。

### 两张 GPU 之间预期使用的传输

本小节是根据源码得出的预期，不是观测结果。库源码在这里只用于预测运行能否启动，不定义任何语义。

来源（均于 2026-11-12 在 github.com 上阅读）：

- NVSHMEM：[`NVIDIA/nvshmem` 的 tag `v3.6.5-0`](https://github.com/NVIDIA/nvshmem/tree/6e53a40d9db16df170b3b0057e94c340d72f77eb)，commit `6e53a40d9db16df170b3b0057e94c340d72f77eb`。固定版本 Triton-distributed 的 `python/setup.py` 要求 `nvidia-nvshmem-cu12==3.6.5`。
- NCCL：[`NVIDIA/nccl` 的 tag `v2.27.3-1`](https://github.com/NVIDIA/nccl/tree/72d2432094d6ae36abd6e511c3a16a2d052dbf94)，commit `72d2432094d6ae36abd6e511c3a16a2d052dbf94`。PyTorch 2.8.0 对 CUDA 12.6 要求 `nvidia-nccl-cu12==2.27.3`（`.github/scripts/generate_binary_build_matrix.py`）。
- PyTorch：[`pytorch/pytorch` 的 tag `v2.8.0`](https://github.com/pytorch/pytorch/tree/ba56102387ef21a3b04b357e5b183d48f0afefc7)，commit `ba56102387ef21a3b04b357e5b183d48f0afefc7`。

此处未确认：GPU 端实际安装的版本。目前唯一返回的 `triton-dist` `env.json`（`env-check-triton-dist-001`）报告 PyTorch `2.8.0+cu126` 和 NVSHMEM 绑定 `0.3.0`；没有报告 NVSHMEM 库或 NCCL 的版本。

从源码读到：

- NVSHMEM 用 `gethostname()` 的哈希来标识主机（`src/host/util/util.cpp`，`nvshmemu_getHostHash`）。launcher 设置的内容都不参与该哈希。
- 它的 P2P 传输（`src/host/transport/p2p/p2p.cpp`，`nvshmemt_p2p_can_reach_peer`）不访问主机哈希不同的 peer。对哈希相同的 peer，它在本进程可见的设备中查找 peer 的 GPU。找到时，询问 `cudaDeviceCanAccessPeer`。找不到时（源码将其归因于 `CUDA_VISIBLE_DEVICES`），若 CUDA 驱动为 12 或更高版本，或堆不是 VMM 堆，则向 NVML 查询两个设备之间的 P2P 状态，读写均允许时授予映射。否则它发出警告，说明改用网络传输。
- 任何传输都无法到达的 peer 会使初始化以 `Peer GPU … is not accessible` 结束（`src/host/topo/topo.cpp`，`nvshmemi_build_transport_map`）。
- `initialize_distributed` 用 NCCL 建立进程组；NCCL 在主机哈希相同时把两个 rank 视为本地；对本进程不可见的 peer 设备，其 P2P 检查处的注释为 `CUDA 10.1 and later can use P2P with invisible devices`（`src/transport/p2p.cc`）。

推断：两个逻辑节点的主机哈希相同，因此预期 NVSHMEM 在两张 GPU 之间使用 P2P 传输，也就是普通单节点运行所用的、对 peer 对称堆的映射，经由该卡对的 NVLink。由于每个逻辑节点隐藏了对方的 GPU，这一结果经由 NVML 分支得到。预期没有网卡，也没有远程传输（IBRC、UCX、IBGDA、libfabric）承载数据。

### 逻辑节点运行覆盖什么

以下是对固定版本源码的阅读结果，针对两个逻辑节点、每节点一个本地 rank（`WORLD_SIZE=2`、`LOCAL_WORLD_SIZE=1`、`nnodes=2`）：

| 程序 | 两个逻辑节点、每节点一个 rank 时到达的部分 | 仍未到达的部分 |
| --- | --- | --- |
| Tutorial 03 | `NNODES=2`。在 `all_gather_push_2d_kernel` 中，每个 program 都满足 `peer_local_rank == local_rank`：对应另一 rank 的 program 执行跨节点的 `putmem_signal_nbi_block`，对应自身 rank 的 program 等待 peer 的信号。 | 节点内分支，包括先等待来自另一节点的分段再转发。它需要本地 rank 不同的 peer：两个节点各两个本地 rank，共四张 GPU。 |
| Tutorial 06 | `nnodes=2`。启动 `kernel_inter_node_p2p_for_same_local_rank`（`n = 0` 时），结果取自 P2P 缓冲区，`ring_reduce` 在两个节点上进行。 | 向另一本地 rank 的 `intra_node_scatter`：只有一个本地 rank 时它复制到该 rank 自己的缓冲区。 |
| 跨节点 all-to-all kernel | 显式传入 `local_world_size=1` 时 `nnodes=2`，因此 pull 循环会遇到 `src_node != node_id` 并等待另一节点的信号。只阅读了这几行；没有进一步追踪该 kernel。 | 接收节点上不同本地 rank 之间的转移。 |
| 跨节点序列并行 all-gather attention kernel | 其 push kernel 与 tutorial 03 的 2D kernel 结构相同：向相同本地 rank 的跨节点 put。只阅读了这几行。 | 节点内转发分支，与 tutorial 03 相同。 |

该 launcher 也能把四个 rank 启动为两个逻辑节点、每节点两个 rank（`ranks` 为 4、`logical_nodes` 为 2；已用替身测试），这会到达转发分支。但这需要在一个请求中使用四张 GPU；这台主机的规则是两张。

### 它不覆盖什么

- 网卡，以及 RDMA 或 IBGDA 传输：预期数据在一台主机内通过 GPU 点对点映射传递。
- 跨主机的 NVSHMEM bootstrap、NCCL 和 torchrun rendezvous：三者都在一台主机内连接。
- 主机之间的隔离：逻辑节点共用操作系统、主机名、文件系统（包括 Triton 缓存目录）和 CPU 内存。
- 节点间网络的延迟、带宽和时序。
- 故障：节点丢失、链路丢失、网络分区。

因此，逻辑节点结果说明的是 `nnodes > 1` 分支的控制流和 rank 运算，以及发往另一节点 rank 的数据能够到达，所用传输与单节点运行相同。它是关于程序文本的证据，不是关于主机间通信的证据。`gpurun` 会给它打标签：`meta.json` 中有 `"topology": "logical-nodes"` 以及 `logical_nodes` 和 `local_ranks`；`requires.nodes` 和 `GPURUN_NODES` 保持为 1；如果批次要求多于一个物理节点，其中的逻辑节点测试返回 `skipped_env`。

### 只有 GPU 端运行才能确定的事项

1. 同一主机上的两个 torchrun agent 能否在回环端点上汇合，固定端口 `23456` 是否空闲：共享主机上的其他作业，或上一次重复留下的 store，可能占用它。
2. 每个 rank 是否得到预期的 `RANK`、`LOCAL_RANK`、`WORLD_SIZE` 和 `LOCAL_WORLD_SIZE`。
3. NCCL 能否在各自只能看到一张 GPU 的两个 rank 之间建立进程组。
4. NVSHMEM 的 UID bootstrap 能否跨两个 launcher 组完成。按源码，它只看到 `initialize_distributed` 传入的唯一标识、rank 和 rank 总数，看不到分组；这是阅读结论，不是运行结果。
5. NVSHMEM 能否通过 NVML 访问被隐藏的 peer GPU 并映射其堆，还是以 `Peer GPU … is not accessible` 终止；以及安装的 NVSHMEM 库是哪个版本。
6. 发往另一逻辑节点 rank 的 put 是否到达，以及已安装绑定中的 `nvshmem.core.put` 是否接受 harness 的调用。教程是在 kernel 内部执行 put。固定版本中唯一的主机端 put 走底层绑定：`python/triton_dist/kernels/nvidia/reduce_scatter.py` 中的 `pynvshmem.putmem_on_stream`（`reduce_scatter_for_each_node_ring`，位于 `nnodes > 1` 之下），调用时传入四个参数，而 NVSHMEM tag `v3.6.5-0` 的绑定声明了五个参数，最后一个是 stream（`nvshmem4py/nvshmem/bindings/nvshmem.pyx`）。harness 调用的是 `nvshmem.core.put`，它会传入 stream。
7. 两个逻辑节点同时向同一个 Triton 缓存目录编译时是否互相干扰。这一点在教程批次中才开始有影响。

### 首个请求

`internode-logical-env-001`（`experiments/interp_gpu/batches/internode-logical-env-001/`，harness 为 `experiments/interp_gpu/smoke/dist_logical_nodes.py`）在不使用教程 kernel 的情况下处理第 1 至 6 项。每个 rank 通过固定版本的 `initialize_distributed` 初始化；以张量形式报告自己的 rank、world size、`LOCAL_WORLD_SIZE`、推导出的节点数、`LOCAL_RANK` 和 CUDA 设备索引；打印设备名称；在两次 barrier 之间向另一 rank 的对称内存 put 四个值；然后结束。第一次 barrier 用来保证接收方初始化缓冲区之前不会有 put 到达。

该批次有两个使用同一 harness 的测试：`one_node/put_barrier` 使用 `triton-dist`，作为对照；`logical_nodes/put_barrier` 使用新的 launcher。这样可以把 harness 自身的失败与逻辑节点的失败区分开。开发端的预期：对照测试中 `LOCAL_WORLD_SIZE` 为 2，一个节点，设备索引等于 rank；逻辑节点测试中 `LOCAL_WORLD_SIZE` 为 1，两个节点，两个 rank 的设备索引均为 0；两个测试中，每个 rank 都收到 100 加上另一 rank 的编号。该批次已提交但未公告：没有 issue，也没有标签。

## 受影响请求与完成门槛

本文中的批次都尚未运行。`internode-logical-env-001` 已作为 manifest 存在，尚未公告。其余 ID 是建议的名称，并非已存在的 manifest 或结果：

| 建议批次 | 计划检查 | 当前处置 |
| --- | --- | --- |
| `internode-ag03-single-host-001` | Tutorial 03 的双本地 rank 1D/2D 路径 | 等待 `gpurun` 和开发端固定 manifest；只覆盖单节点。 |
| `internode-rs06-single-host-001` | Tutorial 06 的双本地 rank 路径 | 等待 `gpurun` 和 manifest；不覆盖其跨节点 P2P kernel。 |
| `internode-a2av-single-host-001` | 显式使用 `local_world_size=2` 的 all-to-all 核心 | 等待 `gpurun` 及开发端适配器／manifest；现有测试假定 8。 |
| `internode-sp-attn-single-host-001` | 双本地 rank 序列并行 attention | 等待 `gpurun` 和 manifest；异节点分支仍未覆盖。 |
| `internode-ag03a-rocm-001` | 原样执行 tutorial 03a | H800 无法提供；需要 AMD 环境或明确审阅的移植。 |
| `internode-logical-env-001` | 两个逻辑节点能否启动；两次 barrier 之间一次 put；不含 kernel | manifest 已于 2026-11-12 提交；等待协调者公告。 |
| `internode-logical-ag03-001` | 在 `triton-dist-logical-nodes` 下运行 tutorial 03，两个逻辑节点、每节点一个 rank：2D all-gather 的跨节点分支 | 仅有名称；待 `internode-logical-env-001` 有结果之后。只覆盖逻辑节点。 |
| `internode-logical-rs06-001` | 在同一 launcher 下运行 tutorial 06：`kernel_inter_node_p2p_for_same_local_rank` 以及两个节点上的归约 | 仅有名称；条件同上。只覆盖逻辑节点。 |

协议 v1 下，要求两个物理节点的请求仍应返回 `skipped_env`。不能把逻辑节点运行报告为双物理节点结果；其 `meta.json` 带有标签 `"topology": "logical-nodes"`。

本文不构成 #353 运行验收通过。在上述可执行请求产出结果、开发端记录比较与覆盖边界之前，或用户明确接受其他处置之前，Issue 保持开放。用户已明确要求首个正式批次等待 `gpurun`；本次检查不绕过这一依赖。
