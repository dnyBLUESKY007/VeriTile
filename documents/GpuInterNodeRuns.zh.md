[English](GpuInterNodeRuns.md) | [中文](GpuInterNodeRuns.zh.md)

# 跨节点程序在单机上的覆盖范围

Issue：[#353](https://github.com/dnyBLUESKY007/VeriTile/issues/353)。检查日期：2026-11-10；逻辑节点 launcher 于 2026-11-12 加入；首个逻辑节点结果以及 tutorial 03 和 06 的请求于 2026-11-12 记录。状态：源码检查完成。在这一台主机上启动两个逻辑节点的 launcher 已在 GPU 端运行：`internode-logical-env-002` 于 2026-11-12 返回，判定为通过（两个逻辑节点能够启动，二者之间的一次 put 到达）。针对 tutorial 03 和 06 的请求 `internode-logical-ag03-001` 与 `internode-logical-rs06-001` 已写好，尚未运行。Issue 允许的两种结果中，第一种不会成立（见下面的决定）。第二种，即“跨节点代码路径在这一台主机上运行”的检查记录，目前对启动和一次 put 成立，对任何教程 kernel 都还不成立。本文说明这种运行覆盖什么、不覆盖什么，并列出相关批次。

**决定（用户，2026-11-12）。** 不再安排第二个物理节点：合作者的集群已不可用。跨节点程序只以这一台主机上的逻辑节点方式测试。主机之间的传输（网卡、RDMA）不在项目范围内：模型里只有 rank 和 scope，没有节点或链路，证明不依赖它。如果逻辑节点无法运行，跨节点代码路径只做证明，并如实报告为未运行。

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

2026-11-12 加入。本节内容中，launcher 以及两个逻辑节点之间的一次 put 已在 GPU 上运行（`internode-logical-env-002`，见小节“首个请求及其结果”）；尚无教程 kernel 在该 launcher 下运行。launcher 还在没有 GPU 的开发机上用固定脚本的替身做了测试。凡涉及 torchrun、NVSHMEM 或 NCCL 的陈述，都是从所引源码读到的，或由其推断的，并逐条注明是哪一种；小节“只有 GPU 端运行才能确定的事项”说明其中哪些已被那次运行确认。

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

GPU 端安装的版本，据 `internode-logical-env-002` 的 `env.json`：PyTorch `2.8.0+cu126`、NVSHMEM 库 `3.6.5`、NVSHMEM 绑定 `0.3.0`、Triton `3.7.1`。未确认：NCCL 的版本，`env.json` 不报告它。

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
| Tutorial 06 | `nnodes=2`。启动 `kernel_inter_node_p2p_for_same_local_rank`（`n = 0` 时），结果取自 P2P 缓冲区，`ring_reduce` 在两个节点上进行。这需要一个上下文，而教程自己的构造函数在节点 0 之外给不出它（见小节“教程请求”）。 | 向另一本地 rank 的 `intra_node_scatter`：只有一个本地 rank 时它复制到该 rank 自己的缓冲区。 |
| 跨节点 all-to-all kernel | 显式传入 `local_world_size=1` 时 `nnodes=2`，因此 pull 循环会遇到 `src_node != node_id` 并等待另一节点的信号。只阅读了这几行；没有进一步追踪该 kernel。 | 接收节点上不同本地 rank 之间的转移。 |
| 跨节点序列并行 all-gather attention kernel | 其 push kernel 与 tutorial 03 的 2D kernel 结构相同：向相同本地 rank 的跨节点 put。只阅读了这几行。 | 节点内转发分支，与 tutorial 03 相同。 |

该 launcher 也能把四个 rank 启动为两个逻辑节点、每节点两个 rank（`ranks` 为 4、`logical_nodes` 为 2；已用替身测试），这会到达转发分支。但这需要在一个请求中使用四张 GPU；这台主机的规则是两张。小节“教程请求”按行列出 tutorial 03 和 06 的分支，以及阅读过程中的发现。

### 它不覆盖什么

- 网卡，以及 RDMA 或 IBGDA 传输：预期数据在一台主机内通过 GPU 点对点映射传递。
- 跨主机的 NVSHMEM bootstrap、NCCL 和 torchrun rendezvous：三者都在一台主机内连接。
- 主机之间的隔离：逻辑节点共用操作系统、主机名、文件系统（包括 Triton 缓存目录）和 CPU 内存。
- NVSHMEM 眼中的节点边界。NVSHMEM 把主机名哈希相同的 rank 放进同一个节点 team（`init_team_node`，tag `v3.6.5-0` 的 `src/host/team/team_internal.cpp:1106-1130`；2026-11-12 阅读，未运行），因此两个逻辑节点是同一个 NVSHMEM 节点。不读 `LOCAL_WORLD_SIZE`、而向 NVSHMEM 询问哪些 rank 与自己同节点的代码，看到的是一个包含全部 rank 的节点。Tutorial 06 的节点内 barrier 就是这样的代码（见小节“教程请求”）。
- 节点间网络的延迟、带宽和时序。
- 故障：节点丢失、链路丢失、网络分区。

因此，逻辑节点结果说明的是 `nnodes > 1` 分支的控制流和 rank 运算，以及发往另一节点 rank 的数据能够到达，所用传输与单节点运行相同。它是关于程序文本的证据，不是关于主机间通信的证据。`gpurun` 会给它打标签：`meta.json` 中有 `"topology": "logical-nodes"` 以及 `logical_nodes` 和 `local_ranks`；`requires.nodes` 和 `GPURUN_NODES` 保持为 1；如果批次要求多于一个物理节点，其中的逻辑节点测试返回 `skipped_env`。

### 只有 GPU 端运行才能确定的事项

七项中每一项之后是它在 2026-11-12、即 `internode-logical-env-002`（见下一小节）之后的状态。“已确定”指在那一个批次中观察到：两次重复，一对 GPU。

1. 同一主机上的两个 torchrun agent 能否在回环端点上汇合，固定端口 `23456` 是否空闲：共享主机上的其他作业，或上一次重复留下的 store，可能占用它。**对那次运行已确定：**两次重复中两个 agent 都在回环地址的端口 `23456` 上汇合，每个 rank 都启动了。端口是否空闲仍是每次运行各自的问题。
2. 每个 rank 是否得到预期的 `RANK`、`LOCAL_RANK`、`WORLD_SIZE` 和 `LOCAL_WORLD_SIZE`。**已确定：**各 rank 报告了预期的值（见下一小节的表）。
3. NCCL 能否在各自只能看到一张 GPU 的两个 rank 之间建立进程组。**已确定：**`initialize_distributed` 在两个 rank 中都返回了；它在全部 rank 上建立 NCCL 组并在其上执行一次 barrier（固定版本的 `python/triton_dist/utils.py:355-356`）。
4. NVSHMEM 的 UID bootstrap 能否跨两个 launcher 组完成。按源码，它只看到 `initialize_distributed` 传入的唯一标识、rank 和 rank 总数，看不到分组。**已确定：**NVSHMEM 在两个 rank 中都完成了初始化，其后的 barrier 和 put 也都完成。
5. NVSHMEM 能否通过 NVML 访问被隐藏的 peer GPU 并映射其堆，还是以 `Peer GPU … is not accessible` 终止；以及安装的 NVSHMEM 库是哪个版本。**结果已确定：**初始化没有终止，put 到达，日志中既没有该消息，也没有关于网络传输的警告。日志中看不出源码的哪个分支授予了映射。安装的库是 `3.6.5`（`env.json`）。
6. 发往另一逻辑节点 rank 的 put 是否到达，以及已安装绑定中的 `nvshmem.core.put` 是否接受 harness 的调用。**已确定：**每个 rank 都收到了另一 rank 的四个值。未确定，且那个批次没有涉及：固定版本中唯一的主机端 put，它走底层绑定，即 `python/triton_dist/kernels/nvidia/reduce_scatter.py` 中的 `pynvshmem.putmem_on_stream`（`reduce_scatter_for_each_node_ring`，位于 `nnodes > 1` 之下），调用时传入四个参数，而 NVSHMEM tag `v3.6.5-0` 的绑定声明了五个参数，最后一个是 stream（`nvshmem4py/nvshmem/bindings/nvshmem.pyx:1615`）。Tutorial 06 到达不了这个调用（见小节“教程请求”），所以它的请求也不会确定这一点。
7. 两个逻辑节点同时向同一个 Triton 缓存目录编译时是否互相干扰。**未确定：**环境批次没有编译任何 kernel。教程请求是最先编译的：每个 `logical_nodes` 测试的第一次重复中，两个逻辑节点同时编译同一个 kernel。

随教程请求新增，均未确定：

8. Tutorial 03 和 06 的 kernel 在已安装的环境中能否编译并运行：GPU 端还没有运行过其中任何一个。`one_node` 对照测试在逻辑节点之外回答这一点。
9. Triton 的 JIT 能否从一个只含该文件各项定义的模块中编译 tutorial 03 的 kernel；harness 正是这样避免运行教程的程序。开发机上只测试到 `inspect.getsourcelines`（JIT 用它读取 kernel）返回该文件的文本为止。
10. NVSHMEM 的节点 team 在逻辑节点下报告什么（按源码预期：这台主机上的全部 rank），以及使用它的 tutorial 06 节点内 barrier 能否跨两个逻辑节点完成。

### 首个请求及其结果

`internode-logical-env-001`（`experiments/interp_gpu/batches/internode-logical-env-001/`，harness 为 `experiments/interp_gpu/smoke/dist_logical_nodes.py`）在不使用教程 kernel 的情况下处理第 1 至 6 项。每个 rank 通过固定版本的 `initialize_distributed` 初始化；以张量形式报告自己的 rank、world size、`LOCAL_WORLD_SIZE`、推导出的节点数、`LOCAL_RANK` 和 CUDA 设备索引；打印设备名称；在两次 barrier 之间向另一 rank 的对称内存 put 四个值；然后结束。第一次 barrier 用来保证接收方初始化缓冲区之前不会有 put 到达。

该批次有两个使用同一 harness 的测试：`one_node/put_barrier` 使用 `triton-dist`，作为对照；`logical_nodes/put_barrier` 使用新的 launcher。这样可以把 harness 自身的失败与逻辑节点的失败区分开。开发端的预期：对照测试中 `LOCAL_WORLD_SIZE` 为 2，一个节点，设备索引等于 rank；逻辑节点测试中 `LOCAL_WORLD_SIZE` 为 1，两个节点，两个 rank 的设备索引均为 0；两个测试中，每个 rank 都收到 100 加上另一 rank 的编号。

结果。`internode-logical-env-001` 已在 GPU 端运行，每个测试均为 `ok`，但其结果包没有发布：launcher 的日志中有回环地址，而当时公开结果的规则不允许日志中出现地址。`gpurun` 现在把该地址写成 `<loopback>`，同一请求于 2026-11-12 以 `internode-logical-env-002` 再次运行（[issue #430](https://github.com/dnyBLUESKY007/VeriTile/issues/430)；请求 commit `ad26d295f4f8737ab82df8312a7eed21bbc4366c`；结果包为分支 `gpu-results` 的 commit `3ff8914868fa562c7dff6caac9344627732c4ce4` 中的 `results/internode-logical-env-002/`）。两个测试在两次重复中均为 `ok`，两个 rank 均为 `ok`，使用 NVLink 卡对 4–5。开发端解码的结果，两次重复相同：

| 测试 | Rank | `topology`（rank、world size、`LOCAL_WORLD_SIZE`、节点数、`LOCAL_RANK`、设备索引） | `received` |
| --- | --- | --- | --- |
| `one_node/put_barrier` | 0 | 0, 2, 2, 1, 0, 0 | 四个 101 |
| `one_node/put_barrier` | 1 | 1, 2, 2, 1, 1, 1 | 四个 100 |
| `logical_nodes/put_barrier` | 0 | 0, 2, 1, 2, 0, 0 | 四个 101 |
| `logical_nodes/put_barrier` | 1 | 1, 2, 1, 2, 0, 0 | 四个 100 |

每个值都与预期一致，因此该批次判定为通过：两个各含一个 rank 的逻辑节点经由固定的 launcher 在这一台主机上启动，每个 rank 只看到自己的 GPU，运行时推导出两个节点，发往另一逻辑节点上 rank 的 put 到达。上一小节说明这确定了哪些事项。

### 教程请求

写于 2026-11-12；两者都尚未运行。`internode-logical-ag03-001` 和 `internode-logical-rs06-001`（位于 `experiments/interp_gpu/batches/` 下）用 2 个 rank、在固定的小输入上调用 tutorial 03 和 06 的函数。`logical_nodes/` 下的每个测试（launcher `triton-dist-logical-nodes`，两个逻辑节点、每节点一个 rank）在 `one_node/` 下都有一个对照（launcher `triton-dist`），harness、输入和输出相同，这样可以把教程的失败与逻辑节点的失败区分开。每个 rank 返回自己的结果张量以及环境批次中的 `topology` 值。harness 是 `experiments/interp_gpu/internode_logical/harness.py`。同目录的 `judge.py` 用 Python 标准库从重新生成的输入计算预期张量，并按测试、rank 和输出逐项打印 `equal` 或 `different`；`test_internode_logical.py` 在没有 GPU 的情况下测试二者。没有修改或复制任何教程文件：harness 从构建该环境所用的 checkout 中读取它。这两个批次已提交但未公告：没有 issue，也没有标签。

`T03:n`、`T06:n`、`RS:n`、`CO:n` 和 `UT:n` 分别指固定版本中 `tutorials/03-inter-node-allgather.py`、`tutorials/06-inter-node-reduce-scatter.py`、`python/triton_dist/kernels/nvidia/reduce_scatter.py`、`python/triton_dist/kernels/nvidia/common_ops.py` 和 `python/triton_dist/utils.py` 的行，与 [dist/EventControlFlow.md](dist/EventControlFlow.md) 的用法相同。本小节的全部内容都是 2026-11-12 从源码读到的，或由其推断的；其中没有任何内容运行过。

**Tutorial 03，`internode-logical-ag03-001`。** 四个测试：kernel `push_1d` 和 `push_2d`，各在两个 launcher 下。输入：64 字节，每个 rank 相同；rank r 拥有第 32r 至 32r + 31 字节。一个 rank 的缓冲区起初是自己的分段，其余位置为 sentinel；一次 kernel 调用之后作为 `gathered` 返回。预期：每个测试为 `ok`，每个 rank 的 `gathered` 等于 64 个输入字节。

| Kernel 文本 | `one_node`（一个节点、两个 rank） | `logical_nodes`（两个节点、各一个 rank） | 留给两个节点、各两个 rank 的部分 |
| --- | --- | --- | --- |
| `all_gather_push_1d_kernel`（`T03:62-95`）；其中没有按节点数的分支 | 向另一 rank 的带信号 put（`T03:85-95`），以及对其信号的等待（`T03:72-76`）。 | 相同。 | 无。 |
| `all_gather_push_2d_kernel`，本地 rank 相同、rank 不同的 program（`T03:135-147`）：跨节点的 `putmem_signal_nbi_block` | 未到达。 | 到达：每个 program 都满足 `peer_local_rank == local_rank`，对应另一 rank 的 program 把本 rank 的分段 put 到另一节点。 | 无。 |
| 同一 kernel，对应本 rank 自身的 program（`T03:148-155`）：等待每个 peer 的信号 | 到达。 | 到达。 | 无。 |
| 同一 kernel，本地 rank 不同的 program（`T03:156-177`）：节点内的 `putmem_signal_block` | 到达，但没有 `T03:160-167` 的等待，因为 `peer_node_id == node_id`。 | 未到达。 | 在节点内转发之前对来自另一节点的分段的等待（`T03:160-167`）。 |

阅读过程中的发现：

- **该教程在 import 时运行它的整个程序。** 文件没有 `__main__` 保护：`T03:256-299` 启动运行时、生成随机数据、对两个 kernel 计时并停止运行时。因此 harness 只执行该文件顶层的 import、类和函数。它过滤语法树，并以该文件的名字编译，使 Triton 的 JIT 仍然在原来的行上读到未改动的 kernel 文本。harness 自己充当围绕这些定义的程序：每个测试一个 64 字节的缓冲区、一次 kernel 调用，而教程使用双缓冲区和一个计时循环。
- **信号张量：`dist/EventControlFlow.md` 第 8 节第 5 条观察的读法已由源码确认。** `T03:276-278` 以形状 `(1, )` 创建每个信号张量，即一个 8 字节的字（`UT:572`）；`T03:103` 和 `T03:185` 把其中一个作为 `symm_flag` 传入。1D kernel 对每个小于 `WORLD_SIZE` 且不等于本 rank 的 `peer` 等待 `symm_flag + peer`（`T03:68-76`），并对 `symm_flag + segment` 发信号，其中 `segment = rank`（`T03:81`、`T03:91`）。2D kernel 在 `T03:138-143`、`T03:149-154`、`T03:158-163` 和 `T03:173` 做同样的事。两个 rank 时，rank 1 设置 rank 0 上该张量的第 1 个字，rank 0 等待这个字：比教程分配的多出一个字。因此只要 `WORLD_SIZE > 1`，教程的程序就访问张量之外的内存，一个节点时如此，两个节点时也如此；分配器是否恰好不使用那个字，未知。本批次不运行那个程序。harness 为每个信号张量分配 `ranks` 个字，并明确说明这一点。没有选择按原样运行教程，是因为没有任何结果能显示这个缺陷：通过并不能否定它，而挂起、错误的字节或分配器报错都可能另有原因。

**Tutorial 06，`internode-logical-rs06-001`。** 文件有 `__main__` 保护（`T06:448`），整体 import。输入：每个 rank 一个 512 乘 128 的矩阵，元素为 -100 到 100 之间的整数，转换为 bfloat16，即教程的 dtype（`T06:458`）。按教程的 `__main__` 的方式调用 `reduce_scatter_2d_op`（`T06:410-433`），前后是相同的两次 barrier（`T06:474-479`），其返回值作为 `reduced` 返回。rank r 的预期：两个矩阵之和的第 256r 至 256r + 255 行。每个和都是绝对值不超过 200 的整数，bfloat16 可以精确表示，因此比较不带容差。该形状是使 `kernel_ring_reduce_tma` 的每个 tile（256 行乘 64 或 128 列，`RS:749-775`）都落在其张量之内的最小形状，与教程自己的 8192 乘 16384 情况相同；更小的矩阵还会顺带测试 tile 跨越张量边界时的行为，而那不是这里要测的。

| 文本 | `one_node`（一个节点、两个 rank） | `logical_nodes`（两个节点、各一个 rank） | 留给两个节点、各两个 rank 的部分 |
| --- | --- | --- | --- |
| 对节点的循环（`T06:314`） | 一轮：本节点。 | 两轮：先另一节点，再本节点。 | 无。 |
| `intra_node_scatter`（`T06:264-294`），主机端复制 | 复制到另一 rank 的缓冲区。 | 复制到本 rank 自己的缓冲区（`remote_local_rank == local_rank`）。 | 两轮中各有一次向另一本地 rank 的复制。 |
| `barrier_all_on_stream`（`T06:344`） | 节点内 barrier kernel，跨两个 rank。 | 同一 kernel，预期跨两个 rank，而真实的单 rank 节点只会跨一个（见下）。 | 无。 |
| 每个节点的 `ring_reduce`（`T06:349-354`） | 两个块求和。 | 一个块的复制（`num_splits` 为 1）。 | 每轮两个块求和。 |
| `nnodes > 1` 之下（`T06:357`），最后一轮（`T06:358-363`）：把本 rank 自己的结果复制到 P2P 缓冲区 | 未到达。 | 到达，第二轮。 | 无。 |
| `nnodes > 1` 之下，其余各轮（`T06:364-374`）：`kernel_inter_node_p2p_for_same_local_rank`，向另一节点上相同本地 rank 的一次 `putmem_block`（`T06:256-261`） | 未到达。 | 到达，第一轮。 | 无。 |
| 返回值（`T06:377-379`） | 每节点缓冲区。 | P2P 缓冲区。 | 无。 |
| 对节点的 `ring_reduce`（`T06:406`） | 一个块的复制（`num_splits = nnodes = 1`）。 | 两个块求和，其中一个收自另一节点。 | 无。 |

四个测试：`tutorial_context` 和 `harness_context`，各在两个 launcher 下。阅读过程中的发现：

- **教程的上下文构造函数在节点 0 之外会失败：一个预测的失败。** `create_reduce_scater_2d_ctx` 通过 `nvshmem_create_tensors(..., rank, local_world_size)` 得到信号张量，这是一个列表，调用者所在节点的每个 rank 对应一项（`UT:257-272`），随后用全局 rank 取 `signal_bufs[rank]`（`T06:199-202`）。在节点 0 的 rank 中，取到的是该 rank 自己的张量。在其他任何 rank 中，下标超出列表末尾：`IndexError: list index out of range`，此时还没有启动任何 kernel。因此按原样的 tutorial 06 无法在多于一个节点上运行，无论逻辑节点还是物理节点。本请求既暴露这一点，也绕开它，分别放在不同的测试中：
  - `*/tutorial_context` 按教程 `__main__` 的方式调用教程的构造函数（`T06:460-466`）。在 `one_node` 下预期能工作。在 `logical_nodes` 下预期 rank 1 抛出该 `IndexError`，随后 rank 0 预期在 `T06:205` 的 barrier 中等待它，直到该测试的超时 240 秒。`logical_nodes/tutorial_context` 的预期状态是 `timeout` 或 `runtime_error`，其中 rank 1 为 `runtime_error`，`stderr.log` 中有该 `IndexError`。judge 把这种情况称为 `failed as predicted`，其他情况称为 `prediction not met`。这个失败正是该测试想要的结果；批次的 notes 向 GPU 端说明了这一点。
  - `*/harness_context` 在 harness 中构造同样的 `ReduceScatter2DContext`，分配和取值与 `T06:188-229` 相同，只是那一个下标在节点内取（`rank % local_world_size`）。其后的一切都是教程的代码。`logical_nodes/harness_context` 就是到达表中 `nnodes > 1` 分支的那个测试。
- **到达不了 `putmem_on_stream`。** Tutorial 06 定义了自己的 `reduce_scatter_for_each_node` 和 `reduce_scatter_multi_node`（`T06:297`、`T06:382`），从库文件中只 import 了 `ring_reduce`（`T06:49`）。传入四个参数的 `pynvshmem.putmem_on_stream` 调用位于 `reduce_scatter_for_each_node_ring`（`RS:538-543`）。它唯一的调用者是库中的 `reduce_scatter_multi_node`，在 `has_fullmesh_nvlink()` 为假时调用（`RS:807-808`）；tutorial 08 会到达该函数，tutorial 06 不会。因此预期本批次不会出现由该调用引起的 `runtime_error`，本文的所有批次也都不会执行到该调用。
- **节点内 barrier 看不到逻辑节点。** 上下文持有 `BarrierAllContext(True)`（`T06:223`），它的本地 rank 和节点大小取自 NVSHMEM 的节点 team（`CO:238-240`），而不是 `LOCAL_WORLD_SIZE`。两个逻辑节点是同一个 NVSHMEM 节点（见小节“它不覆盖什么”）。因此在 `logical_nodes` 下，预期 `T06:344` 的 barrier 跨两个 rank 执行。两个 rank 调用它的次数相同，每轮一次，所以预期它能完成：对单 rank 节点而言，它同步的范围比文本的意图更大，而不是更小。每个 rank 在 `barrier` 中返回该 barrier 的本地 rank 和节点大小；两个 launcher 下的预期都是 r 和 2。
- **逻辑节点下原生原子操作是被假定的。** 当 `supports_p2p_native_atomic()` 成立时，`barrier_all_on_stream` 启动 compare-and-swap kernel（`CO:257-258`）。进程只看到一张 GPU 时，该函数不询问驱动就返回真（`UT:536-538`），单 rank 逻辑节点的每个 rank 都是这种情况；在 `one_node` 下它向驱动询问这对 GPU（`UT:544-547`）。该 kernel 的 docstring 禁止在 GPU 之间的链路没有原子操作时使用它（`CO:155-157`）。`barrier` 的第三个值是该函数的回答：在 NVLink 卡对上，两个 launcher 下预期都是 1（由该 docstring 推断，它把 PCIe 称为没有原子操作的情况）。如果 `one_node` 下为 0，就意味着 `logical_nodes` 测试在一对报告不支持原子操作的 GPU 上使用了原子操作。该请求应在 NVLink 卡对上运行，与环境批次相同。

**留给每节点两个本地 rank 的分支。** `T03:160-167` 的等待，以及 tutorial 06 中向另一本地 rank 的 scatter 再接 P2P kernel，需要两个逻辑节点、每节点两个 rank：一个请求四张 GPU。launcher 支持这种情况（`ranks` 为 4、`logical_nodes` 为 2）。建议的名称（没有 manifest）：`internode-logical-ag03-2x2-001` 和 `internode-logical-rs06-2x2-001`。四张 GPU 超出这台主机每个请求两张的规则，因此这样的请求需要先得到用户的许可。在这样的运行中，NVSHMEM 的节点 team 会包含全部四个 rank，所以 tutorial 06 的 barrier 同样不会遵循逻辑节点。

## 受影响请求与完成门槛

本文的批次中，`internode-logical-env-001` 和 `internode-logical-env-002` 已经运行，后者有已发布的结果。`internode-logical-ag03-001` 和 `internode-logical-rs06-001` 已作为 manifest 存在，尚未公告。其余 ID 是建议的名称，并非已存在的 manifest 或结果：

| 建议批次 | 计划检查 | 当前处置 |
| --- | --- | --- |
| `internode-ag03-single-host-001` | Tutorial 03 的双本地 rank 1D/2D 路径 | 没有单独的 manifest：`internode-logical-ag03-001` 的 `one_node` 测试就是这项检查。 |
| `internode-rs06-single-host-001` | Tutorial 06 的双本地 rank 路径 | 没有单独的 manifest：`internode-logical-rs06-001` 的 `one_node` 测试就是这项检查。 |
| `internode-a2av-single-host-001` | 显式使用 `local_world_size=2` 的 all-to-all 核心 | 等待 `gpurun` 及开发端适配器／manifest；现有测试假定 8。 |
| `internode-sp-attn-single-host-001` | 双本地 rank 序列并行 attention | 等待 `gpurun` 和 manifest；异节点分支仍未覆盖。 |
| `internode-ag03a-rocm-001` | 原样执行 tutorial 03a | H800 无法提供；需要 AMD 环境或明确审阅的移植。 |
| `internode-logical-env-001` | 两个逻辑节点能否启动；两次 barrier 之间一次 put；不含 kernel | 已运行，每个测试均为 `ok`；结果包未发布。由 `internode-logical-env-002` 取代。 |
| `internode-logical-env-002` | 同一请求，使用新的 ID | 已于 2026-11-12 运行；判定为通过。 |
| `internode-logical-ag03-001` | 在 `triton-dist-logical-nodes` 下运行 tutorial 03，两个逻辑节点、每节点一个 rank：2D all-gather 的跨节点分支；并在 `triton-dist` 下以相同测试作为对照 | manifest 已于 2026-11-12 提交；等待协调者公告。预期：每个测试为 `ok` 且相等。只覆盖逻辑节点。 |
| `internode-logical-rs06-001` | 在同一 launcher 下运行 tutorial 06：`kernel_inter_node_p2p_for_same_local_rank` 以及两个节点上的归约；并在 `triton-dist` 下以相同测试作为对照 | manifest 已于 2026-11-12 提交；等待协调者公告。预期：三个测试为 `ok` 且相等，`logical_nodes/tutorial_context` 按预测失败。只覆盖逻辑节点。 |
| `internode-logical-ag03-2x2-001` | Tutorial 03，两个逻辑节点、每节点两个 rank：节点内转发之前的等待 | 仅有名称。一个请求四张 GPU：需要用户许可。 |
| `internode-logical-rs06-2x2-001` | Tutorial 06，两个逻辑节点、每节点两个 rank：向另一本地 rank 的 scatter，再接 P2P kernel | 仅有名称。条件同上。 |

协议 v1 下，要求两个物理节点的请求仍应返回 `skipped_env`。不能把逻辑节点运行报告为双物理节点结果；其 `meta.json` 带有标签 `"topology": "logical-nodes"`。

本文不构成 #353 运行验收通过。在上述可执行请求产出结果、开发端记录比较与覆盖边界之前，或用户明确接受其他处置之前，Issue 保持开放。用户已明确要求首个正式批次等待 `gpurun`；本次检查不绕过这一依赖。
