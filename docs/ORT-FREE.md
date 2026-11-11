# 全自研推理引擎（ORT-free）阶段一：算子普查、架构、GPT 侧替换

目标：去掉 onnxruntime 依赖，五张图的算子全部自己实现。本阶段做普查、架构定案与 GPT 侧（encoder、first、stage）替换；vits 侧与宿主集成是下一阶段，本阶段 vits 仍走 ORT + 现有 PackedConv，边界在文末标清。

硬线（用户定）：性能不降（GPT 侧逐段对现基线：解码 43–46 ms/步、first 2.3–4.1 s，x86 同机）；内存不许高于现基线（宿主稳态 RSS 270–318 MB 打包 + no-arena 口径，自研只许更低）；数值口径不放宽（tf_check 299/361 与 logits dump 对齐档）；可移植性沿用 `cpu.h` 运行时特性分发；打包格式、权重包、参考包、输出链一概不动；烘焙后路在新引擎里是同引擎跑 fp32 张量（权重 fp32 驻留、内核走非打包分支），保持可切换。

## 一、算子普查

对象：运行时实际执行的图。GPT 三图取打包形态（`host/packed_graphs/` 的 first/stage + `work/wo_graphs/w4/` 的 encoder，即 tf_check 的打包口径），另对照折叠模板（烘焙形态）统计；vits 取打包形态；prompt_encoder 已离线预计算、不在运行时，只列在末尾备查。

### 1. 节点与算子总表

| 图 | 节点数 | 初始器 | 算子种类 | 备注 |
|---|---|---|---|---|
| GPT encoder | 82 | 7 | 21 | 自包含（权重内嵌），含 VQ 码书距离计算 |
| GPT first（打包） | 1,748 | 194 | 32 | 全序列 prefill，S = x 长 + prompts 长 |
| GPT stage（打包） | 1,750 | 194 | 34 | 每步只算 1 个新位置（GEMV 形态） |
| GPT first/stage（烘焙模板） | 1,748 / 1,750 | 291 | 同上 | MatMul 125 + Gemm 24 取代 PackedMatMul 97 + MatMul 52，拓扑同构 |
| vits（打包） | 7,666 | 294 | 37 | 其中 PackedConv 218 + PackedConvTranspose 5 + PackedGather 2 已是本地内核 |
| prompt_encoder（离线） | 273 | 23 | 35 | 不在运行时，本阶段不管 |

GPT 侧（encoder + first + stage）算子种类合计 **39 种**（含烘焙形态的 Gemm 为 40 种）。全运行时（含 vits 打包形态）合计 **49 种**。prompt 独有的 If、Expand、Softplus、STFT、Tile、PRelu、Not 不在运行时，不列入实现清单。

### 2. GPT 侧算子清单（按 stage 图出现次数，first 近同）

热点标注：MAC 量为 stage 图每步静态推算（1 个新位置、T 为当前总长）；计时证据来自 STAGE3 第二、十节的同机实测。

| 算子 | first / stage 次数 | 类别 | 热点与实现 |
|---|---|---|---|
| PackedMatMul | 97 / 97 | 线性（打包） | 每步约 75.5 MMAC（24 层 × in 0.79M + out 0.26M + fc1/fc2 各 1.05M）+ lm_head 0.52M。**已有内核** `packed::gemm_a/b_w4/w8` 直接复用 |
| MatMul | 52 / 48 | 线性（激活×激活） | 注意力 QK^T 与 attn·V，约 2×T×512/层，随 T 增长；另含相对位置等小乘。需新写 fp32 GEMM（带步幅、批量） |
| LayerNormalization | 48 / 48 | 归一化 | 每层 2 次，行长 512，访存型。新写 |
| Softmax | 25 / 25 | 归一化 | 注意力 24 次（行长 T）+ 采样头 1 次（1025）。新写 |
| Relu | 24 / 24 | 逐元素 | MLP 激活，行宽 2048。新写（并入逐元素程序） |
| Add / Mul / Div / Sub / Neg / Pow / Sqrt / Sin / Cos / Cast / Where / Less / Greater / Equal / Or | 合计约 560 / 550 | 逐元素 | 含位置编码 sin/cos、logits 罚分链、掩码构造。统一用逐元素程序步执行（见架构节），标量与广播全覆盖 |
| Transpose / Reshape / Unsqueeze / Squeeze / Slice / Concat / Gather / GatherElements / ScatterElements | 合计约 640 / 660 | 布局与索引 | ORT 里是真拷贝与分发开销的大头；新引擎里 Transpose/Reshape/Unsqueeze/Squeeze/Slice 尽量做视图（步幅张量），Concat/Gather/Scatter 才是拷贝内核 |
| Shape / Constant / ConstantOfShape / CumSum / Identity | 合计约 530 / 560 | 元数据与常量 | Constant 全部编译期折叠；Shape 链产出仿射形状表达式，编译期符号求值；CumSum 只有位置编码与掩码两处是真数据计算 |
| ArgMax / TopK | 1–2 / 2–3 | 归约 | 图内采样头（TopK 15 → 阈值掩码 → Softmax → 除随机数 → ArgMax）只影响输出 y 的末位，该值随后被宿主采样覆盖；仍按原语义实现，RandomNormalLike 用引擎 RNG |
| RandomNormalLike | 1 / 1 | 随机 | 同上，ORT 本身跨运行也不复现此值（无固定种子语义依赖） |
| Conv | encoder 1 | 卷积 | ssl_proj：[768,768,2] 一维卷积，步幅 2。新写直卷 |
| Gather（嵌入） | encoder 7、first/stage 约 30 | 索引 | 词嵌入与位置索引查表，拷贝内核 |
| ReduceSum | encoder 2、stage 1 | 归约 | 码书距离与 stop 条件。新写 |

### 3. 热点占比（结论）

- **解码每步**：静态 MAC 约 78 MMAC（线性 76M + 注意力约 2M），实测 43–46 ms/步只合约 1.7 GMAC/s，说明 ORT 口径下步时由 1,750 个节点的分发、布局拷贝与 KV 张量逐歩传递主导，不是矩阵乘。静态计划 + 视图布局 + 静态 arena 的收益点正在这里。
- **first（prefill）**：线性约 S × 75.5 MMAC（S 为提示总长，tf 行 S 约百量级）+ 注意力 S² 量级，实测 2.3–4.1 s（宿主句口径）。GEMM 效率（现有 block 反量化 GEMM 复用）是主要变量。
- **encoder**：Conv 约 0.24–0.7 GMAC + 码书 MatMul [N,768]×[768,1024]，全程毫秒至百毫秒量级，非热点。
- **vits**：图内卷积总量 139.7 GMAC/样本（STAGE3 第十节逐节点实数），是全引擎绝对热点，本阶段不动（仍 ORT + PackedConv，128–147 ms/词为现值）。

### 4. 缺口清单（本阶段要新写的）

已可复用：PackedMatMul 四种模式（`packed::gemm_*`）、BlobStore 内容寻址池、内核线程池 `packed::kernel_pool`、cpu.h 分发。缺口 = 其余全部执行语义：

1. 图读取与计划编译（新文件，见架构节）。
2. fp32 GEMM/GEMV（带步幅与批量，MatMul/Gemm 共用）。
3. LayerNormalization、Softmax、ReduceSum、ArgMax、TopK、CumSum。
4. 逐元素程序（二元广播 + 一元链融合执行）。
5. 布局与索引：视图化 Transpose/Reshape/Unsqueeze/Squeeze/Slice，拷贝型 Concat/Gather/GatherElements/ScatterElements、ConstantOfShape。
6. Conv1d（仅 encoder 一处）。
7. RandomNormalLike（引擎 RNG 填充）。

vits 侧缺口（下一阶段）：其 37 种里除 3 种 potato 算子外，标准算子 34 种的执行替换（Pad、LeakyRelu、Tanh、Sigmoid、Exp、Split、Max 等），清单在文末阶段二估计里。

## 二、架构定案

采纳父代理倾向：**加载时把图编译成静态执行计划，不做通用图运行时**。理由：五张图的拓扑固定、形状只依赖少数运行时长度（stage 图只有一个 T），通用运行时的分发与形状推断正是现步时的大头；静态计划把这些成本全部前移到加载期。

### 1. 组件（host/src/ 新文件，不动现有出货文件）

- `onnx_model.h`：最小 protobuf 读取器，只覆盖 ModelProto/GraphProto/NodeProto/TensorProto/AttributeProto/TypeProto 的子集（含 raw_data 与分段字段、int64/float/string/ints/floats/tensor 属性）。图全部自包含、无外部数据，已核对。
- `free_engine.h` + `free_engine.cpp`：计划编译器 + 执行器 + 全部新内核。复用 `packed.h` 的 gemm 内核、BlobStore 与线程池（这些函数不引用 ORT 符号，新二进制不链接 libonnxruntime，以 ldd 为据）。
- `tf_check_free.cpp`：与 tf_check.cpp 同协议的验收二进制（encoder + first + stage 全程走新引擎），`--mode packed|baked`、`--dump/--ref` 语义与原版一致，逐行打印 init/steps 毫秒与 RSS。

### 2. 值模型（编译期的三种值）

- **Const**：具体张量（初始器、Constant 节点、以及输入全为 Const 的节点在编译期用同一套内核求值后折叠）。
- **SymInt**：int64 张量，其元素是运行时维度符号的仿射表达式（Shape 的输出、对其的 Gather/算术）。Slice 边界、Reshape 目标形、ConstantOfShape 形状等元数据消费点在绑定时求值成具体整数；仿射装不下的（如先符号后除法）退化为绑定时求值的小型表达式程序，仍不在运行期走通用推理。
- **RT**：运行时张量 = dtype + 每维仿射形状 + arena 分配。Transpose/Reshape/Unsqueeze/Squeeze/Slice（步长 1 时）编译成视图（基值 + 偏移 + 步幅），不产生数据搬运；内核对非连续输入走 `contiguous()` 按需物化。

### 3. 执行计划与绑定

- 编译产物 `Plan`：一串 PlanStep（内核 id、输入值 id、输出值 id、属性与预解析元数据），外加每个 RT 值的形状表达式与生命周期区间。
- 每次调用先 `bind`：代入具体输入形状，求出全部具体形状，按生命周期做空闲表分配，给出 arena 总大小与每值偏移；arena 按最大需求复用一块缓冲（解码逐歩 T 递增，每步重绑，成本是约千步整数运算，微秒级）。
- 图输出与跨步状态（present_k/v、y、y_emb）不进 arena，由调用方（宿主解码环 / tf_check_free）持有，与现行名绑定协议一致：stage 的输入名 past_* 接上一歩输出 present_*，宿主采样后覆写 y 末位再喂回。这保持了与现行 packed 宿主完全相同的状态语义。
- 死代码：不做裁剪，全部节点按图执行（图内采样头的随机分支只影响被覆盖的 y 末位，保留执行以保证输出集合完整、语义与 ORT 一致）。

### 4. 权重绑定与烘焙后路

- 打包模式：计划从打包图编译，PackedMatMul 步在绑定时按节点名查对应池（每图一池、共享 BlobStore，与现行 WeightPool 同一套字节），步内直接调 `packed::gemm_*`。
- 烘焙模式：同一计划（从烘焙模板图编译，拓扑与打包图同构），线性步绑定还原后的 fp32 张量（现行 restore 口径），走普通 fp32 GEMM 步。两模式共用执行器与全部非线性内核，切换只是绑定来源不同。
- 本阶段验收里烘焙模式用 wo_graphs/w4 的烘焙图（权重内嵌为初始器）直接编译，效果等同：计划里线性步绑定的就是 fp32 常量张量。

### 5. 内核分发

沿用 `cpu.h`：标量实现是地板且永远在；GEMM/GEMV 与逐元素热点按 `cpu::features()` 在进程启动时选 AVX2（x86）/ NEON（aarch64）实现，SDOT 只留分发点不实现（与第十节设备表一致）。内核并行用现有 `packed::kernel_pool`（默认线程数与现行一致），避免重蹈 ORT 线程与内核池抢核的覆辙：新引擎内只有一个线程池。

### 6. 与现有宿主的边界

本阶段不改 `main.cpp` 与出货默认布局：新引擎以 `tf_check_free` 独立二进制存在，验收口径与 tf_check 完全同构（同输入、同协议、同 dump 格式）。宿主集成（main.cpp 里以开关把 GPT 三个会话换成新引擎实例）是阶段二与 vits 替换一起做的事，本阶段只在架构上保证接口形状：`Plan::bind + run(名→张量)` 与现行 `packed::NamedSession::run` 同形，集成时一一对应替换。

## 三、GPT 侧替换结果

验收口径与 tf_check 完全同构（同输入、同教师强迫协议、同 dump 格式），ORT 与自研在本机同会话背靠背跑。注意本机当时处于高负载：ORT 自身解码约 183 ms/步（安静时基线 43 至 46），两边的绝对毫秒都被抬高，守线判断以同场比值与逐项对照为准，安静口径的绝对数留待真机标定与阶段二复测。

### 1. 数值对齐（全 6 行、361 步）

| 口径 | 自研 | ORT 对照 | 判定 |
|---|---|---|---|
| 打包一致率 | **299/361（82.83%）** | 299/361 | 逐行全同（57/68、59/68、42/48、57/72、42/58、42/47） |
| 打包 logits 对 ORT 同会话 dump | 最大差 **8.74e-05**，argmax **361/361** | ORT 打包对烘焙为 8.01e-05 | 同档 |
| 烘焙一致率 | **299/361** | 299/361 | 同上 |
| 烘焙 logits 对 ORT 烘焙 dump | 最大差 **5.53e-05**，argmax **361/361** | — | 1e-04 量级内 |
| A57（QEMU cortex-a57）第 1 行 | 57/68；对 x86 自研 dump 最大差 6.19888e-05 | 与 x86 自研对 ORT 的差同值 | A57 与 x86 逐位同数（只验对错，不测速） |

encoder 单独验收：prompts 与 ORT 逐位全等，x 最大差 1.4e-05。图内采样头产出的 y 末位是无种子随机抽样（ORT 自身跨运行也不复现），协议随即以教师 token 覆盖，不进对齐口径。

### 2. 计时（同机同会话，高负载口径）

| 段 | 自研 | ORT | 比值 |
|---|---|---|---|
| 解码 361 步合计 | 约 24.0 s（约 66 ms/步） | 65.9 s（约 183 ms/步） | **约 0.36** |
| init（encoder+first，逐行） | 4.0 至 9.1 s | 6.5 至 12.8 s | 每行都更快 |

按 ORT 安静基线 43 至 46 ms/步等比折算，自研投影约 16 ms/步，低于 46 ms 守线（投影，非实测，安静复测在阶段二与真机标定做）。first 的安静守线 4.1 s 同理以比值领先。

### 3. 内存

- 打包全程 RSS：自研 **315.7 MB**，在 270 至 318 MB 基线带内；ORT 测试台同口径 272 至 274 MB。自研高出的部分是结构性的：双 stage 计划零拷贝 ping-pong 持有两份 KV 输出 arena，ORT 测试台是单会话加逐歩拷贝出口。宿主口径（270 至 318 的原口径）的同构复测在阶段二集成时做。
- 烘焙全程 RSS：自研 1,345 MB（ORT 烘焙约 1,277 MB），多出的是双 stage 计划各持一份图内常量。烘焙后路按定案就是内存宽裕机器的形态，只要求可切换，不守打包 RSS 线。
- 权重池唯一字节 44,173,824 B，与 BlobStore 既有数字一致；`ldd tf_check_free` 无 libonnxruntime，确为真去 ORT。

### 4. 本阶段动过的性能点（数值不变，已逐项复验）

Reshape 对非连续 base 改为执行期拷贝步（同时是正确性修复：原 bind 期物化拷的是未执行缓冲）；copy_tensor 与逐元素改增量步幅漫游、大张量池并行；MatMul 跨 batch 单次池调用且不再强制连续化操作数；LayerNormalization 与 Softmax 小工作量改串行避池屏障。PackedMatMul 内核与线程池原样复用，其每步耗时与 ORT 路径同级（约 19 至 21 ms/步，高负载口径）。

### 5. 边界

vits 本阶段未动，仍走 ORT 加现有 PackedConv（宿主现形态）；新引擎以 `host/build/tf_check_free` 独立二进制与 CMake 新目标 `tf_check_free` 并行存在，出货默认布局未动。为让新引擎能单独链接，`packed.h` 拆为 `packed_core.h`（ORT 无关的池与内核）加原 ORT 胶水，行为不变，现有全部 CMake 目标已重编通过。

## 四、vits 侧下一阶段工作量估计

对象：`host/packed_graphs/vits_wnfold.onnx`，7,666 节点、294 个初始器、37 种算子。阶段一的计划编译器与执行器原样复用，工作量分四块：

1. **本地内核换宿主（小）**：PackedConv 218 + PackedConvTranspose 5 + PackedGather 2 共 225 个节点的内核数学已在 `vits_kernels.cpp` 里且与 ORT 无关，要换的只是算子胶水：把 ORT 自定义算子的入出口改成新引擎的步接口（含 PackedConv 的权重描述符解析，与 GPT 侧 FreePool 同路）。无新算法。
2. **标准算子补缺（小到中）**：vits 的 34 种标准算子里，逐元素族、Gather、Concat、视图族、CumSum、Softmax 等已在阶段一实现。真正新增的只有：Pad（常数/反射/边缘模式）、LeakyRelu（带斜率的一元，挂进一元链）、Split（视图，近零成本）、Expand（广播物化，一种拷贝步）、ReduceMean（ReduceSum 步加均值变体）、fp32 ConvTranspose（仅烘焙后路需要，打包口径走 PackedConvTranspose）。合计约 6 种新步型。
3. **编译与内存（中，是主要风险）**：7,666 节点里大半是 flow 与时长预测器的小算子链，没有新数学，风险在计划编译时间与静态 arena 规模。参照 stage 图 1,750 节点编译约 0.3 s，vits 估计 1–2 s 量级；中间张量是 [192, T] 量级的 flow 激活，arena 估计数十 MB 级。时长预测器含随机噪声输入（RandomNormalLike 已实现），与 ORT 口径一样波形不可逐位比对。
4. **验收（中）**：尺子现成。用 vits_check 的逐层相关性口径对 ORT 路径逐段比对，守全图相关性 0.9599（混合包 E 基线）；最终以 samples-final 听感集复核。速度上要分开报：x86 开发机上 ORT 的 MLAS 有 AVX-512 宽度优势（第十节的 2.5 倍差），新引擎的通用 GEMM 在 x86 上追平不作承诺；目标机 A57 两侧同为 4 宽 NEON 级，该差距结构上不存在，真机数以 phone-calib 回传为准。STAGE3 第十节悬置的 vits 速度裁决在新引擎里按此口径落：打包口径不变、内核在新引擎内重测、A57 为准。

集成形态：`main.cpp` 加 `--engine free|ort` 开关，GPT 三图先换（接口已同形），vits 最后换；切换默认等两边都过验收。集成时按宿主口径重测稳态 RSS 对 270–318 MB 基线（本阶段 tf_check_free 的 315.7 MB 是测试台数字：双 stage 计划零拷贝 ping-pong 持有两份 KV 输出 arena，ORT 测试台是单会话加逐歩拷贝出口，同口径对比要到宿主集成时才成立）。工作量估计：实现约半个阶段一，验收与集成约半个阶段一，架构与编译器不用再动。

## 五、阶段二结果：vits 替换与宿主集成

### 1. 实现清单

- **新步型**：PCONV（PackedConv 与 PackedConvTranspose 共用，pc_transpose 区分）、PGATHER、CONVT（fp32 ConvTranspose）、PAD、EXPAND；LeakyRelu 挂一元链（alpha 随步携带）；Split 实现为视图（view_kind 7，按轴偏移，与 Slice 视图同路在 bind 期解析）；ReduceMean 复用 REDUCESUM 步加均值缩放，含 noop_with_empty_axes 恒等情形。本地三内核只换胶水不换数学：PCONV/PGATHER 执行期经计划已有的 PackedLookup 解析 PackedBinding，以与 ORT 胶水完全相同的参数调用 `packed::conv1d_fast` / `conv_transpose1d_fast` / `gather_packed`。为让 free 二进制不被 ORT 胶水拖入链接，内核声明拆到新头 `vits_kernels.h`（与 packed_core 同样处理），`gather_packed` 定义移入 `vits_kernels.cpp`（逐字搬家）。
- **符号层为 vits 图付的账**（每项都是编译阻塞，逐个实证后修）：Slice 统一 `slice_count` 钳制语义（负步长与 INT64 哨兵值；顺带修掉常量折叠 INT64_MIN/−1 的 SIGFPE，ebin 折叠改 __int128 加饱和）；符号 Concat 接受 SYMVAL 与 CONSTV 混合输入；符号 Gather 处理 0 维标量索引（删轴）；表达式加 MAXK/MINK，Max/Min 与常指数 Pow（展开为连乘，vits 注意力 reshape 目标用 Pow(dim, 2)）进符号求值；ConstantOfShape 全常量规格折叠为 CONSTV（注意力 pads 链的规格是 [0]，不折叠会毒化其后的符号 Concat）。
- **验收工具**：`vits_check_free`（新）与 `vits_check` 同口径——同 feed（`work/memopt/vits_feed.npz`）、同 probes 图、逐层记录格式；`vits_check` 加 `--dump` 产出 ORT 侧逐层基准。FreePool（per-graph 描述符加共享 BlobStore）从 tf_check_free 抽为共享头 `free_pool.h`。
- **通用内核三处性能修复**（数值逐项复验，见第 2 节的逐位对照）：k_conv 重写为直接卷积（sp=1 逐行 SAXPY 形态，与 conv1d_fast 同构；原 im2col 在最内层循环反复分配 vector，编码器的 ssl_proj 单次调用约 540 ms）；k_convtranspose 加相位分解快速路径（与 conv_transpose1d_fast 同构，原 gather 式每个输出元素做上千次整除判定，5 个节点共 145 s）；k_matmul 大 M 行分块（B 行段在 16 行 tile 内复用，prefill 不再每行重读整个 B）。
- **宿主集成**：`tts_host --engine free|ort`（默认 ort，ORT 路径原样保留）。free 模式为打包驻留同构：GPT 编码器（fp32 图）加 first/stage 双计划加 vits 打包计划，权重走 FreePool 三池共享一个 BlobStore；`--vits-baked` 在 free 模式下跑烘焙 fp32 vits 图，权重由 `patch_graph_inplace` 在图内替换为 E 包还原张量（与 ORT 的 load_and_patch 同一组值，不经 protobuf 字节手术）。请求循环的采样、补 y 末位、输出链（deess2air 加软削波）与 ORT 分支共用同一份代码。

### 2. 数值验收

| 口径 | 结果 | 基线/对照 | 判定 |
|---|---|---|---|
| vits 逐层 free 打包 对 ORT 打包 | 226 层全比，worst corr **1.000000**，worst maxerr 0.00119 | — | 同一内核加同档胶水 |
| vits 逐层 free 烘焙 对 ORT 烘焙 | 226 层全比，worst corr **1.000000**，worst maxerr 0.00143 | — | 通用内核对 MLAS 同档 |
| vits 逐层 free 打包 对 free 烘焙 | worst corr **1.000000**，worst maxerr 0.00155 | ORT 打包对烘焙 maxerr 1.96e-05（音频级） | 逐层为累加顺序档 |
| vits 全图音频对 fp32 参考 | 打包 corr **0.959880**、烘焙 **0.959881** | ORT 打包 0.959881、烘焙 0.959880；守线 0.9599 | 守住（第六位小数量级互换） |
| GPT 回归（内核改动后复跑） | 打包与烘焙均 **299/361**；logits 对 ORT dump 最大差打包 **8.73804e-05**、烘焙 **5.53131e-05**，argmax **361/361** | 与阶段一数值逐位相同 | 不退 |

自由跑贪心不作跨引擎口径：实测自研引擎跨进程、跨请求可逐 token 复现，且与 ORT 进程的首跑逐 token 相同；ORT 路径自身在同进程内对同一贪心请求两次运行即漂移（首 token 起分岔，源头是编码器会话的多线程规约不确定）。教师强迫 logits 仍是唯一对齐口径。

### 3. 编译时长与 arena

第四节的估计是编译 1 至 2 s、arena 数十 MB。实测：

- 编译：出货图在宿主内 enc 23 ms、first 4 至 88 ms、stage 4 至 63 ms、vits 15 至 106 ms（随当时负载浮动，量级一致）；probes 图在验收工具内打包 27 至 34 ms、烘焙 0.94 至 1.07 s。烘焙慢在把 466 MB fp32 初始器拷进计划常量快照，打包图只有桩初始器所以近瞬时。估计对打包偏高一个量级、对烘焙落在下沿，原因即在此。
- arena：probes 计划报 778 MB 是验收形态的假象——226 个探针输出全是图输出、全部存活到运行结束，arena 必须同时持有每一层输出。出货图在宿主内的真实 arena（跑过实际请求后）：enc 1.6 MB、first 51.8 MB、stage 每份 22.6 至 22.7 MB、vits 49.7 MB，合计约 148 MB——数十 MB 的估计对单个计划成立，五计划同驻才到这个数，处置见第四节。

### 4. 宿主口径 RSS 与 arena 处置

第一轮集成（arena 全常驻）稳态 393 至 422 MB，超基线带。账算清后处置如下：

- **enc / first / vits：请求内用完即释放**（新增 `Plan::release_arena`）。三者的峰值出现在请求的互斥阶段：enc 在 first 跑完后释放，first 在解码结束后释放，vits 在音频拷出后释放。下个请求重新按形状长出来，成本是毫秒级的页触碰。稳态不再含它们的峰值，运行峰值也只剩当时阶段的那一份。
- **stage ping-pong：保留双份**（22.6 + 22.7 MB）。双计划零拷贝交接状态是阶段一选定的形态；改单份要每步把约 18 MB 的 KV 状态拷进输入暂存，90 步一句约 1.6 GB 拷贝量，直接顶撞解码速度守线。用 22.7 MB 常驻换每步零拷贝，这笔账成立。
- 结果：稳态 RSS **262 至 280 MB**（首请求 262、其后稳定 280），落在 270 至 318 基线带内；运行峰值（vits 阶段：boot 193 + stage 双份 45.4 + vits 49.7 加瞬态）在 290 MB 量级，与采样值相符。tf 测试台口径（不释放、双 stage 全驻）313 至 316 MB，与阶段一的 315.7 一致。退出时诊断行 `[free] final arenas` 可直接读各计划常驻（enc/first/vits 应为 0）。

### 5. 逐段计时（x86 开发机，安静口径）

条件：4 核开发机，整批顺序单跑、无并发任务，负载均值从 2.0 降至 1.05（批内进程即主要负载源）；宿主 `--threads 2`（出货配置），验收工具 `--threads 4`。ORT 对照为同机同批背靠背。

宿主逐段（ORT 打包 对 自研，4 句均值域）：

| 段 | ORT 打包 | 自研 free | 比值 |
|---|---|---|---|
| encoder | 3 至 5 ms | 48 至 51 ms | 通用 GEMM/卷积对 MLAS 的残差（修前 300 至 420 ms） |
| first | 3.19 至 3.23 s | 3.48 至 3.65 s | 约 1.10，在第十节 ORT 基线带 2.3 至 4.1 s 内 |
| 解码每步 | 34.5 至 37.1 ms | 48.1 至 50.6 ms | 约 1.4 |
| vits 每句 | — | — | 同步数句约 1.09 至 1.15（按下行 token 口径） |
| 稳态 RSS | 292 至 315 MB | 262 至 280 MB | 自研更低（arena 释放策略） |

验收工具口径（同 feed，pred 75 token，threads 4；每 token 毫秒 = run 毫秒 / 75）：

| 配置 | ORT run | 自研 run | ORT 每 token | 自研每 token |
|---|---|---|---|---|
| vits 打包 | 10.97 s | 12.15 至 12.48 s | 146 ms | 162 至 166 ms |
| vits 烘焙 | 5.19 s | 19.44 s | 69 ms | 259 ms |

守线读法：打包 vits 的底线（第十节 128 至 147 ms 每词，与本批 ORT 实测 146 同口径）在 x86 上自研约慢 11 至 14%——卷积内核是同一份（PCONV 占打包 run 约 84%），差在通用逐元素与视图胶水；ORT 的这点优势里含 MLAS 的 AVX-512 宽度，目标机 A57 两侧同为 4 宽 NEON 级，该成分结构上不存在。烘焙 vits 自研对 ORT 为 3.7 倍（同为 MLAS 宽度差；相位分解修复前约 73 倍，病理已除）。解码每步同理：PACKED 内核两边共用，自研慢在注意力 MatMul 与胶水，x86 约 1.4 倍。**x86 上自研未全面守住「不慢于 ORT」，差距构成是已知的 MLAS 宽度与胶水两笔；是否成立按 A57 与真机数裁决，不以 x86 单方数据切默认。** tf 测试台解码合计自研 54.8 ms/步（2 线程）与 55.7（4 线程），带宽瓶颈、加线程不涨。

### 6. A57（QEMU）验证

qemu-aarch64-static `-cpu cortex-a57` 加 Alpine 3.24.4 rootfs，只验对错不测速（本节不出现速度数字）：

- **tf_check_free 打包全量**：一致率 **299/361**，与 x86 ORT 打包 dump 的 logits 最大差 **8.65459e-05**、argmax **361/361**——与 x86 自研对同一 dump 的 8.73804e-05 同档，A57 与 x86 同数。
- **宿主端到端冒烟**：`tts_host --engine free` 贪心单句（S0）全程跑通（GPT 加 vits 加 deess2air 加软削波），token 序列与 x86 自研贪心**逐条相同**（79 个含 EOS），音频时长逐位相同（3.09772 s），输出链电平对齐（peak_in 0.591808 对 0.59181、drive 2.02768 全同）；稳态 RSS 269.8 MB（QEMU 进程口径，仅作量级参考）。
- 产物在 `host/dist-aarch64-free/`（构建说明同目录 BUILD.md）：tf_check_free 与 vits_check_free 不链 libonnxruntime，tts_host 因保留 ORT 路径仍链系统 1.24.4。验证所用二进制与最终源码仅差 fp32 ConvTranspose 的速度路径（打包模式不走该路径，两项验证均不受影响），入库产物已按最终源码重建。

### 7. 全引擎总账与设备定位

- 覆盖：五图（GPT 编码器、first、stage、vits 打包与烘焙）全部可在自研引擎运行，运行时算子 49 种全覆盖；`tf_check_free`、`vits_check_free` 的 `ldd` 均无 libonnxruntime。宿主单二进制仍链 ORT（`--engine ort` 默认路径保留），引擎层已无 ORT 依赖。
- 内存总账：权重池唯一字节 87,123,996 B（GPT 加 vits，与第十节一致）；宿主稳态 262 至 280 MB 在基线带内；烘焙模式 RSS 仍为 GB 级（烘焙 vits 计划常量快照 466 MB 起），定位不变——内存宽裕机器的后路。
- 五档设备定位：**无变化**。cpu.h 运行时分发沿用（标量地板、AVX2/NEON 档、SDOT 与 i8mm 只留分发点）；本阶段新写的通用内核（直接卷积、相位转置卷积、MatMul 分块）都是标量可编译、靠自动向量化的形态，没有引入任何新的 ISA 要求，x86-64 桌面仍以 AVX2 为基线。打包格式、参考包、输出链未动。
- 标定包自研变体：`phone-calib-free/` 与 ORT 版并列（数据逐字节相同，bin 换 aarch64 自研构建，脚本走 `--engine free` 并多跑一段 tf_check_free 全量复核）；**默认仍是 ORT 版 `phone-calib`**。
- 切默认建议：数值与内存两关已过；速度关在 x86 上是「打包同档偏慢一成、解码慢四成、烘焙慢 3.7 倍」，在 A57 上预期大幅收窄但未经真机证实。建议先在真机跑 `phone-calib-free` 与 ORT 版对拍：若 A57 上解码与 vits 的 RTF 在 ORT 的一成以内（或更好），再切默认到 `--engine free`；否则默认保留 ORT，自研作可选，同时按 profile 啃注意力 MatMul 与逐元素胶水。等用户点头后执行。

## 六、x86 提速与合并标定包

### 1. Profile 定位（先量后动）

给自研引擎加了按计划分开的 FE_PROFILE 统计（环境变量开启，默认关，不影响数值），在 tf_check_free 打包模式、4 线程下量解码每步的构成；再用 vits_check_free 量 vits。本轮开发机的负载全程不稳（load average 在 0.3 到 9.8 之间跳，格子上限 1.5 个共享 CPU），绝对毫秒普遍膨胀，下面的构成比与同窗 A/B 才是可靠证据，绝对值只作量级参考。

解码 stage 计划每步构成（基线版，负载膨胀单位）：PACKED 约 50%（打包 GEMV，自研与 ORT 共用同一套内核）；MATMUL 约 21%，全部是注意力小乘积（16 头、头维 32、T 约 177，每层每步只有约 9 万 MAC）；BINOP 约 21%（逐元素，含采样头与 KV 拼接周边）；UNOP 约 2%；CONCAT、COPY、SOFTMAX 合计约 1.5%。形状实测（FE_MM_DEBUG 转储）：scores 乘积的 B 是 K 的 [头, 32, T] 布局，last-two 步长 (T, 1)，不匹配已有的转置 GEMV 快路；attn·V 乘积的 B 是 [T, 512] 缓冲的交错视图（头步长 32、行步长 512）。每个注意力乘积只有约 1e5 MAC，而引擎的内核线程池一次汇合的成本在争用下远超乘积本身，这是 MATMUL 占两成的真实构成：不是算得慢，是排队贵。

vits 打包计划构成：PCONV 84.1%（223 次调用，与 ORT 共用的 PackedConv 内核）；其余胶水约 16%，其中 PAD 2.2%（96 次调用，逐元素里程表走法）、UNOP 4.0%、BINOP 2.8%。按安静窗折算，非 PCONV 胶水约 26 ms 每词，对 ORT 的约 23 ms 每词同档；第五节记的 vits 差距（自研 162 至 166 对 ORT 146）主要在 PCONV 内核位宽（本机 MLAS 走 AVX-512，自研打包内核走 AVX2），不在胶水。

first 计划每步构成与 stage 同型（PACKED 约六成、MATMUL 与 BINOP 各约一成）。encoder 只有 6 次 Conv，安静窗 3 至 5 ms，不是热点。

### 2. 动手改动（只动自研引擎通用路径，不动数值口径）

1. MatMul 小乘积串行化：总 MAC 小于 2^20 的 MatMul 不再进线程池，直接在调用线程算（注意力乘积约 1e5 MAC，prefill 的大乘积不受影响）；转置 GEMV 快路从单 batch 推广到任意 batch。
2. 逐元素改 float 求值：BINOP 与 UNOP 的热路径改用 float 运算（ORT 本身也是 float 求值），此前自研按 double 求值，多一倍带宽与转换。
3. 逐元素按行组织：输出最后一维连续、操作数最后一维步长为 0 或 1 时，按行做一维向量化循环，外层偏移每行只分解一次；池化门槛从 65536/8192 元素提到 2^17，stage 的中等张量（约 9 万元素）不再付池汇合。
4. k_pad 行快路：f32 且输入输出都连续时，最后一维按段 memcpy 加短填充（constant/edge/reflect），原逐元素里程表版留作兜底。
5. 诊断保留：FE_PROFILE 改为按计划分开输出，另加 FE_MM_DEBUG 形状转储，两者都是环境变量开启、默认关。

打包格式、权重包、参考包、输出链、cpu.h 运行时分发（AVX2 基线）均未动；打包 GEMV 与 PackedConv 内核未动（与 ORT 共用，动了就不是同口径对照）。

### 3. 数值复验（改后全量）

- tf_check 打包：ORT 299/361、自研 299/361；烘焙：ORT 299/361、自研 299/361。
- logits 最大差（自研对 ORT 同轮 dump）：打包 8.87e-05、烘焙 5.15e-05，argmax 均 361/361，与第五节的 8.74e-05、5.53e-05 同档（1e-4 量级）。自研改后对改前自身的 dump 最大差 6.87e-05，是 float 求值改动的舍入量级。
- vits 全图相关性：自研打包 0.959880、烘焙 0.959881，与第五节逐位同数；逐层对 ORT 最差相关性 1.000000（打包与烘焙各 226 项）。

### 4. 计时结果（如实口径）

同窗 A/B（同一二进制前后版交替跑，负载鲁棒的部分）：stage 注意力 MATMUL 从每步 39 至 68 ms（膨胀单位）降到 1.7 ms，约 25 至 40 倍，这一档差距确认消除；UNOP 从归一化约 5.3 ms 每步降到约 0.85 ms 每步。BINOP 同窗对比在噪声内（每步 46 至 79 ms 膨胀单位，两版重叠），float 化与按行化的收益没能从负载噪声里分离出来，不单独报数。

宿主级同口径对拍（2 线程、标定 4 句、两轮交替）：第一轮（两边都在负载下）自研解码每步 52.8 ms 对 ORT 74.2（比 0.71）、vits 每词 137.6 对 181.9（比 0.76）；第二轮负载翻转（ORT 抓到安静窗、自研撞上忙窗）为 59.0 对 38.8（比 1.52）、168.2 对 131.3（比 1.28）。tf 验收集一对同窗样本：自研每步比 ORT 为 0.57。本轮始终没等到第五节那种 load 约 1.0 的完整安静窗，第五节的安静窗绝对值（ORT 解码每步 34.5 至 37.1、自研 48.1 至 50.6）仍是唯一安静口径；本轮能下的结论只有两条：一是 profile 点名的两个热点（注意力池汇合、UNOP double 求值）已经消除，二是同窗样本里自研三轮有两轮领先、一轮因负载翻转落后，x86 上「追平」不签确定结论，等真机 A57 的数（两边同为 NEON 4 宽、2 线程口径、无 AVX-512 成分）。

追不平部分的构成（预期内，不改口径掩盖）：解码每步剩余大头是 PACKED 打包 GEMV（约五成，与 ORT 共用同一内核，本轮按口径未动）；vits 剩余差距在 PCONV 的 AVX2 对 MLAS AVX-512 位宽差（本机限定，A57 上结构性不存在）；BINOP 逐元素带宽在两边都是内存墙，没有可再压的算法差。

### 5. 内存

宿主稳态 RSS：自研 280 至 281 MB（两轮对拍的 4 句最大值），在 262 至 280 MB 基线带的顶边（其中一轮 281，与带顶差 1 MB，属测量噪声量级，如实记）；ORT 同轮 318 至 319 MB。tf_check_free 打包 RSS 314 至 315 MB，与第五节 315.7 同档。提速没有引入新的常驻结构（串行化与 float 化只减不加），带内结论维持。

### 6. 合并标定包 phone-calib-all/

`phone-calib-all/`（约 555 MB）合并原 `phone-calib/`（ORT）与 `phone-calib-free/`（自研）：数据、权重包、图、参考包与两个单包逐字节相同，`bin/` 四个 aarch64 产物（tts_host 双引擎合一、tf_check、tf_check_free、vits_check_free）。`run-on-phone.sh` 一次运行背靠背跑完两边：ORT 段 `--packed --no-arena`、自研段 `--engine free`，同一组 4 句、同 seed、同计时与 RSS 采样；随后两边的 tf_check 打包全量各跑一遍。唯一交付文件 `calib-result.txt`：逐句明细（每段毫秒、每步解码、RTF、RSS）、4 句合计的各段 RTF 与比值、稳态 RSS（4 句最大值）、首句延迟，末尾按既定判据给一行结论（解码 RTF 比与 vits RTF 比都小于等于 1.10 写「建议切默认到 --engine free」，否则写「暂不建议切默认」）。README 写明 2 线程默认与全程插墙充前提。脚本的解析与判据逻辑已在本地用真实宿主输出加桩二进制双向验过（聚合数与独立计算逐项一致，两个判据分支都走到）。

aarch64 产物按 `host/dist-aarch64-free/BUILD.md` 同一条交叉编译线用提速后源码重建（tts_host sha256 def23792…、tf_check_free 304880ad…、vits_check_free 43fbf009…；tf_check 沿用阶段三构建 a44bc3fc…），`host/dist-aarch64-free/` 已同步更新。

QEMU 冒烟（potato-screen 的 ~/a57 环境，qemu-aarch64-static -cpu cortex-a57 加 rootfs-ort，只验对错不测速）：合并包在机器上以 stage3 数据树为底、按本地包逐文件哈希比对后覆盖差异文件组装（两张解码器图、编码器图、两份 manifest 是退回初代包时重生成的版本，stage3 树里是旧版，已换；其余脚本触及的文件哈希一致），bin 用 binfmt 包装走 QEMU。整本脚本一次跑完：两段宿主各 4 句退出码 0、wav 各 4 条；对比汇总与结论行正常生成（QEMU 的 RTF 比值 0.948、0.967 只证明流程与判据代码通，不作速度证据）；两边 tf 复核在 A57 模拟下同为 299/361。自研侧贪心句时长 3.09772 s，与 x86 及第五节 A57 结果逐位相同；ORT 侧本次贪心为 2.41944 s、65 步，是 ORT 自身多线程规约漂移（第五节已定性，自由跑不作跨引擎判据）。真机回传 calib-result.txt 后按其结论行与原始逐句数决定是否切默认。

## 七、vits PackedConv 的 NEON tile（2026-11-10 真机）

背景：真机 FE_PROFILE 实测中句整请求 424.8 s 里 PackedConv 占 365.3 s（86.0%），而 GPT 解码只占 10.2%。上一轮给 GPT 解码写的 NEON GEMV 实测比 clang -O2 自动向量化的标量循环慢 24%（kbench 231.2 对 185.7 ms 每步），本节先反汇编定性，再动手。

**反汇编证据（动手前）**：aarch64 构建（clang 18、Alpine sysroot）的 `vits_kernels.o` 全对象只有 4 条向量 fmla，与 STAGE3 的旧记录一致。`conv_tile_generic` 被内联进 `conv1d_fast_impl` 的计算 lambda；其内层循环只有一条 4 宽 `fmla v3.4s`，且累加器每步都要经栈槽 ldr/str 往返——generic tile 的 `acc[4][32]` 按运行时通道数 ob 索引，clang 无法把累加器提升进寄存器。win[] 窗口填充是逐元素带边界检查的标量代码。这与 GPT GEMV 的情形相反：那边标量循环早被自动向量化（内层 17 条 FMLA），这边是真没有向量化。

**形状普查（PCONV_CENSUS 埋点，按形状记 calls/ΣLout/MAC/反量化 ms/计算 ms）**：feed 口径（75 帧）总计算约 9.6 s（x86 AVX2），反量化合计只占 0.5%，可忽略。主导形状（计算时间份额）：192×192 K=11 dil=1 L=12000（11.1%）、192×192 K=7 dil=1（8.1%）、192×192 K=3 dil=1（6.3%）、384×384 K=11 dil=1 L=1500（5.4%）、96×96 K=11 dil=1 mode1 L=24000（5.0%）、384×384 K=7（4.7%）、96×96 K=7 mode1（4.1%），其后是同族 dil=3/5 变体。几乎全部节点 stride=1、K∈{3,7,11}、dil∈{1,3,5}，(K−1)·dil 最大 50，全部落在 tile 的窗口包络（≤64）内；转置相（phase=1）合计约 5%。全图 MAC 约 150 GMAC 每 feed 句。

**tile 设计**：`conv_tile_neon`（vits_kernels.cpp），与 AVX2 tile 同一分块契约（stride=1、权重已预反量化进 wbuf），按 A57 设计：累加块 4 通道 × 16 位置共 16 个 q 寄存器，OB 模板展开保证累加器常驻寄存器；每通道用 `vfmaq_n_f32` 广播乘加；尾块（tn<16）走对齐 spill 缓冲，输出写回精确 tn 个；win[] 恒按虚拟宽 16 填充（wn = 16 + (K−1)·dil ≤ 80，逐元素边界检查）。v2 加了内部块快速填充：窗口完全落在行内时用向量拷贝代替逐元素标量填充（实测标量填充约占 v1 动态指令的一半，v2 因此再快约 35%）。分发：bench 覆盖 > 环境变量 POTATO_VITS_TILE=generic|neon > 自动（编译入 POTATO_PACKED_NEON 且运行时检测到 neon 时走 neon），generic 保留回退。逐元素累加顺序与 generic 完全一致（同 (i, kk) 顺序、同 FMA），数值逐位相同。

**tile 级 A/B（真机，threads=2，输出与 generic 逐位相同 maxdiff=0）**：

| 形状 | generic | NEON | 加速 |
|---|---|---|---|
| 512×256 K3 L2048（普查前暂定形状） | 0.922 GMAC/s | 1.355 | 1.47× |
| 192×192 K11 d1 L12000（普查 #1） | 1.515 | 3.065 | 2.02× |
| 192×192 K7 d1 L12000（#2） | 1.076 | 1.342 | 1.25× |
| 192×192 K3 d1 L12000（#3） | 0.161 | 0.391 | 2.42× |
| 384×384 K11 d1 L1500（#4） | 1.698 | 3.379 | 1.99× |
| 96×96 K11 d1 mode1 L24000（#5） | 1.518 | 3.814 | 2.51× |
| 192×192 K11 d5 L12000 | 1.130 | 1.981 | 1.75× |
| 24×24 K11 d1 mode1 L96000 | 0.905 | 2.017 | 2.23× |

弱项是 K=7 与 K=3 dil=1：窗口填充成本按 wn 摊、计算量按 K 摊，K 小则填充占比高，两家都受限于填充；后续若再优化，方向是 dil=1 专用寄存器窗口（EXT 抽头），本轮未做。

**整机真机 A/B（旁路二进制，ttsscan 同口径：seed 42、threads 2、不钉核，栈停机状态下跑）**：受控同句同 seed 对照（短句，token 序列与时长逐位相同）：整请求 107.5 s → 54.2 s，其中 vits 79.8 s → 31.7 s（2.52×）、解码段同轮有环境抖动不作数。ttsscan 四请求口径（elapsed/时长 = RTF）：短句 generic 86.4（首跑离群）/58.4 对 NEON 33.1/41.8；中句 generic 76.3/83.2 对 NEON 57.8/35.1。中句均值 RTF 约 79.8 → 约 46.5（约 1.7×）；手机单轮方差 ±40% 量级，绝对值以同轮对照为准。

**上一轮的雷已拆**：packed_core.h 的 NEON GEMV 改为默认关——`neon_gemv_enabled()` 读环境变量 POTATO_NEON_GEMV=1 才启用，`gemv_a_w4` 与 `gemm_b_w4` 的 M==1 分支两处分发都加了这道门；aarch64 默认恢复标量（即阶段验收时的配置），POTATO_NO_NEON 编译开关保留，x86 不受影响。

**普查埋点的一个真 bug（已修）**：首版普查在整图运行时退出阶段卡死/崩溃，LD_PRELOAD 抓栈定位在 `__cxa_finalize` 里的 `_Rb_tree_increment` 空转。根因是 atexit 注册顺序：pconv_add 里先 call_once 注册 dump、后首次构造 census map，退出时 map 的析构（后注册先执行）先跑，dump 遍历已销毁的红黑树；单形状的 bench 因树小、节点内存未被复用而偶然正常。修法：census map 与 mutex 改为永不析构的堆对象（new 出来故意不 delete），dump 改 raw write 落盘。修复后 x86 普查复验正常（文件完整、退出干净）；真机普查复测在待收尾脚本里。

**正确性**：x86 同源码全量回归：vits_check_free free_packed_vs_fp32 corr 0.959880（与阶段二基线逐位一致）、relrms 0.293293、maxerr 0.339176；tf_check_free 打包一致率 299/361、对照 x86 ORT dump argmax 361/361、logits 最大差 8.87e-05。tile 级 8 个形状与 generic 逐位相同（maxdiff=0）。aarch64 侧全图复跑在真机上因内存压力中止（单进程约 0.5 GB，与整栈争内存；两个并发跑曾触发系统低内存杀进程，连带容器被杀），按「已知不合身就不硬跑」的口径不再补跑；aarch64 正确性证据以 QEMU 子集 argmax 全同、tile 级 maxdiff=0 与 x86 全量逐位一致为准。整栈随后已恢复运行。

## 八、PCONV 的 dil=1 寄存器窗口（EXT 抽头）与线程数（2026-11-10 晚，真机）

背景：第七节的 NEON tile 上线后，生产路径跑的仍是 11 月 9 日的旧宿主（二号刀从未晋升）。本轮先补测新剖面，再动刀。全程遵守当日新规矩：手机上不跑全图校验进程（vits_check_free / tf_check_free 在这台机上单跑涨到 0.8 GB 以上、已把容器杀过一次），正确性门全部放在本地 x86 与轻量 tile bench。

**新剖面（vits3 构建、FE_PROFILE、中句「The weather looks really nice today, so let us go outside together.」、seed 42、2 线程、停栈冷宿主首请求）**：整请求 100.6 s（时长 3.713 s）。构成已翻转：vits 49.3 s（49%），其中 PCONV 30.9 s（整请求 30.7%，第七节时是 86%）、BINOP 10.2 s、UNOP 4.3 s、MATMUL 1.7 s；解码（stage_a+b）42.8 s（42.5%），其中 PACKED 24.5 s、BINOP 8.1 s、注意力 MATMUL 4.9 s；first 6.3 s（PACKED 5.5 s）；encoder 0.16 s。BINOP 三图合计约 19.2 s，为第二大聚合项，但逐元素路径已经历第六节的行-wise float 化，剩余以带宽墙为主，手写 NEON 的期望收益低（GEMV 的前车之鉴），本轮未动。

**线程探针（同句同 seed、冷宿主、无剖面、token 序列与时长逐位相同）**：2 线程整请求 142.1 s（first 25.9 / decode 45.2 / vits 70.2；这一跑撞上环境低谷，first 明显离群），4 线程 83.0 s（7.5 / 32.5 / 42.2），整请求 1.71×，三段同涨；即便拿剖面轮的 2 线程构成（合计 98.4 s）作最宽容基线，4 线程仍领先约 1.2×。结论：并行是当前最大单杠杆，生产配置 tts_threads=2 → 4 的取舍留给用户（全栈同跑时与大脑的 4 线程有争核权衡，本节只记旁路宿主口径）。

**第三刀：dil=1 寄存器窗口（conv_tile_neon 内新增 neon_pos16_ext）**。原 NEON tile 每 tap 从栈上 win[] 重载 4 个位置向量（每输入通道 K×4 次 L1 载入）并付一次填充的存-读往返。本刀对 dil=1、满 16 宽、窗口整段落在行内（另留 2 个浮点的载入余量）的块，把整个窗口（16+K−1 ≤ 26 个浮点，K ∈ {3,7,11}）一次载入 q 寄存器驻留，每 tap 的四个向量用 EXT（vextq_f32）抽头形成，不再碰内存。K 与 tap 序号都是模板参数（EXT 的 lane 下标必须是编译期常量，递归模板展开）。累加顺序与原路径完全一致（(i, kk) 升序、每 (o, group) 一次 vfmaq_n），x 值是同一批浮点，结果逐位相同；改动只在 aarch64 NEON 段内，x86 路径按构造不受影响。

**正确性（全在本地与轻量 bench）**：x86 同源码全量回归与基线逐项同数：vits corr 0.959880、relrms 0.293293、maxerr 0.339176；tf 打包 299/361、argmax 361/361、logits 最大差 8.87e-05。真机 tile bench（vits_bench，threads=2）：8 个普查形状 generic 对 neon 的 maxdiff 全部为 0。本轮真机绝对速率受环境影响明显（同形状跨轮次可差数倍，governor 与后台负载未定性），tile 级绝对值不作跨轮次引用，只认同轮相对值与端到端对照。

**真机端到端 A/B（同会话背靠背、4 线程、seed 42、token 序列与时长逐位相同）**：vits3 整请求 82.2 s（first 7.4 / decode 32.5 / vits 41.6，与本轮早先的 4 线程跑 83.0 严丝合缝），vits4（EXT）75.3 s（first 7.8 / decode 32.6 / vits 34.2）。vits 段 41.6 → 34.2（1.22×），decode 与 first 原地不动（内部对照干净），整请求 −8.4%，中句 RTF（时长 3.759 s）21.9 → 20.0。EXT 版已晋升生产路径：~/potatoamika/bin/tts_host 现为 EXT 构建（811,952 B，sha256 43bc0fde…），晋升前生产二进制（11 月 9 日旧构建）备份为 tts_host.bak-20261110-preneon（sha256 def23792…，与第六节记录的阶段三宿主哈希一致）；旁路件 ~/tts_host_vits4 同哈希。本轮结束时手机按用户新口径停在测试台状态（整栈停机），不做可用性收尾。

**下一步瓶颈**：4 线程口径下 decode 32.6 s 占整请求 43% 为最大段，其中 PACKED GEMV 手写 NEON 已证负收益（第一刀），余下是 BINOP 胶水（剖面 2 线程口径 8.1 s）与注意力 MATMUL（4.9 s）；vits 余 34.2 s 里 PCONV 估计仍占六成上下（EXT 只覆盖 dil=1 的满宽内部块，dil=3/5 与边缘块仍走 win[] 路径），胶水（BINOP+UNOP）约 14 s 量级（2 线程剖面比例折算）。若继续，先量 4 线程口径的 FE_PROFILE 再定刀，不凭 2 线程构成外推。

## 九、PCONV 的 dil=3/5 寄存器窗口（EXT 推广）与 4 线程剖面（2026-11-10 深夜，真机）

**4 线程剖面（vits4 EXT 构建、FE_PROFILE、中句、seed 42、停栈）**：整请求 73,316 ms（时长 3.75925 s，RTF 19.5），RSS 261 MB，token 序列与历轮 A/B 逐位相同。分段：first 7,427、decode 32,748（341.1 ms 每步）、vits 32,394、encoder 203、deess 499。vits 内部 PCONV 223 次调用合计 29,748 ms，占 vits 92%、整请求 41%；其余合计仅约 2.6 s（UNOP 927、BINOP 777、MATMUL 565、PAD 130），第八节按 2 线程比例折算的「胶水约 14 s」在 4 线程口径下不成立，胶水不是靶子。decode 内部（stage_a+b 合计）：PACKED 18,048、BINOP 7,928、注意力 MATMUL 2,161、CONCAT 1,510。first 内部 PACKED 6,082（占 first 82%，是 M>1 预填充 GEMM）。两个附带观察：PCONV 从 2 线程剖面的 30.9 s 到 4 线程只降到 29.7 s，几乎不吃线程（同轮 vits BINOP 从 10.2 s 塌到 0.78 s），静态任务切分摊在 big.LITTLE 上的失衡是嫌疑，未单独定性；PACKED 全部（first+decode）24.1 s 为整请求 33%。

**靶子排除记录**：动刀前查了 PACKED 的 mode 分布（packed_graphs 旁路 JSON 按 rows×cols 加权）：mode 0 占权重元素 91.0%、mode 1 占 8.3%、mode 2 占 0.7%。gemm_b_w4 在 M==1 时整行反量化进临时缓冲的坏路径只覆盖 mode 1，即至多约 8% 的权重流量，奖品太小且要动数值路径，不做；mode 0 的标量 GEMV 已是融合加自动向量化形态（第一刀结论不变）。

**第四刀：EXT 寄存器窗口推广到 dil=3/5**。win[] 窗口本来就是一段连续输入，tap kk 读窗口偏移 kk·dil，第三刀的寄存器窗口结构与膨胀无关：把 neon_pos16_ext 与 neon_ext_tap 模板化加 DIL 参数，抽头偏移改 kk·DIL + {0,4,8,12}，窗口寄存器数 NW = ceil((16 + (K−1)·DIL)/4)。覆盖面受寄存器预算封顶：dil=3 全 K（NW ≤ 12）、dil=5 只 K ∈ {3,7}（K=11 需 NW=17，17 窗口加 16 累加器超过整个寄存器文件，不实例化）。分发守卫比 dil=1 略严：要求取整后的窗口（NW×4 个浮点）整段落在行内，多排除至多一个块宽的边缘块，落选块与全部边缘、尾块继续走 win[] 路径。累加顺序与 x 值与 win 路径完全一致，结果逐位相同；改动只在 aarch64 NEON 段与 vits_bench 的形状表（新增 8 个 dil=3/5 普查族形状），x86 按构造不受影响。

**正确性**：x86 同源码全量回归与基线逐项同数：vits corr 0.959880、relrms 0.293293、maxerr 0.339176；tf 打包 299/361、argmax 361/361、logits 最大差 8.87e-05。真机 tile bench（vits_bench，threads=4）：全部 16 个形状 generic 对 neon 的 maxdiff 为 0。新增 dil 形状的 neon 速率（GMAC/s）：192 K11 d3 9.219、K7 d3 8.894、K7 d5 8.673、K3 d3 6.240、K3 d5 6.452、384 K11 d3 7.306、96 K11 d3 mode1 9.381、96 K7 d5 mode1 8.949，与 dil=1 EXT 同档（6.9 至 10.3），NW=12 的两档没有寄存器溢出塌方；仍走 win 路径的 192 K11 d5 为 5.749（同轮对照）。

**真机端到端 A/B（同会话背靠背、4 线程、seed 42、token 序列与时长逐位相同）**：vits4（对照）整请求 74,974 ms（first 7,915 / decode 32,564 / vits 33,773），vits5（第四刀）73,713 ms（first 8,592 / decode 33,007 / vits 31,329）。被改的 vits 段 −2,444 ms（−7.2%）；first 与 decode 是未改代码路径，同轮反向漂移 +677 与 +443（环境噪声），净整请求 −1,261 ms（−1.7%），中句 RTF 19.94 → 19.61。判定为段内明确赢面且数值零风险，已晋升：~/potatoamika/bin/tts_host 现为第四刀构建（822,648 B，sha256 fba8c4da…），第三刀构建备份为 tts_host.bak-20261110-knife3，旁路件 ~/tts_host_vits5 同哈希。本轮 A/B 的宿主日志未捕获 FE_PROFILE 逐项行（harness 的 stderr 路由与剖面轮不同），候选轮的 PCONV 子项拆分未记录，判定依据是分段总时加 tile bench。如实记：本轮同日用户已拍板生产 tts_threads 改 4 并自行改好调度器配置，本轮 bench 口径与之同为 4 线程。

**剩余项与量级估计**：decode PACKED 约 18 s 是现在最大的单项，但 mode 0 标量 GEMV 已在自动向量化的融合形态上，手写 NEON 是死路，余下只有权重布局或循环重构类的未证方向；first 预填充 PACKED 约 6.1 s 走 gemm_a_w4 的分块反量化路径，向量化反量化或常驻反量化是候选，每请求只付一次；PCONV 残余是 K11 d5 的寄存器窗口（要降 OB 才装得下）、边缘与尾块、以及上面记的线程不扩展嫌疑（若属实，改任务切分比改 kernel 更值）；decode BINOP 7.9 s 在计划执行器层做相邻逐元素算子融合有空间。四项全拿下估计再省整请求的两到三成，中句 RTF 到 14 至 16 量级；到实时（RTF 1）还差约 20 倍，内核微优化到不了，那是模型与步数层面的事，不在本系列内。

## 十、PCONV 线程扩展嫌疑的证伪（2026-11-10 深夜，真机）

用户 23:26 拍板「尽力而为」后开第五轮系列刀，第一件是查第九节留下的嫌疑：PCONV 从 2 线程剖面 30.9 s 到 4 线程只降到 29.7 s、疑似静态切分在 big.LITTLE 上失衡。动手前先读代码：`conv1d_fast_impl` 的计算段根本不是静态切分——它用原子计数器在（批 × 位置瓦片 LT=1024 × 4 通道块）任务上动态取活（该设计源于共享 VM 上静态切分被最慢 worker 拖死的实测），静态 `chunk()` 只用在其余内核（GEMV、BINOP、反量化）上。所以「改 PCONV 的切分」无的放矢，只剩下「PCONV 整体不吃线程」这一经验命题可验。

**验证（同会话、vits_bench、NEON tile、16 个普查形状、GMAC/s）**：t1→t2 干净形状 1.74–2.01×；t2→t4 干净形状 1.52–2.01×；t1→t4 多数形状 2.2–3.6×（192 K7 d1：3.072→5.632→9.738；192 K11 d3：3.181→6.065→9.214；384 K11 d1：2.640→5.298→8.413）；t4→t6 再涨 10–35%（K7 d1 12.041、K3 d1 9.204）。PCONV 在隔离 bench 里扩展性健康。剖面那对 30.9 对 29.7 是两个不同会话的跨会话比较，而本会话内单形状污染实测最高达 5×（t1 的 K11 d1 只有 0.488、t2 的 K11 d5 只有 0.646，同形状在 t4 都回到 9.8 与 5.7），该比较不携带信号。**结论：前提不成立，本候选关闭，不动代码。** 钉核三段（taskset 钉大核/小核）没跑成：容器里没有 taskset 命令（doas 装包不在本轮权限内），但扩展性数据已定案，钉核只剩测大小核速率比的参考价值，不补跑。

**附带观察**：隔离口径下 PCONV 到 6 线程仍在涨（t4→t6 约 +10–35%），即 4 线程并未吃满这台机；但生产 threads=4 是用户按整栈争核（大脑同为 4 线程）拍板的配置，本系列不动配置，只记录。

## 十一、first 预填充反量化：常驻缓存死于内存，NEON 反量化死于量级（2026-11-10 深夜）

候选二，靶子是 first 的 PACKED 6,082 ms（4 线程剖面）。两个子方案分开裁：

**常驻反量化缓存**：权重每图固定，加载时全量反量化成 fp32 常驻，运行时零反量化。内存账先杀它：gpt_first 打包权重 76.0M 元素，fp32 是 304 MB；宿主当前稳态 RSS 261 MB，加上去直接翻倍以上，破 270–318 MB 基线带，手机全栈同跑时（约 950 MB 已驻留）也没有这个余量。不做。

**NEON 向量化反量化**：先量前提。first 的序列长 M 由 ref 语义 token 39 个加文本音素约 60 个组成，M≈100——x86 复刻量测（first 主导形状、反量化段与 apply 段分开计时）显示反量化占比在 M=64 时 21%、M=128 时 5.4%，A57 标量更慢、占比只会更高，表面上值得做。代码已实现：抽出 `dequant_w4_span`（mode 0 的块反量化与 mode 1 的行反量化共用），组对齐的 64 宽主体用 `expand_w4_32` 展开加广播减零点乘尺度，数值逐位一致（nibble 与 zero 是 15 以内整数、减法精确、乘法只舍一次，与标量同序）；x86 回归门同数（299/361、0.959880/0.293293/0.339176）。

**真机微量测杀死它**：`dq_test`（旁路小工具，非全图校验）在手机上验完 11 种 span（对齐、未对齐头、尾巴）与标量逐位相同；速度上整张 512×2048 反量化标量 4.197 ms 对 NEON 1.539 ms（2.73×，单线程）。但绝对账算不过来：标量 4.0 ns 每元素，first 全图 76.0M 元素单线程约 304 ms、按 4 线程池折算约 76 ms，只占 first PACKED 的约 1.3%；就算把块间交错与缓存干扰按四倍悲观放大，NEON 的天花板收益也只有约 0.15–0.2 s，约整请求的 0.25%，远低于同会话 A/B 的 ±2% 漂移地板（第四刀里未改段同轮漂移 ±0.7 s）。decode 侧 mode 1 的 M==1 行反量化同理：每步 48 次调用共约 0.1 ms，可忽略。**结论：前提量测不成立，本候选关闭。** 代码已回退到动刀前原文（x86 门复验同数），候选构建从未推上手机，生产二进制未动。教训与第十节同源：占比直觉必须先折算成绝对毫秒再决定动不动刀。

## 十二、decode BINOP：诊断定位 walk_par，span 步行实测更慢，候选关闭（2026-11-10 深夜，真机）

候选三，靶子是 decode 的 BINOP 7,928 ms（stage 两图合计）。先诊断不动刀：给 `k_binop` 加 FE_PROFILE 门控的路径直方图（路径、次数、总时、元素数、最大单次），真机跑一请求（中句、seed 42、4 线程）。结果：f32_fast 16,918 次 1,192 ms（186.9M 元素）；rows 324 次 322 ms（43.7M）；**walk_par 2,379 次 7,443 ms（283.8M 元素，26 ns 每元素）**；serial_odom 零次；generic_odom 583 次 17 ms。BINOP 的钱几乎全在 walk_par 一条路上：操作数或输出带转置步长、掉出 rows 的最后一轴条件的大张量逐元素 op（平均每次 119K 元素，注意力区形状）。

**试的修法**：`binop_span`——沿输出步长为 1 的任意轴（优先最后一轴）做行分解内层跨度循环，操作数在该轴任意步长用仿射 gather，替代 walk_par 的逐元素里程表。逐元素求值同序同式，x86 门逐项同数（299/361、0.959880/0.293293/0.339176），图级逐位一致成立。

**真机判负**：候选构建（span 加直方图，sha256 38921534…）同会话背靠背 A/B：对照（生产第四刀）整请求 70,378 ms（first 7,350 / decode 32,655 / vits 29,650），候选 73,776 ms（first 8,283 / decode 34,798 / vits 29,968）。最干净的证据在路径表内部对照：同一批 2,379 次调用、283.8M 元素，span 用 8,476 ms，比 walk_par 的 7,443 ms 慢 14%（29.9 对 26.2 ns 每元素）；stage_a 的 BINOP 总时也从 3,739 涨到 4,109 ms。里程表不是这批 op 的成本本体——成本在转置操作数本身的离散访问与大张量的内存流量上，换步行结构省不掉。简报原定的计划级相邻 BINOP 链融合也覆盖不到这批调用：它们的图邻居是 softmax 与 matmul，不是另一个 BINOP，链不存在。把逐元素 op 融进相邻 matmul/softmax 的尾声是另一个量级的引擎改造，不在本候选范围内。**结论：候选关闭。** span 路由与实现已拆除（留着会让后续构建平白变慢），路径直方图保留为 FE_PROFILE 诊断（与 PCONV_CENSUS 同类，关掉时零成本）；拆除后 x86 门复验同数。生产二进制未动，仍是第四刀 fba8c4da。

**附带基建记录**：本轮推二进制时撞上一次混块损坏——两个推送脚本共用同一本地分块目录、跳过检查只验块大小，拼出的文件块序与内容混杂（整文件 sha256 不符被当场抓住，坏文件从未运行）。推送已改加固版：独立分块目录、零填充块名、按序显式拼接、整文件哈希两端核对后才算数。

## 十三、PCONV 残段：K11 dil=5 用 8 位寄存器窗补上 EXT，已升正式（2026-11-10 深夜，真机）

候选四。先定量：生产构建带 PCONV_CENSUS 跑一请求（中句、4 线程），PCONV 总 25,837 ms 里 **K11 dil=5 家族占 2,751 ms（10.7%）**：192 通道 1,180、384 通道 583、96 通道 573、48 通道 270、24 通道 146 ms，约整请求 3.9%。这批形状自第四刀起一直走 win[] 路径（16 位窗 NW=17 装不下寄存器），图内速率约 5.3 GMAC/s。

**做法**：`neon_pos8_ext<11,5,OB>`——位置块减半到 8，窗 8+50=58 个 float（NW=15 个向量寄存器）加 OB×2 累加器装得下；`neon_block` 在 tn==16 时跑两个 8 位半块，两半的取整窗（60 个 float）都须落在行内才走，否则回落 win[]。EXT 抽头每 tap 两次 pick（+0/+4），逐元素累加顺序与 win 路径相同（i、kk 升序、逐 (o,组) 一次 vfmaq_n），理论逐位一致。

**门**：x86 全量回归同数（299/361、0.959880/0.293293/0.339176）；真机 tile 台 16 个形状 maxdiff 全为 0；目标形状速率 5.699 → **7.096 GMAC/s**（+24.5%，未及 EXT 家族 9 级——8 位块的窗重载占比更高，到此为止）。

**A/B（同会话两对，顺序交替，中句、seed 42、4 线程）**：vits 段对照两轮 31,365 / 30,815 ms，候选 31,064 / 28,954 ms，两对同向，均值 **−1,082 ms（vits 的 −3.5%）**，与普查按家族占比折算的预期量级一致；整请求均值 73,076 → 71,413 ms（−2.3%）。**判定明确赢，升正式**：生产二进制 sha256 `c9dd5f962ec5c04b8514b91fe45bc33daff34bed02d77be311a0025f96eb9b65`（828,536 B），旧版备份为 `tts_host.bak-20261110-knife4`。

**没做的**：边缘与尾块没有单独定量手段（普查按形状记账、边缘混在形状总时里），且家族大头已被本刀覆盖，剩余零头按占比不足 1% 跳过，符合简报的跳过线。

## 十四、decode GEMV 权重布局：组面板重排 kbench 判负，候选关闭（2026-11-10 深夜，真机）

候选五，最后一个。mode-0 GEMV 的现行布局是行主序（K 行 × D 列，组沿 D 每 64 一个）：内核按输出组切线程，每个线程对一组输出遍历全部 K 行、每行只取 32 字节切片，跨行步长 D/2 字节。试的布局是**组面板重排**：同一输出组的所有行的 32 字节切片连续存放，线程的读流变顺序流；纯字节置换，累加顺序不变，加载时重排一次即可。独立 bench（不碰引擎代码）同进程交替量四个解码层形状 ×24：

| 形状 | 现行 ms/step 份 | 面板 | 差 |
|---|---|---|---|
| 512×1536 | 54.6 | 62.9 | +15% |
| 512×512 | 47.8 | 49.5 | +3.7% |
| 512×2048 | 55.3 | 47.9 | −13.5% |
| 2048×512 | 37.2 | 48.1 | +29% |
| **合计** | **195.0 ms/step** | **208.4 ms/step** | **+6.9%** |

四形状 maxdiff 全为 0（置换逐位一致成立）。总账面板更慢：行主序下四个线程协同扫整块矩阵、相邻组的缓存行经 L2 共享，常步长访问预取器也跟得上；面板把每个线程的流切成互不重叠的长条反而丢了这层共享。唯一赢的形状撑不起总账。**结论：候选关闭，不采用**，引擎零改动。与第一刀（手写 NEON GEMV 输给自动向量化）合看，decode GEMV 这条路在布局与指令两层都已探到底，现行标量自动向量化路径就是这颗 SoC 上的最优已知形态。

## 十五、本轮五候选收官小结（2026-11-10 深夜）

用户「尽力而为」令下的一轮：五候选逐个量测定靶，只升明确赢的一刀。

| 候选 | 处置 | 关键数 |
|---|---|---|
| 1 PCONV 线程扩展 | 前提不成立关闭 | 已是动态取活；单形状 t1→t4 扩展 2.2–3.6× |
| 2 first 预填充 | 前提不成立关闭 | fp32 缓存 +304 MB 装不下；NEON 反量化上限整请求 0.25% |
| 3 decode BINOP | 实现后实测判负拆除 | span 8,476 ms 对 walk_par 7,443 ms（同批调用） |
| 4 PCONV 残段 K11-d5 | **升正式** | tile 5.699→7.096 GMAC/s；两对 A/B vits 段均值 −1,082 ms |
| 5 GEMV 布局 | kbench 判负关闭 | 面板 208.4 对现行 195.0 ms/step |

现行生产：sha256 `c9dd5f962ec5c04b8514b91fe45bc33daff34bed02d77be311a0025f96eb9b65`。中句（seed 42、4 线程、96 步、3.75925 秒音频）在升正式的两对同会话 A/B 里，候选臂（即现行生产）整请求 72,903 / 69,923 ms，**RTF 19.4 / 18.6**；同会话第四刀构建对照臂 73,611 / 72,541 ms（RTF 19.6 / 19.3）。段构成（候选臂较快一轮）：first 7,395（10.6%）、decode 32,825（46.9%）、vits 28,954（41.4%）、deess 499、encoder 209。

剩下的大头与为什么不再动：decode 32.8 秒里 PACKED 约 18 秒——GEMV 的指令层（第一刀）与布局层（本节）都已探底，BINOP 约 7.9 秒的成本在转置访问与内存流量本身（第十二节），两者都没有已知的可落地杠杆；vits 29 秒的 PCONV 主导形状已全部在 EXT/NEON 路径上（第十三节后 K11-d5 家族补齐），剩余是 NEON 宽度与这颗 ARMv8.0 芯片的算力上限；first 7.4 秒的反量化占比 1.3%（第十一节）。整请求距实时（RTF 1）仍差约 19 倍，这个量级不是内核优化能补的，与此前结论一致。
