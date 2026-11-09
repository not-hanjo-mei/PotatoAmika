# Moonshine 搬进自研引擎：普查与小样验证报告

日期：2026-11-09。范围：只做普查、内存预估、encoder 单图小样。没动 ASR 现有宿主与出货布局，探针代码与产物在 `asr/free-probe/`。

## 结论先行

**停手，本轮不全量移植，ASR 留在 ORT。**

三项判定里两项过、一项决定性不过：

| 判定项 | 结果 | 依据 |
|---|---|---|
| 内存 | 过（估算） | 全量稳态预估约 65 至 90 MB（段长 5 秒口径），ORT 实测 154 至 178 MB |
| 精度 | 过（encoder 层实测） | W8 路径输出比 ORT 自己的量化路径更接近 fp32（见第 5 节） |
| 速度 | **不过（实测）** | 自研 encoder-only RTF 0.41 至 0.96，ORT encoder-only 0.026 至 0.043，慢 12 至 30 倍；ORT 全管线才 0.147 至 0.217 |

用户的原则是做得比 ORT 更好才继续。内存和精度更好、速度差一个量级，整体不算更好。卡点不在架构（图能编译、形状能绑、数值对得上），在三个已有步型的内核质量：波形前端的直卷、瘦小 GEMM、通用逐元素步。它们都是已有步型内部的优化，不需要新架构，但是一轮独立的内核优化工作量，且在 A57 上的收益未证。真要翻案，先把 encoder 的 RTF 在 x86 上追进 ORT 的 2 倍以内，再重跑本报告第 5 节的口径。

## 1. 算子普查

### encoder（量化版 `onnx_encoder_model_quantized.onnx`；fp32 版 554 个节点、25 种）

695 个节点，29 种算子，初始器 172 个共 7.7 MB（uint8 5.97 MB、int8 1.70 MB、fp32 0.06 MB）。

| 算子 | 次数 | 算子 | 次数 |
|---|---|---|---|
| Mul | 141 | ReduceMean | 26 |
| Add | 61 | DynamicQuantizeLinear | 24 |
| Slice | 60 | Div | 21 |
| Concat | 55 | Shape | 20 |
| Unsqueeze | 41 | Gather | 13 |
| Cast | 40 | Sub / Pow / MatMul | 各 13 |
| Reshape | 39 | Sqrt | 13 |
| MatMulInteger | 36 | Neg | 12 |
| Transpose | 32 | Erf | 8 |
| Softmax | 6 | Conv | 3 |
| Tanh / InstanceNormalization / Range / Sin / Cos | 各 1 | | |

输出形状是嵌套 floor 除法链（num_samples 经 /64、/3、/2 三段下采样），不是仿射式。自研引擎的 SymInt 表达式带 DIV 求值，这张图实际编译通过了（见第 5 节），这一条风险已排除。

### decoder（`onnx_decoder_model_merged_quantized.onnx`）

顶层只有 1 个 If 节点加 240 个初始器共 19.5 MB（uint8 19.39 MB、fp32 0.08 MB）。两个分支各自是完整的 6 层解码器：

| 分支 | 节点 | 算子种数 | 与 encoder 的差异算子 |
|---|---|---|---|
| then（use_cache=true） | 1,175 | 33 | 多 Trilu 1、DequantizeLinear 2 |
| else（use_cache=false） | 1,104 | 31 | 多 Constant 12，其余同 |

ASR.md 记的两个坑在图结构上的对应：`use_cache_branch` 是第 27 个输入，决定走哪个分支，宿主逐步可知；encoder KV 只取第 0 步（else 分支）的 present 往后原样重喂。移植形态只能是两个分支各编一个 Plan、宿主按步切换，权重走同一份外层初始器。

### Silero VAD（`silero_vad.onnx`，2.2 MB）

顶层 4 个节点（Identity 2、Equal 1、If 1）。If 的两个分支各 54 个节点、20 种算子，内层还嵌了多层只做形状分发的 If（条件全是常量口径，sr 给定后分支唯一确定）。真正的计算是 Conv 6、Relu 5 和 **LSTM 1**（隐藏 128）。顶层初始器只有 8 字节，权重全在分支里（0.94 与 1.24 MB，是同一套权重的两份表示）。

## 2. 量化形态与承接方式

量化版图里的量化是 ORT 动态量化的标准三件套，每处共 36（encoder）加 108（decoder 两分支合计）组：

```
DynamicQuantizeLinear(激活) → MatMulInteger(激活q, 权重q=uint8 初始器, 两边 zp)
→ Cast(int32→fp32) → Mul(折好的标量 scale = 激活 scale × 权重 scale)
```

权重是逐张量标量 scale（折进最后那个 Mul 常量里），zp 是标量初始器。承接有三条路：

| 路 | 做法 | 裁定 |
|---|---|---|
| A. 反量化回 fp32 | 三件套改写成普通 fp32 MatMul，权重在加载时还原 | 数值最稳，但权重涨到约 110 MB（encoder 约 31、decoder 约 78、VAD 2.2），稳态估算 135 至 150 MB，压不到 100 MB 以内，内存收益基本吃光 |
| B. W8 weight-only 打包（选这条做小样） | 用 fp32 源权重按 TTS 线现成的口径重做逐输出通道对称 W8，三件套换成已有的 PackedMatMul（gemm_b_w8），激活保持 fp32，权重经 BlobStore 内容寻址共享 | 复用现有内核与打包格式，不造新轮子；权重约 28 MB；逐通道比 ORT 的逐张量还细一档 |
| C. 原生 int8 | 新写 DynamicQuantizeLinear 与 MatMulInteger 两个步型 | 本项目已有实测：ARMv8.0 无点积指令时 int8 GEMM 比 fp32 慢 1.65 至 1.76 倍；且与 PackedMatMul 功能重叠，属重复造轮子 |

static int8 变体不在考虑内：它需要 ConvInteger，这版 ORT 都没有实现（ASR.md 第 5 节）。

## 3. 缺口清单

自研引擎已有步型覆盖了三张图的绝大部分：MatMul、Gemm、Conv、LayerNormalization、ReduceMean 与 ReduceSum、Softmax、Gather 与 GatherElements、Concat、逐元素二元与一元族（含 Sigmoid、Tanh、LeakyRelu）、Cast、Where、CumSum、ArgMax、TopK、视图族（Transpose、Reshape、Unsqueeze、Squeeze、Slice、Split）、Shape 符号求值、ConstantOfShape、Expand、Pad、PackedMatMul、PackedGather、ConvTranspose。

真正缺的（全量口径，共 7 项，只有 2 项算中等量级）：

| 缺口 | 出现处 | 量级 | 说明 |
|---|---|---|---|
| Erf | encoder 8、decoder 两分支各约 8 | 极小 | 新增一个 UnOp 枚举，std::erf。小样已实现 |
| InstanceNormalization | encoder 1 | 小 | 按 (n,c) 归一化，与 LN 共用脚手架。小样已实现 |
| Range | encoder 1、decoder 各 2 | 小 | start 与 delta 是常量、limit 是动态标量，输出长度走 SymInt 的 DIV 表达式，执行期按 numel 填。小样已实现 |
| If 分支编 Plan | decoder 顶层、VAD 顶层与内层 | 中 | 分支是现成的 GraphProto，编译器可复用；宿主按 use_cache_branch 选 Plan（decoder）或按 sr 在加载期定死（VAD）。风险在两分支共享外层初始器时常量别被复制两份，TTS 阶段二有过计划常量快照翻倍的教训 |
| LSTM | VAD 1 | 中 | 单层、隐藏 128、每 32 ms 窗跑一次，性能不敏感，朴素实现约百余行 |
| Trilu | decoder then 分支 1 | 小 | 因果掩码，可作带索引的逐元素步，或改写期折成 Where |
| DequantizeLinear | decoder 每分支 2 | 小 | 操作数是初始器，编译期常量折叠即可，不需要运行时步型 |

没有碰到引擎架构装不下的东西：动态形状（含 floor 链）编译通过，KV 协议与 TTS 的 stage 图同形（宿主持、逐步喂回）。缺口总数 7 项，低于「十种以上步型」的停手线，所以做了第 5 节的小样。

## 4. 内存账预估（估算，非实测）

口径：B 路（W8 打包）全量移植，稳态 RSS，VAD 分段后的单段送 encoder、段间 arena 复用。**段长是关键变量**：encoder 的 arena 随音频秒数涨，小样实测斜率约 4.9 MB 每秒（见第 5 节），下表按段长 5 秒估，段长 10 秒约再加 25 MB。

| 构成 | 估算 MB | 依据 |
|---|---|---|
| 权重常驻 | 约 35 | encoder 包 6.03（实测）加图内 fp32 初始器 6.84（实测，含 conv2 的 4.6 MB 卷积核）加 decoder 包约 19.5 加 VAD 2.2 |
| encoder arena（段长 5 秒） | 约 26 | 小样斜率外推 |
| decoder arena 与 KV（段长 5 秒） | 约 7 | encoder KV 约 4.8、decoder KV 与 logits 约 2 |
| VAD 计划与状态 | 约 3 | 权重已计入常驻，此处是激活与状态 |
| 二进制、库、线程栈 | 约 9 | TTS 自研二进制同口径 |
| **合计** | **约 80（区间 65 至 90）** | ORT 实测 154 至 178，约省一半 |

两个前提写明：一是段长要封顶（VAD 在 8 至 10 秒处强制切段，17.9 秒整段送入时 encoder 单图峰值就到 108 MB，见第 5 节）；二是 decoder 两分支的初始器只存一份（靠 BlobStore 内容寻址或计划间共享常量），存两份要再加约 19.5 MB，仍在 100 MB 线内但余量变薄。

## 5. 小样结果（encoder 单图，已实测）

探针在 `free-probe/`：给引擎副本加了 Erf、InstanceNormalization、Range 三个步型（原引擎未动），`enc_probe` 跑图、ORT 输出做对照；W8 包由 `make_packed_encoder.py` 生成（fp32 图的 49 个 MatMul 里 36 个权重是初始器的换成 PackedMatMul，其余 13 个是注意力运行时矩阵乘，保持原样）。

### 数值（测试句 s01，输出 last_hidden_state [1, 71, 288]）

| 对比 | 相关性 | 最大差 | 平均差 |
|---|---|---|---|
| 自研 fp32 对 ORT fp32 | 1.00000000 | 2.86e-05 | 5.1e-07 |
| 自研 W8 对 ORT fp32 | 0.99981243 | 0.107 | 0.0080 |
| ORT 量化版对 ORT fp32（现任口径） | 0.99655804 | 0.464 | 0.0328 |
| 自研 W8 对 ORT 量化版 | 0.99656664 | 0.541 | 0.0328 |

自研 fp32 与 ORT fp32 只差浮点求和顺序量级，三个新步型和动态形状链数值正确。自研 W8 离 fp32 的距离只有现任量化口径的约四分之一：W8 路径的 encoder 输出比现在线上跑的还准。字错率同带的最终判据要等 decoder 接上后跑整管线，本轮只有 encoder 层证据，方向是正的。

### 内存（进程峰值 RSS，实测）

| 跑法 | 输入 | 峰值 RSS |
|---|---|---|
| 自研 W8 | s01（1.75 秒） | 28.5 MB |
| 自研 fp32 | s01 | 69.3 MB |
| 自研 W8 | spliced 整段（17.9 秒） | 108.0 MB |
| 自研 fp32 | spliced 整段 | 148.9 MB |

编译耗时 fp32 图 0.3 至 0.4 秒、W8 图 0.02 至 0.1 秒（本机当时有负载，量级参考）。

### 速度（2 线程，本机当时有其他代理在跑，绝对值带负载噪声，量级可信）

| 跑法 | s01 RTF | spliced RTF |
|---|---|---|
| ORT encoder-only（fp32 与量化同带） | 0.039 至 0.043 | 0.026 至 0.033 |
| 自研 fp32 | 0.68 | 0.41 |
| 自研 W8 | 0.94 | 0.96 |

spliced 一次运行的步型归因（FE_PROFILE）：MatMul 49 次占 41%，Conv 3 次占 35%，BinOp 171 次约 10%，UnOp 36 次约 5%，Softmax 6 次约 5%。前端三层卷积约 2.1 GMAC（conv2 一层就 1.73 GMAC），直卷内核约 0.5 GMAC 每秒；ORT 走 im2col 加 GEMM，同图约 7.5 GMAC 每秒。W8 的 gemm_b_w8 是逐输出点积的小矩阵实现，在这张图的形状上比 fp32 MatMul 还慢。

## 6. 过闸判定与后续

| 闸 | 判定 |
|---|---|
| 稳态 RSS 明显低于 154 至 178、目标 100 以内 | 过（预估约 80，含段长封顶前提） |
| 精度不放宽 | 过（encoder 层实测优于现任口径；整管线字错率未验） |
| 整体做得比 ORT 更好 | **不过**：速度慢 12 至 30 倍，A57 上 RTF 会过 2，监听链路不可用 |

结论：**不搬，ASR 留在 ORT，预算口径维持 ASR.md 的约 160 MB。** 本轮产物保留在 `asr/free-probe/`（普查脚本、三个新步型的引擎副本、W8 打包脚本、探针），日后若先做一轮内核优化（Conv 改走 im2col 复用已有 GEMM、瘦小 GEMM 与逐元素步提速），把 encoder RTF 追进 ORT 的 2 倍以内，移植的其余条件都已验过，可以直接从打包 decoder 分支接着做。
