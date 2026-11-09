# TTS 引擎阶段二报告：C++ 宿主（x86-64 先行验证）

日期：2026-11-08。范围：按阶段一选型写 C++ 宿主，端到端合成加完整外壳行为，先在 x86-64 Linux 上跑通并验证。aarch64/musl 移植是阶段三，本阶段未碰。选型口径中途有三条更新，均已落地：GPT 权重包位宽加载时可选（默认 W4，W8 冒烟通过）；vits 改用变体 E 混合包（enc_p 与 quantizer codebook 用 W8、其余 W4）；输出链在裁静音之前串入 deess2air（见输出链一节）。

配置定案（2026-11-08 13:21 用户拍板）：用户试听未听出 W4 与更高精度之间的区别，按其取舍原则（速度与质量相当时内存占用小为决定性收益），**GPT 主配置正式定为 W4 权重包（92,735,008 字节）**，出货默认与验证主线都按 W4 走；W8 包的加载能力保留为可选项（`--gpt-bits 8`），不删。vits 用 E 混合包（43,523,616 字节）不变。宿主默认即 `--gpt-bits 4`。

## 产物

- 宿主源码：`tts-engine/host/`（CMake 构建，产物 `host/build/tts_host`，254 KB；链接本机 ONNX Runtime 1.22.1，与阶段一 Python 侧同版本，库在 `host/ort/`）。
- 参考特征包：`tts-engine/host/refpacks/*.rpk` 加 `index.json`（离线脚本 `host/precompute_refs.py`）。
- G2P 数据：`tts-engine/host/data/`（symbols.txt、cmudict.tsv 143,199 词、namedict.tsv 19,929 词，由 `host/prep_assets.py` 从 Genie 词典实例导出）。
- 宿主样本：`tts-engine/host-samples/`（见该目录 README）。
- 验证脚本与日志：`host/validate_patch.py`、`host/validate_e2e.py`（Part A）、`host/validate_e2e_b.py`（Part B/C/D）、`host/validation_partA.log`、`host/validation_partBCD.log`。

## 权重加载（位宽可选，逐张量混宽）

启动参数 `--gpt-bits 4|8` 选 GPT 三张图（encoder、first_stage、stage_decoder）的权重包；vits 固定用混合包 `wo-packs/sovits_vits_w4hybrid.npz`（清单 `wo-packs/manifest_vits_w4hybrid.json`）。加载器按 manifest 的逐张量条目读：每条带 name、shape、scheme（w4 分组 64 不对称 / w8 逐通道对称 / fp32-residual），宿主在进程内反量化成 fp32 后按名注入 `work/folded/` 的 fp32 图模板（protobuf 线格式层替换 initializer，其余字节原样），再建 ORT 会话。同一套代码路径吃 W4 包、W8 包和混合包，不存在整包单位宽假设。

一处格式推广已处理：vits 包把 3D 卷积张量按（输出通道，输入轴展平）视图打包（见 VITS-W4.md），且 npz 条目可能存展平形状；宿主还原一律以 manifest 的原始 shape 为准（W4 的分组轴用展平后的列数，2D 时与 GPT 包逐位一致）。

加载报告（stderr，每次启动打印）示例（W4 主配置）：

| 图 | 模板 | 权重包 | 替换 initializer | 注入 fp32 |
|---|---|---|---|---|
| t2s_encoder | 11,479,616 B | gpt_encoder_w4.npz | 7 | 11,465,732 B |
| t2s_first_stage_decoder | 307,212,868 B | gpt_first_w4.npz | 291 | 306,827,268 B |
| t2s_stage_decoder | 307,213,820 B | gpt_stage_w4.npz | 291 | 306,827,268 B |
| vits（wnfold） | 250,037,477 B | sovits_vits_w4hybrid.npz | 519 | 248,561,536 B |

prompt_encoder 的处理：ge 与 ge_advanced 只依赖参考音，与文本无关，阶段二把它们连同 ref_seq、ref_bert、ssl_content 一起离线预计算进参考特征包（用阶段一的 fp32 prompt 图计算），运行时不加载 prompt_encoder。这样同时绕开了阶段一测出的「prompt_encoder 的 int8 QDQ 图是全集相对误差最大的一张（相关性 0.956）」这一项，运行时图里不再有它。

## 参考特征预计算

`host/precompute_refs.py` 按阶段一 inputs.py 的口径（Genie ReferenceAudio 前端：中文音素序列、RoBERTa 特征、HuBERT ssl_content）逐条处理 69 条参考音，另加说话人向量喂 fp32 prompt_encoder 得到 ge / ge_advanced。每条参考一个 `.rpk` 文件（自研小格式：magic、张量名、dtype、shape、原始数据），`index.json` 做标签到文件的索引。宿主运行时按标签懒加载并缓存。

完成情况：65/69 已产出。其余 4 条（sexual_desire1、sexual_desire3、sleepy1、tsun3）是 Genie 中文 G2P 本身的崩溃：ToneSandhi._merge_continuous_three_tones_2 对这四条参考文本抛 IndexError（已定性到具体函数与文本，钉住的 Genie 版本下原版外壳建这些参考的特征时会撞同一个错）。不改参考文本、不给上游打补丁的前提下这 4 条产不出特征包，失败名单记在 index.json 的 `_failed` 字段；宿主对这 4 个标签按未知标签同路回退 neutral1。过程备注：脚本早期几轮曾被系统杀掉（本机 7 GB 内存，验证环节三套 GPT 图并发时内存压力），已加断点续跑与逐条容错（单条失败记录后继续），最终以前台分批跑完。

## 宿主行为（对照改造清单第 7、8 条外壳规格）

- 常驻进程，stdin 逐行一个 JSON 请求（text、tag、speed、top_k、top_p、temperature、seed，可选 greedy），每行合成一条 wav 写到 `--outdir`，stdout 回一行 JSON 结果（文件路径、时长、总耗时、逐段计时、token 序列、实际用标签、G2P 回退计数、RSS、notes）。合成完一句立刻返回，不攒批。
- 情绪标签选参考特征包；标签不存在时回退 neutral1 并在 notes 里写明。
- 句首补 “.”（与 Genie 前置 “。” 经 REP_MAP 变 “.” 等价）。
- 输出裁静音：峰值 −50 dB 阈值，两端各最多裁 1.3 秒。
- 解码循环由宿主控制：encoder、first_stage、stage_decoder 逐个调，KV 由宿主在请求间持有并逐歩传递；图内采样出的 token 每步被宿主选定的 token 覆盖（与阶段一 greedy.py 同一协议）。EOS=1024 或 500 步停。
- 采样五参数：top_k、top_p、temperature、seed 真生效（宿主侧对 stage 图输出的 logits_penalized 做温度缩放、top-k 截断、softmax、top-p 核采样，随机源为按请求 seed 播种的 mt19937_64；temperature≤0 或 greedy=true 时走 argmax）。**speed 不生效，点名说明**：导出的四张图没有任何输入承载 speed（GPT 图无时长输入，vits 无 speed 输入），宿主接受该字段、在 notes 里标注未应用，语义上留给阶段三接时长模型或重采样时处理。
- pred_semantic 取生成序列在首个 EOS 前的部分，与阶段一 drive.py 口径一致。

## 输出链：deess2air（DEESS.md v2，用户听感定版）

vits 输出的整句音频在裁静音之前依次过两级（`host/src/deess.h`，逐句处理、状态每句清零）：

1. 分频动态 de-esser：5 kHz 零相位分频（6 阶 Butterworth 高通幅度形状，高频支路 = 高通、低频支路 = 原信号减高频，增益 1 时精确重构；整句 FFT 实现，与原型同一幅度形状，未用因果 biquad 互补）；检测量为高频与宽带的功率包络之比 R（包络为 tau 3 ms 的截断指数核平均），R 超过 −5 dB 的部分按斜率 2.0 换算衰减、封顶 10 dB；增益在 dB 域做攻击 0.5 ms / 释放 50 ms 平滑（0.5 ms 网格加线性插值到逐样本），只压高频支路，低频支路直通。
2. 静态空气架：零相位 FFT 幅度实现；11 kHz 以下增益恰为 1，11 到 16 kHz 按升余弦升起，16 kHz（奈奎斯特）达 +3 dB 平台。

参数全部按 Hz/ms 物理量写、系数按实际采样率（本管线 32000 Hz）现场换算。请求字段 `deess`（默认 true）可关掉整级，结果 JSON 回报 deess_applied 与该级耗时（实测 1.9 秒句长约 201 ms，宿主内的 FFT 是朴素实现；相对每句 6 至 15 秒的推理仍可忽略，剖析表未单列，阶段三可随手优化）。

移植验证（不重跑全套，只验这一级）：宿主离线模式 `--deess-file` 对阶段一的 `samples/w4_s0_neutral1.wav` 与 `samples/fp32_s0_neutral1.wav` 跑 deess2air，与 DEESS 原型自己产出的定版文件 `samples-deessed/*_deess2air.wav` 逐值比对：最大绝对差 3.2e-05 与 3.1e-05（约 1 个 int16 LSB，即量化地板），相关性 1.00000000。唯一与原型的实现差是 FFT 长度取 2 的幂（零填充）而非填充后原长，幅度响应曲线同一条，差异落在 1 LSB 内。宿主样本按要求出两份：默认带 deess2air 的与 `"deess": false` 的未处理版并存，命名区分（见 host-samples/README）。

## 英文 G2P

实现 Genie EnglishG2P 的主干：文本级展开（标点 REP_MAP、数字基数词与小数、时间 H:MM、序数词、Dr./Mr. 等缩写子集）→ TweetTokenizer 近似分词 → 同形异音词表（无 POS 标注器，取非特殊读音，即 pron2）→ cmudict 首选读音查表（数据为 Genie 实例实际词典的导出，含其删改）→ 专名 Title Case 查 namedict → 's 所有格规则 → 连字符分拆 → 过滤与符号映射（与 symbols_v2 对齐）。词典查不到的词走简单的字母发音回退（LTS），并逐请求计数（结果 JSON 的 g2p_fallback）。

与 Genie 的已知偏差（探针 20 句验证，逐项见下）：一是同形异音词没有 POS 信息，past tense 的 read 读成 R IY1 D 而非 R EH1 D；二是序数词宿主说正确的 third，Genie 的序数展开有 bug（“3rd” 变成 “three-red”），宿主不复制这个 bug；三是日期串（2025-01-15）Genie 有专门的日期展开，宿主按数字组读，两边不同；四是生造词（Zxqwj）Genie 走神经 G2P，宿主走 LTS，回退结果不同，这类词会被计数。

## 验证（逐项）

1. 图级（权重还原与注入）：宿主 `--emit-models` 导出的四张补丁图，与 numpy 独立实现的还原（同 quant_wo / vits_w4_pack 公式）逐张量比对：1,108 个包张量最大差 0.0（逐位一致）；非包 initializer 与 fp32 模板逐位一致。脚本 `host/validate_patch.py`。
2. 解码级（教师强迫，与阶段一同一口径）：在宿主导出的 W4 图上跑 greedy.py 的教师强迫协议（2 条参考 × 3 句，沿 fp32 贪心轨迹喂 token）：每步 logits 与阶段一烘焙 W4 图的最大差 0.0（361 步）；对 fp32 下一 token 的 argmax 一致率 299/361 = 82.83%，与阶段一 W4 的数字分毫不差。说明宿主内存里组装出的模型与阶段一管线数值等价。
3. 端到端音频：宿主贪心合成 S0 后，取宿主输出的 token 序列在 Python 侧用同一张 vits 图合成并按宿主规则裁剪，比对结构指标。先说清一个前提：vits 图内噪声按进程随机（实测两个 Python 进程、同图同输入，波形相关性只有 0.44），所以波形逐值比对对参考管线本身也不成立；阶段一「重跑差 0.0」的观察应是同进程内的会话。可比的是结构量，本次实测（同一 pred 序列，宿主 vs Python）：未裁剪长度同为 pred 长度的确定函数；RMS 包络相关性 0.987 至 0.994（两次运行）；频谱重心 4614 对 4635 Hz、5486 对 5377 Hz；F0 中位数 287.0 对 293.6 Hz、277.1 对 274.7 Hz；RMS 0.1044 对 0.1043。结论：宿主音频与参考管线统计等价，差异来自图内噪声，不是宿主实现偏差。
4. G2P 探针：20 句与 Genie 标准答案比，整句全对 16/20，逐位置音素一致 579/650 = 89.1%（被日期句一条拉低）。4 句差异即上面列的四类已知偏差。
5. 采样与 seed：宿主采样器由请求 seed 播种，给定相同 logits 时结果可复现（mt19937_64 加固定分布算法）；但端到端 token 序列不保证跨运行一致，根源在图内：第一个 token 由 first_stage 图内部采样产生、无 seed 输入，阶段一已测过采样路径本身不可复现。实测两种情况都出现过：一次同 seed 两请求 54/54 token 全同，另一次出现分叉，分叉点在图内采样一侧，不在宿主采样器。seed 不同（7 对 8）输出不同，greedy 与采样输出不同，五参数中除 speed 外均确认真实影响输出。
6. W8 冒烟：`--gpt-bits 8` 同宿主跑贪心 S0，67 步、2.63 秒，EOS 正常终止，token 轨迹与 fp32 贪心同族（第 2 位起出现 127、366、256、500、910、75、17 的同序片段）。vits 在 W8 档下同样加载混合包并正常出声。

## 性能剖析（x86-64，4 线程，W4 主配置）

机器为本机（7 GB RAM 的 x86-64 盒子），测试时后台同时在跑参考特征预计算，数字按偏上限看；分段结构是给后面手搓 kernel 的输入。同进程连发三条（neutral1）：

| 请求 | 步数 | G2P | GPT 编码 | first stage | GPT 解码 | 解码/步 | vits | 总计 |
|---|---|---|---|---|---|---|---|---|
| S0 贪心 | 61 | 0.07 ms | 6.3 ms | 677 ms | 2,725 ms | 44.7 ms | 3,079 ms | 6,488 ms |
| S0 采样 seed42 | 61 | 0.05 ms | 8.6 ms | 1,015 ms | 3,901 ms | 64.0 ms | 4,375 ms | 9,306 ms |
| S2 采样 seed7 | 48 | 0.04 ms | 23.0 ms | 890 ms | 2,300 ms | 47.9 ms | 2,405 ms | 5,619 ms |

读数要点：G2P 与 GPT 编码可忽略。first stage 一次要把整个参考 prompt（ssl 212 帧量级）过一遍 24 层解码器，是 prefill 型开销，约 0.7 至 1.0 秒。解码每步 45 至 65 ms（序列长约 270 至 330 时对全 KV 做注意力，GEMV 主导），与既定计划一致，是手搓 kernel 的第一目标。vits 约每语义 token 50 至 70 ms，卷积主导。音频时长与总耗时之比约 1:2.9（S0 贪心 2.28 秒音频、6.5 秒墙钟），x86 上非实时；手机端更慢，逐句流水线仍是必选项。

## 驻留（如实报，超预算）

稳态 RSS 约 1.83 至 1.88 GB（启动加载后 1,776 MB，每请求结果里回报实时值）。**远超 300 MB 预算，本阶段按要求不为凑预算砍功能，先把账算清**：

- 磁盘上权重包很小：GPT W4 三包合计 92.7 MB，vits 混合包 43.5 MB，总计约 136 MB。
- 但当前实现是启动时全量反量化烘焙成 fp32 再建会话，注入的 fp32 权重合计 873,681,804 字节（约 833 MiB）。其中 first_stage 与 stage_decoder 两张图各存一份解码器权重（约 293 MiB 是重复），阶段一的最小档曾用共享 bin 去重，权重包路线下这份重复还在。
- RSS 比烘焙量还多约 1 GB：ORT 会话侧的图优化副本、内存池与执行缓冲，加上参考特征包缓存（单个约 1.9 MB，可忽略）。

也就是说，W4/W8 包的体积收益目前只落在磁盘上，没有落到驻留上。要让驻留降下来，靠的是阶段三的融合反量化 kernel（权重以打包形态驻留、GEMV 时边反量化边算），这正是调研报告里给 weight-only 路线定的前提；在本阶段的 ORT 宿主口径下做不到加载时不烘焙。中间档可选项（未做，留给阶段三决策）：两张解码器图共享权重存储、ORT 会话的 arena 收缩、按图懒加载 vits。

## 已知限制清单

- speed 参数不生效（图 IO 无此输入，已点名）。
- G2P 四类偏差：同形异音无 POS、序数词与 Genie 的 bug 不对齐（宿主是对的）、日期串不展开、生造词 LTS 与神经 G2P 不同（计数回报）。
- 端到端 token 级复现不保证：第一个 token 由图内采样决定，无 seed 输入。
- 参考特征包未全量（见上，续跑中）；缺标签回退 neutral1。
- first_stage 与 stage_decoder 权重重复驻留（约 293 MiB），见驻留一节。
- 本阶段只在 x86-64/glibc 验证；aarch64/musl、手机实测、融合反量化 kernel 均为阶段三。
