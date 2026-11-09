# 耳朵原型报告：Moonshine Tiny + Silero VAD（同一 ORT 进程）

日期：2026-11-09。范围：改造清单第 5、16、18 条的原型验证。结论先行：**Moonshine Tiny 的 ONNX 路线走通了，不用切 SenseVoice**。宿主已落地，x86 与 aarch64 两边都跑通（aarch64 一致性见第 6 节）。唯一没达标的是常驻内存：实测峰值 RSS 154 至 178 MB，超约 80 MB 预算约一倍，超支构成在第 5 节逐项列清。

## 1. 模型来源与许可（下载前已核）

| 模型 | 来源 | 许可 | 取的文件 | 大小 |
|---|---|---|---|---|
| Moonshine Tiny ONNX | hf-mirror 镜像的 `onnx-community/moonshine-tiny-ONNX`（基模型 `UsefulSensors/moonshine-tiny`，仓库 sha a6da1241） | MIT（仓库 card 与 tag 均标 mit） | encoder、decoder_merged，各取 fp32 与动态量化（quantized）两份 | fp32：30,882,331 + 78,227,550 B；量化：7,937,661 + 20,243,286 B |
| Silero VAD ONNX | hf-mirror 镜像的 `onnx-community/silero-vad` | MIT（仓库内有 LICENSE 文件） | `onnx/model.onnx`（v5 接口：512 样本窗、state [2,1,128]、sr 输入） | 2,243,022 B |

下载全程走 hf-mirror.com。另下过 static int8 变体做内存对比，结论在第 5 节：这版 ORT 没有 `ConvInteger` 的 CPU 实现，编码器直接建不了会话，static int8 路线在本环境死路。uint8 变体与 quantized 是同一种动态量化，会话成本相同。

词表：`tokenizer.json` 是 SentencePiece 式 BPE（32000 条，`▁` 表空格，带 `<0xXX>` 字节回退）。`scripts/export_vocab.py` 把它导成 `models/vocab.tsv`（id 加 UTF-8 十六进制），宿主只做解码侧拼接，不需要完整的 tokenizer 库。

## 2. 宿主

`src/asr_host.cpp`（单文件，约 500 行），照 tts-engine/host 的做法用 onnxruntime C++ 接口，不用 sherpa-onnx。链法：x86 链 `tts-engine/host/ort` 的 ORT 1.22.1；aarch64 按 tts 线的 clang 交叉编译法链 Alpine rootfs 的系统 ORT 1.24.4（musl）。

管线：16 kHz 单声道 WAV → Silero VAD 按 512 样本（32 ms）逐窗出概率 → 分段（概率 ≥ 0.5 起段；连续低于 0.5 满 800 ms 关段，段尾取静音段起点；短于 200 ms 的段丢弃）→ 每段前后各垫 160 ms → Moonshine 编码器 → 合并解码器贪心解码（宿主持 KV）→ stdout 一行 `[起 止] 文本`，stderr 一行统计（段数、音频时长、各段耗时、RTF、getrusage 峰值 RSS）。

参数：`--quant`（默认 fp32）、`--no-vad`、`--threads`（默认 2）、`--threshold`、`--min-silence-ms`（默认 800）、`--opt disable|basic|extended`、`--no-arena`。

解码环有两个这个合并图特有的坑，都已在代码里注明：

1. `use_cache_branch` 是第 27 个输入（bool），第 0 步传 false、之后传 true。
2. `use_cache=true` 分支返回的 encoder 侧 present 是坏形状（实测 [0, 8, 1, 36]），不能回喂。正确协议是 encoder KV 只取第 0 步的 present、之后每步原样重喂，只有 decoder KV 递进。按其他喂法第 2 步就在 encoder 注意力 MatMul 上报广播错误。

另有一处 ORT 1.22 的毛病：对解码器 KV 输入调 `GetInputTypeInfo().GetShape()` 会抛 `std::length_error`，零长 KV 的形状按 config 写死为 [1, 8, 0, 36]（8 头 × 36 维）。

## 3. 测试方法

测试音频用现成 `tts_host`（neutral1 参考、贪心、seed 42）合成 8 句已知英文（第 4 句贪心撞空序列，换 seed 7 采样补跑），32 kHz 输出经 ffmpeg 重采样到 16 kHz。句子与真值在 `test/truth.json`，逐句文件在 `test/sentences/`。

拼接文件三份：`spliced.wav` 是前 6 句以 300/500/800/1200/2000 ms 静音依次拼接；`spliced2.wav`、`spliced3.wav` 用能量裁出的真实语音段拼接，专验 800 ms 规则的两侧（见第 4 节）。

字错率口径：小写化、去标点后算字符错误率（CER），另附词错误率（WER）。`scripts/cer.py` 复算。

注意一处读数方法：贪心解码在多线程归约下有逐次翻转，同一文件不同运行会在个别难词上给出不同结果（例如第 2 句的 “quick brown” 在不同配置下出过全对、“click-band fox”、“click brown” 三种）。CER 因此是一个带，不是单点，下面的数字注明是哪次运行。

## 4. 分段验证（800 ms 规则）

`spliced.wav`（名义间隔 300 至 2000 ms）：检出 6 段，与 6 句一一对应。检出段尾与文件段尾的差在 0.03 至 0.28 s 内。名义间隔 300、500 ms 也断开了，原因是 TTS 句子起音软：下一句起音初期概率低于 0.5，这段被算进静音跑长，实际触发关段的概率静音都超过 800 ms。段起检出比能量起点晚 0.4 至 0.8 s，靠 160 ms 前垫，转写没有丢句首词（第 3 句句首 “Please” 被听成 “It's”，与软起音有关，记在第 5 节的错误清单里）。

规则本身用裁好的语音段隔离验过：

| 拼接 | 间隔 | 期望 | 实测 |
|---|---|---|---|
| s02 语音 + 静音 + s04 语音（s04 起音硬，单独检出延迟约 64 ms） | 200 ms | 合并为 1 段 | **1 段** [0.704, 4.960]，真值段尾 4.968，两句转写全对 |
| 同上后半 | 1000 ms | 断开 | **断开**，s03 单独成段 |
| s01 + 静音 + s02（s02 起音软，概率低谷约 0.7 s） | 200 ms | 按概率口径会断 | 断成 2 段（概率静音 = 间隔 + 软起音低谷 ≈ 900 ms ≥ 800） |

结论：800 ms 规则在概率流上行为正确，子 800 ms 的真实间隔能合并、超 800 ms 能断开。实际部署里断点取决于下一句的起音强度，软起音会让等效断点提前，这是 VAD 概率口径的固有性质，不是实现错误。

## 5. 识别质量、速度、内存（x86，ORT 1.22.1，2 线程）

逐句（量化版、opt basic、每句单独走完整 VAD 管线）：

| 句 | CER | 错处 |
|---|---|---|
| s01 Hello, how are you today? | 0 | |
| s02 The quick brown fox… | 0 | |
| s03 Please turn off the lights… | 0.139 | Please → It's |
| s04 I would like a cup of coffee, please. | 0 | |
| s05 The weather is nice and sunny… | 0.029 | sunny → funny |
| s06 Can you tell me what time it is? | 0 | |
| s07 She sells seashells by the seashore. | 0.367 | 整句大半听错（绕口令，TTS 发音本身也糊） |
| s08 Thank you very much for your help. | 0 | |

**合计 CER 0.0726（17/234 字），WER 0.1017（6/59 词）**。8 句里 5 句全对。

拼接流式（量化版、opt basic）：6 段全检出，CER 0.1130（20/177）。比逐句口径差的量主要来自逐次翻转（第 2 句 “quick brown” 这次出成 “click-band fox”、第 6 句 “Can you” 出成 “You need to”），不是分段切坏了音频：同一段在逐句运行里能全对。

fp32 对照（拼接、默认 opt）：CER 同档（“click brown fox”、“turn of the lights”、“And you tell me” 这类错照样有），**错误形态与量化版一致，量化不是误差源**，误差在模型对合成语音的识别能力本身。真人语音的数字要等真机麦克风链路有了再测，本报告的 CER 只证明管线和量级。

速度：拼接文件（17.9 s 音频）量化版 compute RTF 0.147 至 0.217（VAD 约 0.2 至 0.7 s、ASR 约 2.2 至 3.2 s，随 opt 等级与逐次负载浮动）；逐句 RTF 0.21 至 0.35；fp32 RTF 0.184。x86 上离实时很远不是问题，手机（A57）速度按 TTS 线的口径只能真机标定，QEMU 不测速。

内存（如实报，超预算）：

| 配置 | 峰值 RSS |
|---|---|
| 量化版，逐句运行（8 次取最大） | 176.5 MB |
| 量化版，拼接运行（三种 opt 等级） | 154 至 162 MB |
| fp32 版，拼接运行 | 369.2 MB |

超支构成（分阶段探针实测，会话创建口径）：裸进程 5.8 MB；建 ORT env 后 14.5 MB；加量化编码器会话 61.2 MB（默认 opt）或 41.1 MB（关 opt）；再加量化解码器会话 145.6 MB（默认 opt）或 89.3 MB（关 opt）；VAD 会话约 0 增量。模型文件本体只有 30.3 MB（量化两件加 VAD）。差额是 ORT 的会话开销：图优化副本、初始器复制与 MatMul 预打包缓冲，这些页在创建时是虚拟占用、跑起来才全部驻留，所以三种 opt 等级在实际运行峰值上拉不开（154 至 162 MB 同带）。试过的压法：关 CPU arena 无效（157.8 MB 且 RTF 恶化到 0.74）；关预打包无效（136 MB 创建口径、运行口径同带）；关图优化只降创建口径。80 MB 预算按清单第 3 条是立项时的估算账，本次是第一次实测：**在 ORT 里跑 Moonshine Tiny，常驻就是 150 MB 量级，预算要么改到约 160 MB，要么换运行时（不在本原型范围）**。手机上是 musl ORT 1.24.4，数字可能不同，要真机标定才算数。

## 6. aarch64 一致性（QEMU cortex-a57，只验对错）

交叉编译用 tts 线同一条线（clang 18、Alpine 3.24.4 rootfs 作 sysroot、链系统 ORT 1.24.4 musl 版），产物 `build-aarch64/asr_host`（aarch64 ELF、解释器 ld-musl-aarch64、89 KB）。在 `tts-engine/a57/rootfs` 环境里以 `qemu-aarch64-static -cpu cortex-a57` 跑同一份量化配置（opt basic、2 线程）处理 `test/spliced.wav`：

实测结果（墙钟 2 分 27 秒，纯模拟耗时，不作速度结论）：检出 **6 段，时间戳与 x86 逐字相同**（[0.480, 1.760]、[2.784, 5.056]、[6.464, 7.968]、[8.896, 10.560]、[12.256, 13.888]、[16.704, 17.920]）。文本 6 段里 5 段与 x86 的 opt basic 运行逐字相同；第 1 句 aarch64 出 “Hello?”、x86 basic 出 “Hello,”，这正是第 3 节记的贪心逐次翻转（x86 默认 opt 的那次运行同样出 “Hello?”），不是架构差异。其余错误形态（“click-band fox”、“It's turn off”、“funny”、“You need to tell me”）两边完全一致。宿主自报峰值 RSS 174.6 MB（含 QEMU 进程开销，与 x86 的 154 至 178 MB 同带）。**一致性判定：通过**，aarch64/musl 路径与 x86 输出等价（贪心翻转带内）。

## 7. 选型结论

1. 耳朵定 Moonshine Tiny 量化版（动态量化 encoder + decoder_merged，合计 28.1 MB）加 Silero VAD（2.2 MB），同一个 ORT 进程，宿主已可直接往调度器里接：输入 16 kHz PCM、输出带时间戳的句子行。
2. 备选 SenseVoice int8 不需要启用：触发条件（Moonshine ONNX 路径走不通）不成立。本次唯一被堵死的量化路线是 Moonshine 自己的 static int8 变体（ConvInteger 无实现），与 SenseVoice 无关。
3. 内存预算要改口径：清单第 3 条的 ASR 约 80 MB 改为约 160 MB（x86 实测带），或在总账里从别处匀。真机（A57 加 musl ORT 1.24.4）的常驻数随真机标定回填。
4. 已知短板两个：软起音句子句首词容易丢或听错（s03 的 Please）；绕口令类难句 CER 高（s07 0.367）。都是模型能力项，管线没有可改的对应项。

## 8. 流式模式与前端网关接通（2026-11-09）

耳朵接到了前端的 `/ws/asr` 后面，语音上行到文本这条链路点通。形状没有动前端：`app.js` 零改动，页面服务器 `server.js` 只把原来的计数桩换成一条原始套接字透传（见下面第 3 点，为什么绕不开这一处）。

1. **asr_host 流式模式**：新增 `--stream`（与 `--wav` 二选一，原文件模式不动）。stdin 持续读 16 kHz 单声道 s16le PCM，进程内 Silero VAD 按 512 样本逐窗跑，断句规则与文件模式共用同一套参数（阈值 0.5、800 ms 静音关段、段前后各垫 160 ms、短于 200 ms 丢弃；增量分段器 `StreamSegmenter` 是 `segment_probs` 的逐块等价实现）。每断出一句立刻识别，stdout 按行吐 JSON：`{"text": ..., "start_s": ..., "end_s": ...}`，时间是流内相对秒。SIGUSR1 是 flush 信号（网关在 ptt_end 时发）：先把管道里已写入的字节排空，再把当前未关的段按文件模式结尾段的同一规则收掉；EOF 同样 flush 后退出。stderr 照旧打统计。
2. **适配进程 `gateway/asr_gateway`**（Go，标准库，静态）：独立小进程，直接提供 `/ws/asr` 的既定形状（PCM 帧上行、`settings` 回 `ready`、`ptt_start`/`ptt_end`、下行 `asr_final{text}`）。它拉起一个 `asr_host --stream --quant --threads 2 --opt basic` 子进程，PCM 原样写子进程 stdin，子进程每行 JSON 转成一条 `asr_final` 下发（空文本段不下发，只记日志）；`ptt_end` 翻译成给子进程的 SIGUSR1；子进程退出会在下一次连接或 PCM 到达时自动拉起。产物：`gateway/asr_gateway`（x86_64，5.2 MB）与 `gateway/asr_gateway-arm64`（aarch64 静态，5.0 MB），监听默认 `127.0.0.1:18120`。
3. **server.js 的改动（已是最小）**：`app.js` 用 `location.host` 加配置里的路径拼 WebSocket 地址，配置只能改路径、指不到另一个端口，所以「页面服务器零改动、靠配置指向后端」做不到，必须有人把这个升级请求交到后端进程手里。改法照 `/api/sched` 的样子做纯透传：升级请求原样重放到后端、两条套接字对拷，不解析任何一帧、不含识别逻辑。删掉了计数桩和它专用的 WebSocket 帧代码（约 115 行），新增透传约 35 行加 `net` 引用一行、头部注释与启动日志各一处；后端地址可用 `ASR_BACKEND` 环境变量改。转发逻辑没有进网关，协议全在适配进程里。
4. **对账（同一构建、量化、opt basic、2 线程）**：文件模式对 `test/spliced.wav` 的新鲜运行（存 `test/out_stream_expected.txt`）是 6 段；流式直灌原始 PCM（EOF 收尾）出 6 段，起止时间逐段相同、文本逐段相同。经网关形状灌（`test/stream_gateway_test.js`：经页面服务器透传、按实时节奏 512 样本一帧发全文件，前 5 段靠 800 ms 规则自关、第 6 段靠文件发完后的 ptt_end flush）：6 段全到，起止与文件模式一致，文本 5 段逐字相同，第 6 段出 “And you tell me what time it is.”、文件模式是 “You need to tell me what time it is.”，正是第 3、5 节记过的贪心逐次翻转（fp32 运行里两个形态都出现过），不是流式引入的差异。结果存 `test/stream_gateway_result.json`。
5. **延迟（开发机实测，当时机器上还有调度器演示栈在抢 CPU，数字偏保守）**：VAD 自关的 5 段，从语音说完到 `asr_final` 到页面侧是 900 至 1004 ms，其中 800 ms 是断句规则本身，规则判定点之后的处理（排空、识别、转发）是 93 至 202 ms。ptt_end flush 路径：第 6 段 ptt_end 后 159 ms 到；单独一句 s01 说完立刻 ptt_end，124 ms 到，文本全对。
6. **内存**：流式稳态（全部段处理完后采样）asr_host VmRSS 161 MB（VmHWM 同为 161 MB），与文件模式的 154 至 178 MB 同带，流式没有额外常驻；流式直灌自报峰值 176.2 MB。适配进程自身 VmRSS 约 2 MB。
7. **aarch64**：`build-aarch64/asr_host` 已按附录的 clang 交叉编译法重建（含流式模式；这次的确切命令行补在下面附录里），QEMU cortex-a57 冒烟：s01 的 PCM 走流式加 EOF，出 1 段 `[0.480 1.760] Hello, how are you today?`，与 x86 逐字一致（QEMU 只验对错，load_s、asr_s 不作数）。适配进程的 aarch64 版是纯 Go 交叉编译，无系统依赖。
8. **联调脚本同步**：`ui-proto/test-client.js` A 部分原来的桩断言（asr_final 回帧数）换成真识别断言（灌 s01 的 PCM 帧、ptt_end 后回的文本要含 hello 且不带 stub 字样），A 部分全过。跑前端时起进程的顺序变成：先 `asr_gateway`、再调度器、再页面服务器，UI.md 的跑法一节已同步。

## 附：复现命令

```sh
# x86 构建
cmake -S . -B build && cmake --build build -j4
# 量化版跑拼接测试（流式分段 + 转写）
./build/asr_host --model-dir models --wav test/spliced.wav --quant --threads 2 --opt basic
# 字错率
<tts venv python> scripts/cer.py test/truth.json test/out_spliced_quant_basic.txt --spliced
# aarch64 交叉编译（R 为 Alpine 3.24.4 aarch64 rootfs；2026-11-09 流式版
# 重建时的确切命令行，-nostdinc 系是为了不让 clang 混进宿主 glibc 头，
# -Wno-invalid-constexpr 压 libstdc++ 15 与 clang 18 的 cmath 误报）
clang++ --target=aarch64-linux-musl --sysroot=$R -nostdinc -nostdinc++ \
  -I$(clang -print-resource-dir)/include -I$R/usr/include/c++/15.2.0 \
  -I$R/usr/include/c++/15.2.0/aarch64-alpine-linux-musl -I$R/usr/include \
  -I$R/usr/include/onnxruntime -std=c++17 -O2 -Wno-invalid-constexpr \
  -c src/asr_host.cpp -o build-aarch64/asr_host.o
clang++ --target=aarch64-linux-musl --sysroot=$R --ld-path=/usr/bin/aarch64-linux-gnu-ld -L$R/usr/lib build-aarch64/asr_host.o -lonnxruntime -o build-aarch64/asr_host
qemu-aarch64-static -cpu cortex-a57 -L $R ./asr_host --model-dir models --wav test/spliced.wav --quant --opt basic
# 流式模式与网关（x86）
./build/asr_host --model-dir models --stream --quant --threads 2 --opt basic < spliced.raw
cd gateway && go build -o asr_gateway . && ./asr_gateway -asr-bin ../build/asr_host -model-dir ../models
node test/stream_gateway_test.js   # 经页面服务器 /ws/asr 对账与延迟，结果 test/stream_gateway_result.json
```
