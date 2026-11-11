# Omni 集成记录（2026-11-11）

LFM3VL-SPIKE.md 的三道门（LLM、ASR、VLM 均 REPLACE: YES）全过后，
按既定口径把 LFM3-Omni-2B 集进调度器与 ASR 网关，在本机（开发 VM）
全链验证。TTS 不换，仍是自研引擎与 Amika 的声音。手机部署不在
本次范围（手机通道当时不通）；没有推仓，没有动前端。

结论先行：

- 三条链都通了。大脑换 Omni（含 mmproj 常驻，拍照不再换载），
麦克风经 /ws/asr 网关改走 Omni 的 input_audio 识别，/v1/look
常驻直问大脑，409 语义保持。
- 两个行为问题必须知道，都不是通路问题，是模型与提示层的账：
一是事实归属倒置，用户介绍自己的狗与过敏时，她会顶回来或以
第一人称认领，回访时内容答对、归属不稳；二是记忆抽取把双方
的话都以第一人称存，还存自相矛盾的包（细节在第 4 节）。改
抽取与人设提示不在本次集成范围，先记账。
- 本机所有耗时数字都是这台 VM 的（1.5 个共享 vCPU），只能作
通路证据。设备口径以 spike 的手机实测为准：预填充 22.0 tok/s、
生成 6.2 tok/s（-t 4、-c 12288）。

## 1. 改了什么

调度器（scheduler/）：

- `config.go`：新增 `brain_mmproj`（大脑的视觉投影文件，空即
Minico 时代行为）与 `brain_thinking`（`effort` 为默认、发
reasoning_effort；`lfm3` 发 enable_thinking=false，且给大脑
服务器加 `--reasoning off`）。
- `servers.go`：大脑启动参数在 brain_mmproj 非空时加 `--mmproj`；
新增常驻判定；describe 在常驻模式发给大脑客户端、换载模式
发给视觉客户端（原有行为）。
- `vision.go`：/v1/look 常驻路径：直接问大脑，timings 只有
vision_answer 与 total，没有停载与恢复步骤，也就不再有恢复
失败的 500；历史落账与换载路径共用同一函数。换载路径一字未动。
- `brain.go`：思考键按 brain_thinking 切换；chat 请求（流式与
整段两处）加 `cache_prompt: true`；新增结构帧处理：Omni 会
真发人设规定的结构帧（status/tone/mode/reply 四行），调度器
只取 reply 字段进 TTS 与历史，帧的 tone 字段不消费（语气仍
走独立的 ClassifyTone 调用，与既定设计一致）。
- `session.go`：流式可释放文本改为帧感知：帧头未齐时整段扣住，
只逐字放行 reply 字段，前缀稳定；流结束兜底同样走帧提取。
拍照进行中 /v1/chat 的 503 文案改短（不再声称大脑被卸载）。
- `frame_test.go`（新增）：纯文本、整帧、多行回复、帧未齐、
流式前缀五类用例加 releasableText 回归。`go test./` 全过。
- 配置：`config.dev.json` 改为 Omni 配置（见第 2 节）；改前
的 Minico 配置整份保为 `config.minico.json`，并补上它原本
在用的视觉换载键，它是完整的回退配置。SCHEDULER.md 末尾加
指针节，API.md 的 /v1/look 一节加常驻模式说明。
- 二进制已重编：`bin/scheduler-x86_64`、`bin/scheduler-aarch64`
（CGO_ENABLED=0、-trimpath，与既有约定一致）。

ASR 网关（asr/gateway/main.go，后端重写，WS 层逐字未动）：

- 后端不再拉 asr_host 子进程。网关内做能量 VAD 断句（32 ms 帧
RMS，默认阈值 500；800 ms 静音关段，与旧规则同口径；200 ms
预卷、100 ms 尾垫、100 ms 最短段、30 s 段长上限；转写队列
深度 4，满了丢段记日志）。每段编成 WAV 发给大脑服务器的
/v1/chat/completions，用 spike 验过的 input_audio 形状，
温度 0，enable_thinking=false，单次转写超时 180 s。
- ptt_end 与客户端断线都直接关当前段（替代原来的 SIGUSR1）。
转写在单个工作协程里串行做，一段完才下一段。
- 对前端的 /ws/asr 合同一字未动：二进制 PCM 上行、settings 回
`{"type":"ready","role":"asr-gateway"}`、ptt_start/ptt_end、
下行 `{"type":"asr_final","text":...}`。
- 已知取舍写在代码头注与 ASR.md 第 9 节：能量 VAD 比 Silero
简单，轻声起音可能被削，稳态噪声会把段一直挂住。asr_host 与
Moonshine 模型原样留树里作回退。
- 二进制已重编：`asr_gateway` 9,950,157 B、`asr_gateway-arm64`
9,191,115 B（比旧版大是因为进了 net/http）。`go vet` 干净。

## 2. 配置键

调度器相关键（config.dev.json 现行值；brain_mmproj 与
brain_thinking 为本次新增，其余是既有键换了值）：

| 键 | 值 | 含义 |
|---|---|---|
| `llama_server_bin` | `lfm3vl-spike/build-x86/bin/llama-server` | spike 的 llama.cpp 构建（commit f3b407c，≥b11960） |
| `brain_model` | `lfm3vl-spike/models/LFM3-Omni-2B-Q4_0.gguf` | 大脑权重 |
| `brain_mmproj` | `lfm3vl-spike/models/mmproj-LFM3-Omni-2B-F16.gguf` | 非空即常驻视觉模式 |
| `brain_thinking` | `lfm3` | 思考键发 enable_thinking=false 并给服务器加 --reasoning off |
| `brain_threads` | 见第 5 节 | 线程取舍 |

其余键（嵌入、tts、persona、scenes、12k 上下文）与 Minico 时代
相同。回退 = 换用 config.minico.json，代码不用动。

网关新旗标：`-brain-url`（默认 http://127.0.0.1:18091）、
`-vad-threshold`（默认 500）。旧的 `-asr-bin`、`-model-dir`、
`-threads` 仍接受、忽略、启动时记一行日志，所以 deploy.sh 的
网关起法不用改。

## 3. 验证环境

开发 VM（1.5 共享 vCPU、6 GB），验证配置在
`scheduler/omni-val/`（config.json 由 config.dev.json 派生，
只改 memory_dir、tts_outdir 与 vision_timeout_s，见 4.4）。
验证脚本与全部原始输出都在该目录：drive2.out、drive3.out、
ws_test2.out、spoken_turn.out、gateway.log、scheduler*.log。
期间 VM 重启过两次（约 13:36 与 15:37），第一轮驱动的后半程
输出随 /tmp 丢失，关键结果在丢失前已读到并记在下面；验证工件
随后全部改放 workspace，重启后逐项复跑补齐。记忆文件跨两次
重启都完整保留（重启后 healthz 的 memory_n 连续）。

## 4. 验证记录

### 4.1 启动与常驻

- 起栈到 healthz 全绿约 40 s。`{"brain_up":true,"memory_n":0,
"ok":true,"tools":false,"tts":true,"vision":true}`，vision 在
常驻模式下为 true。
- 系统提示在本机以 Omni 分词器精确计数 5,145 token（人设文件
4,954 token / 8,665 字）。Minico 时代同一人设是 5,611 token
（Minico 分词器），分词器不同，数字不可直接比大小。
- 空载 VmHWM：大脑 2,210,116 kB（约 2.11 GiB）、嵌入 285,952 kB
（约 279 MB）、tts_host 197,928 kB（约 193 MB）、调度器
8,936 kB（约 9 MB）。全链跑完后：大脑 2,917,984 kB（约
2.78 GiB，12k KV 与视觉通路都用过之后的工作集）、tts_host
454,400 kB（约 444 MB）、嵌入 286,348 kB、调度器 10,976 kB。
Minico 时代同机：大脑空载 537 MB（12k）、嵌入峰值 283 MB。
手机口径（spike 实测）：Omni 服务器纯文本约 1.26 GiB、带图
约 2.19 GiB。BUDGET.md 的旧总账在 Omni 下不再成立，部署时
要重排（不在本次范围）。

### 4.2 对话轮（真人设，真会话）

名字轮跑了两遍（两轮驱动各一遍），结果一致：

- 问 “Hi there. What is your name?”，回复 “I'm Sakurai Mika.”
与 “I am Sakurai Mika.”。原始输出是完整结构帧，回复与 TTS
里只有 reply 字段，status/tone/mode 三行没有漏出（帧提取
端到端验过）。语气标签 firm1（独立 ClassifyTone 调用在
enable_thinking=false 与 json_schema 约束下正常工作，不是
neutral1 兜底）。一句话、出了 wav（合成 14.8 至 15.7 s）。
- 服务器侧：冷启动首轮 prompt 处理 5,165 token、26 至 30
tok/s；生成约 26 至 30 token 在本机约 141 至 188 s
（0.16 至 0.18 tok/s，VM 口径，见第 5 节）。
- `cache_prompt` 验通：第二轮服务器只新算了 601 个 token
（42.3 s、14.21 tok/s），5,145 token 的系统前缀整段复用。
- 每轮调度器分项（ms，两轮驱动实测的带）：generate 150,645 至
520,618、tone_classify 62,303 至 86,000 量级、extract
377,675 至 723,970、tts_drain 14,976 至 37,606。抽取在本机
占了每轮的大头（它的生成量最大），这是 VM 速度的账，不是
集成引入的开销项。

### 4.3 事实与回访（两个独立会话各跑一遍）

- 事实轮（“My name is Sam, my dog is called Biscuit, and I'm
allergic to shellfish.”）：两遍回复分别是 “I'm not Sam —
I'm Sakurai Mika. I don't have a dog or shellfish allergy.”
与 “I'm Sakurai Mika, and I'm allergic to shellfish.”。第一
遍是顶撞（把用户的自我介绍当成对她的断言），第二遍是认
领（过敏成了她自己的）。归属倒置两种形态都出现了。
- 回访轮（“So, what's my dog called, and what food shouldn't
I eat?”）：两遍回复分别是 “My dog's name is Biscuit. I'm
allergic to shellfish.” 与 “My dog is called Biscuit, and
you should not eat shellfish.”。内容两遍全对（Biscuit、贝
类），记忆链（抽取、存储、召回、注入、作答）在 Omni 下
通；归属仍不稳（狗两遍都认成自己的，过敏第二遍归对了）。
- 抽取产物（第一轮驱动落盘的 14 个记忆包，原文）：含
“Name: Sakurai Mika”、“My name is Sam”、“My dog's name
is Biscuit.”、“I do not have a dog.”、“I am allergic to
shellfish.”、“No events mentioned”、“No specific plans
mentioned”、“I prefer not to mention my name in the
conversation.” 等。问题有三：用户与助手双方的话都抽、都写
成第一人称；自相矛盾的包并存（有狗与没狗、两个名字）；
空事实也成包。Minico 时代抽取本就是已知的弱项（稳定性
A/B 记过命中波动），但第一人称认领这个形态是 Omni 这轮
的实测新账。抽取提示与人设提示的修法不在本次范围。
- 回访轮分别在拍照失败（4.4 的 502）之后与拍照成功之后跑，
两遍大脑与历史都完好，healthz 全程 ok。

### 4.4 拍照 /v1/look

- 回合中拍照：回合开始 20 s 时发 /v1/look，回 409
（“a turn is in progress; let it finish before taking a
photo”），0.02 s 返回。不排队、不掐话，与定死语义一致。
- 静息拍照第一次失败：502，300.11 s 超时。默认
vision_timeout_s=300 是按设备口径定的，这台 VM 生成太慢，
答案没出完。502 形状正确（带 partial timings），大脑在位、
后续轮次照常。验证配置把该值调到 1500 后重跑。
- 静息拍照成功：200，466.43 s（VM 口径），timings 只有
vision_answer 与 total（常驻模式形状）。答复原文：“The
image shows three solid shapes on a white background: a
yellow square in the top‑left corner, a red circle centered
in the middle, and a green rectangle (horizontal band)
filling the bottom half of the image. The upper portion is
blue, with the yellow square and red circle placed within
it.” 对真值（640×480，上半蓝、下半绿、左上黄方块、中央红
圆盘）：黄方块位置、红圆居中、绿下半、蓝上半全对；错处是
开头 “on a white background”（图里没有白色）与把红圆说成
在蓝色区内（圆盘是骑在中线上的）。spike 手机那遍的幻影
蓝带这次没有出现。仍是单图结论，不外推。

### 4.5 语音输入（网关 + 口语轮）

用 asr/test 的现成测试音（16 kHz 单声道），直连网关
ws://127.0.0.1:18120/ws/asr：

- 握手：settings 回 `{"type":"ready","role":"asr-gateway"}`，
合同形状不变。
- 测试 A（s01 + 1.5 s 静音，靠 800 ms 规则自关段）：
asr_final = “Hello, how are you today?”，与真值逐字一致；
段界 [2.144 3.456]、段长 1.61 s，转写 88,307 ms（VM 口径）。
- 测试 B（s03 后立刻 ptt_end 冲刷）：asr_final = “Please
turn off the lights before you leave.”，与真值逐字一致；
ptt_end 后 99.07 s 到（转写 98,944 ms，VM 口径）。
- 断线关段也验过：流中途断开，未关的段被收掉并转写出
“Hello, how are you today?”（95,048 ms），因无客户端在
线，结果按设计丢弃并记日志。
- 口语轮：把测试 B 的 asr_final 原文当作一轮 /v1/chat 发给
调度器（前端就是这样消费的），200，回复 “Turn off the
lights before you leave.”，出了 wav（合成 18,818 ms）。
链路全通；应答质量记一笔：她把指令复述了一遍，没有作
确认应答，与 4.3 的归属问题同属提示层账。
- 设备口径提醒：本机单段转写约 88 至 99 s 是 VM 的生成
速度；手机上 ASR 门实测整段约秒级至十秒级（spike），但
网关这条完整链路（VAD 关段到 asr_final）在设备上的延迟
还没有实测数，要等手机部署时补。

### 4.6 进程与稳态

- 验证期间调度器按 config 拉起的大脑进程参数实证：
`--mmproj <F16 投影> --reasoning off -c 12288 -t <threads>`，
与设计一致。
- 记忆跨两次 VM 重启保留（14 包 → 重启后 memory_n 14 →
后续涨到 22），memory.json 在盘上是唯一来源，与既有设计
一致。

## 5. tg 6.2 的口径与线程取舍

- 本机生成速度的三个数，分开看：llama-cli 短上下文（约 30
token 上下文、-c 2048、-t 4）打出 Generation 11.7 t/s；
服务器冷首轮在 5.2k token 上下文深度下生成约 0.16 至
0.18 tok/s（多槽并行、语气与抽取调用排队、1.5 个共享
vCPU）。两者都不是设备数字：前者没有真实上下文深度，
后者是这台 VM 的抢占现实。
- 长提示微测（用人设表层全文作提示、-c 12288、生成 32
token，同一台 VM、llama-cli 单跑）：-t 2 预填充 55.3 t/s、
生成 10.6 t/s；-t 4 预填充 48.0 t/s、生成 5.5 t/s；-t 6
预填充 27.6 t/s、生成 0.1 t/s。这台 VM 只有 1.5 个共享
vCPU，线程越多抢得越凶，-t 6 直接塌掉。本次全链验证的
栈是按暂定的 -t 6 跑的，所以第 4 节的服务内生成速度是
被线程数拖低后的保守值，不是 Omni 在本机的上限（-t 2
同口径生成约为它的 60 倍）。每档只跑了一遍，绝对值随
VM 负载浮动，但 2、4、6 的排序与差距量级与过订阅的
物理一致。
- 设备锚点只有一个，就是 spike 钉的：Dimensity 8100、
-t 4（四个 A78 大核）、-c 12288，预填充 22.0 tok/s、生成
6.2 tok/s。按它估设备上的量级：三句封顶的回复约 60 至
90 token，生成约 10 至 15 s；人设冷预填充 5,145 token
约 234 s，只在调度器启动时付一次，跨轮靠 cache_prompt
摊掉（4.2 已验）；语气与抽取这类小调用在设备上各是
几十 token 级生成，按 6.2 tok/s 是十秒量级，未实测。
- brain_threads 定版：config.dev.json 定为 4。依据是设备
  锚点：四个 A78 大核，spike 的 6.2 tok/s 正是在 -t 4 下
  测的。本机微测里 -t 2 更快（生成 10.6 对 5.5 t/s），那
  是 1.5 个共享 vCPU 的形状，只说明以后纯在本机跑活时
  可以临时降到 2，不构成改设备口径的证据。-t 6 在本机
  实测有害（生成塌到 0.1 t/s），设备上也没有任何支持
  它的数字，不要再用。

## 6. 剩余缺口

1. 事实归属与抽取噪声（4.3）：要改抽取提示（区分说话人、
第三人称写用户事实、空事实不成包）与人设里的事实处理
段，属于提示层工作，本次未动，需要用户点头后单开。
2. 设备侧全链未验：手机部署（deploy 包换 Omni 配置、网关
新二进制、BUDGET 重排）是下一步，本次范围外。设备上的
ASR 端到端延迟、拍照耗时、整栈 RSS 都还没有 Omni 口径
的实测数。
3. vision_timeout_s 默认 300 s 是设备口径；若以后还拿这台
VM 做验证，验证配置里要带大值（本次 omni-val/config.json
用了 1500，只在验证配置里）。
4. 网关能量 VAD 的两条已知弱点（轻声起音、稳态噪声）只在
办公室安静环境验过测试音，真人麦克风未验。
5. 口语轮的复述式应答（4.5）与归属问题同源，提示层修时
一并看。
6. deploy.sh 本次未改：它起网关的旧旗标仍被新网关接受，
但部署包里的调度器配置仍是 Minico 时代的，手机部署时
必须换 Omni 配置与新二进制，别直接拿旧包上机。
