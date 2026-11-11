# PotatoAmika 调度器（Go）与主链路

日期：2026-11-09。范围：改造清单第 1、3、4、10、11、12、13、17 条里调度器
负责的部分。前端接口以 API.md 为准。本文件记架构、实测账目、演示结果和
已知缺口。所有耗时都是这台开发机的数（1.5 个共享 CPU、演示期间还有别的
任务在抢核），不是手机数，手机数等真机标定。

## 架构

一个 Go 静态二进制（CGO 关，零第三方依赖，只用标准库），自己拉起三个
子进程，对外只暴露一个 HTTP 口：

```
前端 ──HTTP/SSE──> scheduler ──┬─ llama-server(Minico, :18091)   大脑
                               ├─ llama-server(Jina,   :18092)   嵌入
                               └─ tts_host(--engine 透传)         嘴
```

- 大脑与嵌入也可以 attach 已有实例（config 里 spawn=false 加 url）。
- 调度器收 SIGTERM 时带走自己拉起的子进程。
- 会话状态（历史）只在内存里，按 session_id 分；记忆是磁盘文件，跨会话。

### 一轮的流程（session.go RunTurn）

1. 召回：用户这句话嵌向量，和全部记忆包暴力余弦，取前 4 包，总注入
   长度不超过 3600 字符（清单口径）。
2. 拼 prompt：静态 system（见下）+ 记忆 system 块 + 近期窗口。窗口先
   按最近 15 轮截，再按 token 预算从最老的轮次往下丢，预算 = 4096
   总量减去 512 的生成预留。system 与记忆是钉住的，不参加裁剪。
3. 生成：llama-server 流式，effort=off 经 chat_template_kwargs 逐请求
   传，无 grammar（清单第 4 条）。流式文本先过 think 剥离的安全线才许
   切句：出现 think 标记时按剥离后的文本放行；完全没标记时长输出超过
   400 字符才放行；流结束时用完整 stripThink 兜底。这样旁白永远进不了
   TTS。剥离规则：整段 <think>…</think> 删除；残留的 </think> 之前全
   丢；首部空行去掉。单元测试覆盖了筛选里记录的几种外泄形态。
4. 定调：第一句切出时，用「用户这句话 + 第一句」调一次分类，输出被
   json_schema 枚举约束在 65 个 TTS 标签内（标签集直接读 tts-engine 的
   refpacks/index.json）。带 grammar 的调用 effort 用 low 不用 off
   （off 加 grammar 时 llama-server 回 400，清单第 4 条的实测）。
   整个回合只定一次，之后每句共用。这个标签同时选 TTS 参考、发 tone
   事件给前端驱动表情（清单第 17 条）。
5. 逐句合成：句子一切出就异步入队给 tts_host（单 worker 串行执行，
   句 N 合成时句 N+1 已经在队列里），结果按序回收并逐句推
   sentence_audio 事件。TTS 请求参数 temperature 0.85、top_k 20、
   top_p 0.9、seed 用 config 值。
6. 提取：全部句子回收后，同步调一次提取（大脑当背景工人，effort=low），
   解析按对象写（见 memory 一节），新事实嵌向量后原子写入记忆文件。
   提取不挡说话，挡的是 turn_done 和 /v1/chat 的返回。

### 静态 system prompt（persona.go）

- 表层.txt 原样读入，只做 {user_name}、{city} 两个占位符替换（原版
  加载器的做法），一字不改。
- 实测体量：全文 5,451 token（llama-server /tokenize 精确数）。4k 窗口
   装不下全文加任何历史，所以加载时按段落边界裁到 2,300 token 预算内
  （只裁长度，不改字句），其后拼调度器自己的运行时框：只说英文、回复
  简短、当天日期、scenes.json 的 surface daily_context 按日期和时段
  固定抽一条。拼完共 2,473 token（精确数）。
- 思考.txt、例子.txt 是原版深层和思考层的资产：v1 是单段生成（清单
  第 3 条），这两份不进运行时，留在原地。

### 记忆（memory.go）

- 存储：`memory_dir/memory.json`，结构 `{packets: [{id, text, vec, created}]}`。
  写入走临时文件加 fsync 加 rename，原子替换。从零建。
- 提取解析：Minico 返回的是带 ```json 围栏的分类对象
  （`{"分类": ["事实", ...]}`），不是清单。解析器三种形态都吃：分类
  对象逐类展平、纯列表、带围栏或前后杂文先剥壳。去重按归一化文本的
  sha1，语义重复（同义不同字）不去，靠召回排序自然沉降。
- 嵌入：Jina embeddings v5 text nano 的 retrieval 变体 Q4_K_M GGUF
  （156,966,976 字节），走第二个 llama-server 的 /embedding。注意它的
  返回是嵌套数组 `[[768 个数]]`，解析按两层写。向量 768 维。
- 闲时工人留了 Extractor 接口，v1 是同步实现替换点在 main.go。

### 工具层（tools.go）

只有 MCP 客户端接口和休眠实现。没有 MCP 配置文件时工具阶段根本不
进入（RunTurn 里先判开关）；有配置文件 v1 也仍休眠，因为启用方案是
两段式（意图分类加 grammar 填参），不在本轮。原生 tool_calls 全链路
没有任何地方读。

### TTS 接入（tts.go）

tts_host 子进程，stdin 逐行 JSON 协议（STAGE2.md 的口径），`--engine`
和 `--threads`、`--root`、`--outdir` 从 config 透传。客户端有一条
防御：等结果时跳过任何不以 `{` 开头的行并打日志，防止宿主的杂行让
请求和结果错位（见演示一节的那次解析失败）。

## 构建与体积

在本机用 Go 1.27.3 构建（工具链在 ~/workspace/tools/，工程本身只依赖
标准库，任何 Go ≥1.23 都能编）：

```sh
CGO_ENABLED=0 go build -trimpath -o bin/scheduler-x86_64 .
GOOS=linux GOARCH=arm64 CGO_ENABLED=0 go build -trimpath -o bin/scheduler-aarch64 .
```

| 产物 | 字节 | 说明 |
|---|---|---|
| bin/scheduler-x86_64 | 10,884,675 | 静态链接，ldd 显示 not a dynamic executable |
| bin/scheduler-aarch64 | 10,115,167 | 静态链接，musl 上直接跑（纯 Go 无 libc 依赖） |

单元测试 5 个（think 剥离、分句、事实解析、去重 ID、token 估算），
`go test ./...` 全过。

## 驻留账目（开发机实测）

| 组件 | RSS | 备注 |
|---|---|---|
| 调度器本体 | 约 8 MB | 空载，/proc VmRSS 7,808 至 9,952 kB |
| llama-server（Minico Q4_K_M，-c 4096，3 线程） | 423 MB | 清单预算约 230 至 308 MB，这是开发机 llama.cpp 的数，手机包 b9564 的数以真机为准 |
| llama-server（Jina Q4_K_M，嵌入） | 150 MB | 清单口径 138 MB 是 ONNX q4 的数，GGUF Q4_K_M 文件本身 157 MB |
| tts_host（--engine free） | 254 MB | 单独冒烟时宿主自报 rss_mb |

## 端到端演示（demo/run_demo.sh，日志 demo/demo.log）

全栈从零起（调度器自己拉三个子进程），同一会话 3 轮，开发机当时有
别的任务抢核，耗时按上限看：

| 轮 | 用户输入要点 | 回合总耗时 | 生成 | 定调 | TTS 段 | 提取 | 新增事实 |
|---|---|---|---|---|---|---|---|
| 1 | 搬到 Portland、平面设计师、贝类过敏 | 149.9 s | 35.0 s | 1.8 s | 65.4 s（3 句） | 47.7 s | 3 |
| 2 | 报了 10 月的 10k，周一开练，紧张 | 522.9 s | 62.3 s | 9.8 s | 440.9 s（10 句） | 19.2 s | 5 |
| 3 | 你记得我什么，我住哪，忌什么口 | 270.0 s | 28.8 s | 2.7 s | 227.9 s（8 句） | 12.9 s | 5 |

观察到的事实，不粉饰：

- 第 2、3 轮的回复确实带上了 Portland（历史窗口在起作用）。
- 第 3 轮的回复开头是一整段没有 </think> 标记的旁白（「Okay, the user
  just moved to Portland and is nervous…」），剥离规则抓不到无标记的
  旁白，它被当成正文合成播放了。这是 effort=off 外泄的第二种形态，
  v1 没解，见缺口。
- 第 1 轮回复是中文，第 2 轮起转英文。人设资产还是中文（清单第 13 条
  的翻译没做），350M 的大脑跟着人设的语言走，运行时框的英文要求压
  不住整段中文人设。
- 三轮的标签都是 neutral1。分类调用本身通（枚举内合法值），但这个
  大脑在这个人设下产出的回复语气平，分类没有区分度可抓；也可能分类
  输入只有第一句太少。没有单独验过分类的准确率，别把 neutral1 当结论。
- 第 1 轮第 3 句出现一次 TTS 结果解析失败（invalid character 'i'）。
  之后给客户端加了跳过非 JSON 行的防御并复验：验证轮 8 句全过，未再
  出现，也没再抓到杂行，那一行的内容没能复现定位。
- 提取出的事实里有幻觉条目（比如「喜欢 Cozy cafés」是第 2 轮回复里
  的话被当成用户事实）。提取提示词没区分「用户说的」和「回复里编的」，
  v1 按原样存了。
- 第 2 轮回复在 220 token 上限处被截断，最后一句是半句话，照样送了
  合成。回复上限和分句对半句话没有保护。

## 修复后验证（demo/verify*.log 与 verify-turn*.json）

演示里发现嵌入返回形状不对（事实全存了、向量全空、召回恒空），修了
三处后单独复验（attach 已有大脑和嵌入服务）：

1. Embed 按嵌套数组解析。验证轮 1：提取 5 条事实，全部带 768 维向量
   落盘。
2. 验证轮 2 提问住址和过敏：召回返回 4 条（正好上限），其中含
   「Moved to Portland last month」和「Allergic to shellfish」，回复
   正确用上。召回本身 31 ms。
3. 同一轮里 TTS 的 8 句全部合成成功，无解析失败（那次是带 TTS 的
   验证轮 1，日志 demo/verify.log）。

## 已知缺口

1. 无标记旁白剥不掉。effort=off 的外泄有两种：带 </think> 的已解；
   不带标记、整段都是旁白的（演示第 3 轮）目前只能靠人设框和采样压
   概率，没有可靠的剥离办法。候选办法是生成后再调一次分类判「这段
   是旁白还是台词」，要多花一次调用，没做。
2. 人设裁剪是临时账。全文 5,451 token 进不了 4k，现在按段落裁到
   2,300。裁掉的后半段包含大量行为规则，角色行为会和原版有差。根治
   要么清单第 13 条的英文版人设本来就更短，要么上下文口径放宽，两件
   都不在本轮。
3. 回复语言不稳。人设是中文、运行时框要求英文时，大脑在两种语言间
   漂。清单第 13 条的资产翻译做完前，这个问题只能缓解（把运行时框
   的英文要求放进每一轮的 user 侧提醒，没做，怕污染对话记录）。
4. 提取质量：事实里混回复侧的编造内容；同义事实重复存（第 1 轮和
   第 3 轮各存了一条 Portland）。要加一道「只提取用户说过的话」的
   约束和语义去重，都没做。
5. 半句话会送合成。生成撞 token 上限时最后一句不完整就进了 TTS；
   另外回复带 markdown 列表时按句切会把列表项逐条念出来（演示第 2
   轮）。念稿前的文本清洗（去 markdown、合并短句）没做。
6. tone 分类恒 neutral1 的病根已定位并修复（2026-11-09，详见文末
   「tone 分类诊断」一节）：分类调用的 24 token 预算被 effort=low 的
   思考吃光，JSON 从未生成，每次都走解析失败回退。修后管线通了，
   但分类准确率本身一般（15 条带标签场景同族命中 7/15，中性输入全
   错），是 350M 的能力上限问题，待定夺；另建议把 session.go 里被
   吞掉的 tagErr 记进日志，这次没动那个文件。
7. ASR 不在本轮。接口只收文本，浏览器采集和 ASR 接入后从 /v1/chat
   进，接口不变（API.md 已注明）。
8. 会话历史不落盘。调度器重启即丢对话，只保记忆文件。原版的行为没
   查对，v1 先这样。
9. 打断已修复并实测（2026-11-09）。原实现的洞：/v1/interrupt 处理
   函数无条件自己先发一条 interrupted，而 RunTurn 的句子回收循环
   不看取消信号，生成结束后落在回收阶段的打断完全无效（前端联调
   B 部分实测：打断回了 200、事件也到了，之后又推 12 句、全合成、
   以 turn_done 收尾）。修法：Session 改为登记在途回合（active
   turnState），处理函数只取消 ctx、只在确有回合在途时才调
   DropQueued，不再发事件；RunTurn 在生成、tone 分类、回收、提取
   每个阶段的边界查 ctx，所有打断出口收敛到一个 unwind，只发一条
   interrupted 收尾；回收循环的等待改为带 ctx 的 select，排队句子
   靠各自 ctx 在 TTS worker 取活时作废，正在合成的一句在宿主里
   合成完，但结果落进无人读的缓冲通道、不推送；历史改为提取之后
   才提交，被打断的回合不写历史、不触发记忆提取。语义已定死进
   API.md（收尾二选一、502 partial 的形状、无回合时打断不发事件）。
   复验（interrupt-test.js 对 attach 模式实例 :18103，日志
   interrupt-verify.log，全 PASS）：回收阶段打断（狐狸故事长回复，
   第一句 sentence_audio 后打断），后续 sentence_audio 0 条（原 bug
   是 12 条），interrupted 唯一且收尾，无 turn_done，/v1/chat 立即
   以 502 partial settle，记忆包 0 增；生成早期打断（turn_started
   后 1.5 秒），chat 7 秒 settle（整轮在开发机要数分钟），0 句
   音频，记忆包 0 增；同一会话随后正常跑完一轮（200、4 句、
   turn_done）。不写历史这一条没有外部可观测面，靠代码构造保证
   （提交点在提取与取消检查之后）加记忆包 0 增旁证。正在合成的
   一句停不下来仍是 tts_host 协议的限制，结果丢弃不推送。

## 12k 切换与人设全量加载（2026-11-09 复验）

用户 2026-11-09 定上下文从 4k 拉到 12k，多出的内存已认账（改造清单第 3 条）。本节数字取代上文 4k 时代的口径：上文「静态 system」一节的 2,473 / 2,300 与已知问题第 2 条（人设裁剪）自此为历史记录。

改动三处：

1. 大脑 llama-server 上下文 4,096 改 12,288（config 的 BrainCtx 与 ContextTokens 默认值）。嵌入的 llama-server 未动。会话窗口逻辑读 ContextTokens，自动跟随。
2. 人设改全量加载：buildSystemPrompt 不再默认裁剪。裁剪保留为配置开关 persona_trim（默认关），开时仍按 SystemBudgetTokens（2,300）走原来的 trimToTokenBudget，随时可回退。
3. 修场景注入读错键：sceneSnippet 原读 surface.daily_context，scenes.json 里没有这个键，场景行自始为空。改读 surface.context（占位符 {user_name}、{city}），scenes_en.json 同步核对，键名与结构一致。原先误用的 periodForHour 已删。

复验实测（开发机，/tokenize 精确计数；消息的 chat 模板开销每条几个 token，未计入）：

1. 启动日志精确计数：表层全文 5,451 token；静态 system 合计 5,611，运行时框 160。其中场景串本身（替换占位符后）120 token，加「Your day so far: 」前缀构成场景行。框从 4k 时代的约 35 涨到 160，增量正好是场景行，注入非空已证。
2. 逐轮 token 账（演示同前 3 轮原文，记忆全新）：

| 轮 | 静态 | 召回 | 窗口 | 用户输入 | 提示合计 | 输出 | 总计 |
|---|---|---|---|---|---|---|---|
| 1 | 5,611 | 0 | 0 | 31 | 5,642 | 71 | 5,713 |
| 2 | 5,611 | 0 | 102 | 23 | 5,736 | 115 | 5,851 |
| 3 | 5,611 | 47 | 240 | 21 | 5,919 | 88 | 6,007 |

3. 第 3 轮总计 6,007，占 12,288 的 49%，余量 6,281。3 轮后窗口共 349 token；按此密度 15 轮窗口约 1,700 至 1,800。buildMessages 的提示预算是 12,288 减 512 等于 11,776，静态 5,611 加召回上限（3,600 字符约 1,000 token 以内）加满窗口加用户输入，离预算还远，窗口不会被挤。
4. 大脑 llama-server RSS：空载（全新启动、尚未生成）549,596 kB，约 537 MB；演示第 1 轮进行中抽样 596,336 kB，约 582 MB。对 4k：本报告前文记的 423 MB 口径下空载增约 114 MB；按主会话单独量的 4k 空载 439 MB 口径增约 98 MB，与定 12k 时预估的约 97 MB 对得上。手机口径仍待真机标定。

演示结论（开发机同时有其他线在跑，耗时只作上限口径：三轮墙钟 212 / 388 / 135 秒，TTS 等待占大头，与 4k 轮的 150 / 523 / 270 不可直接比）：

1. 三轮端到端通过，记忆提取与召回照常工作：共存 9 包，第 3 轮召回 4 条，「Allergic to shellfish」在内。
2. 语言与上一轮同口径对比：本轮三条回复全英文；4k 轮第 1 条中文、第 2 与第 3 条英文。人设与语言行为不在本任务范围，此处只记观察，样本只有一轮，不下定论。
3. 4k 轮第 3 条出现的无标记旁白外泄，本轮未出现；本轮三条回复都是正文。本轮无 </think> 残段。
4. 质量观察如实记，不在本任务处理：第 1 轮提取 0 条事实（4k 轮第 1 轮提取 3 条）；第 1 条回复与过敏事实相矛盾（说了「you'll enjoy shrimp」）；第 3 轮召回过敏事实后回复口径正确。
5. 既有现象，非本次引入：一轮的最后一句偶尔拿到空的 TTS 结果（无 wav、无报错），本轮第 2 轮 sent8 与第 3 轮 sent4 各一次；4k 轮日志里也有 1 次。留作已知项。

## 眼睛拍照换载（2026-11-09 实现并验通）

清单第 6 条的集成步：视觉模型（LFM2.5-VL-450M Q4_K_M 加 mmproj F16，从 potato-screen 取回本机 scheduler/models/，sha256 与源逐字一致）不常驻，拍照走 BUDGET.md 定的换载流程。入口是调度器侧 POST /v1/look（API.md 有契约），前端 v1 没有 camera 通道，ui-proto 一行未动。

实现（vision.go、servers.go）：

1. 子进程收归 serverManager 统一管理（大脑、嵌入、视觉三台 llama-server 的拉起与停止只此一处），启动、SIGTERM 收尾、换载重载共用同一套参数：大脑仍是 -c 12288 加 --jinja，嵌入参数与原先逐字相同。重载用的就是启动参数，不存在第二份口径。
2. Look 流程：先查有没有回合在途（任一会话 active 非空就回 409，不掐说一半的话）；换载全程置 swapping 标志，/v1/chat 在此期间直接 503、RunTurn 入口再兜一层；拍照请求之间用 TryLock 串行，第二个回 409。
3. 停载等进程真退出才往下走（SIGTERM、15 秒宽限、然后 SIGKILL 并等回收），内存真让出来了才加载视觉。RSS 采样证实三台进程的存活区间互不重叠。
4. 失败回滚是结构性的：视觉加载或答复一失败，先走 restoreAfterVision（卸载视觉、按原参数重载大脑与嵌入、等健康）再返回错误；恢复跑在一个全新的后台 ctx 上，拍照请求被取消或超时也掐不断回滚。只有恢复本身失败才回 500。等待健康的 waitUp 带进程退出检测：子进程启动期就死（比如模型路径错）几秒内报错，不死等 180 秒。
5. 历史与记忆都在调度器侧，换载不碰：拍照一轮以 (showed a photo) 加问题、答复的形式进会话历史，不跑记忆提取。配置在 config.vision.json（测试实例，端口 18120 至 18123、TTS 关）；视觉参数进 Config（vision_model 等），模型路径为空或 attach 模式时 /v1/look 回 503、health 的 vision 为 false。

开发机实测（同机有其他线抢核，耗时按上限口径；各 llama-server 2 线程，模型走页缓存热载）：

| 步骤 | 拍照轮 1 | 拍照轮 2（冒烟） |
|---|---|---|
| 停大脑加嵌入 | 8,605 ms | 303 ms |
| 视觉加载到健康 | 4,084 ms | 4,532 ms |
| 视觉答复（上限 300 token） | 15,984 ms | 9,559 ms |
| 卸载视觉 | 121 ms | 186 ms |
| 重载大脑加嵌入到健康 | 5,083 ms | 5,524 ms |
| 整轮 | 33.9 s | 20.1 s |

停的那一档波动最大（0.1 至 8.6 秒）：刚生成完的进程收尾慢、空闲进程秒退，两轮和失败轮都见过。答复内容与 VISION-REF 同档：主色块对（上蓝下绿、黄方块），红圆一轮认成底部红带、一轮认对，450M 的细节上限。

RSS 峰值（/proc VmHWM，4 Hz 采样，贯穿全部测试轮）：视觉 682 MiB（VISION-REF 在 potato-screen 量的是 685 MiB，对得上）；大脑空载基线 537 MiB（与 12k 一节的数一致），生成后涨到 581 至 656 MiB；嵌入峰值 283 MiB，比驻留账目表里的 150 MB 高，那是单点采样、这是全程峰值，口径不同，账目表未改，特此记明。

两条验收：

1. 历史与召回跨重载：第 1 轮埋事实（狗名 Biscuit、柯基，提取存 6 包），拍照一轮，第 3 轮问狗名与品种，召回 4 包含 Biscuit 条目、回复答对。召回要现嵌查询向量，等于把重载后的嵌入也一并验了。拍照后 /v1/memory 计数与拍照前一致。
2. 失败回滚两条都验了：坏图片（合法 base64、内容非图片）视觉回 400，整轮 7.7 秒以 502 返回、大脑已恢复，随后一轮对话正常；坏 mmproj 路径（独立实例 config.vision-badmodel.json）视觉进程启动期退出，3.1 秒以 502 返回，日志里大脑与嵌入的进程号在回滚前后换新（42327/42348 → 42462/42477），回滚后对话 200。

附带修掉一个真隐患：第一次跑坏路径实例时，上一轮崩掉的调度器留了两个孤儿 llama-server 占着端口，新实例的子进程绑不上当场死，健康检查却被孤儿应答骗过，整轮换载在假象里空转（停和重载都是毫秒级假数）。waitUp 因此加了确认窗口：我们自拉的进程在首次健康应答后还要活过 1 秒才算启动成功。孤儿是测试环境事故（调度器在拉起子进程后、记忆目录 mkdir 撞配额挂掉，子进程没人收），生产口径下调度器退出有 stopAll 收尾，但这个确认窗口留着防同类假象。

构建产物随本节更新：bin/scheduler-x86_64 10,947,804 字节、bin/scheduler-aarch64 10,128,534 字节，go vet 与单元测试全过。

## tone 分类诊断（2026-11-09）

起因：历次演示与原版页 5 轮实聊共 40 次分类全出 neutral1（前端显示
casual_neutral），用户明确低落与开心的输入也纹丝不动。把 ClassifyTone
从代码里摘出来单独喂（同模型、同 --jinja 服务、请求逐字段复刻：提示词、
max_tokens、temp 0、top_k 50、repeat_penalty 1.05、effort low、65 标签
json_schema 枚举），15 条情绪指向明确的英文输入（含 CONVO-TEST 第 2、5
轮原句，分类输入按生产口径取回复第一句），结论如下。

病根是管线，不是模型：原样复刻的 15 条全部 finish_reason=length、
message.content 为空。effort=low 的思考走 reasoning_content 单独字段，
实测要 38 至 86 个 token，JSON 本体只占约 7 个；原预算 max_tokens=24 在
思考没结束时就耗尽，JSON 从未写进 content。chatOnce 只读 content，得空
串，解析必然失败，ClassifyTone 每次返回 error；session.go 的回退把标签
定为 neutral1，且 tagErr 被 `_ = tagErr` 吞掉、无任何日志，所以这个失败
无声地持续了全部演示。思考片段本身证明模型在正确分析情绪线索（sad 输入
的思考里写着 "feeling down"、angry 写着 "infuriating"），是结果出不来。

逐项核对的其余嫌疑均排除：标签集与 refpacks/index.json 的 65 键一致、
解析后成员校验正常；枚举顺序无位置偏好（枚举整表反序复跑，分布逐字
不变；生产侧 loadTags 是 Go map 随机序，因此也无碍）；stripThink 作用于
content 不是元凶；effort 改 off 加 grammar 在本机构建上实测 HTTP 400
（sampler 初始化失败），代码注释属实，off 这条路不存在。

修复（只动 brain.go 的 ClassifyTone）：max_tokens 24 改 160。实测单次
用量 45 至 93 个 token，160 留约 1.7 倍余量。go build 通过。同组 15 条
复验：回退 15/15 降为 0，标签分布散开到 12 种（fear1、sad1×2、
disappointed1、whisper1、happy1、happy2×2、frustrated1×2、annoyed1、
surprise1、lonely1、breathy1、embarrassed1 各 1），与 256 预算探针逐条
一致，证明 160 已不截断。

修后准确率如实记（按情绪族算命中）：强信号条目全中（sad2、sad3、happy2、
happy3、angry1、angry2、excited1），合计 7/15。错得离谱的有：开心收尾
（CONVO 第 5 轮）给 whisper1、蜘蛛那条给 lonely1、三条中性输入分别给
frustrated1、breathy1、embarrassed1（中性族一次都没被选中）、害羞两条给
sad1 与 surprise1。管线修好后剩下的就是 350M 在这个提示下的分类能力，
与选型期「模型自带 tone 字段语义错 6 条」的先例同性质。给每标签配示例、
多步分类等改法未擅自上线，等定夺。另一件建议：session.go 回退时把
tagErr 写进日志，避免同类静默失败再藏这么久（该文件有其他工单在改，
本次未动）。

### 续：预览机复核（2026-11-09 晚）

限长复测（orig-ui/CONVO-TEST2）五轮 tone 仍全 neutral，复核结论：预览机
跑的根本不是含修复的二进制，与整回合路径无关，没有第二层病根。

证据两条。哈希：机器 bin/scheduler 与本机 bin/scheduler-x86_64 的旧构建
逐字相同，sha256 均为 4a517798…，即限长工单 07:11 构建、07:21 部署的那
件；tone 修复约 07:15 才落进源码，那次构建没赶上。反汇编：对该件的
Brain.ClassifyTone 在调 chatOnce 前取 maxTokens 的指令是
mov $0x18,%r9d，即 24，旧值实锤；本机现源码构建的同位置是 $0xa0（160）。

处置：以当前源码（含 tone 160 与限长两项，config 默认
reply_brevity=true、reply_max_sentences=3）重构建，go vet 与单元测试
通过，新件 sha256 972cae53…（x86_64 10,988,834 字节；aarch64
917fc633…，10,200,020 字节）。反汇编复验新件 ClassifyTone 为 $0xa0 后
部署预览机（旧件备份为 bin/scheduler.bak-pretone，缺 LD_LIBRARY_PATH 的
老坑按进程环境原样带回），整套重启后 healthz 全绿。

整回合路径复验（:8372 原版页 wss 链路、真回合）：输入明确低落一句，
tone 事件 casual_sad、三句同标；输入明确开心一句，tone casual_smile、
三句同标。tone 在生产路径上已离开 neutral，与直调分类步的结论一致。

内存附账（同轮普查，07:35 UTC）：逐角色实例数与设计一致，无重复进程。
llama-server 恰 2 份（大脑 43127，684 MB；嵌入 43158，285 MB，均为现任
调度器子进程、占 18091/18092）；tts_host 恰 2 份（调度器 43205，343 MB；
适配器 tts_say 自拉 30522，375 MB，设计内）；asr_host、网关、页面、
适配器、cloudflared 各 1。日志显示限长部署在 07:22 有过两次启动
（07:22:13 与 07:22:43，第一发 tts_host 因缺 LD_LIBRARY_PATH 秒退），
第一发的 llama 一对（pid 35909/35956）与第二发的（36178/36196）均已逐个
验死，无视觉工单那次的孤儿残留。对照驻留账：未重启过的件在数小时运行
后持平或低于账面（asr_host 151 对 163、页面 node 28 对 52、cloudflared
21 对 41）；调度器 tts_host 重启前（CONVO-TEST2 五轮后）约 340 MB、重启
后一轮约 343 MB，同值。整机内存台阶式上升定性为工作集到顶型：12k KV
随生成逐页撑开（大脑空载 537 至满载约 684）、tts arena 撑开、适配器
say 宿主峰值 HWM 曾达 641 MB 的瞬时值，外加驻留账写成之后新增的适配器
一对（约 420 MB）不在原账内；不是随轮次无界累积。记忆包 100 个
（data/memory 单文件 1.3 MB，落盘增长属设计内），调度器 RSS 18 MB，
会话留存不构成内存项。重启前 Gen A/B 大脑的逐进程 RSS 未在重启前抓取，
图中 2.28 GiB 的瞬时峰值无法逐项重构，以上归因中瞬时叠加部分为推断，
现行普查与日志时间线为实测。

## /v1/say 与原版页上手机（2026-11-09）

用户定：手机上也要原版 Use 页，适配器为此魔改后进部署包，预览与手机同构。

- 调度器新增 POST /v1/say（契约见 API.md）：回合外独立合成走常驻 tts_host，
  不写历史、不提取、不发事件。与回合的互斥定死为 409 不排队：say 的调用方
  （向导台词、名字试听）按构造都在无回合时说话，撞上回合立即回绝、页面跳过
  该句；排队在手机上最坏等一整轮且等完仍可能被打断丢弃。本机与预览机均实测：
  正常出真 wav、空文本 400、回合中 409、回合结束恢复可用。
- 适配器（orig-ui/adapter.js）删掉自拉的私有 say tts_host（预览实测 375 MB，
  手机账出不起），/api/tts_say 改调 /v1/say。魔改版已同步上预览机：验收时
  全机 tts_host 恰一份（调度器的），公网 /use 200、wss 一轮事件序列完整
  （ready→commit→tts_audio×3→ai_surface_done→ai_end）、tts_say 出 2.62 秒
  真 wav（happy2）；RSS 调度器 18 MB、适配器 54 MB（空载 38 MB）。
- 部署包已重打（deploy/MANIFEST.md）：适配器 8372 为正式前端，ui-proto
  退役出包；Live2D 模型不进 tar，装机时按 orig-ui/live2d-files.txt 从
  hf-mirror 逐件拉并验哈希，自检含 /v1/say 与模型发现两条。手机常驻账
  更新为约 1,019 至 1,075 MB、余 236 至 292 MB（BUDGET.md）。

## 大脑换 Omni 与三合一（2026-11-11，详见 OMNI-INTEGRATION.md）

LFM3VL 三门全过后按既定口径集成：`config.dev.json` 现为 Omni 配置
（大脑 LFM3-Omni-2B Q4_0 + F16 mmproj 常驻、spike 的 llama.cpp 构建、
思考键 `brain_thinking: "lfm3"` 即 enable_thinking=false），原 Minico
配置整份保为 `config.minico.json` 回退。新增代码点：chat 请求加
`cache_prompt: true`（第二轮起系统前缀整段复用，实测第二轮只算
601 个新 token）；Omni 会真发人设的结构帧（status/tone/mode/reply），
调度器在流式与整段两处只取 reply 字段进 TTS 与历史（frame_test.go
覆盖）；/v1/look 改常驻直问大脑（API.md 已补常驻模式一节）；ASR
网关后端改走 Omni 的 input_audio（ASR.md 第 9 节）。TTS 不动，仍是
自研引擎与 Amika 的声音。验证在本机全链跑通（名字、回访、拍照、
409、语音输入轮），已知质量账（抽取噪声、事实归属倒置）与数字全
在 OMNI-INTEGRATION.md。
