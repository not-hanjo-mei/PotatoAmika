# PotatoAmika 调度器前端接口（v1，以本文为准）

前端（浏览器页 + Live2D）只和调度器说话。传输是 HTTP + SSE：请求走 HTTP，
回合内的推送走一条 SSE 事件流。事件对象的字段就是契约本身；日后换
WebSocket 传输时，事件对象原样搬过去，字段不变。

Base：`http://<host>:18100`（config 的 listen）。

## 端点

### GET /healthz

存活与组件状态。

```json
{"ok": true, "brain_up": true, "tts": true, "tools": false, "memory_n": 3, "vision": true}
```

`tools` 在 v1 恒为 false：工具层休眠，没有 MCP 配置时工具阶段不启动。
`vision` 表示拍照接口可用（视觉模型已配置且大脑、嵌入是调度器自拉的）。

### POST /v1/sessions

建会话，返回 `{"session_id": "<hex>"}`。不建也能聊：/v1/chat 不带
session_id 时用 `default` 会话。

### POST /v1/chat

发一轮对话，阻塞到本轮全部句子合成完并返回完整结果。本轮的
`sentence_audio` 事件同时在该会话的 SSE 流上推送，前端可以不等这个
响应就开始播第一句。

请求：

```json
{"session_id": "abc123", "text": "I just moved to Portland last month."}
```

响应（TurnResult）：

```json
{
  "session_id": "abc123",
  "reply": "Portland, huh. New place, new mess to unpack.",
  "tag": "playful1",
  "sentences": [
    {"index": 0, "text": "Portland, huh.", "tag": "playful1",
     "wav": "/abs/path/demo/tts-out/req001.wav", "duration_s": 1.42, "synth_ms": 5210},
    {"index": 1, "text": "New place, new mess to unpack.", "tag": "playful1",
     "wav": "/abs/path/demo/tts-out/req002.wav", "duration_s": 1.87, "synth_ms": 6300}
  ],
  "recalled_facts": ["The user lives in Portland"],
  "facts_added": 2,
  "timings_ms": {"recall": 41.2, "tone_classify": 380.5, "generate": 4200.1,
                 "tts_drain": 6300.4, "extract": 1500.0, "total": 12400.9}
}
```

字段说明：

- `tag` 是本轮唯一的情绪标签。它同时做三件事：选 TTS 参考音、通过
  `tone` 事件发给前端驱动 Live2D 表情、写进每个 sentence 对象。
  取值是 TTS 参考标签集（tts-engine 的 refpacks index 的键，共 65 个，
  如 neutral1、happy2、angry4、whisper1）。标签在宿主侧无对应参考包时，
  TTS 宿主自己回退 neutral1 并在该句的 `tag` 里报实际用值（sentence 的
  `tag` 是宿主回报的实际值，可能与回合 `tag` 不同）。
- `sentences[].wav` 是调度器本机路径。前端在同一台设备上时直接按路径
  播；跨设备时由前端线另行约定取文件方式（v1 不提供音频下载端点）。
- `sentences[].error` 非空表示该句合成失败（文本照常在 reply 里）。
- 出错时 HTTP 502，body 是 `{"error": "...", "partial": <TurnResult>}`。

### GET /v1/events?session=<id>

SSE 流。每条事件是一个 `data:` 行，内容是一个 JSON 对象：

```json
{"type": "sentence_audio", "session_id": "abc123",
 "sentence": {"index": 0, "text": "...", "tag": "playful1",
              "wav": "...", "duration_s": 1.42, "synth_ms": 5210}}
```

事件类型：

| type | 时机 | 关键字段 |
|---|---|---|
| turn_started | 一轮开始 | text（用户原话） |
| tone | 本轮标签定下（第一句切出时） | tag |
| sentence_audio | 一句合成完 | sentence 对象（同 /v1/chat 的 sentences 元素） |
| turn_done | 本轮正常结束（含记忆提取） | reply、tag |
| interrupted | 本轮被打断（收尾事件） | 无 |
| error | 本轮失败 | error |

顺序保证：tone 在第一条 sentence_audio 之前。每轮只有一个收尾事件：
turn_done 和 interrupted 二选一，永远是本轮最后一个事件。
断线重连不补发历史事件，状态以 /v1/chat 的响应为准。

### POST /v1/interrupt

打断指定会话正在进行的一轮。请求：`{"session_id": "abc123"}`，返回
`{"ok": true}`。没有正在进行的一轮时也返回 ok，事件流上什么都不发。

打断在生成的任何阶段都生效：

- 大脑还在流式生成：给 llama-server 的请求当场断开，生成停下。
- 正在定 tone：分类请求一并取消。
- 句子在排队合成：还没开始合成的句子全部丢弃。正在合成的那一句
  停不下来，它会在 TTS 宿主里合成完，但结果不推送。已经发出的
  sentence_audio 不回收，停播由前端负责。

收尾语义定死：被打断的回合在事件流上只发一条 interrupted 收尾，
不发 turn_done；被打断的回合不写入会话历史，也不触发记忆提取。
这轮的 /v1/chat 以 502 返回，body 是 `{"error": "context canceled",
"partial": <TurnResult>}`，partial 的 sentences 只含打断前已经推过
sentence_audio 的句子。

### POST /v1/say

回合外的独立合成：原版 Use 页的向导台词与名字试听走这里（经它的适配器
`/api/tts_say` 转发）。用常驻的那一份 tts_host 合成一句，不属于任何
会话：不写历史、不跑记忆提取、不在事件流上发任何事件。合成与回合的
句子共用同一条 TTS 队列（FIFO 串行），音频文件落在同一个输出目录。

请求：

```json
{"text": "Hi, I'm Amika. What's your name?", "tag": "happy1"}
```

`tag` 可省，是 TTS 参考标签（同 /v1/chat 的取值集），省了用 neutral1；
标签在宿主侧无对应参考包时宿主自己回退并在 `tag_used` 里报实际值。
文本上限 500 字符。

响应（200）：

```json
{"file": "/abs/path/tts-out/req042.wav", "duration_s": 2.15,
 "elapsed_ms": 4310.2, "tag": "happy1", "tag_used": "happy1"}
```

错误语义：

- 400：文本为空或超长、body 不是合法 JSON。
- 409：有回合正在进行。与 /v1/look 同一条规则，定死不排队：say 的
  调用场景（向导、试听）按构造就发生在无回合时，真撞上时立即回绝、
  调用方跳过这一句即可；排队在手机上最坏要等一整轮（分钟级）且等完
  仍可能被打断丢弃，页面请求全程悬着，收益为零。检查点在请求到达
  时刻；检查刚过回合才起步的极端竞态下，say 在队列里排在该回合句子
  之前先合成完，不影响回合正确性。
- 502：tts_host 合成失败（error 带宿主原文）。
- 503：本调度器没开 TTS（纯文本模式）。
- 拍照换载期间可用：只用 tts_host，不碰大脑。

### POST /v1/look

拍照问答（调度器侧接口，v1 前端没有 camera 通道，浏览器不调它）。
视觉模型不常驻：一次调用内完成整轮换载（停大脑与嵌入 → 加载视觉 →
答复 → 卸载视觉 → 按原参数重载大脑与嵌入），返回时对话已恢复。

请求：

```json
{"session_id": "abc123", "question": "What is on the table?",
 "image_b64": "<base64>", "mime": "image/png"}
```

`mime` 可省，默认 image/png；图片上限 16 MB。

响应（200）：

```json
{"answer": "A red mug next to a notebook.",
 "timings_ms": {"stop_servers": 300.1, "vision_load": 4500.2,
                "vision_answer": 9600.3, "vision_unload": 180.4,
                "reload_servers": 5500.5, "total": 20100.6}}
```

错误语义：

- 400：参数缺失、base64 非法或图片超限。
- 409：有回合正在进行（不许掐断说一半的话），或另一个拍照还在进行。
- 502：视觉加载或答复失败。大脑与嵌入已先恢复再返回错误，error
  文案带 `(brain+embed restored)`；body 带 `partial`（含已完成步骤
  的 timings_ms）。
- 500：恢复本身失败（整机处于无大脑状态，需人工介入）。
- 503：视觉未配置（/v1/chat 在换载进行中也回 503，稍后重试）。

这一轮进会话历史（user 记为 `(showed a photo)` 加问题，assistant
记答复），后续回合可以指代照片内容；不跑记忆提取（答复是视觉模型
的描述，不是用户陈述）。换载期间会话历史与记忆存储不受影响，它们
都在调度器侧。

常驻模式（2026-11-11，Omni 配置）：配置带 `brain_mmproj` 时大脑
服务器自身带视觉投影，不再换载。/v1/look 直接问大脑，响应
timings_ms 只有 vision_answer 与 total；没有 stop/reload 步骤，
也就没有 500（恢复失败）一档；502 只表示答复本身失败，大脑全程
在位。409 语义不变（回合进行中或另一个拍照进行中均拒绝，不排队）。
/v1/chat 在拍照进行中回 503 的规则不变。healthz 的 vision 字段在
常驻模式下随大脑在位为 true。Minico 回退配置（config.minico.json）
不带 brain_mmproj，走上面写的换载流程，一字未改。

### GET /v1/memory

当前记忆包列表（调试用）：`{"packets": [{"id": "...", "text": "..."}], "count": n}`。
不带向量，向量只存调度器本地文件。

## 一轮的时序

1. 前端 POST /v1/chat（或先连好 /v1/events 再发）。
2. 调度器召回记忆、生成回复（流式）、第一句切出时定 tone 并推送 tone 事件。
3. 每句合成完推送 sentence_audio，前端按 index 顺序播放，同时用该句的
   tag 切换表情，口型按播放音量起伏（清单第 17 条）。
4. 全部句子完后调度器同步跑记忆提取，然后 /v1/chat 返回、turn_done 推送。

## 不在 v1 里的

- 音频上行（ASR）不在这个接口里：浏览器端采集与 ASR 接入是另一条线，
  本接口只收文本。ASR 线接好后，前端把识别文本当 text 发进来即可，
  接口不变。
- 工具调用：v1 没有，前端不会收到任何工具相关事件。
