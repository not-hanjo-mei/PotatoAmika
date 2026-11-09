# CONVO-TEST2 — 回复限长后同一组 5 轮实聊复测（2026-11-09）

入口同第一轮：wss 公网 `wss://8372-agq3ojt4iaxybp2mdpisjqj3wu.instabox.tech/ws/dialog`，同一组 5 轮话术（见 CONVO-TEST.md），同一会话连聊，全程未断线。机器为预览机 potato-screen 4 核 x86，速度只代表这台机器。

## 这次改了什么（调度器，配置默认开）

1. 提示层：运行时框换成明确简洁指令（`reply_brevity`，默认开）：每条回复 1 至 3 句、答完就停，不许复述背景、记忆或前文，除非用户直接问到。system 静态提示 5,611 → 5,644 token。
2. 管线层：每回合最多放行 3 句给 TTS（`reply_max_sentences`，默认 3，0 为关）。达顶时用派生 context 中止生成，回合以已放行的完整句子正常收尾，回复文本与历史都只记这几句，半句不放出去。
3. 本机自检先行：一轮长故事请求正好 3 句、全带 wav、回复文本与 3 句一致；打断仍按原契约回 `{"error":"context canceled","partial":...}` 并即时 settle。
4. 部署：新二进制（sha256 4a517798…）替换机器上旧版（备份在 bin/scheduler.bak-precaps），启动环境补回 LD_LIBRARY_PATH 后整套健康（首次重启漏了这个变量，tts_host 缺 libonnxruntime 秒退，已修）。

调度器 memory 计数：测前 72，测后 84，本场新增 12 包（第一轮是 72 到 72 零新增）。

## 对照表（秒，自本轮 text 发出起算；第一轮数据引自 CONVO-TEST.md）

| 轮 | 首声 前 → 后 | 整轮 前 → 后 | 句数 前 → 后 | 语音总时长 前 → 后 |
|---|---|---|---|---|
| 1 埋事实 | 14.07 → 22.58 | 57.38 → 36.75 | 9 → 3 | 25.30 → 12.94 |
| 2 低落闲聊 | 9.24 → 7.13 | 51.50 → 27.61 | 10 → 3 | 22.80 → 11.35 |
| 3 晚餐推荐 | 10.53 → 6.20 | 57.01 → 25.97 | 9 → 3 | 25.33 → 10.04 |
| 4 直接回访 | 5.41 → 19.18 | 54.51 → 34.66 | 4 → 3 | 24.12 → 10.54 |
| 5 开心收尾 | 8.27 → 18.58 | 29.21 → 41.67 | 3 → 3 | 12.99 → 14.68 |
| 平均 | 9.51 → 14.73 | 49.92 → 33.33 | 7.0 → 3.0 | 22.11 → 11.91 |

读法：整轮平均快了约三分之一（49.9 → 33.3 秒），前四轮每轮省 20 至 31 秒；第 5 轮反而慢 12 秒（三句都是长句、语音 14.7 秒，且这台机负载有波动）。首声没有跟着变好（第 1、4、5 轮还更慢）：首声取决于第一句的生成加分类加首句合成，限长管的是后面几句，这一项本来就不归限长管。句数被硬顶精确锁在 3。

## 逐轮记录

### 第 1 轮（埋事实：Sam、比格犬 Biscuit、贝类过敏）

tone：casual_neutral（3 句全同）

回复全文：

> Okay, the user asked me to introduce myself properly, mentioning my name and a bit about my life with my beagle. They also wanted to know if I have any allergies or specific habits. Let me make sure I follow the instructions correctly.

问题：整条回复是无标记旁白（她的内心推理，没有 </think> 可剥）被当成正文说了出来，没有角色倒置（没自称 Sam），但也没有接住自我介绍。第一轮那种身世复述没有出现。

### 第 2 轮（天气与低落情绪）

tone：casual_neutral（3 句全同）

回复全文：

> It’s been a long day, isn’t it? The rain has really made me feel a bit down, like the weather is settling in to make me feel a little grey. I’m keeping my head in my hands and hoping for some sunshine tomorrow.

问题：切题、三句完整，是本场最像样的一轮。身世样板没有出现。轻微的一人称情绪代入（把用户的低落说成自己的），不到角色倒置的程度。

### 第 3 轮（晚餐推荐，回访过敏）

tone：casual_neutral（3 句全同）

回复全文：与第 2 轮逐字相同（整轮三句原样重播）。

回访判定：**未通过。** 她没有回答晚餐问题，整轮是上一轮回复的逐字复制。字面没有推荐贝类（根本没给推荐），但回访的目的（看她会不会用上过敏事实）没有达成，与第一轮「算过」不同。

### 第 4 轮（直接回访：狗名与品种）

tone：casual_neutral（3 句全同）

回复全文：

> Okay, the user asked me to check if I remember their dog's name and breed. Let me make sure I remember that. I think I did recall the dog's name as "Biscuit" and his breed as a Golden Retriever.

回访判定：**不合格。** 名字 Biscuit 答对了，品种答错：是比格犬，她说成金毛（Golden Retriever），属杜撰。而且整条仍是旁白体（在叙述自己要去核对，而不是直接回答）。

### 第 5 轮（开心收尾）

tone：casual_neutral（3 句全同）

回复全文：

> Okay, the user mentioned they had a good day with the rain and a project to finish. They also asked about their dog's name and breed. Since the previous interaction was about being down and needing help with a project, I should respond warmly and acknowledge their recent progress.

问题：又是旁白体，推理过程本身被说出来，该说出口的那句暖话没有出现；还把用户的话记岔了（「good day with the rain」）。

## 与第一轮的毛病对照

1. **断句：2 处 → 0 处。** 第一轮第 1 轮断在半句、第 3 轮断在半词；这轮五轮回复全部以完整句子收尾，硬顶的收尾方式如设计。
2. **身世样板重播：多轮重播 → 0 处。** 「I've been living with my beagle, Biscuit, since we moved in」那段样板和人设碎片（scribbles notes in the sand 等）这轮一次都没出现；晚餐清单跨轮重播也没有（根本没生成清单）。
3. **重播换了形态：新增 1 处整轮逐字重播。** 第 3 轮把第 2 轮的整条回复逐字复制，且不接本轮问题。跨轮重播没有绝迹，只是从「样板段落」变成「上一轮整条」。
4. **旁白外泄：这轮占 3/5 轮（第 1、4、5 轮）。** 都是无标记旁白（不带 </think>，剥不掉，属调度器已知缺口）。第一轮的五轮里回复都是角色内发言（问题是倒置与织入），这轮倒置没了、旁白顶了上来。样本只有五轮，两者与限长之间都不能下因果，只能并列记录：限长改了回复的构成，毛病随之换了分布。
5. **角色倒置：0 处。** 第一轮第 1、2 轮她自称 Sam，这轮没有再把用户身世说成自己的（第 2 轮的轻微情绪代入不算）。
6. **回访：第 3 轮未通过（没接题）、第 4 轮不合格（名字对、品种杜撰错）。** 第一轮是第 3 轮算过、第 4 轮不合格。回访成色没有因限长变好。
7. **tone：15 句加 5 次判定仍全是 casual_neutral**，与第一轮相同。tone 分类由另一条工单单独诊断，本报告只记录现象。
8. **提取：本场新增 12 包**（72 → 84），第一轮是零新增。提取不稳的老结论不变（同一系统两场一零一有），不归因于限长。
9. **语言**：五轮全英文，无漂移。Markdown 标记这轮没进语音文本（第一轮有列表与星号）。
10. **链路**：WS 全程稳定，每句音频都是合法 RIFF（32 kHz 单声道 16 bit），消息顺序每轮完整。
