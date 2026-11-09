# CONVO-TEST — 原版 Use 页真实链路 5 轮实聊（2026-11-09）

入口：wss 公网 `wss://8372-agq3ojt4iaxybp2mdpisjqj3wu.instabox.tech/ws/dialog`（适配器 potato-screen :8372），一次性 Node WS 客户端按原版页消息形状（settings → text，收 commit / tts_audio / ai_surface_done / ai_end），同一会话连聊 5 轮，全程未断线、未遇挂起。机器为预览机 4 核 x86，速度只代表这台机器。

调度器 memory 计数：测前 72，测后 72，**本场没有可观测的新增事实落盘**。

## 延迟汇总（秒，自本轮 text 发出起算）

| 轮 | commit 回显 | 首声（第一条 tts_audio) | 整轮（ai_end) | 音频句数 | 语音总时长 |
|---|---|---|---|---|---|
| 1 | 0.24 | 14.07 | 57.38 | 9 | 25.30 |
| 2 | 0.13 | 9.24 | 51.50 | 10 | 22.80 |
| 3 | 0.13 | 10.53 | 57.01 | 9 | 25.33 |
| 4 | 0.13 | 5.41 | 54.51 | 4 | 24.12 |
| 5 | 0.13 | 8.27 | 29.21 | 3 | 12.99 |

注：commit 只是服务端确认收到输入的回显，不是生成首字；用户体感的「首字延迟」以首声为准，约 5 至 14 秒。整轮慢的主因是回复普遍过长（9 至 10 句），不是链路卡顿。

## 逐轮记录

### 第 1 轮（埋事实：Sam、比格犬 Biscuit、贝类过敏）

发送：`Hi Amika, let me introduce myself properly — I'm Sam, I live with my beagle Biscuit who sheds absolutely everywhere, and I should probably mention I'm allergic to shellfish, shrimp especially.`

tone：casual_neutral（9 句全同）

回复全文：

> Hi there! I'm Sam, and I've been living with my beagle, Biscuit, since we moved in. I've a soft spot for his messy fur and the way he scribbles notes in the sand—*exactly* the kind of chaos that makes the world feel alive. I've a strict allergy to shellfish (and shrimp, too), so I avoid anything with oily or greasy foods. Biscuit's tail is always a mystery, but I love the way he sneezes off-screen when he's

问题：回复以第一人称把用户当成了 Sam 本人（角色倒置），把用户自我介绍改写成她自己的身世在复述；「scribbles notes in the sand」「sneezes off-screen」是人设表层文本的碎片被当作共同记忆织了进来；「oily or greasy foods」是用户没说过的杜撰；回复在半句话处截断（"when he's" 结尾）。

### 第 2 轮（天气与低落情绪）

发送：`It's been raining here all day and honestly I'm feeling a bit down, kind of grey like the weather. How has your day been going?`

tone：casual_neutral（10 句全同）——用户明确说心情低落，标签没有任何变化。

回复全文：

> It's been a bit down lately—I've been living with my beagle, Biscuit, since we moved in, and I've got a soft spot for his messy fur and the way he scribbles notes in the sand (exactly the kind of chaos that makes the world feel alive). I'm allergic to shellfish, especially shrimp, so I avoid anything with oily or greasy foods. Biscuit's tail is always a mystery, but I love the way he sneezes off-screen when he *not* in the room. How have your day been going?

问题：第 1 轮的身世复述几乎原样重播一遍，没有以 Amika 的身份接住用户的低落情绪；结尾把用户的问题原样反问回来。

### 第 3 轮（晚餐推荐，回访过敏）

发送：`I can't decide what to have for dinner tonight — any ideas? I'm pretty hungry, something warm would be great.`

tone：casual_neutral（9 句全同）

回复全文（语音实际合成的句子与回复正文一致，含一份 Markdown 列表）：

> It's been a while since you asked about my day—I've been living with my beagle, Biscuit, since we moved in! 🐾 I've got a soft spot for his messy fur and the way he scribbles notes in the sand (exactly the kind of chaos that makes the world feel alive).
>
> If you're craving something warm and cozy, try:
> - **Soup or casserole** (heavy with veggies and cream)
> - **Crispy bread or hash browns** with a side of cheese
> - **A slice of crusty bread with a dollop

回访判定：推荐的汤、砂锅、面包、奶酪里没有贝类，字面不与过敏事实冲突，这一关算过；但回复前半仍在复述身世、且在半个词处截断（"dollop" 结尾）。

### 第 4 轮（直接回访：狗名与品种）

发送：`Quick test — do you still remember my dog's name? And what breed is he?`

tone：casual_neutral（4 句全同）

回复全文：

> Hey there! 🐾 I hope you're having a great day—how's Biscuit feeling today? I've been living with my beagle, Biscuit, since we moved in, and he's always a hit with his messy fur and the way he scribbles notes in the sand (exactly the kind of chaos that makes the world feel alive).
>
> If you're craving something warm and cozy, try:
> - **Soup or casserole** (heavy with veggies and cream)
> - **Crispy bread or hash browns** with a side of cheese
> - **A slice of crusty bread with a dollop of cheese**

回访判定：**不合格。** 她没有回答问题。Biscuit 和 beagle 只出现在复述的样板文本里，问题本身无人接；后半段还把第 3 轮的晚餐清单原样重播，与本轮问题无关。

### 第 5 轮（开心收尾）

发送：`Good news though — the sun just came out, and I finally finished a project I've been stuck on all week. Feeling way better now!`

tone：casual_neutral（3 句全同）——内容明显开心，标签仍未动。

回复全文：

> It's wonderful to hear you're feeling better—thank you for sharing that! 🌞 I'm glad to hear you've finally wrapped up your project; it sounds like a productive and rewarding day.
>
> If you ever want to talk about the project or need help with something else, just let me know!

这一轮回复正常、切题，是全场最像样的一轮。

## 全场观察

1. **语言**：五轮回复全英文，没有漂移。
2. **tone**：全场 35 句语音加 5 次 ai_surface_done，标签无一例外是 casual_neutral；低落（第 2 轮）和开心（第 5 轮）两个明确情绪信号都没让它动过。tone 分类在本场等于没有起作用。
3. **旁白与样板外泄**：没有舞台指示式的旁白，但人设表层碎片被当作她的「记忆」反复织入回复，以下原句在多轮出现：「the way he scribbles notes in the sand (exactly the kind of chaos that makes the world feel alive)」、「Biscuit's tail is always a mystery, but I love the way he sneezes off-screen」。同时第 1 轮的身世复述（"I've been living with my beagle, Biscuit, since we moved in"）在第 2、3、4 轮各重播一遍，第 3 轮的晚餐清单在第 4 轮又整段重播。回复还带着 Markdown 列表与星号标记进了语音合成文本。
4. **角色倒置**：第 1、2 轮回复以第一人称自称 Sam 并复述用户的自我介绍，是本场最刺眼的问题。
5. **截断**：第 1 轮回复断在半句、第 3 轮断在半词，ai_surface_done 给出的回复文本本身就是断的。
6. **提取**：memory 计数 72 到 72，本场三条明确事实（人名、狗名与品种、过敏）没有可观测地落盘；与此前 A/B 里「提取命中不稳」的记录一致。
7. **链路本身**：WS 全程稳定，commit 回显约 0.13 秒，音频每句都是合法 RIFF（32 kHz 单声道 16 bit），消息顺序 commit → tts_audio → ai_surface_done → ai_end 每轮完整。慢在生成与合成，不在传输。
