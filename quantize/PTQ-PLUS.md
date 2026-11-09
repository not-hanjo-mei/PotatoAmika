# PTQ-PLUS：GPT W4 包的激活感知加料轮（GPTQ + MSE 裁剪）

日期：2026-11-08。范围：只动 GPT 三个包（encoder / first / stage），不碰 vits、不碰宿主、不改任何定版配置。包格式与现行 W4 包完全兼容（分组 64、不对称、半字节打包、1 维张量 fp32-residual、不可打包形状 w8-fallback），宿主加载器不用改。

## 方法：实际用了 GPTQ，没有退 AWQ

- 图内每个 W4 张量按算子分工，共 97 个/解码图：
  - Gemm（transB=1，权重存 (out,in)，分组沿输入维）24 个：教科书式 GPTQ（Case B），沿输入维顺序量化，误差用 H 的逆 Cholesky 因子反馈给未量化列；分组定界用 diag(H) 加权的 MSE 最优裁剪搜索。
  - MatMul（权重存 (in,out)，Y = X@W，分组沿输出轴、按输入行切段）72 个：转置版 GPTQ（Case A），按输入行顺序处理，整行在其步内量化（该行的全部分组段都在行内），误差反馈给后续行；此布局下每组段的加权误差可证与未加权 MSE 成正比，裁剪搜索用未加权 MSE。
  - Gather 嵌入（解码图 ar_audio_embedding (1025,512)、编码图音素嵌入 (732,512)）：无线性层 Hessian 可用，走 MSE 裁剪 RTN。
- 编码图另有 2 个 GPTQ 目标：quantizer codebook (768,1024) 与 bert_proj (1024,512)，均 Case A。
- 裁剪搜索：缩放网格 1.00 → 0.50、步长 0.025，对 [min,max] 对称收缩后重算 scale/zero 取 MSE 最小者。阻尼 λ = 0.01 · mean(diag H)。H 用 float64 累积。
- 非 W4 条目（fp32-residual、w8-fallback）从现行 W4 包逐字节照抄。
- 玩具验证：强离群合成权重上，加权误差为 RTN 的 0.368 倍（裁剪贡献为主，反馈再加约 4%）。

## 校准数据

- 参考特征从阶段二 refpacks 提取（65 条），与阶段一 inputs.npz 在 neutral1/caring1 上逐值一致已验。
- 只有 65/69：sexual_desire1、sexual_desire3、sleepy1、tsun3 四条与阶段二撞同一堵墙（Genie 中文 G2P 的 ToneSandhi 对其参考文本抛 IndexError），无法生成特征，未纳入校准。
- encoder / first 图：65 参考 × S0/S1/S2 共 195 组教师强迫输入；codebook 累积 43,305 行、bert_proj 18,506 行、first 每线性层 61,811 行。
- stage 图：82 条贪心自回归 rollout（65 参考 × S0、neutral1/caring1 × S1/S2、5 条 w4more 参考 × 2 句、angry4/happy4/fear4 × fat-ass 句），每线性层 6,472 行。
- 注：捕获时的轨迹对照检查有 5/6 组在第 0 步分叉，那是 first 图内采样的既有口径（STAGE1 已记录），不是捕获管线的问题，Hessian 不受影响。

## 结果一：教师强迫一致率（361 步标准集，同口径）

评测门先复现现行 W4：299/361 = 82.83%，分组数字与 STAGE1 逐组一致，口径可信。

| 组 | 现行 W4 | W4-plus |
|---|---|---|
| neutral1 S0 | 57/68 | 60/68 |
| neutral1 S1 | 59/68 | 64/68 |
| neutral1 S2 | 42/48 | 42/48 |
| caring1 S0 | 57/72 | 63/72 |
| caring1 S1 | 42/58 | 51/58 |
| caring1 S2 | 42/47 | 43/47 |
| **合计** | **299/361 = 82.83%** | **323/361 = 89.47%** |

提升 6.64 个百分点，6 组里 5 组涨、1 组持平。参照点 W8 包 97.23%：没有追上，W4 与 W8 之间的差距收窄了约 46%。

## 结果二：体积与还原误差

| 包 | 现行 W4 字节 | W4-plus 字节 |
|---|---|---|
| gpt_encoder | 2,166,744 | 2,166,744 |
| gpt_first | 45,284,276 | 45,284,276 |
| gpt_stage | 45,283,988 | 45,283,988 |
| **合计** | **92,735,008** | **92,735,008** |

体积逐字节同量级（比值 1.000，在 ±2% 要求内）。

还原最大逐元素误差：encoder 2.991（最差张量 codebook）、first/stage 5.942（最差张量为两图共有的 ar_audio_embedding 嵌入，走 RTN 那颗）。现行 W4 为 encoder 1.183、解码 3.476。**逐元素最大误差反而变大了，这是预期内的**：GPTQ 优化的是激活加权的输出误差，不是逐元素最大误差，误差反馈会有意挪动单个权重，裁剪搜索也拿最大误差换整体 MSE。判据看功能指标（一致率、尾巴），逐元素最大误差这一项如实记为变差。

## 结果三：破音尾巴（愤怒版同配置，n=8 各组）

配置：参考 angry4、温度 1.35（plus 图内常数已回读确认）、vits E 包、prompt 最小档、过 deess2air。指标同既有口径：最响 10% 帧的 3–12 kHz 频谱平坦度、8–16 kHz 能量占比。

| 组 | 平坦度 min | 平坦度中位 | 平坦度最差 | hf 占比中位 | hf 占比最差 |
|---|---|---|---|---|---|
| 现行 W4 | 0.0743 | 0.1085 | **0.2563** | 0.0301 | 0.0649 |
| **W4-plus** | 0.0816 | 0.0967 | **0.1177** | 0.0392 | 0.0556 |
| fp32 | 0.0624 | 0.0938 | 0.1659 | 0.0369 | 0.0473 |

尾巴结论：最差平坦度从 0.2563 降到 0.1177（降 54%），已经低于 fp32 组的最差值 0.1659；中位也从 0.1085 改善到 0.0967，接近 fp32 的 0.0938。如实记两点保留：每组只有 8 条抽样，组内波动本来就大（此前已实测组内最轻最重差约 2.7 倍），plus 组的 min（0.0816）比现行组 min（0.0743）略差、hf 占比中位（0.0392）比现行组（0.0301）略高，都在抽样噪声量级；hf 占比最差（0.0556）仍高于 fp32（0.0473）。

边界声明：破音主体是抽样到的 token 序列加声码器在大嗓门段的固有毛刺，此前对照实验已定案，与量化无关。本轮只针对量化组的那条超额尾巴，尾巴确实收了；不许把本结论读成治好了破音。

## 产物

- 包：`wo-packs/gpt_encoder_w4plus.npz`、`gpt_first_w4plus.npz`、`gpt_stage_w4plus.npz`，清单 `wo-packs/manifest_w4plus.json`。
- 烘焙图：`work/wo_graphs/w4plus/`（三图）。
- 评测：`work/tf_wo_w4plus.json`（对照 `work/tf_wo_w4_repro.json` 为现行包复现）、`work/angry_w4plus_takes.json`，音频 `samples-fun/holy_big_fat_ass_angry_w4plus_take1..8.wav`。
- 代码：`work/gptq_pack.py`（GPTQ 打包）、`work/capture_hessians.py`、`work/extract_calib_inputs.py`、`work/eval_tf.py`、`work/synth_angry_w4plus.py`；Hessian 存 `work/ptqplus/`。
- 过程备注：打包中途开发机重启过一次，GPTQ 计算改为逐图落盘后重跑，Hessian 与校准数据未受损（已逐个验有限值）。
