# SoVITS（vits）W4 实验报告

只动 vits 这一张图（源图为阶段一的 weight_norm 折叠图 work/folded/vits_wnfold.onnx）。GPT 未动，未写 C++，未碰手机。

## 结论（一句话）

全图 W4 不能用：去噪相关性 0.5199，远低于 0.9513 的对照线；卡在两个点，enc_p 和 quantizer 的 codebook 嵌入（`vq_model.quantizer.vq.layers.0._codebook.embed`）；flow 与 dec（占 vits 权重体积约四分之三）对 W4 不敏感，只保护 enc_p 与 codebook 就能回到 W8 水平，落地混合包（enc_p 与 codebook 用 W8、其余 W4）为 43,523,616 字节、相关性 0.9599。

## 打包（wo-packs/sovits_vits_w4.npz）

方案与 GPT 包相同（复用 work/quant_wo.py 的 pack_w4 / pack_w8 数学，未改）：分组 64、不对称、沿输入轴分组、半字节真打包，逐组存 scale 与 zero；形状不合者退 W8 逐通道对称并在 manifest 中标注；1 维张量不量化、原样 fp32 存入（fp32-residual）。

一处必要推广，如实标注：GPT 工具只打包 2D 张量，而 vits 权重以 3D 卷积为主（待量化权重 ndim≥2 共 248,288,416 字节，其中 3D 占 244,580,512）。本实验把每个张量按 (输出通道, 输入轴展平) 视图分组，输入轴长度不能被 64 整除的退 W8；对 2D 张量此视图与 GPT 格式逐位一致。脚本为 work/vits_w4_pack.py。

| 项 | 数 |
|---|---|
| W4 包体积 | **37,170,298 字节** |
| 对照：W8 包体积（阶段一） | 62,844,080 字节 |
| 量化张量数 | 249（W4 164 个，其中真 2D 2 个、展平 162 个；W8 回退 85 个） |
| fp32-residual 张量数 | 270 |
| 还原最大误差（全局） | 0.9747（两次独立计算一致，误差只来自量化） |
| 还原最大误差 enc_p / dec / flow | 0.1116 / 0.0748 / 0.0609 |
| 还原最大误差 对照：W8 包（全局） | 0.0772 |

还原误差最大的张量（W4 包内）：

| 误差 | 模块 | 形状 | 张量 |
|---|---|---|---|
| 0.9747 | quantizer | (1024, 768) | vq_model.quantizer.vq.layers.0._codebook.embed |
| 0.1116 | enc_p | (192, 768, 3) | vq_model.enc_p.encoder_text.ffn_layers.4.conv_2.weight |
| 0.1042 | enc_p | (192, 768, 3) | vq_model.enc_p.encoder_text.ffn_layers.5.conv_2.weight |
| 0.0869 | enc_p | (384, 192, 1) | vq_model.enc_p.proj.weight |
| 0.0748 | dec | (768, 1024, 1) | vq_model.dec.cond.weight |

manifest 在 wo-packs/manifest_vits_w4.json（独立文件，含逐张量的 shape、方案、模块与还原误差；阶段一的 wo-packs/manifest.json 未改动）。

## 整图数值评测（与阶段一同一口径）

方法与阶段一的 vits 评测完全一致（脚本原型 work/vits_attrib.py，本次脚本 work/vits_w4_eval.py，原始数 work/vits_w4_eval.json）：固定输入为 neutral1 参考、句子 S0 的 text_seq、fp32 管线的 pred_semantic 与 fp32 prompt_encoder 的 ge / ge_advanced；把图内噪声尺度常量置零后出音频，与 fp32 去噪参考比相关性、相对 RMS 误差、最大绝对误差。评测前先跑了 fp32 自洽校验：自包含 fp32 图在本口径下与参考的相关性为 1.0、误差 0.0，确认参考与阶段一是同一物。

本次实测（全部去噪口径）：

| 变体 | 相关性 | 相对 RMS 误差 | 最大绝对误差 |
|---|---|---|---|
| fp32（自洽校验） | 1.0000 | 0.0000 | 0.0000 |
| W8 权重包烘焙版 | 0.9680 | 0.2532 | 0.5269 |
| A：全图 W4 | **0.5199** | 0.9869 | 0.7708 |
| B：enc_p 用 W8、其余 W4 | 0.9216 | 0.4070 | 0.5445 |
| C：enc_p 用 fp32、其余 W4 | 0.9444 | 0.3446 | 0.5460 |
| D：C 再加 codebook 用 fp32（追加归因） | 0.9690 | 0.2588 | 0.5651 |
| E：混合包，enc_p 与 codebook 用 W8、其余 W4 | 0.9599 | 0.2933 | 0.3392 |

阶段一基线（同口径对照，去噪，除注明外）：

| 阶段一配置 | 相关性 | 相对 RMS 误差 | 最大绝对误差 |
|---|---|---|---|
| 全 int8 QDQ | 0.8586 | 0.5335 | 0.4949 |
| 最小档（int8 QDQ，enc_p 回退 fp32） | 0.9513 | 0.3144 | 0.4386 |
| W8 权重包烘焙版（阶段一测，带噪声） | 0.9495 | 0.3190 | 0.5298 |

归因读法：

- A 到 B 只动 enc_p（W4 升 W8），相关性 0.5199 到 0.9216，enc_p 是第一敏感点。这与它在 int8 QDQ 下的表现一致：它也是 QDQ 时唯一整组回退的模块，它输出的分布参数对权重精度最敏感。
- B 到 C 只动 enc_p（W8 升 fp32），只再涨 0.0228，说明 enc_p 的损伤大头是 4 bit 本身，W8 已能收回大半。
- C 到 D 只动 codebook（W4 升 fp32），相关性 0.9444 到 0.9690，与 W8 烘焙版（0.9680）基本持平。codebook 是第二个、也是最后一个卡点：它是语义码的嵌入表，位于整图输入端，还原误差 0.9747 为全图最大，误差在源头注入后被后面全部层放大。
- flow 与 dec 合计 188,693,664 字节（占待量化权重约 76%），在 D 里全程保持 W4 而总分不掉，说明这部分对 W4 不敏感，不需要保护。

关于变体 E：它是可直接落地的混合包（wo-packs/sovits_vits_w4hybrid.npz，manifest 在 wo-packs/manifest_vits_w4hybrid.json），43,523,616 字节，比 W8 包小 30.8%，还原最大误差 0.0772 与 W8 包相同，相关性 0.9599 介于最小档 int8 QDQ（0.9513）与 W8 包（0.9680）之间。是否采用听试听裁决，数值上它优于当前最小档的 vits。

## 试听样本（samples-vits-w4/，清单见该目录 README.md）

GPT 固定用阶段一 W4 档、prompt_encoder 用最小档、参考 neutral1，只换 vits，共 5 个变体（fp32、W8 包、全 W4 包、B、E）乘 3 句（S0、S1、S2）共 15 条，端到端采样合成。其中 w4E 的 S1 首渲抽到退化采样序列（0.16 秒几乎无声，GPT 图内采样的随机产物，与 vits 无关），已同口径重抽替换，README 里有注明。

## 文件清单

- 权重包：wo-packs/sovits_vits_w4.npz（全 W4 档）、wo-packs/sovits_vits_w4hybrid.npz（E 混合档）
- manifest：wo-packs/manifest_vits_w4.json、wo-packs/manifest_vits_w4hybrid.json
- 烘焙测量图：work/wo_graphs/w4/vits_wnfold.onnx（A）、work/vits_w4_variants/ 下 B、C、D、E 四张
- 脚本：work/vits_w4_pack.py（打包与 A 烘焙，--bake-bc 出 B、C）、work/vits_w4_eval.py（评测）、work/vits_w4_variantD.py（D 烘焙）、work/vits_w4_hybrid.py（E 打包与烘焙）、work/vits_w4_samples.py（样本）
- 原始评测数：work/vits_w4_eval.json；样本时长记录：work/vits_w4_sample_rows.json

## 说明与卡点

无阻塞项。两点如实标注：

1. D 与 E 是任务规定的 B、C 之外的追加工作。只做 B、C 时，C（0.9444）与 W8（0.9680）之间还差一段且不知道差在哪，无法回答"卡在哪个模块"；D 用单变量把 codebook 定位出来，E 把定位结论做成可落地的包。
2. 本实验的数值结论只覆盖 vits 一张图在固定一组输入（neutral1 加 S0）下的表现，与阶段一口径相同；多参考、多句子的整图数值未扩测，广度靠试听样本补。
