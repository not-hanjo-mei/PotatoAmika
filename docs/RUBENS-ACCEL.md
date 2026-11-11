# Rubens GPU/NPU 可達性調查（2026-11-11）

設備：Redmi K50，天璣 8100（SoC 型號 MT6895Z/TCZA，平台 mt6895），Android 14（SDK 34），Termux 用戶空間，鎖 BL。探針全部以 Termux 應用 UID 從普通 native 進程發起，與日後真正跑模型的進程身份一致。探針源碼與原始輸出在 `accel/`（nnapi_probe.c、nnapi_probe2.c、nnapi_probe3.c、dlprobe.c、probe-output.txt、probe2-output.txt、probe3-output.txt）。

## 結論一覽

| 路 | 結論 | 決定性證據 |
|---|---|---|
| NNAPI（公開 NDK API） | **開** | 五個設備可枚舉；綁定加速器的編譯與執行全部 rc=0 且輸出正確（詳下） |
| 聯發科原始 Neuron 庫直調 | **擋**（Termux native 進程內） | dlopen 被鏈接器以命名空間為由拒絕，原文在下面 |
| Mali OpenCL | **擋**（實際上） | 庫能載入，但 clGetPlatformIDs 返回 -1001，零平台 |
| Vulkan 計算 | **開** | dlopen、建實例、枚舉到 Mali-G610 MC6 全部成功 |

此前「沒有從 Termux 到加速硬件的受支持路」這個說法不成立。NNAPI 是 Google 給應用進程的正式 NDK 接口（libneuralnetworks 屬 LL-NDK），在這台機器上從 Termux 進程直接可用，且真的把計算送進了聯發科的加速器驅動。需要修正的是另一半：原始 Neuron 庫確實拿不到，但那是因為有 NNAPI 這扇正門，廠商把私庫鎖在了門後。

## NNAPI 探針結果

`ANeuralNetworks_getDeviceCount` rc=0，設備列表逐字如下（版本字符串均為 5.1.1，featureLevel 31，nnapi-reference 除外）：

```
device[0] name="mtk-dsp_shim"     type=ACCELERATOR(4) version="5.1.1" featureLevel=31
device[1] name="mtk-gpu_shim"     type=GPU(3)         version="5.1.1" featureLevel=31
device[2] name="mtk-mdla_shim"    type=ACCELERATOR(4) version="5.1.1" featureLevel=31
device[3] name="mtk-neuron_shim"  type=ACCELERATOR(4) version="5.1.1" featureLevel=31
device[4] name="nnapi-reference"  type=CPU(2)         version="OS2.0.10.0.ULNCNXM" featureLevel=1000008
```

`service list` 裡四個 AIDL 驅動與此對應：mtk-dsp_shim、mtk-gpu_shim、mtk-mdla_shim、mtk-neuron_shim。

兩張小圖在每個設備上綁定編譯並各執行一次（ADD：float[4]+float[4]；FC：1x4 乘 2x4 全連接加偏置）。結果矩陣（✓ 為編譯加執行全部 rc=0 且輸出逐值正確；✗ 為編譯 finish 返回 rc=4，即 ANEURALNETWORKS_OP_FAILED）：

| 設備 | ADD fp32 | FC fp32 | ADD fp16 | FC fp16 |
|---|---|---|---|---|
| mtk-dsp_shim | ✓ | ✗ | ✓ | ✗ |
| mtk-gpu_shim | ✓ | ✓ | ✓ | ✓ |
| mtk-mdla_shim | ✗ | ✗ | ✓ | ✓ |
| mtk-neuron_shim | ✓ | ✓ | ✓ | ✓ |
| nnapi-reference | ✓ | ✓ | ✓ | ✓ |

最要緊的一行是 mtk-mdla_shim：它就是 APU 裡的深度學習加速器本體。它對 fp32 圖整體拒收（編譯期 rc=4），對 fp16 圖照單全收，FC 的輸出 {1.5, 1.5} 與期望逐值一致。也就是說 NPU 這扇門開著，門票是格式：fp16 或 int8 標準圖可以，fp32 不行。mtk-neuron_shim 是 Neuron 運行時經 NNAPI 暴露的入口，fp32 也能吃（它內部做轉換），是五個設備裡最寬容的一個。不綁定設備的編譯同樣執行正確，運行時自動分區可用。

限制也要寫清。NNAPI 已在 Android 15 起被 Google 棄用，本機是 Android 14，不受影響，但這條路沒有未來版本的新特性。NNAPI 的算子集為卷積時代設計，動態形狀和逐 token 循環不是它的形狀，這一點決定了下面工作負載的適配判斷。聯發科的 NNAPI 驅動歷史上以算子覆蓋參差著稱，本探針只驗了 ADD 與 FC 兩種算子，具體模型的算子覆蓋要在試驗裡逐個撞。

## 原始 Neuron 庫直調：被鏈接器擋住

dlopen 探針（RTLD_NOW，逐字記錄 dlerror）：

```
dlopen("/vendor/lib64/libneuron_runtime.5.so") -> FAIL: dlopen failed: library "/vendor/lib64/libneuron_runtime.5.so" needed or dlopened by "/data/data/com.termux/files/home/accel/dlprobe" is not accessible for the namespace "(default)"
dlopen("/vendor/lib64/libneuron_adapter_mgvi.so") -> FAIL: 同上，namespace "(default)" 不可達
```

加 `LD_LIBRARY_PATH=/vendor/lib64:/system/lib64` 後按 SONAME 再試，兩者仍是同一句拒絕。`libneuron_adapter_mgvi.so` 雖然寫在 `/vendor/etc/public.libraries.txt` 裡，但那份名單服務的是帶 manifest 的 APK 進程（API 31 起還要在 manifest 裡寫 `<uses-native-library>` 才給），Termux 的 native 進程走的是 (default) 命名空間，名單對它不生效。這與聯發科官方文檔的口徑一致：第三方應用廠商不能用 Neuron Runtime API，那套 API 只給 vendor 分區上的應用；第三方走 TFLite 加 NNAPI delegate 或 Neuron delegate。

兩個邊角事實一併記錄。`libapusys.so` 在加了 LD_LIBRARY_PATH 後 dlopen 成功，但它是 APUWare 的 IPC 客戶端庫，公開頭文件與 SDK 走聯發科合作夥伴渠道，載入成功不等於有可用的編程面。`libtflite_mtk.so` 在 LD_LIBRARY_PATH 下過了命名空間檢查，死在自己的依賴上（`libflatbuffers-cpp.so` not found），說明設備自帶的這份 TFLite 不是給應用進程準備的入口，我們要用 TFLite 需自帶運行時。

## OpenCL：庫在，平台不在

`libOpenCL.so` 只在加了 LD_LIBRARY_PATH 後能 dlopen（裸路徑同樣被命名空間拒絕）。載入後 `clGetPlatformIDs` 返回 -1001（CL_PLATFORM_NOT_FOUND_KHR），平台數 0。/vendor 裡這份是 ICD 加載器外殼，它背後的 Mali ICD 沒有向應用進程註冊。結論：這條路在這台機器上實際不通，不值得再花時間。

## Vulkan：開

`libvulkan.so` 裸 dlopen 直接成功（系統公開庫），vkCreateInstance rc=0，枚舉到 1 個物理設備：

```
VK: device[0] name="Mali-G610 MC6" vendorID=0x13b5 deviceID=0xa8670000 apiVersion=1.1.177
```

llama.cpp 的 Vulkan 後端在 Termux 裡跑 Mali 有公開先例，本探針只做枚舉，沒有跑計算負載。

## 工作負載適配

1. **TTS 的 vits 靜態卷積圖：最適配 NNAPI。** 固定形狀、卷積為主、fp16 可接受，正中 MDLA 的設計形狀。經 mtk-neuron_shim 或直接綁 mtk-mdla_shim 都值得試。這是四個負載裡唯一一個形狀、格式、調用模式全對上的。
2. **Moonshine ASR：適配，次一等。** 編碼器是卷積加注意力的固定長圖，可以轉 TFLite 或 NNAPI 圖走同一扇門；分段長度變化要靠分桶或補齊處理。
3. **GPT decode：不適配 NNAPI。** 逐 token 自迴歸，每步一次圖調用的開銷和 KV cache 的動態形狀都撞在 NNAPI 的弱項上。llama.cpp 也沒有 NNAPI 後端，這條不在 NPU 的菜單上。
4. **llama.cpp 的 LLM：走 Vulkan。** Mali-G610 已枚舉到，GGML_VULKAN 在 Termux 有現成構建路徑，這是 GPU 側唯一有生態的入口。

## 建議的第一個真試驗

把 vits 的一張子圖（或整圖）轉成 fp16 的 TFLite，走 TFLite 的 NNAPI delegate 綁 mtk-neuron_shim，再直連 NNAPI C API 綁 mtk-mdla_shim 驗一遍算子覆蓋，量單段延遲，與現在 CPU SDOT 路的 vits 段（短句 4.4 秒那份口徑）同會話對照。算子覆蓋先行：轉換後逐個對照哪些算子被拒、哪些被接受，被拒的算子清單就是這條路的真實邊界。LLM 側另開一條小試驗：llama.cpp 開 GGML_VULKAN 在這台上量 Minico 的 tg，與 CPU 口徑對照，不與 NPU 試驗混在一起。

## 資料來源

- Android NNAPI 官方文檔（NDK C API，應用進程接口；Android 15 起棄用）：https://developer.android.com/ndk/guides/neuralnetworks/
- 聯發科 NeuroPilot 開發指南（開發者角色表：第三方應用廠商用 TFLite 加 NNAPI/Neuron delegate，Neuron Runtime API 僅限 vendor 分區）：https://developer.mediatek.com/ai/6423c916c612745b3a4baa2d.html
- 第三方在 Dimensity 9400 上的同類調查（public.libraries 與 `<uses-native-library>` 的關係、NNAPI AIDL shim 的查法）：https://github.com/ExperimentalMachines/openweights/blob/HEAD/docs/research/mediatek-npu.md
- NNAPI 棄用與 LiteRT 遷移口徑：https://developer.android.com/agents/skills/device-ai/litert/litert-runtime/references/migration
- LiteRT NeuroPilot Accelerator（新路，公開 SoC 名單為天璣 7300、8300、9000 起，未列 8100）：https://www.marktechpost.com/2025/12/09/google-litert-neuropilot-stack-turns-mediatek-dimensity-npus-into-first-class-targets-for-on-device-llms/
