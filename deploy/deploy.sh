#!/bin/sh
# PotatoAmika 整机部署脚本（小米 4c / Alpine v3.24 aarch64 容器内运行）
#
# 瘦身版（2026-11-09）：tar 不带大脑/嵌入 GGUF、ASR 三件与 fallback-gen1，装机时按钉死地址拉齐（见配置区）。
# 用法（在本脚本与 potatoamika-slim.tar（或旧整包 potatoamika.tar / 分片）同目录里跑）：
#   ./deploy.sh              解包 -> 校验 -> 起全套常驻 -> 自检（一轮文本进/wav出 + 一段PCM进/文本出）
#   ./deploy.sh --start      只起服务（已解包时）
#   ./deploy.sh --stop       停全套常驻
#   ./deploy.sh --selftest   只跑自检（服务已在跑时）
#   ./deploy.sh --dry-run    只验文件与哈希，不起服务（POTATO_SRC 指到已解开的目录时验那个目录）
#   ./deploy.sh --uninstall  停全套并删安装目录（回滚）
#
# 幂等：重复跑不会重复起服务（pid 文件在且进程活着就跳过），解包覆盖同名文件，
# data/（记忆）与 logs/ 不在包里、不会被覆盖。
# 任一步失败立即停，打印失败步名与对应日志末尾。

set -u

# ================= 配置区（只改这里） =================
INSTALL_DIR="${POTATO_HOME:-$HOME/potatoamika}"   # 固定安装目录
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
TARBALL="${POTATO_TARBALL:-$SCRIPT_DIR/potatoamika-slim.tar}"
[ -f "$TARBALL" ] || TARBALL="$SCRIPT_DIR/potatoamika.tar"      # 瘦身 tar 优先，也认旧整包名
[ -f "$TARBALL" ] || TARBALL="$SCRIPT_DIR/potatoamika.tar.gz"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-/usr/bin/llama-server}"  # 容器自带 Alpine 官方包 b9564
SCHED_PORT=18100          # 调度器 API
BRAIN_PORT=18091          # 调度器内部拉起的大脑 llama-server（配置里定，勿单独改）
EMBED_PORT=18092          # 调度器内部拉起的嵌入 llama-server
ASR_GW_PORT=18120         # ASR 网关（自拉 asr_host --stream）
ADAPTER_PORT=8372         # 原版 Use 页适配器（纯 Go 静态二进制 bin/adapter，正式前端；原型页 ui-proto 已退役不进包）
L2D_REPO="MekaHime/MekaHime_Amika"   # Live2D 模型装机时从 hf-mirror 按 orig-ui/live2d-files.txt 的钉死 revision 逐件拉
BRAIN_THREADS=2           # 手机定案 2 线程（电池坏，忌满载）
EMBED_THREADS=2
TTS_THREADS=2
ASR_THREADS=2
HEALTH_TIMEOUT_S=600      # 等调度器健康的上限（手机冷加载大脑慢，给足）
SELFTEST_TIMEOUT_S=1800   # 自检一轮对话的上限（手机上 TTS 慢）
VISION_BASE="https://hf-mirror.com/LiquidAI/LFM2.5-VL-450M-GGUF/resolve/1abed04b6fe71314d8c446a1371c03d7c332266d"
VISION1_FILE="LFM2.5-VL-450M-Q4_K_M.gguf"
VISION1_SIZE=229313568
VISION1_SHA256="1093f1331319199bbcacdbd7ecc9aa6e5678db6b55073948d0f508be90c8ab68"
VISION2_FILE="mmproj-LFM2.5-VL-450m-F16.gguf"
VISION2_SIZE=189126080
VISION2_SHA256="51b458cfdbc736982145a35f798ce37611af0aab639e58b33473ba0c7815fd99"
# 瘦身版新增：大脑与嵌入 GGUF 不进 tar，装机时按钉死 revision 从 hf-mirror 拉（哈希已在本机实测与原包逐字节一致）
# 注意 Minico 必须钉 eee16357…：上游 main 在 2026-11-09 又推了一个改 GGUF 元数据的提交，main 上的文件已与原包不一致
BRAIN_FILE="Minico-M1-1105-Q4_K_M.gguf"
BRAIN_SIZE=229316992
BRAIN_SHA256="2b0456131f1efeac1f6a7af85ea4541b6d12b20bcd94af628dfc64ef88460125"
BRAIN_URL="https://hf-mirror.com/Mincofficial/Minico-M1-1105-GGUF/resolve/eee16357785998f7fa035e543eef1b902d77d94d/Minico-M1-1105-Q4_K_M.gguf"
EMBED_FILE="v5-nano-retrieval-Q4_K_M.gguf"
EMBED_SIZE=156966976
EMBED_SHA256="f50822244ba0c7a348c5455b99bb8a0afd182511e8a816888c5dc65d972e51d5"
EMBED_URL="https://hf-mirror.com/jinaai/jina-embeddings-v5-text-nano-retrieval-GGUF/resolve/59cfaceeeb7d738c404659435af4c0da74d06c96/v5-nano-retrieval-Q4_K_M.gguf"
# ASR 模型同样不进 tar（vocab.tsv 是本机 export_vocab.py 从 tokenizer.json 导出的派生文件，无公开原件，仍留在包里）
ASR_BASE_MOON="https://hf-mirror.com/onnx-community/moonshine-tiny-ONNX/resolve/a6da1241cd305dcd64eab1edbd615f2bb9aabb95/onnx"
ASR_ENC_FILE="onnx_encoder_model_quantized.onnx"
ASR_ENC_SIZE=7937661
ASR_ENC_SHA256="c6fc4b7bc5af75c0591fd157a1f3829b533d18e9769a888fd95a62e470dd4f4a"
ASR_DEC_FILE="onnx_decoder_model_merged_quantized.onnx"
ASR_DEC_SIZE=20243286
ASR_DEC_SHA256="eed87831c3a6103534aae7d47a5d485025c659a1323901513961c39fe8a1a367"
ASR_VAD_FILE="silero_vad.onnx"
ASR_VAD_SIZE=2243022
ASR_VAD_SHA256="a4a068cd6cf1ea8355b84327595838ca748ec29a25bc91fc82e6c299ccdc5808"
ASR_VAD_URL="https://hf-mirror.com/onnx-community/silero-vad/resolve/e71cae966052b992a7eca6b17738916ce0eca4ec/onnx/model.onnx"
# ===================================================

STEP="init"
say()  { echo "[deploy] $*"; }
fail() { echo "[deploy] 失败在步骤: $STEP" >&2
         echo "[deploy] $*" >&2
         [ -n "${LOGHINT:-}" ] && { echo "[deploy] 日志末尾（$LOGHINT）:" >&2; tail -n 20 "$LOGHINT" >&2; }
         exit 1; }

pid_alive() { [ -f "$1" ] && kill -0 "$(cat "$1" 2>/dev/null)" 2>/dev/null; }

wait_port() { # $1=port $2=timeout_s $3=label ; 用 /dev/tcp 不可得时退到 nc/curl
  _p=$1; _t=$2; _n=0
  while [ $_n -lt "$_t" ]; do
    if curl -s -o /dev/null -m 2 "http://127.0.0.1:$_p/" 2>/dev/null; then return 0; fi
    if command -v nc >/dev/null 2>&1 && nc -z 127.0.0.1 "$_p" 2>/dev/null; then return 0; fi
    sleep 2; _n=$((_n+2))
  done
  return 1
}

# ---------------- 步骤 ----------------

step_preflight() {
  STEP="preflight 环境核对"
  [ "$(uname -m)" = "aarch64" ] || fail "不是 aarch64（当前 $(uname -m)），本包是 aarch64 构建"
  [ -x "$LLAMA_SERVER_BIN" ] || fail "找不到 $LLAMA_SERVER_BIN（容器应自带 Alpine 官方 llama.cpp 包）"
  if [ ! -e /usr/lib/libonnxruntime.so.1 ]; then
    say "系统 onnxruntime 不在，试 apk add（tts_host 动态链接它，与引擎选择无关）"
    if command -v doas >/dev/null 2>&1; then doas apk add onnxruntime || true; fi
    [ -e /usr/lib/libonnxruntime.so.1 ] || fail "libonnxruntime.so.1 缺失且 apk add 没装上，先手动 doas apk add onnxruntime"
  fi
  command -v curl >/dev/null 2>&1 || fail "缺 curl（健康检查要用），doas apk add curl"
  say "环境核对过：aarch64、llama-server、onnxruntime、curl 都在"
}

PKG_SHA256="10e1d2790ff65685e3f7a00fbdf6988627d7fff145270f833278b0d4fb5198cd"  # 瘦身 tar 的 sha256（去掉两个 GGUF、ASR 三件、fallback-gen1；模型装机时按上方钉死地址拉）

step_extract() {
  STEP="extract 解包"
  if [ ! -f "$TARBALL" ] && [ -f "$SCRIPT_DIR/potatoamika.tar.part-00" ]; then
    say "没有整 tar，从分片拼（potatoamika.tar.part-*）"
    _tmp_tar="$HOME/.potatoamika-assembled.tar"
    cat "$SCRIPT_DIR"/potatoamika.tar.part-* > "$_tmp_tar" || fail "拼分片失败"
    [ "$(sha256sum "$_tmp_tar" | cut -d' ' -f1)" = "$PKG_SHA256" ] || { rm -f "$_tmp_tar"; fail "拼出的 tar 哈希不对，分片不完整或损坏"; }
    TARBALL="$_tmp_tar"; _assembled=1
    say "分片拼好并验过整包哈希"
  fi
  if [ -f "$TARBALL" ]; then
    mkdir -p "$INSTALL_DIR" || fail "建不了 $INSTALL_DIR"
    case "$TARBALL" in *.gz) _tf=xzf ;; *) _tf=xf ;; esac
    tar -$_tf "$TARBALL" -C "$INSTALL_DIR" --strip-components=1 || fail "解 $TARBALL 失败"
    [ "${_assembled:-0}" = 1 ] && rm -f "$TARBALL"
    say "已解包到 $INSTALL_DIR"
  elif [ -f "$INSTALL_DIR/hashes.sha256" ]; then
    say "没有 tarball，但 $INSTALL_DIR 已有解开的树，沿用"
  else
    fail "既没有 $TARBALL，也没有已解开的 $INSTALL_DIR"
  fi
  mkdir -p "$INSTALL_DIR/data/memory" "$INSTALL_DIR/data/orig-ui" "$INSTALL_DIR/logs" "$INSTALL_DIR/run" "$INSTALL_DIR/out/tts" "$INSTALL_DIR/models-live2d"
  chmod +x "$INSTALL_DIR/bin/"* 2>/dev/null || true
}

step_verify() {
  STEP="verify 哈希校验"
  [ -f "$INSTALL_DIR/hashes.sha256" ] || fail "$INSTALL_DIR/hashes.sha256 不在，包不完整"
  ( cd "$INSTALL_DIR" && sha256sum -c hashes.sha256 ) > "$INSTALL_DIR/logs/verify.log" 2>&1 \
    || { tail -n 5 "$INSTALL_DIR/logs/verify.log" >&2; fail "哈希校验有不过的文件，看 logs/verify.log"; }
  say "哈希全过（$(grep -c ': OK' "$INSTALL_DIR/logs/verify.log") 个文件）"
}

step_config() {
  STEP="config 生成调度器配置"
  sed "s|@ROOT@|$INSTALL_DIR|g" "$INSTALL_DIR/config/scheduler.json.tmpl" > "$INSTALL_DIR/config/scheduler.json" \
    || fail "渲染 scheduler.json 失败"
  grep -q "$INSTALL_DIR" "$INSTALL_DIR/config/scheduler.json" || fail "scheduler.json 里没替进安装路径"
  say "scheduler.json 已按 $INSTALL_DIR 生成（12k、persona_trim 关、scenes 英文版、memory_language en）"
}

step_llm_models() {
  STEP="llm-models 大脑与嵌入 GGUF（瘦身包不带，装机时备齐）"
  _d="$INSTALL_DIR/models"
  mkdir -p "$_d" || fail "建不了 $_d"
  _ok=1
  for _spec in "$BRAIN_FILE $BRAIN_SIZE $BRAIN_SHA256 $BRAIN_URL brain" "$EMBED_FILE $EMBED_SIZE $EMBED_SHA256 $EMBED_URL embed"; do
    set -- $_spec; _f=$1; _sz=$2; _sha=$3; _url=$4; _kind=$5
    if [ -f "$_d/$_f" ] && [ "$(stat -c %s "$_d/$_f" 2>/dev/null || echo 0)" = "$_sz" ] \
       && [ "$(sha256sum "$_d/$_f" | cut -d' ' -f1)" = "$_sha" ]; then
      say "模型 $_f 已在（大小与哈希对）"; continue
    fi
    if [ "$_kind" = "brain" ]; then
      # 复用查找两处：$HOME 根与 $HOME/models（用户的 GGUF 实际躺在后者）
      _reused=0
      for _cand in "$HOME/$_f" "$HOME/models/$_f"; do
        if [ -f "$_cand" ] \
           && [ "$(stat -c %s "$_cand" 2>/dev/null || echo 0)" = "$_sz" ] \
           && [ "$(sha256sum "$_cand" | cut -d' ' -f1)" = "$_sha" ]; then
          if cp "$_cand" "$_d/$_f"; then
            say "大脑模型 $_f 从 $_cand 复用（哈希对上，没下载）"; _reused=1; break
          fi
          say "从 $_cand 拷 $_f 失败，改走下载"
        fi
      done
      [ "$_reused" = 1 ] && continue
    fi
    say "模型 $_f 不在，从 hf-mirror 拉（钉死 revision）"
    curl -L -m 1800 -o "$_d/$_f.tmp" "$_url" || { say "下载 $_f 失败，跳过"; _ok=0; continue; }
    if [ "$(stat -c %s "$_d/$_f.tmp" 2>/dev/null || echo 0)" != "$_sz" ] \
       || [ "$(sha256sum "$_d/$_f.tmp" | cut -d' ' -f1)" != "$_sha" ]; then
      say "下载的 $_f 大小或哈希不对，删掉跳过"; rm -f "$_d/$_f.tmp"; _ok=0; continue
    fi
    mv "$_d/$_f.tmp" "$_d/$_f"; say "模型 $_f 已拉到并验哈希"
  done
  [ $_ok = 1 ] || say "注意：大脑或嵌入模型没备齐，调度器起不来；联网后重跑 ./deploy.sh 会续拉补齐，其余步骤不受影响"
}

step_asr_models() {
  STEP="asr-models ASR 模型（瘦身包不带，装机时备齐）"
  _d="$INSTALL_DIR/asr-models"
  mkdir -p "$_d" || fail "建不了 $_d"
  _ok=1
  for _spec in "$ASR_ENC_FILE $ASR_ENC_SIZE $ASR_ENC_SHA256 $ASR_BASE_MOON/encoder_model_quantized.onnx" \
               "$ASR_DEC_FILE $ASR_DEC_SIZE $ASR_DEC_SHA256 $ASR_BASE_MOON/decoder_model_merged_quantized.onnx" \
               "$ASR_VAD_FILE $ASR_VAD_SIZE $ASR_VAD_SHA256 $ASR_VAD_URL"; do
    set -- $_spec; _f=$1; _sz=$2; _sha=$3; _url=$4
    if [ -f "$_d/$_f" ] && [ "$(stat -c %s "$_d/$_f" 2>/dev/null || echo 0)" = "$_sz" ] \
       && [ "$(sha256sum "$_d/$_f" | cut -d' ' -f1)" = "$_sha" ]; then
      say "ASR 模型 $_f 已在（大小与哈希对）"; continue
    fi
    say "ASR 模型 $_f 不在，从 hf-mirror 拉（钉死 revision）"
    curl -L -m 900 -o "$_d/$_f.tmp" "$_url" || { say "下载 $_f 失败，跳过"; _ok=0; continue; }
    if [ "$(stat -c %s "$_d/$_f.tmp" 2>/dev/null || echo 0)" != "$_sz" ] \
       || [ "$(sha256sum "$_d/$_f.tmp" | cut -d' ' -f1)" != "$_sha" ]; then
      say "下载的 $_f 大小或哈希不对，删掉跳过"; rm -f "$_d/$_f.tmp"; _ok=0; continue
    fi
    mv "$_d/$_f.tmp" "$_d/$_f"; say "ASR 模型 $_f 已拉到并验哈希"
  done
  [ $_ok = 1 ] || say "注意：ASR 模型没备齐，语音输入先不可用；联网后重跑 ./deploy.sh 会续拉补齐，其余步骤不受影响"
}

step_vision() {
  STEP="vision 眼睛模型（不常驻，只备文件）"
  _d="$INSTALL_DIR/models-vision"
  mkdir -p "$_d" || fail "建不了 $_d（curl 写 .tmp 会失败）"
  _ok=1
  for _spec in "$VISION1_FILE $VISION1_SIZE $VISION1_SHA256" "$VISION2_FILE $VISION2_SIZE $VISION2_SHA256"; do
    set -- $_spec; _f=$1; _sz=$2; _sha=$3
    if [ -f "$_d/$_f" ] && [ "$(stat -c %s "$_d/$_f" 2>/dev/null || echo 0)" = "$_sz" ]; then
      say "眼睛模型 $_f 已在（大小对）"; continue
    fi
    say "眼睛模型 $_f 不在包里（开发机配额所限没进 tarball），从 hf-mirror 拉"
    curl -L -m 1800 -o "$_d/$_f.tmp" "$VISION_BASE/$_f" || { say "下载 $_f 失败，跳过（眼睛是临时加载，不挡常驻）"; _ok=0; continue; }
    if [ "$(stat -c %s "$_d/$_f.tmp")" != "$_sz" ] || [ "$(sha256sum "$_d/$_f.tmp" | cut -d' ' -f1)" != "$_sha" ]; then
      say "下载的 $_f 大小或哈希不对，删掉跳过"; rm -f "$_d/$_f.tmp"; _ok=0; continue
    fi
    mv "$_d/$_f.tmp" "$_d/$_f"; say "眼睛模型 $_f 已拉到并验哈希"
  done
  [ $_ok = 1 ] || say "注意：眼睛模型没备齐，拍照功能先不可用，其余不受影响"
}

L2D_OK=1
step_l2d() {
  STEP="l2d Live2D 模型（不进 tar，装机时按清单拉）"
  _list="$INSTALL_DIR/orig-ui/live2d-files.txt"
  _d="$INSTALL_DIR/models-live2d"
  if [ ! -f "$_list" ]; then
    say "Live2D 清单不在（orig-ui/live2d-files.txt），跳过：页面能开、对话与语音不受影响，只是舞台上没人"; L2D_OK=0; return 0
  fi
  _fail=0; _got=0; _had=0
  while read -r _model _rev _rel _sz _sha; do
    case "$_model" in ''|'#'*) continue ;; esac
    _t="$_d/$_model/$_rel"
    if [ -f "$_t" ] && [ "$(stat -c %s "$_t" 2>/dev/null || echo 0)" = "$_sz" ] && [ "$(sha256sum "$_t" | cut -d' ' -f1)" = "$_sha" ]; then
      _had=$((_had+1)); continue
    fi
    mkdir -p "$(dirname "$_t")"
    if curl -sL -m 600 -o "$_t.tmp" "https://hf-mirror.com/$L2D_REPO/resolve/$_rev/live2d/$_model/$_rel" \
       && [ "$(stat -c %s "$_t.tmp" 2>/dev/null || echo 0)" = "$_sz" ] \
       && [ "$(sha256sum "$_t.tmp" | cut -d' ' -f1)" = "$_sha" ]; then
      mv "$_t.tmp" "$_t"; _got=$((_got+1))
    else
      rm -f "$_t.tmp"; _fail=$((_fail+1)); say "Live2D 文件没拉到或哈希不对：$_model/$_rel"
    fi
  done < "$_list"
  if [ $_fail -gt 0 ]; then
    L2D_OK=0
    say "注意：Live2D 模型没备齐（失败 $_fail 件、新拉 $_got 件、已有 $_had 件）。后端与页面照常起，只是舞台上没人；联网后重跑 ./deploy.sh 会按清单续拉补齐。"
  else
    say "Live2D 模型齐了（新拉 $_got 件、已有 $_had 件，逐件大小与 sha256 全过）"
  fi
}

step_start_asr() {
  STEP="start-asr 起 ASR 网关"
  LOGHINT="$INSTALL_DIR/logs/asr-gateway.log"
  if pid_alive "$INSTALL_DIR/run/asr_gateway.pid"; then say "ASR 网关已在跑，跳过"; return 0; fi
  # pid 落盘要可靠：cd 必须在子 shell 里先执行、nohup 单独 & 起，$! 才是守护进程
  # 本人且 pid 路径落在 $INSTALL_DIR/run 下（旧写法 `cd && nohup ... & echo $!`
  # 把 cd 圈进了后台作业，$! 与相对路径都落在外层，脚本会误判进程已退出）。
  ( cd "$INSTALL_DIR" || exit 1
    nohup ./bin/asr_gateway \
      -listen "127.0.0.1:$ASR_GW_PORT" \
      -asr-bin "$INSTALL_DIR/bin/asr_host" \
      -model-dir "$INSTALL_DIR/asr-models" \
      -threads "$ASR_THREADS" \
      > logs/asr-gateway.log 2>&1 &
    echo $! > run/asr_gateway.pid )
  wait_port $ASR_GW_PORT 60 "asr" || fail "ASR 网关 60 秒没在 $ASR_GW_PORT 起来"
  say "ASR 网关在 $ASR_GW_PORT 就绪（asr_host 由它自拉）"
}

step_start_scheduler() {
  STEP="start-scheduler 起调度器（自拉大脑、嵌入、TTS）"
  LOGHINT="$INSTALL_DIR/logs/scheduler.log"
  if pid_alive "$INSTALL_DIR/run/scheduler.pid"; then say "调度器已在跑，跳过"; return 0; fi
  # pid 落盘同 step_start_asr 的修法：cd 先在子 shell 里执行、nohup 单独 & 起。
  ( cd "$INSTALL_DIR" || exit 1
    nohup ./bin/scheduler -config config/scheduler.json \
      > logs/scheduler.log 2>&1 &
    echo $! > run/scheduler.pid )
  _n=0
  while [ $_n -lt $HEALTH_TIMEOUT_S ]; do
    if curl -s -m 3 "http://127.0.0.1:$SCHED_PORT/healthz" 2>/dev/null | grep -q '"ok"'; then
      say "调度器健康（/healthz 过，大脑与嵌入的 llama-server、tts_host 都由它拉起）"; return 0
    fi
    pid_alive "$INSTALL_DIR/run/scheduler.pid" || fail "调度器进程中途退出了"
    sleep 5; _n=$((_n+5))
  done
  fail "调度器 $HEALTH_TIMEOUT_S 秒内没健康（手机冷加载慢可调 HEALTH_TIMEOUT_S 再 --start）"
}

step_start_adapter() {
  STEP="start-adapter 起原版页适配器（正式前端）"
  LOGHINT="$INSTALL_DIR/logs/adapter.log"
  if pid_alive "$INSTALL_DIR/run/adapter.pid"; then say "适配器已在跑，跳过"; return 0; fi
  # 适配器是纯 Go 静态二进制（bin/adapter），装机不再需要 node。
  # pid 落盘同 step_start_scheduler 的修法：cd 先在子 shell 里执行、nohup 单独 & 起。
  ( cd "$INSTALL_DIR" || exit 1
    PORT=$ADAPTER_PORT \
      UI_PRISTINE="$INSTALL_DIR/orig-ui/pristine" \
      UI_PATCHED="$INSTALL_DIR/orig-ui/patched" \
      MODELS_DIR="$INSTALL_DIR/models-live2d" \
      STATIC_DIR="$INSTALL_DIR/orig-ui/static" \
      DATA_DIR="$INSTALL_DIR/data/orig-ui" \
      SCHED="http://127.0.0.1:$SCHED_PORT" \
      ASR_WS="ws://127.0.0.1:$ASR_GW_PORT/ws/asr" \
      nohup ./bin/adapter > logs/adapter.log 2>&1 &
    echo $! > run/adapter.pid )
  wait_port $ADAPTER_PORT 30 "adapter" || fail "适配器 30 秒没在 $ADAPTER_PORT 起来"
  say "原版页适配器在 $ADAPTER_PORT 就绪：http://127.0.0.1:$ADAPTER_PORT/use"
}

step_selftest() {
  STEP="selftest 自检（文本进 wav 出）"
  LOGHINT="$INSTALL_DIR/logs/scheduler.log"
  _resp="$INSTALL_DIR/logs/selftest-chat.json"
  curl -s -m $SELFTEST_TIMEOUT_S -X POST "http://127.0.0.1:$SCHED_PORT/v1/chat" \
    -H 'Content-Type: application/json' \
    -d '{"session_id":"deploy-selftest","text":"Hello. Please say one short friendly sentence."}' \
    > "$_resp" || fail "自检对话请求失败或超时（上限 $SELFTEST_TIMEOUT_S 秒）"
  _wav=$(grep -o '"wav": *"[^"]*"' "$_resp" | head -1 | sed 's/.*: *"//; s/"$//')
  [ -n "$_wav" ] || fail "自检回复里没有 wav（回复全文在 logs/selftest-chat.json）"
  [ -f "$_wav" ] || fail "自检 wav 路径不存在：$_wav"
  [ "$(stat -c %s "$_wav")" -gt 1000 ] || fail "自检 wav 太小：$_wav"
  [ "$(head -c 4 "$_wav")" = "RIFF" ] || fail "自检 wav 不是 RIFF：$_wav"
  say "自检一过：文本进、wav 出（$_wav，$(stat -c %s "$_wav") 字节）"

  STEP="selftest 自检（PCM 进文本出）"
  LOGHINT="$INSTALL_DIR/logs/asr-stream.log"
  ( cat "$INSTALL_DIR/test/hello.pcm" | timeout 300 "$INSTALL_DIR/bin/asr_host" \
      --model-dir "$INSTALL_DIR/asr-models" --stream --quant --threads "$ASR_THREADS" --opt basic \
      > "$INSTALL_DIR/logs/asr-stream.out" 2> "$INSTALL_DIR/logs/asr-stream.log" ) || true
  grep -q '"text": *"[^"]' "$INSTALL_DIR/logs/asr-stream.out" \
    || fail "自检 ASR 没吐出文本（输出在 logs/asr-stream.out）"
  say "自检二过：PCM 进、文本出（$(head -1 "$INSTALL_DIR/logs/asr-stream.out")）"

  STEP="selftest 自检（say 回合外独立合成）"
  LOGHINT="$INSTALL_DIR/logs/scheduler.log"
  _sayresp="$INSTALL_DIR/logs/selftest-say.json"
  curl -s -m $SELFTEST_TIMEOUT_S -X POST "http://127.0.0.1:$SCHED_PORT/v1/say" \
    -H 'Content-Type: application/json' \
    -d '{"text":"Self test.","tag":"neutral1"}' \
    > "$_sayresp" || fail "自检 say 请求失败或超时（上限 $SELFTEST_TIMEOUT_S 秒）"
  _swav=$(grep -o '"file": *"[^"]*"' "$_sayresp" | head -1 | sed 's/.*: *"//; s/"$//')
  [ -n "$_swav" ] || fail "自检 say 响应里没有 file（响应在 logs/selftest-say.json）"
  [ -f "$_swav" ] || fail "自检 say 的 wav 路径不存在：$_swav"
  [ "$(head -c 4 "$_swav")" = "RIFF" ] || fail "自检 say 的 wav 不是 RIFF：$_swav"
  say "自检三过：回合外独立合成出真 wav（$_swav，向导台词与名字试听走的就是这条）"

  STEP="selftest 自检（原版页发现 Live2D 模型）"
  if [ -d "$INSTALL_DIR/models-live2d/Amika" ]; then
    curl -s -m 15 "http://127.0.0.1:$ADAPTER_PORT/api/avatar/models" \
      > "$INSTALL_DIR/logs/selftest-models.json" || fail "自检取 /api/avatar/models 失败（适配器没在跑？）"
    grep -q '"id":"Amika"' "$INSTALL_DIR/logs/selftest-models.json" \
      || fail "模型在盘上但 /api/avatar/models 没发现 Amika（响应在 logs/selftest-models.json）"
    say "自检四过：/api/avatar/models 发现 Amika"
  else
    say "自检四跳过：Live2D 模型不在盘上（拉取失败时装机日志里有明示），舞台为空但其余功能不受影响"
  fi
}

step_stop() {
  STEP="stop 停全套"
  for _p in adapter scheduler asr_gateway; do
    if pid_alive "$INSTALL_DIR/run/$_p.pid"; then
      kill "$(cat "$INSTALL_DIR/run/$_p.pid")" 2>/dev/null || true
    fi
    rm -f "$INSTALL_DIR/run/$_p.pid"
  done
  sleep 2
  # 调度器的子进程（llama-server、tts_host）与网关的子进程（asr_host）按安装路径兜底清理
  pkill -f "$INSTALL_DIR" 2>/dev/null || true
  say "已停全套（调度器、ASR 网关、前端及其子进程）"
}

step_dryrun() {
  STEP="dry-run 干跑（只验文件与哈希）"
  _src="${POTATO_SRC:-$INSTALL_DIR}"
  if [ ! -f "$_src/hashes.sha256" ]; then
    # 没解开的树就验 tarball 的内容清单与包内 hashes 表（流式，不落盘解全包）
    if [ ! -f "$TARBALL" ] && [ -f "$SCRIPT_DIR/potatoamika.tar.part-00" ]; then
      say "对着分片干跑：potatoamika.tar.part-*"
      cat "$SCRIPT_DIR"/potatoamika.tar.part-* | tar -tf - | grep -q '^potatoamika/hashes.sha256$' || fail "分片里没有 hashes.sha256"
      say "分片可读、清单在。完整哈希校验在真解包后由 install 流程做。"
      return 0
    fi
    [ -f "$TARBALL" ] || fail "没有 $_src/hashes.sha256，也没有 $TARBALL"
    say "对着 tarball 干跑：$TARBALL"
    case "$TARBALL" in *.gz) _tf=tzf ;; *) _tf=tf ;; esac
    tar -$_tf "$TARBALL" | grep -q '^potatoamika/hashes.sha256$' || fail "tarball 里没有 hashes.sha256"
    say "tarball 可读、清单在。完整哈希校验在真解包后由 install 流程做。"
    return 0
  fi
  say "对着已解开的树干跑：$_src"
  ( cd "$_src" && sha256sum -c hashes.sha256 ) > /tmp/potato-dryrun-verify.log 2>&1 \
    || { grep -v ': OK' /tmp/potato-dryrun-verify.log | head -5 >&2; fail "干跑哈希有不过的文件"; }
  say "文件数与哈希全过（$(grep -c ': OK' /tmp/potato-dryrun-verify.log) 个文件）"
  for _b in scheduler tts_host asr_host asr_gateway adapter; do
    _m=$(od -An -tx1 -N4 "$_src/bin/$_b" 2>/dev/null | tr -d ' ')
    [ "$_m" = "7f454c46" ] || fail "bin/$_b 不是 ELF"
    say "bin/$_b ELF 头在（aarch64 架构在 MANIFEST 里有 file 检验记录）"
  done
  sed "s|@ROOT@|$_src|g" "$_src/config/scheduler.json.tmpl" > /tmp/potato-dryrun-config.json \
    || fail "配置模板渲染失败"
  grep -q "$_src" /tmp/potato-dryrun-config.json || fail "配置模板没替进路径"
  say "配置模板渲染过。干跑结论：文件齐、哈希对、配置能生成，不起服务到此为止。"
}

# ---------------- 主流程 ----------------
case "${1:-install}" in
  --dry-run)   step_dryrun ;;
  --stop)      step_stop ;;
  --uninstall) step_stop; rm -rf "$INSTALL_DIR"; say "已删 $INSTALL_DIR，回滚完成" ;;
  --start)
    step_start_asr; step_start_scheduler; step_start_adapter
    say "全套已起。停：./deploy.sh --stop" ;;
  --selftest)  step_selftest ;;
  install|"")
    step_preflight; step_extract; step_verify; step_config; step_llm_models; step_asr_models; step_vision; step_l2d
    step_start_asr; step_start_scheduler; step_start_adapter; step_selftest
    say "部署完成。页面 http://127.0.0.1:$ADAPTER_PORT/use ，调度器 http://127.0.0.1:$SCHED_PORT/healthz"
    say "停全套：./deploy.sh --stop ；回滚：./deploy.sh --uninstall" ;;
  *) echo "未知参数 $1（看脚本头部用法）" >&2; exit 2 ;;
esac
