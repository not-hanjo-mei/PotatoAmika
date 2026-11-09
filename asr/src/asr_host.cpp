// asr_host: Moonshine Tiny (ONNX) + Silero VAD in one onnxruntime process.
//
// Pipeline: 16 kHz mono PCM -> Silero VAD (512-sample chunks) -> segments
// (a run of >= min_silence_ms of sub-threshold chunks ends a segment) ->
// per-segment Moonshine encoder + greedy decoder with KV cache -> text lines
// with timestamps on stdout. Stats (audio seconds, compute seconds) go to
// stderr; peak RSS is printed by the caller via /usr/bin/time or getrusage.
//
// The tokenizer is SentencePiece-style BPE: decoding concatenates token byte
// strings, turns the U+2581 marker back into a space, and expands <0xXX>
// byte-fallback tokens. Vocab comes from models/vocab.tsv (export_vocab.py).
//
// No sherpa-onnx: plain onnxruntime C++ API, same style as tts-engine/host,
// so the same source builds against Alpine's musl onnxruntime package.

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <sys/resource.h>
#include <unistd.h>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace {

bool g_no_arena = false;
int g_opt = -1;  // -1 = ORT default (ALL), 0 = disable, 1 = basic, 2 = extended

void apply_opt(Ort::SessionOptions& opts) {
  if (g_opt == 0) opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
  else if (g_opt == 1) opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
  else if (g_opt == 2) opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  if (g_no_arena) opts.DisableCpuMemArena();
}

[[noreturn]] void die(const std::string& msg) {
  fprintf(stderr, "[asr] error: %s\n", msg.c_str());
  exit(1);
}

// Stream mode flush request (SIGUSR1, sent by the gateway on ptt_end).
// Installed without SA_RESTART so a blocked read() returns EINTR.
volatile sig_atomic_t g_flush_req = 0;
void on_flush_sig(int) { g_flush_req = 1; }

std::string json_escape(const std::string& s) {
  std::string out;
  for (char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if ((unsigned char)c < 0x20) {
          char b[8];
          snprintf(b, sizeof b, "\\u%04x", c);
          out += b;
        } else {
          out.push_back(c);
        }
    }
  }
  return out;
}

// ---------------- WAV input (PCM16 / float32, mixed down to mono) ----------
struct Wav {
  int sample_rate = 0;
  std::vector<float> samples;  // mono, nominally [-1, 1]
};

Wav read_wav(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) die("cannot open wav: " + path);
  std::vector<char> buf((std::istreambuf_iterator<char>(in)),
                        std::istreambuf_iterator<char>());
  auto u16 = [&](size_t off) {
    return (uint16_t)(uint8_t)buf[off] | ((uint16_t)(uint8_t)buf[off + 1] << 8);
  };
  auto u32 = [&](size_t off) {
    return (uint32_t)u16(off) | ((uint32_t)u16(off + 2) << 16);
  };
  if (buf.size() < 44 || memcmp(buf.data(), "RIFF", 4) != 0) die("not a RIFF wav");
  uint16_t fmt_tag = 0, channels = 0, bits = 0;
  uint32_t rate = 0;
  size_t data_off = 0, data_size = 0;
  for (size_t off = 12; off + 8 <= buf.size();) {
    uint32_t sz = u32(off + 4);
    if (memcmp(buf.data() + off, "fmt ", 4) == 0) {
      fmt_tag = u16(off + 8);
      channels = u16(off + 10);
      rate = u32(off + 12);
      bits = u16(off + 22);
    } else if (memcmp(buf.data() + off, "data", 4) == 0) {
      data_off = off + 8;
      data_size = sz;
      break;
    }
    off += 8 + sz + (sz & 1);
  }
  if (!data_off || channels == 0) die("wav missing data chunk");
  Wav w;
  w.sample_rate = (int)rate;
  size_t frames = 0;
  if (fmt_tag == 1 && bits == 16) {
    frames = data_size / 2 / channels;
    w.samples.resize(frames);
    for (size_t i = 0; i < frames; ++i) {
      double acc = 0;
      for (int c = 0; c < channels; ++c) {
        int16_t v = (int16_t)u16(data_off + (i * channels + c) * 2);
        acc += v / 32768.0;
      }
      w.samples[i] = (float)(acc / channels);
    }
  } else if (fmt_tag == 3 && bits == 32) {
    frames = data_size / 4 / channels;
    w.samples.resize(frames);
    for (size_t i = 0; i < frames; ++i) {
      double acc = 0;
      for (int c = 0; c < channels; ++c) {
        float v;
        memcpy(&v, buf.data() + data_off + (i * channels + c) * 4, 4);
        acc += v;
      }
      w.samples[i] = (float)(acc / channels);
    }
  } else {
    die("unsupported wav format (need PCM16 or float32)");
  }
  return w;
}

// ---------------- Tokenizer (decode side only) ------------------------------
struct Vocab {
  std::map<int64_t, std::string> by_id;  // raw UTF-8 bytes of each token
  static int hexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return 0;
  }
  void load(const std::string& path) {
    std::ifstream in(path);
    if (!in) die("cannot open vocab: " + path);
    std::string line;
    while (std::getline(in, line)) {
      size_t tab = line.find('\t');
      if (tab == std::string::npos) continue;
      int64_t id = std::stoll(line.substr(0, tab));
      std::string hex = line.substr(tab + 1);
      std::string bytes;
      for (size_t i = 0; i + 1 < hex.size(); i += 2)
        bytes.push_back((char)(hexval(hex[i]) * 16 + hexval(hex[i + 1])));
      by_id[id] = bytes;
    }
  }
  // ids -> UTF-8 text. Skips specials (0,1,2) and added tokens (>=32000);
  // <0xXX> byte-fallback tokens become the single byte XX; U+2581 -> space.
  std::string decode(const std::vector<int64_t>& ids) const {
    std::string raw;
    for (int64_t id : ids) {
      if (id < 3 || id >= 32000) continue;
      auto it = by_id.find(id);
      if (it == by_id.end()) continue;
      const std::string& t = it->second;
      if (t.size() == 6 && t[0] == '<' && t[1] == '0' && t[2] == 'x' &&
          t[5] == '>') {
        int v = hexval(t[3]) * 16 + hexval(t[4]);
        raw.push_back((char)v);
      } else {
        raw += t;
      }
    }
    const std::string marker = "\xe2\x96\x81";  // U+2581
    std::string out;
    for (size_t i = 0; i < raw.size();) {
      if (raw.compare(i, 3, marker) == 0) {
        out.push_back(' ');
        i += 3;
      } else {
        out.push_back(raw[i]);
        ++i;
      }
    }
    size_t b = out.find_first_not_of(' ');
    size_t e = out.find_last_not_of(' ');
    return b == std::string::npos ? "" : out.substr(b, e - b + 1);
  }
};

// ---------------- Silero VAD -------------------------------------------------
class Vad {
 public:
  Vad(Ort::Env& env, const std::string& path, int threads) {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(threads);
    opts.SetInterOpNumThreads(1);
    apply_opt(opts);
    session_ = std::make_unique<Ort::Session>(env, path.c_str(), opts);
    state_.assign(2 * 128, 0.0f);
  }
  // prob for one 512-sample chunk (zero-padded by caller if short)
  float step(const float* chunk512) {
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<int64_t, 2> in_shape{1, 512};
    std::array<int64_t, 3> st_shape{2, 1, 128};
    std::array<int64_t, 0> sr_shape{};
    int64_t sr = 16000;
    std::vector<Ort::Value> ins;
    ins.emplace_back(Ort::Value::CreateTensor<float>(mem, const_cast<float*>(chunk512), 512, in_shape.data(), 2));
    ins.emplace_back(Ort::Value::CreateTensor<float>(mem, state_.data(), state_.size(), st_shape.data(), 3));
    ins.emplace_back(Ort::Value::CreateTensor<int64_t>(mem, &sr, 1, sr_shape.data(), 0));
    const char* in_names[] = {"input", "state", "sr"};
    const char* out_names[] = {"output", "stateN"};
    auto outs = session_->Run(Ort::RunOptions{nullptr}, in_names, ins.data(), 3, out_names, 2);
    float prob = outs[0].GetTensorData<float>()[0];
    const float* st = outs[1].GetTensorData<float>();
    std::copy(st, st + state_.size(), state_.data());
    return prob;
  }

 private:
  std::unique_ptr<Ort::Session> session_;
  std::vector<float> state_;
};

struct Segment {
  double start_s, end_s;    // speech span decided by the VAD rules
  size_t start_sample, end_sample;  // padded span actually fed to ASR
};

// Chunk probs -> segments. Speech starts at the first chunk >= threshold;
// a segment ends when sub-threshold chunks run for >= min_silence_ms
// (segment end = start of that silence run). Blips < min_speech_ms dropped.
std::vector<Segment> segment_probs(const std::vector<float>& probs, float thr,
                                   int min_silence_ms, int min_speech_ms,
                                   int preroll_ms, int sr) {
  const int chunk = 512;
  auto ms_of = [&](size_t chunk_idx) { return (int)(chunk_idx * chunk * 1000 / sr); };
  std::vector<Segment> segs;
  bool in_speech = false;
  size_t speech_start = 0, silence_run_start = 0;
  int silence_ms = 0;
  auto close = [&](size_t end_chunk) {
    if (ms_of(end_chunk) - ms_of(speech_start) >= min_speech_ms) {
      Segment s;
      s.start_s = ms_of(speech_start) / 1000.0;
      s.end_s = ms_of(end_chunk) / 1000.0;
      size_t pre = (size_t)preroll_ms * sr / 1000;
      s.start_sample = speech_start * chunk > pre ? speech_start * chunk - pre : 0;
      s.end_sample = end_chunk * chunk + pre;  // symmetric tail padding
      segs.push_back(s);
    }
  };
  for (size_t i = 0; i < probs.size(); ++i) {
    if (!in_speech) {
      if (probs[i] >= thr) {
        in_speech = true;
        speech_start = i;
        silence_ms = 0;
      }
    } else {
      if (probs[i] >= thr) {
        silence_ms = 0;
      } else {
        if (silence_ms == 0) silence_run_start = i;
        silence_ms = ms_of(i + 1) - ms_of(silence_run_start);
        if (silence_ms >= min_silence_ms) {
          close(silence_run_start);
          in_speech = false;
          silence_ms = 0;
        }
      }
    }
  }
  if (in_speech) close(probs.size());
  return segs;
}

// ---------------- Moonshine --------------------------------------------------
class Moonshine {
 public:
  Moonshine(Ort::Env& env, const std::string& enc_path, const std::string& dec_path,
            int threads) {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(threads);
    opts.SetInterOpNumThreads(1);
    apply_opt(opts);
    enc_ = std::make_unique<Ort::Session>(env, enc_path.c_str(), opts);
    dec_ = std::make_unique<Ort::Session>(env, dec_path.c_str(), opts);
    Ort::AllocatorWithDefaultOptions alloc;
    size_t n_in = dec_->GetInputCount();
    for (size_t i = 0; i < n_in; ++i) {
      std::string name = dec_->GetInputNameAllocated(i, alloc).get();
      dec_in_names_.push_back(name);
      if (name.rfind("past_key_values.", 0) == 0) {
        // Zero-length past for step 0. Shape is [1, heads, 0, head_dim] with
        // heads=8, head_dim=36 for moonshine-tiny (hidden 288). Reading this
        // back via GetInputTypeInfo(i).GetShape() throws std::length_error on
        // ORT 1.22 for these symbolic shapes, so it is pinned from the config.
        past_shapes_[name] = {1, 8, 0, 36};
      }
    }
    size_t n_out = dec_->GetOutputCount();
    for (size_t i = 0; i < n_out; ++i)
      dec_out_names_.push_back(dec_->GetOutputNameAllocated(i, alloc).get());
  }

  std::vector<int64_t> transcribe(const float* audio, size_t n, const Vocab& vocab,
                                  std::string& text_out) {
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    // encoder
    std::array<int64_t, 2> in_shape{1, (int64_t)n};
    std::vector<Ort::Value> ein;
    ein.emplace_back(Ort::Value::CreateTensor<float>(mem, const_cast<float*>(audio), n, in_shape.data(), 2));
    const char* ein_name = "input_values";
    const char* eout_name = "last_hidden_state";
    auto eout = enc_->Run(Ort::RunOptions{nullptr}, &ein_name, ein.data(), 1, &eout_name, 1);
    auto einfo = eout[0].GetTensorTypeAndShapeInfo();
    std::vector<int64_t> eshape = einfo.GetShape();
    size_t enc_elems = einfo.GetElementCount();
    std::vector<float> enc_data(eout[0].GetTensorData<float>(),
                                eout[0].GetTensorData<float>() + enc_elems);
    // greedy decoder loop
    const int64_t kStart = 1, kEos = 2;
    const int kMaxTokens = 256;
    std::map<std::string, Ort::Value> past;  // decoder past_key_values.* -> value
    // Encoder KV is taken from step 0's presents and re-fed verbatim on every
    // later step: in the use_cache=true branch this merged graph returns
    // malformed encoder presents (shape [0, 8, 1, 36]), so they are never
    // fed back. Only the decoder KV advances.
    std::map<std::string, Ort::Value> enc_past;
    std::vector<int64_t> ids;
    int64_t cur = kStart;
    bool use_cache = false;  // use_cache_branch input: false on the first step
    bool use_cache_val = false;
    std::vector<const char*> in_names, out_names;
    for (auto& s : dec_in_names_) in_names.push_back(s.c_str());
    for (auto& s : dec_out_names_) out_names.push_back(s.c_str());
    for (int step = 0; step < kMaxTokens; ++step) {
      std::vector<Ort::Value> ins;
      ins.reserve(dec_in_names_.size());
      // Ort::Value does not copy shape arrays: every shape handed to
      // CreateTensor must outlive the Run call, so they are kept here at
      // step scope, not inside the per-input branches.
      std::array<int64_t, 2> ids_shape{1, 1};
      std::array<int64_t, 1> ucb_shape{1};
      std::vector<std::vector<int64_t>> past_shapes_keep;
      for (const std::string& name : dec_in_names_) {
        if (name == "input_ids") {
          ins.emplace_back(Ort::Value::CreateTensor<int64_t>(mem, &cur, 1, ids_shape.data(), 2));
        } else if (name == "encoder_hidden_states") {
          ins.emplace_back(Ort::Value::CreateTensor<float>(mem, enc_data.data(), enc_elems, eshape.data(), eshape.size()));
        } else if (name == "use_cache_branch") {
          use_cache_val = use_cache;
          ins.emplace_back(Ort::Value::CreateTensor<bool>(mem, &use_cache_val, 1, ucb_shape.data(), 1));
        } else {
          auto& src = (name.find(".encoder.") != std::string::npos) ? enc_past : past;
          auto it = src.find(name);
          if (it != src.end()) {
            ins.emplace_back(std::move(it->second));
            src.erase(it);
          } else {
            // first step: zero-length past, pinned shape [1, 8, 0, 36],
            // with a real (dummy) data pointer rather than nullptr.
            past_shapes_keep.push_back(past_shapes_[name]);
            const std::vector<int64_t>& sh = past_shapes_keep.back();
            ins.emplace_back(Ort::Value::CreateTensor<float>(mem, &dummy_, 0, sh.data(), sh.size()));
          }
        }
      }
      if (step <= 2 && getenv("ASR_DEBUG")) {
        fprintf(stderr, "[dbg] eshape:");
        for (auto d : eshape) fprintf(stderr, " %lld", (long long)d);
        fprintf(stderr, " enc_elems=%zu\n", enc_elems);
        for (size_t i = 0; i < ins.size(); ++i) {
          auto sh = ins[i].GetTensorTypeAndShapeInfo().GetShape();
          fprintf(stderr, "[dbg] in %s shape:", dec_in_names_[i].c_str());
          for (auto d : sh) fprintf(stderr, " %lld", (long long)d);
          fprintf(stderr, " type=%d\n", (int)ins[i].GetTensorTypeAndShapeInfo().GetElementType());
        }
        fprintf(stderr, "[dbg] ucb tensor data: %d (use_cache=%d)\n",
                (int)ins.back().GetTensorMutableData<bool>()[0], (int)use_cache);
      }
      if (getenv("ASR_DEBUG")) fprintf(stderr, "[dbg] run step %d cur=%lld\n", step, (long long)cur);
      auto outs = dec_->Run(Ort::RunOptions{nullptr}, in_names.data(), ins.data(),
                            ins.size(), out_names.data(), out_names.size());
      // reclaim the encoder KV tensors lent to this step's inputs
      for (size_t i = 0; i < dec_in_names_.size(); ++i)
        if (dec_in_names_[i].find(".encoder.") != std::string::npos &&
            enc_past.find(dec_in_names_[i]) == enc_past.end() && step > 0)
          enc_past[dec_in_names_[i]] = std::move(ins[i]);
      // collect present.* back under past_key_values.* names; encoder
      // presents are only trustworthy on step 0 (see enc_past note above)
      for (size_t i = 0; i < dec_out_names_.size(); ++i) {
        const std::string& on = dec_out_names_[i];
        if (on == "logits") continue;
        std::string pn = "past_key_values." + on.substr(std::string("present.").size());
        if (pn.find(".encoder.") != std::string::npos) {
          if (step == 0) enc_past[pn] = std::move(outs[i]);
        } else {
          past[pn] = std::move(outs[i]);
        }
      }
      const float* logits = outs[0].GetTensorData<float>();
      auto linfo = outs[0].GetTensorTypeAndShapeInfo();
      size_t vocab_n = (size_t)linfo.GetShape().back();
      size_t last = linfo.GetElementCount() - vocab_n;
      int64_t best = 0;
      float bv = logits[last];
      for (size_t i = 1; i < vocab_n; ++i)
        if (logits[last + i] > bv) {
          bv = logits[last + i];
          best = (int64_t)i;
        }
      if (best == kEos) break;
      ids.push_back(best);
      cur = best;
      use_cache = true;
    }
    text_out = vocab.decode(ids);
    return ids;
  }

 private:
  std::unique_ptr<Ort::Session> enc_, dec_;
  float dummy_ = 0.0f;
  std::vector<std::string> dec_in_names_, dec_out_names_;
  std::map<std::string, std::vector<int64_t>> past_shapes_;
};

// ---------------- Streaming mode --------------------------------------------
// Incremental twin of segment_probs(): same thresholds, same chunk math,
// same padding. feed() returns true when the silence rule closes a segment;
// flush() closes whatever speech is open (ptt_end / EOF), applying the same
// min-speech rule segment_probs applies to its trailing segment.
class StreamSegmenter {
 public:
  StreamSegmenter(float thr, int min_silence_ms, int min_speech_ms,
                  int preroll_ms, int sr)
      : thr_(thr), min_sil_(min_silence_ms), min_speech_(min_speech_ms),
        preroll_ms_(preroll_ms), sr_(sr) {}

  bool feed(float prob, size_t chunk_idx, Segment& out) {
    if (!in_speech_) {
      if (prob >= thr_) {
        in_speech_ = true;
        speech_start_ = chunk_idx;
        silence_ms_ = 0;
      }
      return false;
    }
    if (prob >= thr_) {
      silence_ms_ = 0;
      return false;
    }
    if (silence_ms_ == 0) silence_run_start_ = chunk_idx;
    silence_ms_ = ms_of(chunk_idx + 1) - ms_of(silence_run_start_);
    if (silence_ms_ >= min_sil_) {
      bool ok = close(silence_run_start_, out);
      in_speech_ = false;
      silence_ms_ = 0;
      return ok;
    }
    return false;
  }

  bool flush(size_t upto_chunk, Segment& out) {
    if (!in_speech_) return false;
    bool ok = close(upto_chunk, out);
    in_speech_ = false;
    silence_ms_ = 0;
    return ok;
  }

 private:
  int ms_of(size_t chunk_idx) const { return (int)(chunk_idx * 512 * 1000 / sr_); }
  bool close(size_t end_chunk, Segment& s) const {
    if (ms_of(end_chunk) - ms_of(speech_start_) < min_speech_) return false;
    size_t pre = (size_t)preroll_ms_ * sr_ / 1000;
    s.start_s = ms_of(speech_start_) / 1000.0;
    s.end_s = ms_of(end_chunk) / 1000.0;
    s.start_sample = speech_start_ * 512 > pre ? speech_start_ * 512 - pre : 0;
    s.end_sample = end_chunk * 512 + pre;
    return true;
  }
  float thr_;
  int min_sil_, min_speech_, preroll_ms_, sr_;
  bool in_speech_ = false;
  size_t speech_start_ = 0, silence_run_start_ = 0;
  int silence_ms_ = 0;
};

// --stream: raw s16le 16 kHz mono PCM on stdin, one JSON object per closed
// segment on stdout: {"text": ..., "start_s": ..., "end_s": ...} with
// stream-relative timestamps. SIGUSR1 flushes the open segment (the gateway
// sends it on ptt_end); EOF does the same and exits. Stats go to stderr.
int run_stream(const std::string& model_dir, bool quant, bool no_vad,
               int threads, float thr, int min_silence_ms) {
  struct sigaction sa {};
  sa.sa_handler = on_flush_sig;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = 0;  // no SA_RESTART: read() must wake on SIGUSR1
  sigaction(SIGUSR1, &sa, nullptr);

  Vocab vocab;
  vocab.load(model_dir + "/vocab.tsv");
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "asr_host_stream");
  auto t0 = std::chrono::steady_clock::now();
  std::string enc_f = quant ? "/onnx_encoder_model_quantized.onnx" : "/onnx_encoder_model.onnx";
  std::string dec_f = quant ? "/onnx_decoder_model_merged_quantized.onnx" : "/onnx_decoder_model_merged.onnx";
  Moonshine asr(env, model_dir + enc_f, model_dir + dec_f, threads);
  std::unique_ptr<Vad> vad;
  if (!no_vad) vad = std::make_unique<Vad>(env, model_dir + "/silero_vad.onnx", threads);
  fprintf(stderr, "[asr] stream ready (load_s=%.3f)\n",
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  fflush(stderr);

  StreamSegmenter segmenter(thr, min_silence_ms, 200, 160, 16000);
  std::vector<float> buf;         // samples not yet dropped
  size_t buf_base = 0;            // global sample index of buf[0]
  size_t total_samples = 0;       // global samples received
  size_t next_chunk = 0;          // next 512-sample chunk to run through VAD
  size_t n_segments = 0;
  double asr_s = 0, vad_s = 0;

  auto emit_span = [&](size_t start_sample, size_t end_sample, double start_s, double end_s) {
    if (end_sample > total_samples) end_sample = total_samples;
    if (start_sample < buf_base) start_sample = buf_base;  // preroll already dropped
    if (end_sample <= start_sample) return;
    std::string text;
    auto ta0 = std::chrono::steady_clock::now();
    asr.transcribe(buf.data() + (start_sample - buf_base),
                   end_sample - start_sample, vocab, text);
    asr_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - ta0).count();
    printf("{\"text\":\"%s\",\"start_s\":%.3f,\"end_s\":%.3f}\n",
           json_escape(text).c_str(), start_s, end_s);
    fflush(stdout);
    n_segments++;
    // Everything up to the unpadded speech end can be dropped: the next
    // segment's padded start reaches back at most the 160 ms preroll, and
    // emit clamps to buf_base if that region is already gone.
    size_t drop = (size_t)(end_s * 16000.0);
    if (drop > total_samples) drop = total_samples;
    if (drop > buf_base) {
      buf.erase(buf.begin(), buf.begin() + (drop - buf_base));
      buf_base = drop;
    }
  };
  auto emit_segment = [&](const Segment& s) {
    emit_span(s.start_sample, s.end_sample, s.start_s, s.end_s);
  };

  // Run every complete 512-sample chunk now in buf through the VAD.
  std::vector<float> chunk(512);
  auto process_chunks = [&]() {
    while ((next_chunk + 1) * 512 <= total_samples) {
      size_t off = next_chunk * 512 - buf_base;
      std::copy(buf.begin() + off, buf.begin() + off + 512, chunk.begin());
      auto tv0 = std::chrono::steady_clock::now();
      float prob = vad->step(chunk.data());
      vad_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - tv0).count();
      Segment s;
      bool closed = segmenter.feed(prob, next_chunk, s);
      next_chunk++;
      if (closed) emit_segment(s);
    }
  };
  // Zero-pad and run one trailing partial chunk (mirrors file mode at EOF).
  auto process_partial_chunk = [&]() {
    if (no_vad || total_samples <= next_chunk * 512) return;
    std::fill(chunk.begin(), chunk.end(), 0.0f);
    size_t off = next_chunk * 512 - buf_base;
    size_t cnt = total_samples - next_chunk * 512;
    std::copy(buf.begin() + off, buf.begin() + off + cnt, chunk.begin());
    auto tv0 = std::chrono::steady_clock::now();
    float prob = vad->step(chunk.data());
    vad_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - tv0).count();
    Segment s;
    bool closed = segmenter.feed(prob, next_chunk, s);
    next_chunk++;
    if (closed) emit_segment(s);
  };
  auto flush_open = [&]() {
    if (no_vad) {
      if (total_samples > buf_base)
        emit_span(buf_base, total_samples, buf_base / 16000.0, total_samples / 16000.0);
      return;
    }
    process_partial_chunk();
    Segment s;
    if (segmenter.flush(next_chunk, s)) emit_segment(s);
  };
  uint8_t carry = 0;
  bool have_carry = false;
  auto append_bytes = [&](const char* p, size_t n) {
    for (size_t i = 0; i < n; ++i) {
      uint8_t b = (uint8_t)p[i];
      if (!have_carry) {
        carry = b;
        have_carry = true;
      } else {
        int16_t v = (int16_t)((uint16_t)carry | ((uint16_t)b << 8));
        buf.push_back(v / 32768.0f);
        total_samples++;
        have_carry = false;
      }
    }
  };

  char rb[8192];
  bool eof = false;
  while (!eof) {
    if (g_flush_req) {
      g_flush_req = 0;
      // Drain bytes the writer already handed to the pipe, then flush.
      int flags = fcntl(0, F_GETFL, 0);
      if (flags >= 0) {
        fcntl(0, F_SETFL, flags | O_NONBLOCK);
        for (;;) {
          ssize_t r = read(0, rb, sizeof rb);
          if (r > 0) {
            append_bytes(rb, (size_t)r);
            if (!no_vad) process_chunks();
          } else break;
        }
        fcntl(0, F_SETFL, flags);
      }
      flush_open();
      continue;
    }
    ssize_t r = read(0, rb, sizeof rb);
    if (r < 0) {
      if (errno == EINTR) continue;  // flush flag handled at loop top
      die(std::string("stdin read failed: ") + strerror(errno));
    }
    if (r == 0) {
      eof = true;
      break;
    }
    append_bytes(rb, (size_t)r);
    if (!no_vad) process_chunks();
  }
  flush_open();
  struct rusage ru {};
  getrusage(RUSAGE_SELF, &ru);
  fprintf(stderr,
          "[asr] stream done segments=%zu audio_s=%.3f vad_s=%.3f asr_s=%.3f "
          "peak_rss_mb=%.1f\n",
          n_segments, total_samples / 16000.0, vad_s, asr_s, ru.ru_maxrss / 1024.0);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_dir, wav_path;
  bool quant = false, no_vad = false, stream = false;
  int threads = 2;
  float thr = 0.5f;
  int min_silence_ms = 800;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() { return std::string(argv[++i]); };
    if (a == "--model-dir") model_dir = next();
    else if (a == "--wav") wav_path = next();
    else if (a == "--stream") stream = true;
    else if (a == "--quant") quant = true;
    else if (a == "--no-vad") no_vad = true;
    else if (a == "--no-arena") g_no_arena = true;
    else if (a == "--opt") { std::string v = next(); g_opt = v == "disable" ? 0 : v == "basic" ? 1 : v == "extended" ? 2 : -1; }
    else if (a == "--threads") threads = std::stoi(next());
    else if (a == "--threshold") thr = std::stof(next());
    else if (a == "--min-silence-ms") min_silence_ms = std::stoi(next());
    else die("unknown arg: " + a);
  }
  if (model_dir.empty() || (!stream && wav_path.empty()))
    die("usage: asr_host --model-dir DIR (--wav FILE | --stream) [--quant] [--no-vad]");
  if (stream)
    return run_stream(model_dir, quant, no_vad, threads, thr, min_silence_ms);

  Wav w = read_wav(wav_path);
  if (w.sample_rate != 16000) die("wav must be 16000 Hz (got " + std::to_string(w.sample_rate) + ")");
  double audio_s = (double)w.samples.size() / 16000.0;

  Vocab vocab;
  vocab.load(model_dir + "/vocab.tsv");
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "asr_host");

  auto t0 = std::chrono::steady_clock::now();
  std::string enc_f = quant ? "/onnx_encoder_model_quantized.onnx" : "/onnx_encoder_model.onnx";
  std::string dec_f = quant ? "/onnx_decoder_model_merged_quantized.onnx" : "/onnx_decoder_model_merged.onnx";
  Moonshine asr(env, model_dir + enc_f, model_dir + dec_f, threads);
  auto t_load = std::chrono::steady_clock::now();

  std::vector<Segment> segs;
  double vad_s = 0;
  if (no_vad) {
    Segment s{0.0, audio_s, 0, w.samples.size()};
    segs.push_back(s);
  } else {
    Vad vad(env, model_dir + "/silero_vad.onnx", threads);
    auto tv0 = std::chrono::steady_clock::now();
    size_t nchunks = (w.samples.size() + 511) / 512;
    std::vector<float> probs(nchunks);
    std::vector<float> chunk(512);
    for (size_t i = 0; i < nchunks; ++i) {
      std::fill(chunk.begin(), chunk.end(), 0.0f);
      size_t off = i * 512, cnt = std::min<size_t>(512, w.samples.size() - off);
      std::copy(w.samples.begin() + off, w.samples.begin() + off + cnt, chunk.begin());
      probs[i] = vad.step(chunk.data());
    }
    vad_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - tv0).count();
    segs = segment_probs(probs, thr, min_silence_ms, 200, 160, 16000);
    for (auto& s : segs) {
      if (s.end_sample > w.samples.size()) s.end_sample = w.samples.size();
    }
  }

  double asr_s = 0;
  for (auto& s : segs) {
    std::string text;
    auto ta0 = std::chrono::steady_clock::now();
    asr.transcribe(w.samples.data() + s.start_sample, s.end_sample - s.start_sample, vocab, text);
    asr_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - ta0).count();
    printf("[%.3f %.3f] %s\n", s.start_s, s.end_s, text.c_str());
    fflush(stdout);
  }
  auto t_all = std::chrono::steady_clock::now();
  struct rusage ru {};
  getrusage(RUSAGE_SELF, &ru);
  fprintf(stderr,
          "[asr] segments=%zu audio_s=%.3f load_s=%.3f vad_s=%.3f asr_s=%.3f "
          "compute_s=%.3f rtf=%.4f peak_rss_mb=%.1f\n",
          segs.size(), audio_s,
          std::chrono::duration<double>(t_load - t0).count(), vad_s, asr_s,
          vad_s + asr_s, (vad_s + asr_s) / (audio_s > 0 ? audio_s : 1),
          ru.ru_maxrss / 1024.0);
  return 0;
}
