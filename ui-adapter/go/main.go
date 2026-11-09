package main

/*
 * PotatoAmika adapter for the ORIGINAL OpenAmika Use page — Go port of
 * orig-ui/adapter.js (same behaviour, single static binary, stdlib only).
 *
 * Serves the original page files (pristine, except use.html / use.js /
 * common.js which carry the documented removals — see REMOVALS.md) and
 * translates the endpoints the page expects to the PotatoAmika scheduler
 * (API.md) and the ASR gateway. Standalone speech (the wizard's lines,
 * name previews) goes through the scheduler's POST /v1/say, which uses
 * the one resident tts_host — this adapter spawns no TTS process of its
 * own.
 *
 * Capabilities PotatoAmika definitively does not have are NOT faked
 * here: their page parts were removed, and their endpoints return
 * explicit 501/404 (see REMOVALS.md). Nothing in this file invents data.
 */

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Directory defaults are relative to the adapter binary, mirroring the
// previous Node build's staged defaults; deploy.sh sets all five
// explicitly, so installed behaviour does not depend on them.
var exePath, _ = os.Executable()
var baseDir = filepath.Dir(exePath)

var (
	port       = envInt("PORT", 8372)
	uiPristine = envStr("UI_PRISTINE", filepath.Join(baseDir, "pristine"))
	uiPatched  = envStr("UI_PATCHED", filepath.Join(baseDir, "patched"))
	modelsDir  = envStr("MODELS_DIR", filepath.Join(baseDir, "models-live2d"))
	staticDir  = envStr("STATIC_DIR", filepath.Join(baseDir, "static"))
	dataDir    = envStr("DATA_DIR", filepath.Join(baseDir, "data"))
	schedBase  = envStr("SCHED", "http://127.0.0.1:18100")
	asrWS      = envStr("ASR_WS", "ws://127.0.0.1:18120/ws/asr")
)

func envStr(key, dflt string) string {
	if v, ok := os.LookupEnv(key); ok {
		return v
	}
	return dflt
}

func envInt(key string, dflt int) int {
	if v, ok := os.LookupEnv(key); ok {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return dflt
}

func logf(format string, args ...interface{}) {
	fmt.Printf("%s %s\n", time.Now().UTC().Format("2006-01-02T15:04:05.000Z"), fmt.Sprintf(format, args...))
}

/* ------------------------------------------------------------------ *
 * Emotion vocabularies (copied from the original speech/voice_adapter.py)
 * ------------------------------------------------------------------ */

// orderedMap preserves the JS object's insertion order for the V1->V2
// derivation and for /api/voice_map output.
type kv struct{ k, v string }

var defaultMapList = []kv{
	{"polite_neutral", "neutral1"}, {"polite_smile", "happy1"}, {"polite_happy", "happy1"},
	{"polite_caring", "caring1"}, {"polite_sad", "sad1"}, {"polite_worried", "worried1"},
	{"polite_cold", "cold1"}, {"polite_angry", "annoyed1"}, {"polite_embarrassed", "embarrassed1"},
	{"polite_surprise", "surprise1"}, {"polite_fear", "fear1"},
	{"casual_neutral", "neutral1"}, {"casual_smile", "happy1"}, {"casual_happy", "happy2"},
	{"casual_very_happy", "happy3"}, {"casual_laughing", "playful3"}, {"casual_playful", "playful2"},
	{"casual_caring", "caring2"}, {"casual_relieved", "relieved1"}, {"casual_shy", "shy2"},
	{"casual_sad", "sad2"}, {"casual_worried", "worried1"}, {"casual_frustrated", "frustrated1"},
	{"casual_angry", "angry2"}, {"casual_jealous", "jealous2"}, {"casual_embarrassed", "embarrassed1"},
	{"casual_surprise", "surprise1"}, {"casual_fear", "fear2"}, {"casual_tired", "tired1"},
	{"casual_mumbling", "mumbling1"},
	{"serious_neutral", "firm1"}, {"serious_protective", "protective1"}, {"serious_cold", "cold3"},
	{"serious_worried", "worried2"}, {"serious_anxious", "anxiety1"}, {"serious_disappointed", "disappointed1"},
	{"serious_sad", "sad2"}, {"serious_angry", "angry3"}, {"serious_very_angry", "angry4"},
	{"serious_fear", "fear3"}, {"serious_surprise", "surprise2"},
	{"intimate_neutral", "neutral1"}, {"intimate_smile", "happy1"}, {"intimate_happy", "happy2"},
	{"intimate_very_happy", "happy3"}, {"intimate_caring", "caring3"}, {"intimate_very_caring", "caring3"},
	{"intimate_playful", "playful1"}, {"intimate_shy", "shy2"}, {"intimate_very_shy", "shy3"},
	{"intimate_relieved", "relieved2"}, {"intimate_sad", "sad2"}, {"intimate_very_sad", "sad3"},
	{"intimate_tearful", "sad3"}, {"intimate_worried", "worried1"}, {"intimate_jealous", "jealous1"},
	{"intimate_tired", "tired2"}, {"intimate_sleepy", "sleepy1"}, {"intimate_mumbling", "mumbling1"},
	{"whisper_neutral", "whisper1"}, {"whisper_smile", "whisper2"}, {"whisper_happy", "happy1"},
	{"whisper_caring", "caring1"}, {"whisper_relieved", "relieved2"}, {"whisper_shy", "shy1"},
	{"whisper_sad", "sad1"}, {"whisper_tearful", "sad1"}, {"whisper_worried", "worried1"},
	{"whisper_sleepy", "sleepy1"},
}

var defaultMap = func() map[string]string {
	m := make(map[string]string, len(defaultMapList))
	for _, e := range defaultMapList {
		m[e.k] = e.v
	}
	return m
}()

var emotions = func() []string {
	out := make([]string, len(defaultMapList))
	for i, e := range defaultMapList {
		out[i] = e.k
	}
	return out
}()

var v1IDs = func() map[string]bool {
	s := map[string]bool{}
	for _, e := range defaultMapList {
		s[e.v] = true
	}
	for _, t := range []string{"awe1", "lonely1", "awkwardness1", "sarcastic1", "pout1", "proud1", "breathy1", "monotone1"} {
		s[t] = true
	}
	return s
}()

// V1 -> V2, preferring the casual register, then polite/intimate/serious/whisper.
var v1ToV2Map = func() map[string]string {
	m := map[string]string{}
	for _, reg := range []string{"casual", "polite", "intimate", "serious", "whisper"} {
		for _, e := range defaultMapList {
			if strings.HasPrefix(e.k, reg+"_") {
				if _, taken := m[e.v]; !taken {
					m[e.v] = e.k
				}
			}
		}
	}
	return m
}()

func v1toV2(tag string) string {
	if v, ok := v1ToV2Map[tag]; ok {
		return v
	}
	return "casual_neutral"
}

/* ------------------------------------------------------------------ *
 * JSON plumbing. Bodies are handled as ordered raw objects where the
 * JS passed values through untouched; field order in objects we build
 * ourselves follows the JS literals.
 * ------------------------------------------------------------------ */

type rawObj = map[string]json.RawMessage

func marshalJSON(v interface{}) []byte {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	_ = enc.Encode(v)
	return bytes.TrimRight(buf.Bytes(), "\n")
}

func marshalIndent(v interface{}) []byte {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	_ = enc.Encode(v)
	return bytes.TrimRight(buf.Bytes(), "\n")
}

// rawString coerces a JSON value the way JS String(v || "") + lookups did:
// only true JSON strings yield their text; numbers/bools yield their
// literal; anything else yields "".
func rawString(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}
	switch string(raw) {
	case "null", "{}", "[]":
		return ""
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	var v interface{}
	if err := dec.Decode(&v); err == nil {
		switch t := v.(type) {
		case json.Number:
			return t.String()
		case bool:
			if t {
				return "true"
			}
			return "false"
		}
	}
	return ""
}

func rawStrField(obj rawObj, key string) (string, bool) {
	raw, ok := obj[key]
	if !ok {
		return "", false
	}
	var s string
	if err := json.Unmarshal(raw, &s); err != nil {
		return "", false
	}
	return s, true
}

func rawBoolField(obj rawObj, key string) (bool, bool) {
	raw, ok := obj[key]
	if !ok {
		return false, false
	}
	var b bool
	if err := json.Unmarshal(raw, &b); err != nil {
		return false, false
	}
	return b, true
}

func rawNumField(obj rawObj, key string) (float64, bool) {
	raw, ok := obj[key]
	if !ok {
		return 0, false
	}
	var f float64
	if err := json.Unmarshal(raw, &f); err != nil {
		return 0, false
	}
	return f, true
}

/* ------------------------------------------------------------------ *
 * HTTP helpers (mirror sendJSON / sendFile / readBody / readJSON)
 * ------------------------------------------------------------------ */

var mimeTypes = map[string]string{
	".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
	".css": "text/css; charset=utf-8", ".json": "application/json; charset=utf-8",
	".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
	".webp": "image/webp", ".gif": "image/gif", ".svg": "image/svg+xml",
	".moc3": "application/octet-stream", ".wav": "audio/wav", ".mp3": "audio/mpeg",
	".ogg": "audio/ogg", ".txt": "text/plain; charset=utf-8", ".md": "text/plain; charset=utf-8",
}

type respWriter struct {
	http.ResponseWriter
	sent bool
}

func (w *respWriter) WriteHeader(code int) {
	w.sent = true
	w.ResponseWriter.WriteHeader(code)
}

func (w *respWriter) Write(b []byte) (int, error) {
	w.sent = true
	return w.ResponseWriter.Write(b)
}

func sendJSON(w http.ResponseWriter, code int, v interface{}) {
	body := marshalJSON(v)
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.Header().Set("Content-Length", strconv.Itoa(len(body)))
	w.WriteHeader(code)
	_, _ = w.Write(body)
}

func sendDetail(w http.ResponseWriter, code int, detail string) {
	sendJSON(w, code, orderedJSON{{"detail", detail}})
}

// orderedJSON marshals as a JSON object with keys in slice order.
type orderedJSON [][2]interface{}

func (o orderedJSON) MarshalJSON() ([]byte, error) {
	var buf bytes.Buffer
	buf.WriteByte('{')
	for i, pair := range o {
		if i > 0 {
			buf.WriteByte(',')
		}
		kb, err := json.Marshal(pair[0].(string))
		if err != nil {
			return nil, err
		}
		buf.Write(kb)
		buf.WriteByte(':')
		vb := marshalJSON(pair[1])
		buf.Write(vb)
	}
	buf.WriteByte('}')
	return buf.Bytes(), nil
}

func sendFile(w http.ResponseWriter, file string) {
	st, err := os.Stat(file)
	if err != nil || !st.Mode().IsRegular() {
		sendDetail(w, 404, "Not found.")
		return
	}
	f, err := os.Open(file)
	if err != nil {
		sendDetail(w, 404, "Not found.")
		return
	}
	defer f.Close()
	ct, ok := mimeTypes[strings.ToLower(filepath.Ext(file))]
	if !ok {
		ct = "application/octet-stream"
	}
	w.Header().Set("Content-Type", ct)
	w.Header().Set("Content-Length", strconv.FormatInt(st.Size(), 10))
	w.WriteHeader(200)
	_, _ = io.Copy(w, f)
}

var errBodyTooLarge = fmt.Errorf("body too large")

func readBody(r *http.Request, capBytes int64) ([]byte, error) {
	defer r.Body.Close()
	var buf bytes.Buffer
	n, err := io.Copy(&buf, io.LimitReader(r.Body, capBytes+1))
	if err != nil {
		return nil, err
	}
	if n > capBytes {
		return nil, errBodyTooLarge
	}
	_ = n
	return buf.Bytes(), nil
}

func readJSONBody(r *http.Request) (rawObj, error) {
	buf, err := readBody(r, 30*1024*1024)
	if err != nil {
		return nil, err
	}
	if len(bytes.TrimSpace(buf)) == 0 {
		return rawObj{}, nil
	}
	var obj rawObj
	if err := json.Unmarshal(buf, &obj); err != nil {
		// JS JSON.parse accepts any JSON value; a non-object body just
		// yields no fields. Only a syntax error is a 500 there.
		var anyv interface{}
		if json.Unmarshal(buf, &anyv) == nil {
			return rawObj{}, nil
		}
		return nil, err
	}
	if obj == nil {
		obj = rawObj{}
	}
	return obj, nil
}

/* ------------------------------------------------------------------ *
 * Scheduler client
 * ------------------------------------------------------------------ */

var httpClient = &http.Client{}

type schedResp struct {
	status int
	body   rawObj // nil when the response was not a JSON object
}

func schedJSON(method, p string, body interface{}) (schedResp, error) {
	var rd io.Reader
	var data []byte
	if body != nil {
		data = marshalJSON(body)
		rd = bytes.NewReader(data)
	}
	req, err := http.NewRequest(method, schedBase+p, rd)
	if err != nil {
		return schedResp{}, err
	}
	if data != nil {
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Content-Length", strconv.Itoa(len(data)))
	}
	res, err := httpClient.Do(req)
	if err != nil {
		return schedResp{}, err
	}
	defer res.Body.Close()
	raw, _ := io.ReadAll(res.Body)
	out := schedResp{status: res.StatusCode}
	var obj rawObj
	if err := json.Unmarshal(raw, &obj); err == nil && obj != nil {
		out.body = obj
	}
	return out, nil
}

var (
	healthzMu    sync.Mutex
	healthzAt    time.Time
	healthzCache rawObj
)

func schedulerHealth() (rawObj, error) {
	healthzMu.Lock()
	defer healthzMu.Unlock()
	if healthzCache != nil && time.Since(healthzAt) < time.Second {
		return healthzCache, nil
	}
	r, err := schedJSON("GET", "/healthz", nil)
	if err != nil {
		return nil, err
	}
	if r.status != 200 || r.body == nil {
		return nil, fmt.Errorf("scheduler healthz failed")
	}
	healthzCache = r.body
	healthzAt = time.Now()
	return r.body, nil
}

/* ------------------------------------------------------------------ *
 * Adapter-local persisted state: persona identity + stage
 * ------------------------------------------------------------------ */

type personaT struct {
	UserName  string `json:"user_name"`
	City      string `json:"city"`
	Timezone  string `json:"timezone"`
	Onboarded bool   `json:"onboarded"`
	SayAs     string `json:"say_as"`
}

type stageT struct {
	Blur float64  `json:"blur"`
	Bg   *string  `json:"bg"`
}

var stateMu sync.Mutex
var personaState personaT
var stageState = stageT{Blur: 12}

func loadStateFile(name string, into interface{}) {
	raw, err := os.ReadFile(filepath.Join(dataDir, name))
	if err != nil {
		return
	}
	_ = json.Unmarshal(raw, into)
}

func saveStateFile(name string, v interface{}) {
	_ = os.WriteFile(filepath.Join(dataDir, name), marshalIndent(v), 0644)
}

func initState() {
	_ = os.MkdirAll(dataDir, 0755)
	loadStateFile("persona.json", &personaState)
	loadStateFile("stage.json", &stageState)
}

func main() {
	initState()
	srv := &http.Server{
		Addr:    fmt.Sprintf("0.0.0.0:%d", port),
		Handler: http.HandlerFunc(rootHandler),
	}
	logf("orig-ui adapter (go) on :%d (ui=%s patched=%s models=%s)", port, uiPristine, uiPatched, modelsDir)
	if err := srv.ListenAndServe(); err != nil {
		logf("listen failed: %v", err)
		os.Exit(1)
	}
}
