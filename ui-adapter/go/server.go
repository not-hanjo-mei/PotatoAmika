package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"
)

/* ------------------------------------------------------------------ *
 * HTTP routing (transcribed from adapter.js, branch order preserved)
 * ------------------------------------------------------------------ */

var patchedFiles = map[string]bool{"use.html": true, "use.js": true, "common.js": true}

func notSupported(what string) orderedJSON {
	return orderedJSON{{"detail", "Not supported in PotatoAmika: " + what}}
}

var (
	reMotions = regexp.MustCompile(`^/api/avatar/models/([^/]+)/motions$`)
	reBanks   = regexp.MustCompile(`^/api/avatar/models/([^/]+)/banks$`)
	reAvFiles = regexp.MustCompile(`^/avatar_models/([^/]+)/(.+)$`)
)

type unsupportedRule struct {
	re   *regexp.Regexp
	what string
}

var unsupportedRules = []unsupportedRule{
	{regexp.MustCompile(`^/api/build_f/models$`), "switching brain providers from the page (the brain is the local Minico)"},
	{regexp.MustCompile(`^/api/build_f/openrouter_key$`), "OpenRouter accounts (all inference is local)"},
	{regexp.MustCompile(`^/api/build_f/ollama(/.*)?$`), "Ollama (the brain runs in llama.cpp)"},
	{regexp.MustCompile(`^/api/build_f/preset$`), "brain presets"},
	{regexp.MustCompile(`^/api/build_f/meter/reset$`), "a usage meter (local inference is not metered)"},
	{regexp.MustCompile(`^/api/build_f/persona/save$`), "editing persona files from the page"},
	{regexp.MustCompile(`^/api/build_f/persona/simple$`), "the simple persona editor"},
	{regexp.MustCompile(`^/api/build_f/messengers$`), "messenger bots (cut from PotatoAmika)"},
	{regexp.MustCompile(`^/api/build_f/tools(/.*)?$`), "the tool layer (disabled in v1)"},
	{regexp.MustCompile(`^/api/build_f/wipe$`), "memory wipe from the page"},
	{regexp.MustCompile(`^/api/build_f/reset$`), "full reset from the page"},
	{regexp.MustCompile(`^/api/build_f/iot(/.*)?$`), "IoT devices (not part of PotatoAmika)"},
	{regexp.MustCompile(`^/api/voice_map(/reset)?$`), "editing the voice map (it ships fixed)"},
}

func rootHandler(w http.ResponseWriter, r *http.Request) {
	rw := &respWriter{ResponseWriter: w}
	defer func() {
		if rec := recover(); rec != nil {
			logf("request error %s %v", r.URL.String(), rec)
			if !rw.sent {
				sendDetail(rw, 500, fmt.Sprint(rec))
			}
		}
	}()
	if isUpgrade(r) {
		// The raw ResponseWriter: the respWriter wrapper's method set does
		// not include http.Hijacker (it is not part of ResponseWriter), so
		// asserting on the wrapper would silently fail.
		handleUpgrade(w, r)
		return
	}
	p := r.URL.Path
	switch {
	case p == "/" || p == "/use":
		sendFile(rw, filepath.Join(uiPatched, "use.html"))
	case p == "/about":
		sendFile(rw, filepath.Join(uiPristine, "about.html"))
	case p == "/setup" || p == "/tools":
		sendDetail(rw, 404, "This page is not part of the PotatoAmika build (see REMOVALS.md).")
	case p == "/flow":
		sendFile(rw, filepath.Join(staticDir, "flow.html"))
	case strings.HasPrefix(p, "/ui/"):
		rel := p[len("/ui/"):]
		if strings.Contains(rel, "..") {
			sendDetail(rw, 404, "Not found.")
			return
		}
		if patchedFiles[rel] {
			sendFile(rw, filepath.Join(uiPatched, filepath.FromSlash(rel)))
		} else {
			sendFile(rw, filepath.Join(uiPristine, filepath.FromSlash(rel)))
		}
	case strings.HasPrefix(p, "/static/"):
		rel := p[len("/static/"):]
		if strings.Contains(rel, "..") {
			sendDetail(rw, 404, "Not found.")
			return
		}
		sendFile(rw, filepath.Join(staticDir, filepath.FromSlash(rel)))
	case strings.HasPrefix(p, "/api/") || strings.HasPrefix(p, "/avatar_models/") || p == "/health" || p == "/sfx_manifest":
		if err := handleAPI(rw, r, p); err != nil {
			logf("request error %s %v", r.URL.String(), err)
			if !rw.sent {
				sendDetail(rw, 500, err.Error())
			}
		}
	default:
		sendDetail(rw, 404, "Not found.")
	}
}

func isUpgrade(r *http.Request) bool {
	return strings.EqualFold(r.Header.Get("Upgrade"), "websocket") &&
		strings.Contains(strings.ToLower(r.Header.Get("Connection")), "upgrade")
}

func handleUpgrade(w http.ResponseWriter, r *http.Request) {
	key := r.Header.Get("Sec-WebSocket-Key")
	hj, ok := w.(http.Hijacker)
	if !ok {
		return
	}
	conn, rw, err := hj.Hijack()
	if err != nil {
		return
	}
	if r.URL.Path != "/ws/dialog" || key == "" {
		// /ws/camera and anything else: no such channel in this build.
		conn.Close()
		return
	}
	fmt.Fprintf(rw, "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: %s\r\n\r\n", wsAcceptKey(key))
	if err := rw.Flush(); err != nil {
		conn.Close()
		return
	}
	var wmu sync.Mutex
	writeFrame := func(frame []byte) {
		wmu.Lock()
		defer wmu.Unlock()
		_, _ = conn.Write(frame)
	}
	d := newDialog(writeFrame)
	parser := &wsParser{
		onMessage: func(s string) { d.onText(s) },
		onBinary:  func(b []byte) { d.onBinary(b) },
		onClose: func() {
			d.destroy()
			conn.Close()
		},
	}
	parser.onPing = func(pl []byte) { writeFrame(wsEncode(pl, 10, false)) }
	go func() {
		if err := d.init(); err != nil {
			logf("dialog init failed: %v", err)
			d.send(orderedJSON{{"type", "ai_error"}, {"message", "scheduler unreachable"}})
		}
	}()
	if n := rw.Reader.Buffered(); n > 0 {
		head := make([]byte, n)
		_, _ = rw.Reader.Read(head)
		parser.push(head)
	}
	buf := make([]byte, 32*1024)
	for {
		n, err := rw.Reader.Read(buf)
		if n > 0 {
			parser.push(buf[:n])
		}
		if err != nil {
			d.destroy()
			conn.Close()
			return
		}
	}
}

/* ------------------------------------------------------------------ */

func handleAPI(w *respWriter, r *http.Request, p string) error {
	method := r.Method

	/* ---- avatar ---- */
	if p == "/api/avatar/models" && method == "GET" {
		sendJSON(w, 200, orderedJSON{{"ok", true}, {"models", avatarModelsList()}, {"emotions", emotions}})
		return nil
	}
	if m := reMotions.FindStringSubmatch(p); m != nil && method == "GET" {
		id := m[1]
		dir, ok := modelDir(id)
		if !ok {
			sendDetail(w, 404, "Unknown model.")
			return nil
		}
		out := []interface{}{}
		for _, rel := range walkFiles(dir, []string{".motion3.json", ".exp3.json"}) {
			isMotion := strings.HasSuffix(rel, ".motion3.json")
			typ, suffix := "expression", ".exp3.json"
			if isMotion {
				typ, suffix = "motion", ".motion3.json"
			}
			base := rel[strings.LastIndex(rel, "/")+1:]
			out = append(out, orderedJSON{
				{"type", typ},
				{"name", strings.TrimSuffix(base, suffix)},
				{"file", base},
				{"path", "/avatar_models/" + id + "/" + rel},
			})
		}
		sendJSON(w, 200, out)
		return nil
	}
	if m := reBanks.FindStringSubmatch(p); m != nil {
		id := m[1]
		if method == "GET" {
			doc, ok := banksDoc(id)
			if !ok {
				sendDetail(w, 404, "Unknown model.")
				return nil
			}
			sendJSON(w, 200, doc)
			return nil
		}
		if method == "POST" {
			dir, ok := modelDir(id)
			if !ok {
				sendDetail(w, 404, "Unknown model.")
				return nil
			}
			body, err := readJSONBody(r)
			if err != nil {
				return err
			}
			banksRaw, has := body["banks"]
			var banksIn map[string]json.RawMessage
			if has {
				_ = json.Unmarshal(banksRaw, &banksIn)
			}
			if banksIn == nil {
				sendDetail(w, 400, "Body must carry a banks object.")
				return nil
			}
			cur, _ := banksDoc(id)
			var curCfg map[string]json.RawMessage
			if cur != nil {
				_ = json.Unmarshal(cur["config"], &curCfg)
			}
			if curCfg == nil {
				curCfg = map[string]json.RawMessage{}
			}
			clean := rawObj{}
			for key, bankRaw := range banksIn {
				var bank rawObj
				if json.Unmarshal(bankRaw, &bank) != nil || bank == nil {
					continue
				}
				emotes := []interface{}{}
				if rawArr, ok := bank["emotes"]; ok {
					var arr []json.RawMessage
					if json.Unmarshal(rawArr, &arr) == nil {
						for _, item := range arr {
							var s string
							if json.Unmarshal(item, &s) == nil {
								emotes = append(emotes, s)
							}
						}
					}
				}
				movs := []interface{}{}
				if rawArr, ok := bank["movements"]; ok {
					var arr []json.RawMessage
					if json.Unmarshal(rawArr, &arr) == nil {
						for _, item := range arr {
							var mv rawObj
							if json.Unmarshal(item, &mv) != nil || mv == nil {
								continue
							}
							name, ok := rawStrField(mv, "name")
							if !ok {
								continue
							}
							lh := 0
							if lv, ok := mv["likelihood"]; ok {
								lh = jsParseInt(lv)
							}
							if lh < 0 {
								lh = 0
							}
							if lh > 100 {
								lh = 100
							}
							movs = append(movs, orderedJSON{{"name", name}, {"likelihood", lh}})
						}
					}
				}
				clean[key] = marshalJSON(orderedJSON{{"emotes", emotes}, {"movements", movs}})
			}
			cfg := curCfg
			if cfgRaw, ok := body["config"]; ok {
				var bodyCfg rawObj
				if json.Unmarshal(cfgRaw, &bodyCfg) == nil && bodyCfg != nil {
					for _, k := range []string{"display_name", "mouth_open_param", "mouth_form_param", "scale", "nx", "ny", "head_y", "busy_expression"} {
						if v, has := bodyCfg[k]; has {
							cfg[k] = v
						}
					}
				}
			}
			doc := rawObj{"config": marshalJSON(cfg), "banks": marshalJSON(clean)}
			if err := os.WriteFile(filepath.Join(dir, "banks.json"), marshalIndent(doc), 0644); err != nil {
				return err
			}
			sendJSON(w, 200, orderedJSON{{"ok", true}, {"banks", doc}})
			return nil
		}
	}

	/* ---- avatar model files (with the original's model3 transform) ---- */
	if m := reAvFiles.FindStringSubmatch(p); m != nil && method == "GET" {
		id, rel := m[1], m[2]
		dir, ok := modelDir(id)
		if !ok || strings.Contains(rel, "..") {
			sendDetail(w, 404, "Not found.")
			return nil
		}
		file := filepath.Join(dir, filepath.FromSlash(rel))
		if !strings.HasPrefix(file, dir) {
			sendDetail(w, 404, "Not found.")
			return nil
		}
		if _, err := os.Stat(file); err != nil {
			sendDetail(w, 404, "Not found.")
			return nil
		}
		if strings.HasSuffix(rel, ".model3.json") {
			if data, err := versionedModel3(id, rel); err == nil {
				sendJSON(w, 200, data)
				return nil
			}
			sendFile(w, file)
			return nil
		}
		sendFile(w, file)
		return nil
	}

	/* ---- stage ---- */
	if p == "/api/stage" && method == "GET" {
		stateMu.Lock()
		bg := stageState.Bg
		blur := stageState.Blur
		stateMu.Unlock()
		var bgVal interface{}
		if bg != nil {
			if st, err := os.Stat(filepath.Join(dataDir, *bg)); err == nil {
				bgVal = "/api/stage/background?v=" + strconv.FormatInt(st.ModTime().UnixMilli()/1000, 10)
			}
		}
		sendJSON(w, 200, orderedJSON{{"ok", true}, {"background", bgVal}, {"blur", blur}})
		return nil
	}
	if p == "/api/stage" && method == "POST" {
		body, err := readJSONBody(r)
		if err != nil {
			return err
		}
		stateMu.Lock()
		if raw, ok := body["clear_background"]; ok && jsTruthy(raw) {
			if stageState.Bg != nil {
				_ = os.Remove(filepath.Join(dataDir, *stageState.Bg))
			}
			stageState.Bg = nil
		}
		if v, ok := rawNumField(body, "blur"); ok {
			if v < 0 {
				v = 0
			}
			if v > 40 {
				v = 40
			}
			stageState.Blur = v
		}
		saveStateFile("stage.json", stageState)
		var bgVal interface{}
		if stageState.Bg != nil {
			bgVal = "/api/stage/background"
		}
		blur := stageState.Blur
		stateMu.Unlock()
		sendJSON(w, 200, orderedJSON{{"ok", true}, {"blur", blur}, {"background", bgVal}})
		return nil
	}
	if p == "/api/stage/background" && method == "GET" {
		stateMu.Lock()
		bg := stageState.Bg
		stateMu.Unlock()
		if bg == nil {
			sendDetail(w, 404, "No background set.")
			return nil
		}
		sendFile(w, filepath.Join(dataDir, *bg))
		return nil
	}
	if p == "/api/stage/background" && method == "POST" {
		return stageBackgroundUpload(w, r)
	}

	/* ---- health / status / probe ---- */
	if p == "/health" && method == "GET" {
		sendJSON(w, 200, orderedJSON{
			{"status", "ok"}, {"app", "PotatoAmika"},
			{"model", "Minico-M1-1105 (llama.cpp, local)"}, {"device", "cpu"},
			{"cuda_available", false}, {"gpu_name", nil},
		})
		return nil
	}
	if p == "/api/build_f/status" && method == "GET" {
		hz, err := schedulerHealth()
		if err != nil {
			sendDetail(w, 502, "Scheduler unreachable.")
			return nil
		}
		events := json.RawMessage("0")
		if raw, ok := hz["memory_n"]; ok {
			events = raw
		}
		sendJSON(w, 200, orderedJSON{
			{"ok", true}, {"app", "PotatoAmika scheduler"}, {"template_check", orderedJSON{}},
			{"memory", orderedJSON{{"events", events}, {"degraded", false}}},
		})
		return nil
	}
	if p == "/api/tts_probe" && method == "GET" {
		hz, err := schedulerHealth()
		if err != nil {
			sendJSON(w, 200, orderedJSON{
				{"ok", false}, {"url", "scheduler tts_host (local)"},
				{"character", "MekaHime_Amika"}, {"error", "scheduler unreachable"},
			})
			return nil
		}
		tts, _ := rawBoolField(hz, "tts")
		status := 503
		if tts {
			status = 200
		}
		sendJSON(w, 200, orderedJSON{
			{"ok", tts}, {"url", "scheduler tts_host (local)"},
			{"character", "MekaHime_Amika"}, {"status", status},
		})
		return nil
	}

	/* ---- tts_say (first-run wizard + name preview) ---- */
	if p == "/api/tts_say" && method == "POST" {
		return ttsSay(w, r)
	}

	/* ---- build_f: models / persona / pronunciation ---- */
	if p == "/api/build_f/models" && method == "GET" {
		opt := orderedJSON{
			{"key", "local-minico"}, {"label", "Minico-M1-1105 · llama.cpp (local)"},
			{"transport", "llama.cpp"}, {"model", "Minico-M1-1105-Q4_K_M"},
		}
		roles := []string{"surface", "deep", "reflection", "memory_write"}
		options := orderedJSON{}
		selections := orderedJSON{}
		for _, role := range roles {
			options = append(options, [2]interface{}{role, []interface{}{opt}})
			selections = append(selections, [2]interface{}{role, "local-minico"})
		}
		sendJSON(w, 200, orderedJSON{{"options", options}, {"selections", selections}})
		return nil
	}
	if p == "/api/build_f/persona" && method == "GET" {
		stateMu.Lock()
		ps := personaState
		stateMu.Unlock()
		sendJSON(w, 200, orderedJSON{
			{"ok", true}, {"onboarded", ps.Onboarded}, {"persona_name", "Amika"},
			{"user_name", ps.UserName}, {"city", ps.City}, {"timezone", ps.Timezone},
			{"timezones", ianaZones},
		})
		return nil
	}
	if p == "/api/build_f/persona/identity" && method == "POST" {
		body, err := readJSONBody(r)
		if err != nil {
			return err
		}
		stateMu.Lock()
		if v, ok := rawStrField(body, "user_name"); ok {
			personaState.UserName = v
		}
		if v, ok := rawStrField(body, "city"); ok {
			personaState.City = v
		}
		if v, ok := rawStrField(body, "timezone"); ok {
			personaState.Timezone = v
		}
		if v, ok := rawBoolField(body, "onboarded"); ok {
			personaState.Onboarded = v
		}
		saveStateFile("persona.json", personaState)
		stateMu.Unlock()
		sendJSON(w, 200, orderedJSON{{"ok", true}})
		return nil
	}
	if p == "/api/name_pronunciation" && method == "GET" {
		stateMu.Lock()
		ps := personaState
		stateMu.Unlock()
		sendJSON(w, 200, orderedJSON{{"ok", true}, {"name", ps.UserName}, {"say_as", ps.SayAs}})
		return nil
	}
	if p == "/api/name_pronunciation" && method == "POST" {
		body, err := readJSONBody(r)
		if err != nil {
			return err
		}
		stateMu.Lock()
		if v, ok := rawStrField(body, "say_as"); ok {
			personaState.SayAs = v
		}
		if v, ok := rawStrField(body, "name"); ok {
			personaState.UserName = v
		}
		saveStateFile("persona.json", personaState)
		stateMu.Unlock()
		sendJSON(w, 200, orderedJSON{{"ok", true}})
		return nil
	}
	if p == "/api/voice_map" && method == "GET" {
		pairs := make(orderedJSON, len(defaultMapList))
		for i, e := range defaultMapList {
			pairs[i] = [2]interface{}{e.k, e.v}
		}
		sendJSON(w, 200, orderedJSON{
			{"ok", true}, {"enabled", true}, {"voice", "MekaHime_Amika"}, {"map", pairs},
		})
		return nil
	}

	/* ---- shaped, truthful empties ---- */
	if p == "/api/preflight" && method == "GET" {
		sendJSON(w, 200, orderedJSON{
			{"at", time.Now().UTC().Format("2006-01-02T15:04:05.000Z")},
			{"checks", []interface{}{}}, {"summary", orderedJSON{}},
		})
		return nil
	}
	if p == "/api/build_f/messengers" && method == "GET" {
		sendJSON(w, 200, orderedJSON{{"ok", true}, {"enabled", false}, {"messengers", []interface{}{}}})
		return nil
	}
	if p == "/api/build_f/tools" && method == "GET" {
		sendJSON(w, 200, orderedJSON{
			{"ok", true}, {"enabled", false}, {"servers", orderedJSON{}},
			{"dialogue", []interface{}{}}, {"jobs", []interface{}{}}, {"pending", []interface{}{}},
		})
		return nil
	}
	if p == "/api/build_f/openrouter_key" && method == "GET" {
		sendJSON(w, 200, orderedJSON{{"ok", true}, {"present", false}, {"hint", ""}})
		return nil
	}
	if p == "/api/build_f/ollama" && method == "GET" {
		sendJSON(w, 200, orderedJSON{
			{"ok", true}, {"reachable", false}, {"base_url", ""}, {"live_version", nil},
			{"models", []interface{}{}}, {"in_use", orderedJSON{}},
		})
		return nil
	}
	if p == "/api/vision" && method == "GET" {
		sendJSON(w, 200, orderedJSON{
			{"ok", true}, {"enabled", false}, {"url", nil}, {"service", nil},
			{"description", nil}, {"last_description", nil}, {"description_age_s", nil},
			{"camera_clients", 0}, {"subscribers", 0}, {"her_line", nil}, {"language", "en"},
		})
		return nil
	}
	if p == "/sfx_manifest" && method == "GET" {
		sendJSON(w, 200, orderedJSON{{"dir", "/static/sfx/"}, {"files", []interface{}{}}})
		return nil
	}

	/* ---- explicit unsupported ---- */
	for _, rule := range unsupportedRules {
		if rule.re.MatchString(p) {
			sendJSON(w, 501, notSupported(rule.what))
			return nil
		}
	}
	sendDetail(w, 404, "Not found.")
	return nil
}

// jsParseInt mirrors JS parseInt(x, 10) for JSON values (numbers are
// stringified first, so 50.7 -> 50; non-numeric -> NaN -> 0 via || 0).
func jsParseInt(raw json.RawMessage) int {
	s := strings.TrimSpace(string(raw))
	if len(s) > 0 && s[0] == '"' {
		var str string
		if json.Unmarshal(raw, &str) == nil {
			s = str
		}
	}
	i := 0
	if i < len(s) && (s[i] == '+' || s[i] == '-') {
		i++
	}
	start := i
	for i < len(s) && s[i] >= '0' && s[i] <= '9' {
		i++
	}
	if i == start {
		return 0
	}
	n, err := strconv.Atoi(s[:i])
	if err != nil {
		return 0
	}
	return n
}

func jsTruthy(raw json.RawMessage) bool {
	s := strings.TrimSpace(string(raw))
	switch s {
	case "", "null", "false", "0", `""`:
		return false
	}
	return true
}

var reBoundary = regexp.MustCompile(`boundary=(.+)$`)

func stageBackgroundUpload(w *respWriter, r *http.Request) error {
	ct := r.Header.Get("Content-Type")
	bm := reBoundary.FindStringSubmatch(ct)
	if bm == nil {
		sendDetail(w, 400, "Expected multipart form data.")
		return nil
	}
	buf, err := readBody(r, 26*1024*1024)
	if err != nil {
		return err
	}
	boundary := []byte("--" + strings.TrimSpace(bm[1]))
	saved := ""
	pos := 0
	for {
		idx := bytes.Index(buf[pos:], boundary)
		if idx < 0 {
			break
		}
		pos += idx
		headEnd := bytes.Index(buf[pos:], []byte("\r\n\r\n"))
		if headEnd < 0 {
			break
		}
		headEnd += pos
		head := string(buf[pos:headEnd])
		nidx := bytes.Index(buf[headEnd+4:], boundary)
		if nidx < 0 {
			break
		}
		next := headEnd + 4 + nidx
		dataEnd := next - 2
		if fm := regexp.MustCompile(`filename="([^"]*)"`).FindStringSubmatch(head); fm != nil && fm[1] != "" {
			ext := strings.ToLower(filepath.Ext(fm[1]))
			if ext == "" {
				ext = ".png"
			}
			switch ext {
			case ".png", ".jpg", ".jpeg", ".webp", ".gif", ".avif":
			default:
				sendDetail(w, 400, "Unsupported image type.")
				return nil
			}
			name := "stage-bg" + ext
			start := headEnd + 4
			if dataEnd < start {
				dataEnd = start
			}
			if err := os.WriteFile(filepath.Join(dataDir, name), buf[start:dataEnd], 0644); err != nil {
				return err
			}
			stateMu.Lock()
			stageState.Bg = &name
			saveStateFile("stage.json", stageState)
			stateMu.Unlock()
			saved = name
			break
		}
		pos = next
	}
	if saved == "" {
		sendDetail(w, 400, "No image file in upload.")
		return nil
	}
	sendJSON(w, 200, orderedJSON{{"ok", true}, {"background", "/api/stage/background"}})
	return nil
}

/* tts_say with the streamed keepalive: the public proxy drops a POST
 * whose first byte takes longer than ~20 s, and a long wizard line's
 * synthesis can exceed that. The JSON head goes out at once and a space
 * (insignificant whitespace between JSON tokens) follows every 5 s
 * while synthesis runs; the page's response.json() parses the assembled
 * body exactly as before. */
func ttsSay(w *respWriter, r *http.Request) error {
	body, err := readJSONBody(r)
	if err != nil {
		return err
	}
	text := strings.TrimSpace(rawString(body["text"]))
	if text == "" {
		sendDetail(w, 400, "Empty text.")
		return nil
	}
	tag := "neutral1"
	voice, _ := rawStrField(body, "voice")
	emotion, _ := rawStrField(body, "emotion")
	if v1IDs[voice] {
		tag = voice
	} else if v, ok := defaultMap[emotion]; ok {
		tag = v
	}
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(200)
	flusher, _ := w.ResponseWriter.(http.Flusher)
	write := func(s string) bool {
		if _, err := w.Write([]byte(s)); err != nil {
			return false
		}
		if flusher != nil {
			flusher.Flush()
		}
		return true
	}
	write(`{"ok":`)
	type outcome struct {
		payload *sayPayload
		err     error
	}
	done := make(chan outcome, 1)
	go func() {
		p, e := sayCached(text, tag, emotion)
		done <- outcome{p, e}
	}()
	gone := false
	ctxDone := r.Context().Done()
	tick := time.NewTicker(5 * time.Second)
	defer tick.Stop()
	for {
		select {
		case out := <-done:
			if !gone {
				if out.err != nil {
					msg, _ := json.Marshal(out.err.Error())
					write("false,\"error\":" + string(msg) + "}")
				} else {
					rest := marshalJSON(out.payload)
					if len(rest) > 0 && rest[0] == '{' {
						rest = rest[1:]
					}
					write("true," + string(rest))
				}
			}
			return nil
		case <-tick.C:
			if !gone {
				if !write(" ") {
					gone = true
				}
			}
		case <-ctxDone:
			gone = true
			ctxDone = nil
		}
	}
}
