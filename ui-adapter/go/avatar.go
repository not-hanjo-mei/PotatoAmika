package main

import (
	"encoding/json"
	"math"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// Avatar models (port of the original speech/api.py behaviour, as
// implemented by adapter.js).

func walkFiles(dir string, exts []string) []string {
	out := []string{}
	var rec func(d string)
	rec = func(d string) {
		entries, err := os.ReadDir(d)
		if err != nil {
			return
		}
		for _, e := range entries {
			full := filepath.Join(d, e.Name())
			if e.IsDir() {
				rec(full)
			} else {
				lower := strings.ToLower(e.Name())
				for _, x := range exts {
					if strings.HasSuffix(lower, x) {
						rel, err := filepath.Rel(dir, full)
						if err == nil {
							out = append(out, filepath.ToSlash(rel))
						}
						break
					}
				}
			}
		}
	}
	rec(dir)
	sort.Strings(out)
	return out
}

func modelDir(id string) (string, bool) {
	d := filepath.Join(modelsDir, id)
	if path.Base(id) != id {
		return "", false
	}
	st, err := os.Stat(d)
	if err != nil || !st.IsDir() {
		return "", false
	}
	return d, true
}

// banksDoc returns {config, banks} with any missing emotion bank filled
// from the model's motion files, exactly like adapter.js (in memory
// only; GET never writes).
func banksDoc(id string) (rawObj, bool) {
	dir, ok := modelDir(id)
	if !ok {
		return nil, false
	}
	doc := rawObj{
		"config": json.RawMessage(`{}`),
		"banks":  json.RawMessage(`{}`),
	}
	if raw, err := os.ReadFile(filepath.Join(dir, "banks.json")); err == nil {
		var parsed rawObj
		if err := json.Unmarshal(raw, &parsed); err == nil && parsed != nil {
			doc = parsed
		}
	}
	getObj := func(key string) map[string]json.RawMessage {
		var m map[string]json.RawMessage
		if raw, ok := doc[key]; ok {
			_ = json.Unmarshal(raw, &m)
		}
		if m == nil {
			m = map[string]json.RawMessage{}
		}
		return m
	}
	config := getObj("config")
	banks := getObj("banks")
	stems := map[string]bool{}
	stemList := []string{}
	for _, rel := range walkFiles(dir, []string{".motion3.json"}) {
		stem := strings.TrimSuffix(path.Base(rel), ".motion3.json")
		if !stems[stem] {
			stems[stem] = true
			stemList = append(stemList, stem)
		}
	}
	_ = stemList
	fill := func(key string) json.RawMessage {
		names := []string{}
		for s := range stems {
			if s == key || strings.HasPrefix(s, key+".") {
				names = append(names, s)
			}
		}
		sort.Strings(names)
		movs := make([]interface{}, 0, len(names))
		for _, n := range names {
			movs = append(movs, orderedJSON{{"name", n}, {"likelihood", 100}})
		}
		return marshalJSON(orderedJSON{{"emotes", []interface{}{}}, {"movements", movs}})
	}
	for _, key := range emotions {
		if _, has := banks[key]; !has {
			banks[key] = fill(key)
		}
	}
	if _, has := banks["__idle__"]; !has {
		names := []string{}
		for s := range stems {
			if strings.HasPrefix(s, "idle") {
				names = append(names, s)
			}
		}
		sort.Strings(names)
		movs := make([]interface{}, 0, len(names))
		for _, n := range names {
			movs = append(movs, orderedJSON{{"name", n}, {"likelihood", 100}})
		}
		banks["__idle__"] = marshalJSON(orderedJSON{{"emotes", []interface{}{}}, {"movements", movs}})
	}
	doc["config"] = marshalJSON(config)
	doc["banks"] = marshalJSON(banks)
	return doc, true
}

func cfgRawString(cfg map[string]json.RawMessage, key string) (string, bool) {
	raw, ok := cfg[key]
	if !ok {
		return "", false
	}
	var s string
	if err := json.Unmarshal(raw, &s); err != nil {
		return "", false
	}
	return s, true
}

func avatarModelsList() []interface{} {
	models := []interface{}{}
	entries, err := os.ReadDir(modelsDir)
	if err != nil {
		return models
	}
	ids := []string{}
	for _, e := range entries {
		if e.IsDir() {
			ids = append(ids, e.Name())
		}
	}
	sort.Strings(ids)
	for _, id := range ids {
		dir, ok := modelDir(id)
		if !ok {
			continue
		}
		model3s := walkFiles(dir, []string{".model3.json"})
		if len(model3s) == 0 {
			continue
		}
		model3 := model3s[0]
		cfg := map[string]json.RawMessage{}
		if doc, ok := banksDoc(id); ok {
			_ = json.Unmarshal(doc["config"], &cfg)
		}
		name := id
		if v, ok := cfgRawString(cfg, "display_name"); ok && v != "" {
			name = v
		}
		strOr := func(key, dflt string) interface{} {
			if v, ok := cfgRawString(cfg, key); ok && v != "" {
				return v
			}
			return dflt
		}
		rawOrNull := func(key string) interface{} {
			if raw, ok := cfg[key]; ok {
				return raw
			}
			return json.RawMessage("null")
		}
		stage := orderedJSON{
			{"scale", rawOrNull("scale")},
			{"nx", rawOrNull("nx")},
			{"ny", rawOrNull("ny")},
			{"head_y", rawOrNull("head_y")},
			{"busy_expression", strOr("busy_expression", "")},
		}
		// busy_expression falls back to null in the JS (cfg.x || null).
		if _, ok := cfg["busy_expression"]; !ok {
			stage[4][1] = json.RawMessage("null")
		} else if v, _ := cfgRawString(cfg, "busy_expression"); v == "" {
			stage[4][1] = json.RawMessage("null")
		}
		models = append(models, orderedJSON{
			{"id", id},
			{"name", name},
			{"model3_url", "/avatar_models/" + id + "/" + model3},
			{"motions_url", "/api/avatar/models/" + id + "/motions"},
			{"banks_url", "/api/avatar/models/" + id + "/banks"},
			{"assets_prefix", "/avatar_models/" + id + "/"},
			{"mouth_open_param", strOr("mouth_open_param", "ParamMouthOpenY")},
			{"mouth_form_param", strOr("mouth_form_param", "ParamMouthForm")},
			{"role", strOr("role", "character")},
			{"stage", stage},
		})
	}
	return models
}

// posixRel mimics path.posix.relative for two slash-separated relative
// paths, as used by versionedModel3 in adapter.js.
func posixRel(from, to string) string {
	clean := func(p string) []string {
		parts := []string{}
		for _, seg := range strings.Split(p, "/") {
			switch seg {
			case "", ".":
			case "..":
				if len(parts) > 0 {
					parts = parts[:len(parts)-1]
				}
			default:
				parts = append(parts, seg)
			}
		}
		return parts
	}
	f, t := clean(from), clean(to)
	i := 0
	for i < len(f) && i < len(t) && f[i] == t[i] {
		i++
	}
	out := []string{}
	for j := i; j < len(f); j++ {
		out = append(out, "..")
	}
	out = append(out, t[i:]...)
	return strings.Join(out, "/")
}

func versionedModel3(id, rel string) (interface{}, error) {
	dir, _ := modelDir(id)
	file := filepath.Join(dir, filepath.FromSlash(rel))
	raw, err := os.ReadFile(file)
	if err != nil {
		return nil, err
	}
	var data rawObj
	if err := json.Unmarshal(raw, &data); err != nil || data == nil {
		return nil, errParse
	}
	refsRaw, has := data["FileReferences"]
	if !has {
		return data, nil
	}
	var refs rawObj
	if err := json.Unmarshal(refsRaw, &refs); err != nil || refs == nil {
		return data, nil
	}
	base := path.Dir(rel)
	stamp := func(r string) string {
		v := "0"
		if st, err := os.Stat(filepath.Join(dir, filepath.FromSlash(base), filepath.FromSlash(r))); err == nil {
			v = strconv.FormatInt(int64(math.Floor(float64(st.ModTime().UnixMilli())/1000.0)), 10)
		}
		return r + "?v=" + v
	}
	if rawArr, ok := refs["Textures"]; ok {
		var tex []string
		if err := json.Unmarshal(rawArr, &tex); err == nil && tex != nil {
			stamped := make([]string, len(tex))
			for i, t := range tex {
				stamped[i] = stamp(t)
			}
			refs["Textures"] = marshalJSON(stamped)
		}
	}
	for _, k := range []string{"Moc", "Physics", "DisplayInfo"} {
		if rawV, ok := refs[k]; ok {
			var s string
			if err := json.Unmarshal(rawV, &s); err == nil {
				refs[k] = marshalJSON(stamp(s))
			}
		}
	}
	nonEmpty := func(raw json.RawMessage) bool {
		if len(raw) == 0 {
			return false
		}
		var obj map[string]json.RawMessage
		if err := json.Unmarshal(raw, &obj); err != nil {
			return false
		}
		for _, v := range obj {
			var arr []interface{}
			if err := json.Unmarshal(v, &arr); err == nil {
				if len(arr) > 0 {
					return true
				}
				continue
			}
			var s string
			if err := json.Unmarshal(v, &s); err == nil {
				if s != "" {
					return true
				}
				continue
			}
			var b bool
			if err := json.Unmarshal(v, &b); err == nil {
				if b {
					return true
				}
				continue
			}
			var n float64
			if err := json.Unmarshal(v, &n); err == nil {
				if n != 0 {
					return true
				}
				continue
			}
			// objects count as truthy in JS
			return true
		}
		return false
	}
	if !nonEmpty(refs["Motions"]) {
		files := walkFiles(dir, []string{".motion3.json"})
		def := make([]interface{}, 0, len(files))
		for _, f := range files {
			rf := posixRel(base, f)
			if rf == "" {
				rf = f
			}
			def = append(def, orderedJSON{{"File", rf}, {"FadeInTime", 0.5}, {"FadeOutTime", 0.5}})
		}
		refs["Motions"] = marshalJSON(orderedJSON{{"Default", def}})
	}
	if !nonEmpty(refs["Expressions"]) {
		files := walkFiles(dir, []string{".exp3.json"})
		exprs := make([]interface{}, 0, len(files))
		for _, f := range files {
			rf := posixRel(base, f)
			if rf == "" {
				rf = f
			}
			exprs = append(exprs, orderedJSON{{"Name", strings.TrimSuffix(path.Base(f), ".exp3.json")}, {"File", rf}})
		}
		refs["Expressions"] = marshalJSON(exprs)
	}
	data["FileReferences"] = marshalJSON(refs)
	return data, nil
}

var errParse = &plainError{"parse error"}

type plainError struct{ s string }

func (e *plainError) Error() string { return e.s }
