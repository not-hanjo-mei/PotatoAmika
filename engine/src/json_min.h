// Minimal JSON parser: enough for manifest.json and per-line request objects.
// Supports objects, arrays, strings (with \uXXXX BMP escapes), numbers, bool, null.
#pragma once
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

struct Json {
    enum Type { Null, Bool, Num, Str, Arr, Obj } type = Null;
    bool b = false;
    double num = 0;
    std::string str;
    std::vector<Json> arr;
    std::map<std::string, Json> obj;
    const Json* find(const std::string& key) const {
        if (type != Obj) return nullptr;
        auto it = obj.find(key);
        return it == obj.end() ? nullptr : &it->second;
    }
    std::string as_str(const std::string& dflt = "") const {
        return type == Str ? str : dflt;
    }
    double as_num(double dflt = 0) const { return type == Num ? num : dflt; }
};

class JsonParser {
  public:
    explicit JsonParser(const std::string& s) : s_(s) {}
    Json parse() {
        skip();
        Json v = value();
        return v;
    }

  private:
    const std::string& s_;
    size_t p_ = 0;
    void skip() {
        while (p_ < s_.size() && (s_[p_] == ' ' || s_[p_] == '\t' || s_[p_] == '\n' || s_[p_] == '\r')) p_++;
    }
    char peek() { return p_ < s_.size() ? s_[p_] : '\0'; }
    Json value() {
        skip();
        char c = peek();
        if (c == '{') return object();
        if (c == '[') return array();
        if (c == '"') { Json j; j.type = Json::Str; j.str = string(); return j; }
        if (c == 't') { expect("true"); Json j; j.type = Json::Bool; j.b = true; return j; }
        if (c == 'f') { expect("false"); Json j; j.type = Json::Bool; j.b = false; return j; }
        if (c == 'n') { expect("null"); return Json(); }
        return number();
    }
    void expect(const char* lit) { p_ += strlen(lit); }
    Json object() {
        Json j; j.type = Json::Obj;
        p_++; skip();
        if (peek() == '}') { p_++; return j; }
        while (true) {
            skip();
            std::string key = string();
            skip();
            if (peek() == ':') p_++;
            j.obj[key] = value();
            skip();
            if (peek() == ',') { p_++; continue; }
            if (peek() == '}') p_++;
            break;
        }
        return j;
    }
    Json array() {
        Json j; j.type = Json::Arr;
        p_++; skip();
        if (peek() == ']') { p_++; return j; }
        while (true) {
            j.arr.push_back(value());
            skip();
            if (peek() == ',') { p_++; continue; }
            if (peek() == ']') p_++;
            break;
        }
        return j;
    }
    static void utf8_encode(std::string& out, unsigned cp) {
        if (cp < 0x80) out += (char)cp;
        else if (cp < 0x800) { out += (char)(0xC0 | (cp >> 6)); out += (char)(0x80 | (cp & 0x3F)); }
        else { out += (char)(0xE0 | (cp >> 12)); out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F)); }
    }
    std::string string() {
        std::string out;
        if (peek() == '"') p_++;
        while (p_ < s_.size()) {
            char c = s_[p_++];
            if (c == '"') break;
            if (c == '\\' && p_ < s_.size()) {
                char e = s_[p_++];
                switch (e) {
                    case 'n': out += '\n'; break;
                    case 't': out += '\t'; break;
                    case 'r': out += '\r'; break;
                    case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break;
                    case '/': out += '/'; break;
                    case '\\': out += '\\'; break;
                    case '"': out += '"'; break;
                    case 'u': {
                        unsigned cp = 0;
                        for (int i = 0; i < 4 && p_ < s_.size(); i++)
                            cp = cp * 16 + (unsigned)std::stoi(s_.substr(p_++, 1), nullptr, 16);
                        if (cp >= 0xD800 && cp <= 0xDBFF && p_ + 1 < s_.size() && s_[p_] == '\\' && s_[p_ + 1] == 'u') {
                            p_ += 2;
                            unsigned lo = 0;
                            for (int i = 0; i < 4 && p_ < s_.size(); i++)
                                lo = lo * 16 + (unsigned)std::stoi(s_.substr(p_++, 1), nullptr, 16);
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            // 4-byte UTF-8
                            out += (char)(0xF0 | (cp >> 18)); out += (char)(0x80 | ((cp >> 12) & 0x3F));
                            out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F));
                        } else utf8_encode(out, cp);
                        break;
                    }
                    default: out += e; break;
                }
            } else out += c;
        }
        return out;
    }
    Json number() {
        size_t start = p_;
        while (p_ < s_.size() && (isdigit((unsigned char)s_[p_]) || strchr("+-.eE", s_[p_]))) p_++;
        Json j; j.type = Json::Num;
        j.num = std::stod(s_.substr(start, p_ - start));
        return j;
    }
};

inline std::string json_escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\t': out += "\\t"; break;
            case '\r': out += "\\r"; break;
            default: out += c;
        }
    }
    return out;
}
