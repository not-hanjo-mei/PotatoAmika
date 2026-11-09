#include "g2p.h"
#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <functional>
#include <sstream>

static std::string lower_of(std::string s) {
    for (auto& c : s) c = (char)tolower((unsigned char)c);
    return s;
}
static bool has_alpha(const std::string& s) {
    for (char c : s) if (isalpha((unsigned char)c)) return true;
    return false;
}
static std::vector<std::string> split_ws(const std::string& s) {
    std::vector<std::string> out;
    std::istringstream is(s);
    std::string w;
    while (is >> w) out.push_back(w);
    return out;
}

// ---- number words (mirrors Genie _number_to_words_custom: space-joined, no "and") ----
static std::string cardinal_ll(long long n) {
    static const char* units[] = {"", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"};
    static const char* teens[] = {"ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"};
    static const char* tens[] = {"", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"};
    static const char* thou[] = {"", "thousand", "million", "billion", "trillion"};
    if (n == 0) return "zero";
    std::function<std::string(long long)> lt1000 = [&](long long v) -> std::string {
        if (v == 0) return "";
        if (v < 10) return units[v];
        if (v < 20) return teens[v - 10];
        if (v < 100) { std::string r = tens[v / 10]; if (v % 10) r += std::string(" ") + units[v % 10]; return r; }
        std::string r = std::string(units[v / 100]) + " hundred";
        if (v % 100) r += " " + lt1000(v % 100);
        return r;
    };
    std::vector<std::string> parts;
    int i = 0;
    while (n > 0) {
        if (n % 1000) { std::string p = lt1000(n % 1000); if (i) p += std::string(" ") + thou[i]; parts.insert(parts.begin(), p); }
        n /= 1000; i++;
    }
    std::string out;
    for (size_t k = 0; k < parts.size(); k++) { if (k) out += " "; out += parts[k]; }
    return out;
}

static std::string ordinal_word(const std::string& cardinal) {
    // Replace the final word of a cardinal phrase with its ordinal form.
    auto pos = cardinal.find_last_of(' ');
    std::string head = pos == std::string::npos ? "" : cardinal.substr(0, pos + 1);
    std::string last = pos == std::string::npos ? cardinal : cardinal.substr(pos + 1);
    static const std::map<std::string, std::string> irr = {
        {"one", "first"}, {"two", "second"}, {"three", "third"}, {"five", "fifth"},
        {"eight", "eighth"}, {"nine", "ninth"}, {"twelve", "twelfth"}};
    auto it = irr.find(last);
    if (it != irr.end()) return head + it->second;
    if (!last.empty() && last.back() == 'y') return head + last.substr(0, last.size() - 1) + "ieth";
    return head + last + "th";
}

// Text-level expansions on the raw string (subset of Genie Normalization):
// rep-map punctuation, abbreviations dr/mr, times, decimals, ordinals, cardinals.
static std::string text_expand(const std::string& in) {
    std::string s = in;
    // REP_MAP (byte-level; '。' is the UTF-8 sequence E3 80 82)
    auto repl_all = [&](const std::string& from, const std::string& to) {
        size_t pos = 0;
        while ((pos = s.find(from, pos)) != std::string::npos) { s.replace(pos, from.size(), to); pos += to.size(); }
    };
    repl_all("\xE3\x80\x82", ".");   // 。
    repl_all("\xEF\xBC\x81", "!");   // ！
    repl_all("\xEF\xBC\x9F", "?");   // ？
    repl_all("\xEF\xBC\x9B", ",");   // ；
    repl_all("\xEF\xBC\x8C", ",");   // ，
    repl_all(";", ",");
    repl_all(":", ",");              // Genie REP_MAP maps [;:：，；] -> ',' ; ':' alone is not in the map but ':' as a
                                     // standalone token is dropped downstream anyway; keep text-level simple:
                                     // (reverted below if it was part of a time — times are expanded first in Genie;
                                     //  we expand times BEFORE this in the caller order — see normalize())
    repl_all("\xE2\x80\x9D", "'");   // ”
    repl_all("\xE2\x80\x99", "'");   // ’
    repl_all("\"", "'");
    return s;
}

// Expand digit sequences in text: times H:MM, decimals, ordinals, cardinals.
// Runs BEFORE punctuation rep-map so ':' in times survives until consumed here.
static std::string expand_numbers(const std::string& in) {
    std::string out;
    size_t i = 0, n = in.size();
    auto isdig = [&](size_t k) { return k < n && isdigit((unsigned char)in[k]); };
    while (i < n) {
        if (isdig(i)) {
            size_t j = i;
            while (isdig(j)) j++;
            std::string digits = in.substr(i, j - i);
            // time H:MM
            if (in[j] == ':' && j + 2 < n + 1 && isdig(j + 1) && isdig(j + 2) && !(j + 3 < n && isdigit((unsigned char)in[j + 3]))) {
                long long hh = std::stoll(digits), mm = std::stoll(in.substr(j + 1, 2));
                out += cardinal_ll(hh);
                if (mm == 0) out += " o'clock";
                else if (mm < 10) out += " oh " + cardinal_ll(mm);
                else out += " " + cardinal_ll(mm);
                i = j + 3;
                continue;
            }
            // decimal
            if (in[j] == '.' && isdig(j + 1)) {
                size_t k = j + 1;
                while (isdig(k)) k++;
                out += cardinal_ll(std::stoll(digits)) + " point";
                for (size_t d = j + 1; d < k; d++) out += " " + cardinal_ll(in[d] - '0');
                i = k;
                continue;
            }
            // ordinal suffix
            if ((in.substr(j, 2) == "st" || in.substr(j, 2) == "nd" || in.substr(j, 2) == "rd" || in.substr(j, 2) == "th") &&
                !(j + 2 < n && isalpha((unsigned char)in[j + 2]))) {
                out += ordinal_word(cardinal_ll(std::stoll(digits)));
                i = j + 2;
                continue;
            }
            // plain cardinal (strip commas inside digit runs)
            out += cardinal_ll(std::stoll(digits));
            i = j;
            continue;
        }
        if (in[i] == ',' && i > 0 && isdig(i - 1) && isdig(i + 1)) { i++; continue; }  // 1,000 -> 1000 handled digit-run-wise approx
        out += in[i++];
    }
    return out;
}

static std::string expand_abbrev(const std::string& in) {
    // Small verified subset (probed against Genie): Dr. -> doctor, Mr. -> mister.
    std::string s = in;
    auto rep_word = [&](const std::string& abbr, const std::string& full) {
        size_t pos = 0;
        while (true) {
            pos = s.find(abbr, pos);
            if (pos == std::string::npos) break;
            bool left_ok = pos == 0 || !(isalpha((unsigned char)s[pos - 1]));
            size_t after = pos + abbr.size();
            bool right_ok = after >= s.size() || !(isalpha((unsigned char)s[after]));
            if (left_ok && right_ok) { s.replace(pos, abbr.size(), full); pos += full.size(); }
            else pos = after;
        }
    };
    rep_word("Dr.", "doctor"); rep_word("dr.", "doctor");
    rep_word("Mr.", "mister"); rep_word("mr.", "mister");
    rep_word("Mrs.", "misses"); rep_word("mrs.", "misses");
    return s;
}

std::vector<std::string> G2P::tokenize(const std::string& text) const {
    std::vector<std::string> toks;
    size_t i = 0, n = text.size();
    auto is_word_char = [&](char c) { return isalnum((unsigned char)c); };
    while (i < n) {
        unsigned char c = (unsigned char)text[i];
        if (isspace(c)) { i++; continue; }
        if (isalpha(c) || isdigit(c)) {
            size_t j = i;
            while (j < n) {
                char cj = text[j];
                if (is_word_char(cj)) { j++; continue; }
                if ((cj == '\'' || cj == '-') && j + 1 < n && isalpha((unsigned char)text[j + 1]) && isalpha((unsigned char)text[j - 1])) { j++; continue; }
                break;
            }
            toks.push_back(text.substr(i, j - i));
            i = j;
            continue;
        }
        if (c >= 0x80) {  // one UTF-8 character as a punctuation token
            size_t len = (c >= 0xF0) ? 4 : (c >= 0xE0) ? 3 : 2;
            toks.push_back(text.substr(i, len));
            i += len;
            continue;
        }
        toks.push_back(text.substr(i, 1));
        i++;
    }
    return toks;
}

std::vector<std::string> G2P::lts_fallback(const std::string& w) const {
    std::vector<std::string> out;
    std::string s = lower_of(w);
    if (!s.empty() && s.back() == 'e' && s.size() > 2 && !strchr("aeiou", s[s.size() - 2])) s = s.substr(0, s.size() - 1);  // silent e
    static const std::vector<std::pair<std::string, std::vector<std::string>>> dig = {
        {"tion", {"SH", "AH0", "N"}}, {"sion", {"ZH", "AH0", "N"}}, {"ch", {"CH"}}, {"sh", {"SH"}},
        {"th", {"TH"}}, {"ph", {"F"}}, {"wh", {"W"}}, {"ck", {"K"}}, {"ng", {"NG"}},
        {"qu", {"K", "W"}}, {"ee", {"IY0"}}, {"oo", {"UW0"}}, {"ou", {"AW0"}}, {"ow", {"AW0"}},
        {"ai", {"EY0"}}, {"ay", {"EY0"}}, {"au", {"AO0"}}, {"aw", {"AO0"}}, {"oi", {"OY0"}},
        {"oy", {"OY0"}}, {"ea", {"EH0"}}, {"er", {"ER0"}}, {"gh", {}},
    };
    size_t i = 0;
    while (i < s.size()) {
        bool matched = false;
        for (auto& [pat, phs] : dig) {
            if (s.compare(i, pat.size(), pat) == 0) { out.insert(out.end(), phs.begin(), phs.end()); i += pat.size(); matched = true; break; }
        }
        if (matched) continue;
        char c = s[i];
        auto push = [&](std::initializer_list<std::string> v) { out.insert(out.end(), v.begin(), v.end()); };
        switch (c) {
            case 'a': push({"AE0"}); break;
            case 'b': push({"B"}); break;
            case 'c': push({(i + 1 < s.size() && strchr("eiy", s[i + 1])) ? "S" : "K"}); break;
            case 'd': push({"D"}); break;
            case 'e': push({"EH0"}); break;
            case 'f': push({"F"}); break;
            case 'g': push({(i + 1 < s.size() && strchr("eiy", s[i + 1])) ? "JH" : "G"}); break;
            case 'h': push({"HH"}); break;
            case 'i': push({"IH0"}); break;
            case 'j': push({"JH"}); break;
            case 'k': push({"K"}); break;
            case 'l': push({"L"}); break;
            case 'm': push({"M"}); break;
            case 'n': push({"N"}); break;
            case 'o': push({"AA0"}); break;
            case 'p': push({"P"}); break;
            case 'q': push({"K"}); break;
            case 'r': push({"R"}); break;
            case 's': push({"S"}); break;
            case 't': push({"T"}); break;
            case 'u': push({"AH0"}); break;
            case 'v': push({"V"}); break;
            case 'w': push({"W"}); break;
            case 'x': push({"K", "S"}); break;
            case 'y': push({i == 0 ? "Y" : (i == s.size() - 1 ? "IY0" : "IH0")}); break;
            case 'z': push({"Z"}); break;
            default: break;
        }
        i++;
    }
    if (out.empty()) out.push_back("AH0");
    return out;
}

std::vector<std::string> G2P::query_word(const std::string& o_word, int& fallback_words) const {
    std::string word = lower_of(o_word);
    // Homographs: no POS tagger in the host; use the non-special reading (pron2).
    static const std::map<std::string, std::vector<std::string>> homographs = {
        {"read", {"R", "IY1", "D"}},
        {"complex", {"K", "AA1", "M", "P", "L", "EH0", "K", "S"}},
        {"lead", {"L", "EH1", "D"}},
        {"presents", {"P", "R", "EH1", "Z", "AH0", "N", "T", "S"}},
    };
    auto hit = homographs.find(word);
    if (hit != homographs.end()) return hit->second;
    auto it = cmu_.find(word);
    if (it != cmu_.end()) {
        if (o_word == "A") return {"AH0"};
        return it->second;
    }
    bool title = !o_word.empty() && isupper((unsigned char)o_word[0]) &&
                 std::all_of(o_word.begin() + 1, o_word.end(), [](char c) { return !isupper((unsigned char)c); });
    if (title) {
        auto ni = namedict_.find(word);
        if (ni != namedict_.end()) return ni->second;
    }
    if (word.size() > 2 && word.compare(word.size() - 2, 2, "'s") == 0) {
        auto base = query_word(word.substr(0, word.size() - 2), fallback_words);
        if (!base.empty()) {
            const std::string& last = base.back();
            if (last == "S" || last == "Z" || last == "SH" || last == "ZH" || last == "CH" || last == "JH") { base.push_back("AH0"); base.push_back("Z"); }
            else if (last == "P" || last == "T" || last == "K" || last == "F" || last == "TH") base.push_back("S");
            else base.push_back("Z");
            return base;
        }
    }
    if (word.find('-') != std::string::npos && word.size() > 1) {
        std::vector<std::string> out;
        for (auto& part : split_ws([&] { std::string t = word; for (auto& c : t) if (c == '-') c = ' '; return t; }())) {
            auto sub = query_word(part, fallback_words);
            out.insert(out.end(), sub.begin(), sub.end());
        }
        if (!out.empty()) return out;
    }
    fallback_words++;
    return lts_fallback(word);
}

std::vector<std::string> G2P::filter_phones(const std::vector<std::string>& raw) const {
    static const std::unordered_set<std::string> undesired = {" ", "<pad>", "UW", "</s>", "<s>"};
    std::vector<std::string> out;
    for (auto ph : raw) {
        if (undesired.count(ph)) continue;
        if (ph == "<unk>") ph = "UNK";
        if (ph == "'") ph = "-";
        if (symbol_to_id_.count(ph)) out.push_back(ph);
    }
    return out;
}

std::vector<int64_t> G2P::text_to_ids(const std::string& text, int& fallback_words) const {
    fallback_words = 0;
    std::string expanded = expand_numbers(text);       // digits first (needs ':' intact)
    expanded = text_expand(expanded);                   // rep-map punctuation
    expanded = expand_abbrev(expanded);
    // collapse consecutive punctuation (Genie replace_consecutive_punctuation)
    {
        std::string out;
        for (size_t i = 0; i < expanded.size();) {
            char c = expanded[i];
            if (c == '.' || c == ',' || c == '!' || c == '?') {
                out += c;
                size_t j = i + 1;
                while (j < expanded.size() && (expanded[j] == '.' || expanded[j] == ',' || expanded[j] == '!' || expanded[j] == '?' || isspace((unsigned char)expanded[j]))) {
                    if (!isspace((unsigned char)expanded[j])) { j++; continue; }
                    // space then more punctuation -> still collapses in Genie ([punct\s][punct]+); simple approx: stop at space-run end check
                    size_t k = j;
                    while (k < expanded.size() && isspace((unsigned char)expanded[k])) k++;
                    if (k < expanded.size() && (expanded[k] == '.' || expanded[k] == ',' || expanded[k] == '!' || expanded[k] == '?')) { j = k; continue; }
                    break;
                }
                i = j;
            } else { out += c; i++; }
        }
        expanded = out;
    }
    auto toks = tokenize(expanded);
    std::vector<std::string> raw;
    for (auto& tok : toks) {
        std::vector<std::string> pron;
        if (!has_alpha(tok)) pron = {tok};
        else pron = query_word(tok, fallback_words);
        raw.insert(raw.end(), pron.begin(), pron.end());
    }
    auto phones = filter_phones(raw);
    std::vector<int64_t> ids;
    for (auto& ph : phones) ids.push_back(symbol_to_id_.at(ph));
    return ids;
}

void G2P::load(const std::string& data_dir) {
    {
        std::ifstream f(data_dir + "/symbols.txt");
        std::string line;
        int id = 0;
        while (std::getline(f, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            symbols_.push_back(line);
            symbol_to_id_[line] = id++;
        }
    }
    auto load_dict = [&](const std::string& path, std::unordered_map<std::string, std::vector<std::string>>& dst) {
        std::ifstream f(path);
        std::string line;
        while (std::getline(f, line)) {
            auto tab = line.find('\t');
            if (tab == std::string::npos) continue;
            dst[line.substr(0, tab)] = split_ws(line.substr(tab + 1));
        }
    };
    load_dict(data_dir + "/cmudict.tsv", cmu_);
    load_dict(data_dir + "/namedict.tsv", namedict_);
}
