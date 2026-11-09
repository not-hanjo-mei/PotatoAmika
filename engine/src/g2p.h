// English G2P for the C++ host: dictionary lookup (effective Genie cmudict,
// dumped by prep_assets.py) + homograph defaults + possessive/hyphen rules +
// a simple letter-to-sound fallback for OOV words (counted).
// Mirrors genie_tts G2P.English flow where the task scope allows; deviations
// (no POS tagger -> homographs take the non-special reading; no neural G2P ->
// LTS fallback; ordinal words are correct English, Genie's are buggy) are
// documented in STAGE2.md.
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class G2P {
  public:
    void load(const std::string& data_dir);
    // Returns phone ids for the text (no leading punctuation added by caller
    // other than what synthesis flow prepends). fallback_words counts words
    // that went through the letter-to-sound fallback.
    std::vector<int64_t> text_to_ids(const std::string& text, int& fallback_words) const;

  private:
    std::vector<std::string> symbols_;
    std::unordered_map<std::string, int> symbol_to_id_;
    std::unordered_map<std::string, std::vector<std::string>> cmu_;
    std::unordered_map<std::string, std::vector<std::string>> namedict_;

    std::vector<std::string> normalize(const std::string& text) const;  // returns token stream pre-split on spaces after text-level expansions
    std::vector<std::string> tokenize(const std::string& text) const;
    std::vector<std::string> query_word(const std::string& o_word, int& fallback_words) const;
    std::vector<std::string> lts_fallback(const std::string& word) const;
    std::vector<std::string> filter_phones(const std::vector<std::string>& raw) const;
};
