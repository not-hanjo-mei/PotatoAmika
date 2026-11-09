// Minimal protobuf wire-format patcher for ONNX models: replaces initializer
// payloads (by tensor name) with fp32 raw_data, leaving everything else
// byte-identical. No protobuf library needed.
#pragma once
#include <cstdint>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include "npz.h"  // RestoredTensor

namespace pb {

inline void put_varint(std::string& out, uint64_t v) {
    while (v >= 0x80) { out += (char)((v & 0x7F) | 0x80); v >>= 7; }
    out += (char)v;
}
inline void put_key(std::string& out, uint32_t field, int wt) { put_varint(out, ((uint64_t)field << 3) | (uint64_t)wt); }
inline void put_len_field(std::string& out, uint32_t field, const std::string& payload) {
    put_key(out, field, 2);
    put_varint(out, payload.size());
    out += payload;
}

struct Seg {
    uint32_t field;
    int wt;
    size_t begin;  // start of key
    size_t end;    // one past the value
    size_t vbegin; // value start (for wt2: payload start)
    size_t vend;   // value end (== end)
    uint64_t var = 0; // for wt0
};

inline bool read_varint(const uint8_t* d, size_t len, size_t& pos, uint64_t& v) {
    v = 0; int shift = 0;
    while (pos < len) {
        uint8_t b = d[pos++];
        v |= (uint64_t)(b & 0x7F) << shift;
        if (!(b & 0x80)) return true;
        shift += 7;
        if (shift > 63) return false;
    }
    return false;
}

inline std::vector<Seg> scan(const uint8_t* d, size_t len) {
    std::vector<Seg> out;
    size_t pos = 0;
    while (pos < len) {
        Seg s; s.begin = pos;
        uint64_t key;
        if (!read_varint(d, len, pos, key)) throw std::runtime_error("pb: bad key");
        s.field = (uint32_t)(key >> 3); s.wt = (int)(key & 7);
        switch (s.wt) {
            case 0: { uint64_t v; if (!read_varint(d, len, pos, v)) throw std::runtime_error("pb: bad varint"); s.var = v; s.vbegin = s.vend = pos; break; }
            case 1: pos += 8; s.vbegin = pos - 8; s.vend = pos; break;
            case 5: pos += 4; s.vbegin = pos - 4; s.vend = pos; break;
            case 2: {
                uint64_t n; if (!read_varint(d, len, pos, n)) throw std::runtime_error("pb: bad len");
                s.vbegin = pos; pos += (size_t)n; s.vend = pos; break;
            }
            default: throw std::runtime_error("pb: unsupported wire type");
        }
        if (pos > len) throw std::runtime_error("pb: overrun");
        s.end = pos;
        out.push_back(s);
    }
    return out;
}

// Build a TensorProto message: dims (packed), data_type=FLOAT(1), name, raw_data.
inline std::string build_tensor(const std::string& name, const RestoredTensor& t) {
    std::string msg;
    std::string dims;
    for (auto d : t.shape) put_varint(dims, (uint64_t)d);
    put_len_field(msg, 1, dims);
    put_key(msg, 2, 0); put_varint(msg, 1);
    put_len_field(msg, 8, name);
    std::string raw((const char*)t.data.data(), t.data.size() * 4);
    put_len_field(msg, 9, raw);
    return msg;
}

inline std::string tensor_name(const uint8_t* d, size_t len) {
    for (auto& s : scan(d, len))
        if (s.field == 8 && s.wt == 2) return std::string((const char*)d + s.vbegin, s.vend - s.vbegin);
    return "";
}

// Replace initializers of the (single) graph in `model` whose names appear in
// `repl`. Returns patched model bytes; counts via out-params.
inline std::string patch_model(const std::string& model,
                               const std::map<std::string, RestoredTensor>& repl,
                               int& replaced, size_t& replaced_fp32_bytes) {
    replaced = 0; replaced_fp32_bytes = 0;
    const uint8_t* d = (const uint8_t*)model.data();
    auto msegs = scan(d, model.size());
    std::string out;
    bool done = false;
    for (auto& ms : msegs) {
        if (ms.field == 7 && ms.wt == 2 && !done) {
            const uint8_t* g = d + ms.vbegin;
            size_t glen = ms.vend - ms.vbegin;
            auto gsegs = scan(g, glen);
            std::string newgraph;
            for (auto& gs : gsegs) {
                if (gs.field == 5 && gs.wt == 2) {
                    std::string name = tensor_name(g + gs.vbegin, gs.vend - gs.vbegin);
                    auto it = repl.find(name);
                    if (it != repl.end()) {
                        std::string seg;
                        put_len_field(seg, 5, build_tensor(name, it->second));
                        newgraph += seg;
                        replaced++;
                        replaced_fp32_bytes += it->second.data.size() * 4;
                        continue;
                    }
                }
                newgraph.append((const char*)g + gs.begin, gs.end - gs.begin);
            }
            put_len_field(out, 7, newgraph);
            done = true;
        } else {
            out.append(model.data() + ms.begin, ms.end - ms.begin);
        }
    }
    if (!done) throw std::runtime_error("pb: no graph found");
    return out;
}

}  // namespace pb
