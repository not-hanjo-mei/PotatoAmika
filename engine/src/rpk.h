// Reader for the offline reference feature packs (.rpk), written by
// precompute_refs.py. See that script for the format definition.
#pragma once
#include <cstdint>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

struct RpkTensor {
    int dtype = 0;  // 0=f32, 1=i64
    std::vector<int64_t> shape;
    std::vector<uint8_t> data;
    const float* f32() const { return (const float*)data.data(); }
    const int64_t* i64() const { return (const int64_t*)data.data(); }
    int64_t numel() const {
        int64_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }
};

inline std::map<std::string, RpkTensor> load_rpk(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open rpk: " + path);
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf(sz);
    if (sz > 0 && fread(buf.data(), 1, sz, f) != (size_t)sz) { fclose(f); throw std::runtime_error("short read: " + path); }
    fclose(f);
    if (sz < 8 || memcmp(buf.data(), "RPK1", 4) != 0) throw std::runtime_error("rpk: bad magic: " + path);
    size_t pos = 4;
    auto need = [&](size_t n) { if (pos + n > buf.size()) throw std::runtime_error("rpk: overrun"); };
    auto u8 = [&]() { need(1); return buf[pos++]; };
    auto u16 = [&]() { need(2); uint16_t v = rd16(&buf[pos]); pos += 2; return v; };
    auto u32 = [&]() { need(4); uint32_t v = rd32(&buf[pos]); pos += 4; return v; };
    auto u64 = [&]() { need(8); uint64_t v = (uint64_t)rd32(&buf[pos]) | ((uint64_t)rd32(&buf[pos + 4]) << 32); pos += 8; return v; };
    uint32_t count = u32();
    std::map<std::string, RpkTensor> out;
    for (uint32_t i = 0; i < count; i++) {
        uint16_t nl = u16();
        need(nl);
        std::string name((const char*)&buf[pos], nl); pos += nl;
        RpkTensor t;
        t.dtype = u8();
        uint8_t nd = u8();
        for (uint8_t k = 0; k < nd; k++) t.shape.push_back((int64_t)u32());
        uint64_t nb = u64();
        need((size_t)nb);
        t.data.assign(&buf[pos], &buf[pos] + nb);
        pos += (size_t)nb;
        out[name] = std::move(t);
    }
    return out;
}
