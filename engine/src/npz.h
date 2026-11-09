// Minimal NPZ (zip, stored entries only) + NPY reader, and the weight-only
// pack restore logic (W8 per-channel symmetric / W4 group-64 asymmetric),
// mirroring wo-packs/dequant_wo.py exactly.
#pragma once
#include <cstdint>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

struct NpyTensor {
    std::string descr;              // '<f4', '<f8', '<i1', '|u1', '<i8', '<i4'
    std::vector<int64_t> shape;
    std::vector<uint8_t> data;      // raw element bytes, C order
    int64_t numel() const {
        int64_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }
    int elem_size() const {
        if (descr == "<f4" || descr == "<i4") return 4;
        if (descr == "<f8" || descr == "<i8") return 8;
        return 1;
    }
    double get_double(int64_t i) const {
        if (descr == "<f4") { float v; memcpy(&v, data.data() + i * 4, 4); return v; }
        if (descr == "<f8") { double v; memcpy(&v, data.data() + i * 8, 8); return v; }
        if (descr == "<i8") { int64_t v; memcpy(&v, data.data() + i * 8, 8); return (double)v; }
        if (descr == "<i4") { int32_t v; memcpy(&v, data.data() + i * 4, 4); return (double)v; }
        if (descr == "<i1") { int8_t v; memcpy(&v, data.data() + i, 1); return (double)v; }
        return (double)data[i];
    }
};

inline uint16_t rd16(const uint8_t* p) { return (uint16_t)(p[0] | (p[1] << 8)); }
inline uint32_t rd32(const uint8_t* p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); }

class Npz {
  public:
    // entries_: name (without .npy) -> tensor
    std::map<std::string, NpyTensor> entries;

    void load(const std::string& path) {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) throw std::runtime_error("cannot open npz: " + path);
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        std::vector<uint8_t> buf(sz);
        if (sz > 0 && fread(buf.data(), 1, sz, f) != (size_t)sz) { fclose(f); throw std::runtime_error("short read: " + path); }
        fclose(f);
        // find EOCD
        size_t eocd = SIZE_MAX;
        for (size_t i = buf.size() >= 22 ? buf.size() - 22 : 0; ; --i) {
            if (rd32(&buf[i]) == 0x06054b50) { eocd = i; break; }
            if (i == 0) break;
        }
        if (eocd == SIZE_MAX) throw std::runtime_error("npz: no EOCD in " + path);
        uint32_t cd_off = rd32(&buf[eocd + 16]);
        uint16_t count = rd16(&buf[eocd + 10]);
        size_t p = cd_off;
        for (uint16_t i = 0; i < count; i++) {
            if (rd32(&buf[p]) != 0x02014b50) throw std::runtime_error("npz: bad central dir");
            uint16_t method = rd16(&buf[p + 10]);
            uint32_t csize = rd32(&buf[p + 20]);
            uint16_t nlen = rd16(&buf[p + 28]), elen = rd16(&buf[p + 30]), clen = rd16(&buf[p + 32]);
            uint32_t lhoff = rd32(&buf[p + 42]);
            std::string name((const char*)&buf[p + 46], nlen);
            if (method != 0) throw std::runtime_error("npz: entry not stored: " + name);
            // local header
            if (rd32(&buf[lhoff]) != 0x04034b50) throw std::runtime_error("npz: bad local header");
            uint16_t lnlen = rd16(&buf[lhoff + 26]), lelen = rd16(&buf[lhoff + 28]);
            size_t data_off = lhoff + 30 + lnlen + lelen;
            parse_npy(name, &buf[data_off], csize);
            p += 46 + nlen + elen + clen;
        }
    }

  private:
    void parse_npy(const std::string& zipname, const uint8_t* d, size_t len) {
        if (len < 10 || memcmp(d, "\x93NUMPY", 6) != 0) throw std::runtime_error("npy: bad magic for " + zipname);
        uint16_t hlen = rd16(d + 8);
        std::string header((const char*)d + 10, hlen);
        NpyTensor t;
        auto get = [&](const std::string& key) -> std::string {
            auto pos = header.find("'" + key + "'");
            if (pos == std::string::npos) return "";
            pos = header.find(":", pos) + 1;
            auto end = header.find(",", pos);
            std::string v = header.substr(pos, end - pos);
            size_t a = v.find_first_not_of(" '\""), b = v.find_last_not_of(" '\"");
            return v.substr(a, b - a + 1);
        };
        t.descr = get("descr");
        auto spos = header.find("'shape'");
        spos = header.find("(", spos);
        auto epos = header.find(")", spos);
        std::string shp = header.substr(spos + 1, epos - spos - 1);
        size_t cur = 0;
        while (cur < shp.size()) {
            auto comma = shp.find(",", cur);
            std::string tok = shp.substr(cur, comma == std::string::npos ? comma : comma - cur);
            size_t a = tok.find_first_not_of(" "), b = tok.find_last_not_of(" ");
            if (a != std::string::npos) t.shape.push_back(std::stoll(tok.substr(a, b - a + 1)));
            if (comma == std::string::npos) break;
            cur = comma + 1;
        }
        if (t.shape.empty()) t.shape.push_back(1);
        size_t data_len = (size_t)t.numel() * t.elem_size();
        t.data.assign(d + 10 + hlen, d + 10 + hlen + data_len);
        std::string name = zipname;
        if (name.size() > 4 && name.substr(name.size() - 4) == ".npy") name = name.substr(0, name.size() - 4);
        entries[name] = std::move(t);
    }
};

// Restore one pack per dequant_wo.py: returns name -> fp32 tensor (shape preserved).
struct RestoredTensor {
    std::vector<int64_t> shape;
    std::vector<float> data;
};

inline std::map<std::string, RestoredTensor> restore_pack(const Npz& npz,
                                                          const std::vector<std::pair<std::string, std::string>>& name_scheme,
                                                          const std::map<std::string, std::vector<int64_t>>& shapes) {
    std::map<std::string, RestoredTensor> out;
    for (auto& [name, scheme] : name_scheme) {
        RestoredTensor rt;
        // The manifest shape is authoritative: pack entries may store a
        // flattened (rows, rest) view of tensors with ndim > 2.
        auto shape_it = shapes.find(name);
        if (scheme == "fp32-residual") {
            const NpyTensor& t = npz.entries.at(name);
            rt.shape = (shape_it != shapes.end()) ? shape_it->second : t.shape;
            rt.data.resize(t.numel());
            for (int64_t i = 0; i < t.numel(); i++) rt.data[i] = (float)t.get_double(i);
        } else if (scheme.rfind("w8", 0) == 0) {
            const NpyTensor& q = npz.entries.at(name);
            const NpyTensor& sc = npz.entries.at(name + "@scale");
            rt.shape = (shape_it != shapes.end()) ? shape_it->second : q.shape;
            int64_t rows = q.shape[0];
            int64_t per = q.numel() / rows;
            rt.data.resize(q.numel());
            for (int64_t r = 0; r < rows; r++) {
                double s = sc.get_double(r);
                for (int64_t i = 0; i < per; i++) {
                    int64_t idx = r * per + i;
                    rt.data[idx] = (float)((double)(int8_t)q.data[idx] * s);
                }
            }
        } else if (scheme.rfind("w4", 0) == 0) {
            const NpyTensor& pk = npz.entries.at(name);
            const NpyTensor& sc = npz.entries.at(name + "@scale");
            const NpyTensor& ze = npz.entries.at(name + "@zero");
            const auto& shape = shapes.at(name);
            // Weight-only packs view every tensor as (first dim, flattened rest);
            // for 2D weights this is exactly the GPT pack layout, and the vits
            // hybrid pack flattens conv kernels the same way (see VITS-W4.md).
            int64_t numel = 1;
            for (auto d : shape) numel *= d;
            int64_t rows = shape[0], cols = numel / rows;
            const int64_t G = 64;
            if (cols % G != 0) throw std::runtime_error("w4 restore: flattened cols not divisible by 64: " + name);
            int64_t ngroups = cols / G;
            rt.shape = shape;
            rt.data.resize(rows * cols);
            for (int64_t r = 0; r < rows; r++) {
                for (int64_t c = 0; c < cols; c++) {
                    int64_t g = c / G;
                    size_t byte_idx = (size_t)(r * (cols / 2) + c / 2);
                    uint8_t byte = pk.data[byte_idx];
                    int nib = (c % 2 == 0) ? (byte >> 4) : (byte & 0xF);
                    double s = sc.get_double(r * ngroups + g);
                    double z = ze.get_double(r * ngroups + g);
                    rt.data[r * cols + c] = (float)(((double)nib - z) * s);
                }
            }
        } else {
            throw std::runtime_error("unknown scheme: " + scheme);
        }
        out[name] = std::move(rt);
    }
    return out;
}
