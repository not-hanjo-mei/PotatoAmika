// Minimal ONNX protobuf reader for the ORT-free engine (free_engine).
//
// Covers exactly the ModelProto subset our five graphs use: nodes with
// string/int/float/tensor/list attributes, initializers (raw_data or typed
// repeated fields), and input/output ValueInfo with static or symbolic
// dims. No external data (all five graphs are self-contained; verified
// 2026-11-08). This is a data reader only -- it makes no execution
// decisions; the plan compiler in free_engine.cpp consumes it.
#pragma once
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace onnxm {

struct Reader {
    const uint8_t* p = nullptr;
    const uint8_t* end = nullptr;
    Reader(const uint8_t* b, size_t n) : p(b), end(b + n) {}
    bool empty() const { return p >= end; }
    uint64_t varint() {
        uint64_t v = 0;
        int shift = 0;
        while (true) {
            if (p >= end) throw std::runtime_error("onnx: varint overrun");
            uint8_t b = *p++;
            v |= (uint64_t)(b & 0x7F) << shift;
            if (!(b & 0x80)) return v;
            shift += 7;
            if (shift > 63) throw std::runtime_error("onnx: varint too long");
        }
    }
    // Returns (field, wire). Caller consumes the value by wire type.
    std::pair<int, int> tag() {
        uint64_t t = varint();
        return {(int)(t >> 3), (int)(t & 7)};
    }
    Reader sub() {  // length-delimited submessage
        uint64_t n = varint();
        if (p + n > end) throw std::runtime_error("onnx: submessage overrun");
        Reader r(p, (size_t)n);
        p += n;
        return r;
    }
    std::string bytes() {
        Reader r = sub();
        return std::string((const char*)r.p, (size_t)(r.end - r.p));
    }
    float f32() {
        if (p + 4 > end) throw std::runtime_error("onnx: f32 overrun");
        float v;
        memcpy(&v, p, 4);
        p += 4;
        return v;
    }
    double f64() {
        if (p + 8 > end) throw std::runtime_error("onnx: f64 overrun");
        double v;
        memcpy(&v, p, 8);
        p += 8;
        return v;
    }
    void skip(int wire) {
        switch (wire) {
            case 0: (void)varint(); break;
            case 1: p += 8; break;
            case 2: { uint64_t n = varint(); p += n; } break;
            case 5: p += 4; break;
            default: throw std::runtime_error("onnx: bad wire type");
        }
        if (p > end) throw std::runtime_error("onnx: skip overrun");
    }
};

struct Tensor {
    std::string name;
    int data_type = 0;  // ONNX enum: 1 f32, 2 u8, 3 i8, 6 i32, 7 i64, 9 bool, 11 f64
    std::vector<int64_t> dims;
    std::vector<uint8_t> data;  // element bytes, little-endian, C order
    int64_t numel() const {
        int64_t n = 1;
        for (auto d : dims) n *= d;
        return n;
    }
};

struct Attr {
    std::string name;
    int type = 0;  // 1 f, 2 i, 3 s, 4 tensor, 6 floats, 7 ints, 8 strings
    float f = 0;
    int64_t i = 0;
    std::string s;
    std::shared_ptr<Tensor> t;
    std::vector<float> floats;
    std::vector<int64_t> ints;
    std::vector<std::string> strings;
};

struct Node {
    std::string name, op, domain;
    std::vector<std::string> inputs, outputs;
    std::vector<Attr> attrs;
    const Attr* attr(const char* n) const {
        for (auto& a : attrs)
            if (a.name == n) return &a;
        return nullptr;
    }
    int64_t attr_i(const char* n, int64_t dflt) const {
        auto* a = attr(n);
        return a ? a->i : dflt;
    }
    float attr_f(const char* n, float dflt) const {
        auto* a = attr(n);
        return a ? a->f : dflt;
    }
};

struct ValueInfo {
    std::string name;
    int elem_type = 0;
    // Per dim: dim_value >= 0 when static, else -1 with dim_param set.
    std::vector<int64_t> dims;
    std::vector<std::string> dim_params;
    bool has_type = false;
};

struct Graph {
    std::string name;
    std::vector<Node> nodes;
    std::vector<Tensor> initializers;
    std::vector<ValueInfo> inputs, outputs;
};

inline size_t elem_size(int dt) {
    switch (dt) {
        case 1: case 6: return 4;   // f32, i32
        case 7: case 11: return 8;  // i64, f64
        case 10: return 2;          // f16
        default: return 1;          // u8, i8, bool
    }
}

inline Tensor parse_tensor(Reader r) {
    Tensor t;
    std::vector<float> fdata;
    std::vector<int32_t> i32data;
    std::vector<int64_t> i64data;
    std::vector<double> ddata;
    std::string raw;
    bool has_raw = false;
    while (!r.empty()) {
        auto [field, wire] = r.tag();
        switch (field) {
            case 1:  // dims (packed varint)
                if (wire == 2) {
                    Reader s = r.sub();
                    while (!s.empty()) t.dims.push_back((int64_t)s.varint());
                } else
                    t.dims.push_back((int64_t)r.varint());
                break;
            case 2: t.data_type = (int)r.varint(); break;
            case 4:  // float_data
                if (wire == 2) {
                    Reader s = r.sub();
                    while (!s.empty()) fdata.push_back(s.f32());
                } else if (wire == 5)
                    fdata.push_back(r.f32());
                else
                    r.skip(wire);
                break;
            case 5:  // int32_data
                if (wire == 2) {
                    Reader s = r.sub();
                    while (!s.empty()) i32data.push_back((int32_t)s.varint());
                } else
                    i32data.push_back((int32_t)r.varint());
                break;
            // TensorProto field numbers (per the onnx descriptor):
            // dims 1, data_type 2, float_data 4, int32_data 5,
            // int64_data 7, name 8, raw_data 9, double_data 10,
            // uint64_data 11.
            case 7:  // int64_data
                if (wire == 2) {
                    Reader s = r.sub();
                    while (!s.empty()) i64data.push_back((int64_t)s.varint());
                } else
                    i64data.push_back((int64_t)r.varint());
                break;
            case 8: t.name = r.bytes(); break;
            case 9: raw = r.bytes(); has_raw = true; break;
            case 10:  // double_data
                if (wire == 2) {
                    Reader s = r.sub();
                    while (!s.empty()) ddata.push_back(s.f64());
                } else if (wire == 1)
                    ddata.push_back(r.f64());
                else
                    r.skip(wire);
                break;
            case 11:  // uint64_data: stored into the i64 buffer (same width)
                if (wire == 2) {
                    Reader s = r.sub();
                    while (!s.empty()) i64data.push_back((int64_t)s.varint());
                } else
                    i64data.push_back((int64_t)r.varint());
                break;
            default: r.skip(wire); break;
        }
    }
    if (has_raw) {
        t.data.assign(raw.begin(), raw.end());
    } else if (!fdata.empty()) {
        t.data.resize(fdata.size() * 4);
        memcpy(t.data.data(), fdata.data(), t.data.size());
    } else if (!i64data.empty()) {
        t.data.resize(i64data.size() * 8);
        memcpy(t.data.data(), i64data.data(), t.data.size());
    } else if (!i32data.empty()) {
        t.data.resize(i32data.size() * 4);
        memcpy(t.data.data(), i32data.data(), t.data.size());
    } else if (!ddata.empty()) {
        t.data.resize(ddata.size() * 8);
        memcpy(t.data.data(), ddata.data(), t.data.size());
    }
    return t;
}

inline Attr parse_attr(Reader r) {
    Attr a;
    while (!r.empty()) {
        auto [field, wire] = r.tag();
        // AttributeProto field numbers: name 1, f 2, i 3, s 4, t 5,
        // floats 7, ints 8, strings 9, type 20.
        switch (field) {
            case 1: a.name = r.bytes(); break;
            case 2: a.f = r.f32(); break;
            case 3: a.i = (int64_t)r.varint(); break;
            case 4: a.s = r.bytes(); break;
            case 5: a.t = std::make_shared<Tensor>(parse_tensor(r.sub())); break;
            case 7:
                if (wire == 2) {
                    Reader s = r.sub();
                    while (!s.empty()) a.floats.push_back(s.f32());
                } else
                    a.floats.push_back(r.f32());
                break;
            case 8:
                if (wire == 2) {
                    Reader s = r.sub();
                    while (!s.empty()) a.ints.push_back((int64_t)s.varint());
                } else
                    a.ints.push_back((int64_t)r.varint());
                break;
            case 9: a.strings.push_back(r.bytes()); break;
            case 20: a.type = (int)r.varint(); break;
            default: r.skip(wire); break;
        }
    }
    return a;
}

inline Node parse_node(Reader r) {
    Node n;
    while (!r.empty()) {
        auto [field, wire] = r.tag();
        switch (field) {
            case 1: n.inputs.push_back(r.bytes()); break;
            case 2: n.outputs.push_back(r.bytes()); break;
            case 3: n.name = r.bytes(); break;
            case 4: n.op = r.bytes(); break;
            case 5: n.attrs.push_back(parse_attr(r.sub())); break;
            case 7: n.domain = r.bytes(); break;
            default: r.skip(wire); break;
        }
    }
    return n;
}

inline ValueInfo parse_value_info(Reader r) {
    ValueInfo vi;
    while (!r.empty()) {
        auto [field, wire] = r.tag();
        if (field == 1) {
            vi.name = r.bytes();
        } else if (field == 2) {  // TypeProto
            Reader t = r.sub();
            while (!t.empty()) {
                auto [tf, tw] = t.tag();
                if (tf == 1) {  // tensor_type
                    Reader tt = t.sub();
                    while (!tt.empty()) {
                        auto [xf, xw] = tt.tag();
                        if (xf == 1) {
                            vi.elem_type = (int)tt.varint();
                            vi.has_type = true;
                        } else if (xf == 2) {  // shape
                            Reader sh = tt.sub();
                            while (!sh.empty()) {
                                auto [df, dw] = sh.tag();
                                if (df == 1) {
                                    Reader d = sh.sub();
                                    int64_t dv = -1;
                                    std::string dp;
                                    while (!d.empty()) {
                                        auto [qf, qw] = d.tag();
                                        if (qf == 1) dv = (int64_t)d.varint();
                                        else if (qf == 2) dp = d.bytes();
                                        else d.skip(qw);
                                    }
                                    vi.dims.push_back(dv);
                                    vi.dim_params.push_back(dp);
                                } else
                                    sh.skip(dw);
                            }
                        } else
                            tt.skip(xw);
                    }
                } else
                    t.skip(tw);
            }
        } else
            r.skip(wire);
    }
    return vi;
}

inline Graph parse_graph(Reader r) {
    Graph g;
    std::map<std::string, bool> init_names;
    while (!r.empty()) {
        auto [field, wire] = r.tag();
        switch (field) {
            case 1: g.nodes.push_back(parse_node(r.sub())); break;
            case 2: g.name = r.bytes(); break;
            case 5: {
                Tensor t = parse_tensor(r.sub());
                init_names[t.name] = true;
                g.initializers.push_back(std::move(t));
                break;
            }
            case 11: g.inputs.push_back(parse_value_info(r.sub())); break;
            case 12: g.outputs.push_back(parse_value_info(r.sub())); break;
            default: r.skip(wire); break;
        }
    }
    // ONNX lists initializers also as graph inputs; the compiler wants
    // only true runtime inputs here.
    std::vector<ValueInfo> real;
    for (auto& vi : g.inputs)
        if (!init_names.count(vi.name)) real.push_back(vi);
    g.inputs = std::move(real);
    return g;
}

inline Graph load_model(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("onnx: cannot open " + path);
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf((size_t)sz);
    if (sz > 0 && fread(buf.data(), 1, (size_t)sz, f) != (size_t)sz) {
        fclose(f);
        throw std::runtime_error("onnx: short read " + path);
    }
    fclose(f);
    Reader r(buf.data(), buf.size());
    Graph g;
    bool have = false;
    while (!r.empty()) {
        auto [field, wire] = r.tag();
        if (field == 7) {
            g = parse_graph(r.sub());
            have = true;
        } else
            r.skip(wire);
    }
    if (!have) throw std::runtime_error("onnx: no graph in " + path);
    return g;
}

}  // namespace onnxm
