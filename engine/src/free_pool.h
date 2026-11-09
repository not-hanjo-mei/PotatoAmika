// Per-graph packed descriptors over a shared content-addressed store
// for the ORT-free engine (mirrors packed::WeightPool::load, but hands
// out raw pointers for the free engine's PackedLookup instead of
// Ort::Value views). Extracted from tf_check_free.cpp in phase 2 so the
// host and the vits validation tool bind vits pools the same way.
#pragma once

#include <map>
#include <memory>
#include <string>

#include "free_engine.h"
#include "json_min.h"
#include "npz.h"
#include "packed_core.h"

struct FreePool {
    struct Desc {
        const uint8_t* pk = nullptr;
        const float* sc = nullptr;
        const uint8_t* ze = nullptr;
    };
    std::shared_ptr<packed::BlobStore> store;
    std::map<std::string, Desc> descs;
    std::map<std::string, bool> filter;
    size_t entries = 0, logical = 0;
    explicit FreePool(std::shared_ptr<packed::BlobStore> s) : store(std::move(s)) {}
    void load(const std::string& root, const Json& manifest, const std::string& scheme,
              const std::string& section) {
        const Json* secnode = manifest.find(section.c_str())->find(scheme.c_str());
        std::string packfile = secnode->find("pack_file")->as_str();
        if (!packfile.empty() && packfile[0] != '/') packfile = root + "/wo-packs/" + packfile;
        Npz npz;
        npz.load(packfile);
        for (auto& e : secnode->find("manifest")->arr) {
            std::string name = e.find("name")->as_str();
            std::string sch = e.find("scheme")->as_str();
            if (!(sch.rfind("w4", 0) == 0 || sch.rfind("w8", 0) == 0)) continue;
            if (!filter.empty() && !filter.count(name)) continue;
            const Json* pn = e.find("pack_name");
            std::string lookup = pn ? pn->as_str() : name;
            const NpyTensor& pk = npz.entries.at(lookup);
            const NpyTensor& sc = npz.entries.at(lookup + "@scale");
            const NpyTensor* ze = nullptr;
            if (sch.rfind("w4", 0) == 0) ze = &npz.entries.at(lookup + "@zero");
            logical += pk.data.size() + sc.data.size() + (ze ? ze->data.size() : 0);
            auto c = store->intern(pk, sc, ze);
            descs[name] = Desc{c.pk, c.sc, c.ze};
            entries++;
        }
    }
    freeeng::PackedLookup lookup() {
        return [this](const std::string& base, freeeng::PackedBinding& out) {
            auto it = descs.find(base);
            if (it == descs.end()) return false;
            out.pk = it->second.pk;
            out.sc = it->second.sc;
            out.ze = it->second.ze;
            return true;
        };
    }
};
