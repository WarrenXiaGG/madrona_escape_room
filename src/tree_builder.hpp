#pragma once

#include <array>
#include <charconv>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <madrona/math.hpp>
#include <madrona/span.hpp>
#include <madrona/utils.hpp>
#include <madrona/importer.hpp>
#include <madrona/mesh_bvh.hpp>

namespace EmbreeTreeBuilder {

struct MeshBVHOffsets {
    uint32_t nodesOffset;
    uint32_t leafGeosOffset;
    uint32_t leafMatsOffset;
    uint32_t verticesOffset;
};

bool loadAndConvert(std::string path, madrona::imp::SourceObject& object,
                    std::vector<madrona::phys::MeshBVH>& bvhs,
                    std::vector<madrona::phys::MeshBVH::Node>& nodes,
                    std::vector<madrona::phys::MeshBVH::LeafGeometry>& leafGeos,
                    std::vector<madrona::phys::MeshBVH::LeafMaterial>& leafMaterials,
                    std::vector<madrona::math::Vector3>& verticesOut, 
                    madrona::math::AABB &aabbOut,
                    bool regenerate, bool cache);

};

