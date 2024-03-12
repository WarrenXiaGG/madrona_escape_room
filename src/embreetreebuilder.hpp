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

#define BVH_IMPLEMENTATION madrona::phys::MeshBVH

namespace EmbreeTreeBuilder {
    bool loadAndConvert(std::string path, madrona::imp::SourceObject& object,
                               std::vector<BVH_IMPLEMENTATION>& bvhs,
                               std::vector<BVH_IMPLEMENTATION::Node>& nodes,
                               std::vector<BVH_IMPLEMENTATION::LeafGeometry>& leafGeos,
                               std::vector<BVH_IMPLEMENTATION::LeafMaterial>& leafMaterials,
                               std::vector<madrona::math::Vector3>& verticesOut, bool regenerate, bool cache);
};

