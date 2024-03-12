#include "embreetreebuilder.hpp"

#include <iostream>
#include <embree4/rtcore.h>
#include <embree4/rtcore_common.h>
#include <assimp/scene.h>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assert.h>
#include <fstream>
#include <cfloat>



namespace EmbreeTreeBuilder{
    constexpr int numTrisPerLeaf = 8;
    constexpr int nodeWidth = 4;
    static constexpr inline int32_t sentinel = (int32_t)0xFFFF'FFFF;

    struct LeafGeometry {
        uint64_t packedIndices[numTrisPerLeaf];
    };

    struct LeafMaterial {
        uint32_t material[numTrisPerLeaf];
    };

    struct Vector3 {
        float x;
        float y;
        float z;
    };

    struct Node2 {
        float minX[nodeWidth];
        float minY[nodeWidth];
        float minZ[nodeWidth];
        float maxX[nodeWidth];
        float maxY[nodeWidth];
        float maxZ[nodeWidth];
        int32_t children[nodeWidth];
        int32_t parentID;
    };

    struct NodeCompressed {
        float minX;
        float minY;
        float minZ;
        int8_t expX;
        int8_t expY;
        int8_t expZ;
        uint8_t internalNodes;
        uint8_t qMinX[nodeWidth];
        uint8_t qMinY[nodeWidth];
        uint8_t qMinZ[nodeWidth];
        uint8_t qMaxX[nodeWidth];
        uint8_t qMaxY[nodeWidth];
        uint8_t qMaxZ[nodeWidth];
        int32_t children[nodeWidth];
        int32_t parentID;
    };

    struct RTC_ALIGN(16) BoundingBox{
        float lower_x, lower_y, lower_z, align0;
        float upper_x, upper_y, upper_z, align1;
    };

    inline float area(BoundingBox box){
        float spanX = box.upper_x - box.lower_x;
        float spanY = box.upper_y - box.lower_y;
        float spanZ = box.upper_z - box.lower_z;
        return spanX * spanY * 2 + spanY * spanZ * 2 + spanX * spanZ * 2;
    }

    inline BoundingBox merge(BoundingBox box1, BoundingBox box2){
        return BoundingBox{std::min(box1.lower_x,box2.lower_x),std::min(box1.lower_y,box2.lower_y),std::min(box1.lower_z,box2.lower_z),0,
                           std::max(box1.upper_x,box2.upper_x),std::max(box1.upper_y,box2.upper_y),std::max(box1.upper_z,box2.upper_z),0};
    }

    BoundingBox empty = {};

    bool memoryMonitor(void* userPtr, ssize_t bytes, bool post) {
        return true;
      }

      bool buildProgress (void* userPtr, double f) {
        return true;
      }

      void splitPrimitive (const RTCBuildPrimitive* prim, unsigned int dim, float pos, RTCBounds* lprim, RTCBounds* rprim, void* userPtr)
      {
        assert(dim < 3);
        assert(prim->geomID == 0);
        *(BoundingBox*) lprim = *(BoundingBox*) prim;
        *(BoundingBox*) rprim = *(BoundingBox*) prim;
        (&lprim->upper_x)[dim] = pos;
        (&rprim->lower_x)[dim] = pos;
      }

      struct Node
      {
        bool isLeaf;
        virtual float sah() = 0;
      };

      struct InnerNode : public Node
      {
        BoundingBox bounds[4];
        Node* children[4];
        int numChildren;
        unsigned int id = -1;

        InnerNode() {
          bounds[0] = bounds[1] = bounds[2] = bounds[3] = empty;
          children[0] = children[1] = children[2] = children[3] = nullptr;
          numChildren = 0;
          isLeaf = false;
        }

        float sah() {
            float cost = 0;
            BoundingBox total{INFINITY,INFINITY,INFINITY,0,-INFINITY,-INFINITY,-INFINITY,0};
            for(int i=0;i<4;i++){
                if(children[i] != nullptr){
                    cost += children[i]->sah() * area(bounds[i]);
                    total = merge(bounds[i],total);
                }
            }
            //if(area(total) == 0)
            //std::cerr << total.lower_x << "," << total.lower_y << "," << total.lower_z << "," << total.upper_x << "," << total.upper_y <<
            //"," << total.upper_z << std::endl;
            assert(area(total) >= 0);
            if(area(total) == 0){
                return 1;
            }
            return 1+ cost/area(total);
        }

        static void* create (RTCThreadLocalAllocator alloc, unsigned int numChildren, void* userPtr)
        {
          assert(numChildren > 0);
          void* ptr = rtcThreadLocalAlloc(alloc,sizeof(InnerNode),16);
          return (void*) new (ptr) InnerNode;
        }

        static void  setChildren (void* nodePtr, void** childPtr, unsigned int numChildren, void* userPtr)
        {
          assert(numChildren > 0);
          for (size_t i=0; i<numChildren; i++)
            ((InnerNode*)nodePtr)->children[i] = (Node*) childPtr[i];
          ((InnerNode*)nodePtr)->numChildren = numChildren;
        }

        static void  setBounds (void* nodePtr, const RTCBounds** bounds, unsigned int numChildren, void* userPtr)
        {
          assert(numChildren > 0);
          for (size_t i=0; i<numChildren; i++)
            ((InnerNode*)nodePtr)->bounds[i] = *(const BoundingBox*) bounds[i];
        }
      };

      struct LeafNode : public Node
      {
        unsigned int id[8];
        unsigned int numPrims;
        BoundingBox bounds;
        unsigned int lid = -1;

        LeafNode (const BoundingBox& bounds)
          : bounds(bounds) {
            isLeaf=true;
        }

        float sah() {
          return 1.0f;
        }

        static void* create (RTCThreadLocalAllocator alloc, const RTCBuildPrimitive* prims, size_t numPrims, void* userPtr)
        {
          assert(numPrims > 0);
          void* ptr = rtcThreadLocalAlloc(alloc,sizeof(LeafNode),16);
          LeafNode* leaf = new (ptr) LeafNode(*(BoundingBox*)prims);
          leaf->numPrims = numPrims;
          for(int i=0;i<numPrims;i++){
              leaf->id[i] = prims[i].primID;
          }

          return (void*) leaf;
        }
      };

    bool loadAndConvert(std::string path, madrona::imp::SourceObject& object,
                        std::vector<BVH_IMPLEMENTATION>& bvhs,
                        std::vector<BVH_IMPLEMENTATION::Node>& nodes,
                        std::vector<BVH_IMPLEMENTATION::LeafGeometry>& leafGeos,
                        std::vector<BVH_IMPLEMENTATION::LeafMaterial>& leafMaterials,
                        std::vector<madrona::math::Vector3>& verticesOut,
                        madrona::math::AABB &aabbOut,
                        bool regenerate, bool cache) {
        uint32_t current_node_offset = nodes.size();

        //Assimp::Importer importer;
        //importer.SetPropertyFloat("AI_CONFIG_GLOBAL_SCALE_FACTOR_KEY",100);
        //const aiScene* scene = importer.ReadFile("/home/warrenxia/Desktop/MadronaBVH/madrona_escape_room/data/glbtestobject.glb",
        //                                         aiProcess_Triangulate | aiProcess_PreTransformVertices | aiProcess_GlobalScale);

        int numTriangles = 0;
        int numVertices = 0;
        long offsets[object.meshes.size()+1];
        long triOffsets[object.meshes.size()+1];
        offsets[0] = 0;
        triOffsets[0] = 0;
        for(int i=0;i<object.meshes.size();i++){
            numTriangles += object.meshes[i].numFaces;
            numVertices += object.meshes[i].numVertices;
            offsets[i+1] = object.meshes[i].numVertices;
            triOffsets[i+1] = object.meshes[i].numFaces;
        }

        //std::cout << "Num tris: " << numTriangles;

        RTCDevice device = rtcNewDevice(NULL);
        RTCBVH bvh = rtcNewBVH(device);
        std::vector<RTCBuildPrimitive> prims_i;
        prims_i.reserve(numTriangles);
        prims_i.resize(numTriangles);

        std::vector<Vector3> vertices;
        vertices.reserve(numVertices);
        vertices.resize(numVertices);

        std::vector<uint64_t> prims_compressed;
        prims_compressed.reserve(numTriangles);
        prims_compressed.resize(numTriangles);

        int index = 0;
        for(int i=0;i<object.meshes.size();i++){
            auto mesh = object.meshes[i];
            for(int i2=0;i2<mesh.numFaces;i2++){
                if (mesh.faceCounts != nullptr) {
                    FATAL("MeshBVH only supports triangular meshes");
                }
                int32_t base = 3 * i2;
                uint32_t mesh_a_idx = mesh.indices[base + 0];
                uint32_t mesh_b_idx = mesh.indices[base + 1];
                uint32_t mesh_c_idx = mesh.indices[base + 2];

                auto v1 = mesh.positions[mesh_a_idx];
                auto v2 = mesh.positions[mesh_b_idx];
                auto v3 = mesh.positions[mesh_c_idx];

                uint32_t global_a_idx = mesh_a_idx + offsets[i];
                uint32_t global_b_idx = mesh_b_idx + offsets[i];
                uint32_t global_c_idx = mesh_c_idx + offsets[i];

                int32_t b_diff = (int32_t)global_b_idx - (int32_t)global_a_idx;
                int32_t c_diff = (int32_t)global_c_idx - (int32_t)global_a_idx;
                assert(abs(b_diff) < 32767 && abs(c_diff) < 32767);

                prims_compressed[triOffsets[i] + i2] =
                        (uint64_t(global_a_idx) << 32) |
                        (uint64_t((uint16_t)b_diff) << 16) |
                        uint64_t((uint16_t)c_diff);

                float minX = std::min(std::min(v1.x,v2.x),v3.x);
                float minY = std::min(std::min(v1.y,v2.y),v3.y);
                float minZ = std::min(std::min(v1.z,v2.z),v3.z);

                float maxX = std::max(std::max(v1.x,v2.x),v3.x);
                float maxY = std::max(std::max(v1.y,v2.y),v3.y);
                float maxZ = std::max(std::max(v1.z,v2.z),v3.z);

                printf("(second check) %f %f %f -> %f %f %f\n",
                        minX, minY, minZ,
                        maxX, maxY, maxZ);

                RTCBuildPrimitive prim;
                prim.lower_x = minX;
                prim.lower_y = minY;
                prim.lower_z = minZ;
                prim.geomID = 0;
                prim.upper_x = maxX;
                prim.upper_y = maxY;
                prim.upper_z = maxZ;
                prim.primID = index;
                prims_i[index] = prim;
                index++;
            }
        }

        std::vector<RTCBuildPrimitive> prims;
        prims.reserve(numTriangles);
        prims.resize(numTriangles);

        /* settings for BVH build */
        RTCBuildArguments arguments = rtcDefaultBuildArguments();
        arguments.byteSize = sizeof(arguments);
        arguments.buildFlags = RTC_BUILD_FLAG_NONE;
        arguments.buildQuality = RTC_BUILD_QUALITY_HIGH;
        arguments.maxBranchingFactor = 4;
        arguments.maxDepth = 1024;
        arguments.sahBlockSize = 1;
        arguments.minLeafSize = 4;
        arguments.maxLeafSize = 8;
        arguments.traversalCost = 1.0f;
        arguments.intersectionCost = 8.0f;
        arguments.bvh = bvh;
        arguments.primitives = prims.data();
        arguments.primitiveCount = prims.size();
        arguments.primitiveArrayCapacity = prims.capacity();
        arguments.createNode = InnerNode::create;
        arguments.setNodeChildren = InnerNode::setChildren;
        arguments.setNodeBounds = InnerNode::setBounds;
        arguments.createLeaf = LeafNode::create;
        arguments.splitPrimitive = splitPrimitive;
        arguments.buildProgress = buildProgress;
        arguments.userPtr = nullptr;

        Node* root;
        for (size_t i=0; i<10; i++)
        {
          /* we recreate the prims array here, as the builders modify this array */
          for (size_t j=0; j<prims.size(); j++) prims[j] = prims_i[j];

          //std::cout << "iteration " << i << ": building BVH over " << prims.size() << " primitives, " << std::flush;
          double t0 = 0;
          root = (Node*) rtcBuildBVH(&arguments);
          double t1 = 0;
          const float sah = root ? root->sah() : 0.0f;
          //std::cout << 1000.0f*(t1-t0) << "ms, " << 1E-6*double(prims.size())/(t1-t0) << " Mprims/s, sah = " << sah << " [DONE]" << std::endl;
        }

        std::vector<Node*> stack;
        stack.push_back(root);

        std::vector<InnerNode*> innerNodes;
        std::vector<LeafNode*> leafNodes;

        int childrenCounts[]{0,0,0,0,0};
        int leafCounts[]{0,0,0,0,0,0,0,0,0};

        int leafID = 0;
        int innerID = 0;

        while(!stack.empty()){
            Node* node = stack.back();
            stack.pop_back();
            if(!node->isLeaf){
                auto* inner = (InnerNode*)node;
                for(int i=0;i<4;i++){
                    if(inner->children[i] != nullptr){
                        stack.push_back(inner->children[i]);
                    }
                }
                if(inner->id == -1){
                    inner->id = innerID;
                    innerNodes.push_back(inner);
                    innerID++;
                }
                childrenCounts[inner->numChildren]++;
            }else{
                auto* leaf = (LeafNode*)node;
                /*std::cout << "Leaf node ";
                for(int i=0;i<leaf->numPrims;i++){
                    std::cout << leaf->id[i] << " ";
                }
                leafCounts[leaf->numPrims]++;
                std::cout << std::endl;*/
                if(leaf->lid == -1){
                    leaf->lid = leafID;
                    leafNodes.push_back(leaf);
                    leafID++;
                }
            }
        }


        /*for(int i=0;i<5;i++){
            std::cout<<childrenCounts[i]<<",";
        }
        std::cout << std::endl;
        for(int i=0;i<9;i++){
            std::cout<<leafCounts[i]<<",";
        }*/

        //std::cerr << innerNodes[0]->bounds[0].upper_z << std::endl;
        //std::cerr << innerID << "," << leafID <<  "," <<vertices.size() <<std::endl;
        madrona::Optional<std::ofstream> out = madrona::Optional<std::ofstream>::none();
        if(regenerate) {
            out = std::ofstream("out.bin", std::ios::out | std::ios::binary);
            int size = vertices.size();
            out->write((char *) &innerID, sizeof(int));
            out->write((char *) &leafID, sizeof(int));
            out->write((char *) &size, sizeof(size));
        }

        if(innerID == 0){
            BVH_IMPLEMENTATION::Node node;
            for(int j=0;j<nodeWidth;j++){
                int32_t child;
                if(j < leafNodes.size()) {
                    LeafNode *iNode = (LeafNode *) leafNodes[j];
                    child = 0x80000000 | iNode->lid;
                    BoundingBox box = iNode->bounds;
                    node.minX[j] = box.lower_x;
                    node.minY[j] = box.lower_y;
                    node.minZ[j] = box.lower_z;
                    node.maxX[j] = box.upper_x;
                    node.maxY[j] = box.upper_y;
                    node.maxZ[j] = box.upper_z;
                }else{
                    child = sentinel;
                }
                node.children[j] = child;
                //node.children[j] = 0xBBBBBBBB;
            }
            nodes.push_back(node);
        }
    //#define COMPRESSED
    #ifdef COMPRESSED
        for(int i=0;i< innerID;i++){
            NodeCompressed node;
            float minX = FLT_MAX,
            minY = FLT_MAX,
            minZ = FLT_MAX,
            maxX = FLT_MIN,
            maxY = FLT_MIN,
            maxZ = FLT_MIN;

            for(int i2=0;i2<nodeWidth;i2++){
                if(innerNodes[i]->children[i2] != nullptr) {
                    minX = fminf(minX, innerNodes[i]->bounds[i2].lower_x);
                    minY = fminf(minY, innerNodes[i]->bounds[i2].lower_y);
                    minZ = fminf(minZ, innerNodes[i]->bounds[i2].lower_z);
                    maxX = fmaxf(maxX, innerNodes[i]->bounds[i2].upper_x);
                    maxY = fmaxf(maxY, innerNodes[i]->bounds[i2].upper_y);
                    maxZ = fmaxf(maxZ, innerNodes[i]->bounds[i2].upper_z);
                }
            }
            //printf("%f,%f,%f | %f,%f,%f\n",minX,minY,minZ,maxX,maxY,maxZ);

            int8_t ex = ceilf(log2f((maxX-minX)/(powf(2, 8) - 1)));
            int8_t ey = ceilf(log2f((maxY-minY)/(powf(2, 8) - 1)));
            int8_t ez = ceilf(log2f((maxZ-minZ)/(powf(2, 8) - 1)));
            //printf("%d,%d,%d\n",ex,ey,ez);
            node.minX = minX;
            node.minY = minY;
            node.minZ = minZ;
            node.expX = ex;
            node.expY = ey;
            node.expZ = ez;
            node.parentID = -1;
            for(int i2=0;i2<nodeWidth;i2++){
                node.qMinX[i2] = floorf((innerNodes[i]->bounds[i2].lower_x - minX) / powf(2, ex));
                node.qMinY[i2] = floorf((innerNodes[i]->bounds[i2].lower_y - minY) / powf(2, ey));
                node.qMinZ[i2] = floorf((innerNodes[i]->bounds[i2].lower_z - minZ) / powf(2, ez));
                node.qMaxX[i2] = ceilf((innerNodes[i]->bounds[i2].upper_x - minX) / powf(2, ex));
                node.qMaxY[i2] = ceilf((innerNodes[i]->bounds[i2].upper_y - minY) / powf(2, ey));
                node.qMaxZ[i2] = ceilf((innerNodes[i]->bounds[i2].upper_z - minZ) / powf(2, ez));
            }
            for(int j=0;j<nodeWidth;j++){
                int32_t child;
                if(j<innerNodes[i]->numChildren) {
                    Node *node2 = innerNodes[i]->children[j];
                    if (!node2->isLeaf) {
                        InnerNode *iNode = (InnerNode *) node2;
                        child = iNode->id;
                    } else {
                        LeafNode *iNode = (LeafNode *) node2;
                        child = 0x80000000 | iNode->lid;
                    }
                }else{
                    child = sentinel;
                }
                node.children[j] = child;
                //node.children[j] = 0xBBBBBBBB;
            }
            if(regenerate)
                out->write((char*)&node, sizeof(NodeCompressed));
        }
    #else
        for(int i=0;i<innerID;i++){
            BVH_IMPLEMENTATION::Node node;
            node.parentID = -1;
            for(int i2=0;i2<nodeWidth;i2++){
                BoundingBox box = innerNodes[i]->bounds[i2];
                node.minX[i2] = box.lower_x;
                node.minY[i2] = box.lower_y;
                node.minZ[i2] = box.lower_z;
                node.maxX[i2] = box.upper_x;
                node.maxY[i2] = box.upper_y;
                node.maxZ[i2] = box.upper_z;
            }
            for(int j=0;j<nodeWidth;j++){
                int32_t child;
                if(j<innerNodes[i]->numChildren) {
                    Node *node2 = innerNodes[i]->children[j];
                    if (!node2->isLeaf) {
                        InnerNode *iNode = (InnerNode *) node2;
                        child = iNode->id;
                    } else {
                        LeafNode *iNode = (LeafNode *) node2;
                        child = 0x80000000 | iNode->lid;
                    }
                }else{
                    child = sentinel;
                }
                node.children[j] = child;
                //node.children[j] = 0xBBBBBBBB;
            }

            nodes.push_back(node);

            if(regenerate)
                out->write((char*)&node, sizeof(Node2));
        }
        //printf("Sizes:  %d,%d,%d,%d\n",nodes.size(),current_node_offset,innerID,leafNodes.size());
        auto *root_node = &nodes[current_node_offset];

        // Create root AABB
        madrona::math::AABB merged = {
            .pMin = { root_node->minX[0], root_node->minY[0], root_node->minZ[0] },
            .pMax = { root_node->maxX[0], root_node->maxY[0], root_node->maxZ[0] },
        };

        for (int aabb_idx = 1; aabb_idx < nodeWidth; ++aabb_idx) {
            if (root_node->hasChild(aabb_idx)) {
                madrona::math::AABB child_aabb = {
                    .pMin = { root_node->minX[aabb_idx], root_node->minY[aabb_idx], root_node->minZ[aabb_idx] },
                    .pMax = { root_node->maxX[aabb_idx], root_node->maxY[aabb_idx], root_node->maxZ[aabb_idx] },
                };

                merged = madrona::math::AABB::merge(merged, child_aabb);
            }
        }

        aabbOut = merged;

    #endif
        for(int i=0;i<leafID;i++){
            LeafNode* node = leafNodes[i];
            BVH_IMPLEMENTATION::LeafGeometry geos;
            for(int i2=0;i2<numTrisPerLeaf;i2++){
                if(i2<node->numPrims){
                    geos.packedIndices[i2] = prims_compressed[node->id[i2]];
                }else{
                    geos.packedIndices[i2] =0xFFFF'FFFF'FFFF'FFFF;
                }
            }
            leafGeos.push_back(geos);

            if(regenerate) {
                out->write((char *) &geos, sizeof(LeafGeometry));
            }
        }
        for(int i=0;i<leafID;i++){
            LeafNode* node = leafNodes[i];
            BVH_IMPLEMENTATION::LeafMaterial geos;
            for(int i2=0;i2<numTrisPerLeaf;i2++){
                geos.material[i2] = 0xAAAAAAAA;
            }
            leafMaterials.push_back(geos);
            if(regenerate)
                out->write((char*)&geos, sizeof(LeafMaterial));
        }

        for(int i=0;i<vertices.size();i++) {
            verticesOut.push_back({vertices[i].x,vertices[i].y,vertices[i].z});
        }
        if(regenerate) {
            out->write((char *) &vertices[0], vertices.size() * sizeof(Vector3));
            out->close();
        }

        rtcReleaseBVH(bvh);

        return 0;
    }
}
