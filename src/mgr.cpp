#include "mgr.hpp"
#include "sim.hpp"
#include "import.hpp"

#include <random>
#include <numeric>
#include <algorithm>

#include <madrona/utils.hpp>
#include <madrona/importer.hpp>
#include <madrona/physics_loader.hpp>
#include <madrona/tracing.hpp>
#include <madrona/mw_cpu.hpp>
#include <madrona/render/api.hpp>
#include <madrona/physics_assets.hpp>

#include <array>
#include <charconv>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>

#ifdef MADRONA_CUDA_SUPPORT
#include <madrona/mw_gpu.hpp>
#include <madrona/cuda_utils.hpp>
#endif

#define MADRONA_VIEWER

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;
using namespace madrona::render;
using namespace madrona::py;

namespace madEscape {

struct RenderGPUState {
    render::APILibHandle apiLib;
    render::APIManager apiMgr;
    render::GPUHandle gpu;
};


static inline Optional<RenderGPUState> initRenderGPUState(
    const Manager::Config &mgr_cfg)
{
    if (!mgr_cfg.headlessMode) {
        if (mgr_cfg.extRenderDev || !mgr_cfg.enableBatchRenderer) {
            return Optional<RenderGPUState>::none();
        }
    }

    auto render_api_lib = render::APIManager::loadDefaultLib();
    render::APIManager render_api_mgr(render_api_lib.lib());
    render::GPUHandle gpu = render_api_mgr.initGPU(mgr_cfg.gpuID);

    return RenderGPUState {
        .apiLib = std::move(render_api_lib),
        .apiMgr = std::move(render_api_mgr),
        .gpu = std::move(gpu),
    };
}

static inline Optional<render::RenderManager> initRenderManager(
    const Manager::Config &mgr_cfg,
    const Optional<RenderGPUState> &render_gpu_state)
{
    if (mgr_cfg.headlessMode && !mgr_cfg.enableBatchRenderer) {
        return Optional<render::RenderManager>::none();
    }

    if (!mgr_cfg.headlessMode) {
        if (!mgr_cfg.extRenderDev && !mgr_cfg.enableBatchRenderer) {
            return Optional<render::RenderManager>::none();
        }
    }

    render::APIBackend *render_api;
    render::GPUDevice *render_dev;

    if (render_gpu_state.has_value()) {
        render_api = render_gpu_state->apiMgr.backend();
        render_dev = render_gpu_state->gpu.device();
    } else {
        render_api = mgr_cfg.extRenderAPI;
        render_dev = mgr_cfg.extRenderDev;
    }

    return render::RenderManager(render_api, render_dev, {
        .enableBatchRenderer = mgr_cfg.enableBatchRenderer,
        .agentViewWidth = mgr_cfg.batchRenderViewWidth,
        .agentViewHeight = mgr_cfg.batchRenderViewHeight,
        .numWorlds = mgr_cfg.numWorlds,
        .maxViewsPerWorld = consts::maxAgents,
        .maxInstancesPerWorld = 1024,
        .execMode = mgr_cfg.execMode,
        .voxelCfg = {},
    });
}

struct Manager::Impl {
    Config cfg;
    PhysicsLoader physicsLoader;
    WorldReset *worldResetBuffer;
    Action *agentActionsBuffer;
    Optional<RenderGPUState> renderGPUState;
    Optional<render::RenderManager> renderMgr;
    uint32_t raycastOutputResolution;
    bool headlessMode;

    inline Impl(const Manager::Config &mgr_cfg,
                PhysicsLoader &&phys_loader,
                WorldReset *reset_buffer,
                Action *action_buffer,
                Optional<RenderGPUState> &&render_gpu_state,
                Optional<render::RenderManager> &&render_mgr,
                uint32_t raycast_output_resolution)
        : cfg(mgr_cfg),
          physicsLoader(std::move(phys_loader)),
          worldResetBuffer(reset_buffer),
          agentActionsBuffer(action_buffer),
          renderGPUState(std::move(render_gpu_state)),
          renderMgr(std::move(render_mgr)),
          raycastOutputResolution(raycast_output_resolution),
          headlessMode(mgr_cfg.headlessMode)
    {}

    inline virtual ~Impl() {}

    virtual void run() = 0;

    virtual Tensor exportTensor(ExportID slot,
        TensorElementType type,
        madrona::Span<const int64_t> dimensions) const = 0;

    static inline Impl * init(const Config &cfg);
};

struct Manager::CPUImpl final : Manager::Impl {
    using TaskGraphT =
        TaskGraphExecutor<Engine, Sim, Sim::Config, Sim::WorldInit>;

    TaskGraphT cpuExec;

    inline CPUImpl(const Manager::Config &mgr_cfg,
                   PhysicsLoader &&phys_loader,
                   WorldReset *reset_buffer,
                   Action *action_buffer,
                   Optional<RenderGPUState> &&render_gpu_state,
                   Optional<render::RenderManager> &&render_mgr,
                   TaskGraphT &&cpu_exec)
        : Impl(mgr_cfg, std::move(phys_loader),
               reset_buffer, action_buffer,
               std::move(render_gpu_state), std::move(render_mgr),
               mgr_cfg.raycastOutputResolution),
          cpuExec(std::move(cpu_exec))
    {}

    inline virtual ~CPUImpl() final {}

    inline virtual void run()
    {
        cpuExec.run();
    }

    virtual inline Tensor exportTensor(ExportID slot,
        TensorElementType type,
        madrona::Span<const int64_t> dims) const final
    {
        void *dev_ptr = cpuExec.getExported((uint32_t)slot);
        return Tensor(dev_ptr, type, dims, Optional<int>::none());
    }
};

#ifdef MADRONA_CUDA_SUPPORT
struct Manager::CUDAImpl final : Manager::Impl {
    MWCudaExecutor gpuExec;
    MWCudaLaunchGraph stepGraph;
    MWCudaLaunchGraph renderGraph;

    inline CUDAImpl(const Manager::Config &mgr_cfg,
                   PhysicsLoader &&phys_loader,
                   WorldReset *reset_buffer,
                   Action *action_buffer,
                   Optional<RenderGPUState> &&render_gpu_state,
                   Optional<render::RenderManager> &&render_mgr,
                   MWCudaExecutor &&gpu_exec,
                   MWCudaLaunchGraph &&step_graph,
                   MWCudaLaunchGraph &&render_graph)
        : Impl(mgr_cfg, std::move(phys_loader),
               reset_buffer, action_buffer,
               std::move(render_gpu_state), std::move(render_mgr),
               mgr_cfg.raycastOutputResolution),
          gpuExec(std::move(gpu_exec)),
          stepGraph(std::move(step_graph)),
          renderGraph(std::move(render_graph))
    {}

    inline virtual ~CUDAImpl() final {}

    inline virtual void run()
    {
        gpuExec.run(stepGraph);
        gpuExec.run(renderGraph);
    }

    virtual inline Tensor exportTensor(ExportID slot,
        TensorElementType type,
        madrona::Span<const int64_t> dims) const final
    {
        void *dev_ptr = gpuExec.getExported((uint32_t)slot);
        return Tensor(dev_ptr, type, dims, cfg.gpuID);
    }
};
#endif

// #define LOAD_ENV 2

struct LoadResult {
    std::vector<ImportedInstance> importedInstances;
    std::vector<UniqueScene> uniqueSceneInfos;
};

#if 1
static imp::ImportedAssets loadScenes(
        Optional<render::RenderManager> &render_mgr,
        uint32_t first_unique_scene,
        uint32_t num_unique_scenes,
        LoadResult &load_result,
        const std::string &green_grid_path,
        const std::string &smile_path)
{
    const char *cache_everything = getenv("MADRONA_CACHE_ALL_BVH");
    const char *proc_thor = getenv("MADRONA_PROC_THOR");

    std::string hssd_scenes = std::filesystem::path(DATA_DIR) /
        "hssd-hab/scenes";
    std::string procthor_scenes = std::filesystem::path(DATA_DIR) /
        "ai2thor-hab/ai2thor-hab/configs/scenes/ProcTHOR/5";
    std::string procthor_root = std::filesystem::path(DATA_DIR) /
        "ai2thor-hab/ai2thor-hab/configs";

    //Use Uncompressed because our GLTF loader doesn't support loading compressed vertex formats
    std::string procthor_obj_root = std::filesystem::path(DATA_DIR) /
        "ai2thor-hab/ai2thorhab-uncompressed/configs";

    //Uncomment this for procthor
    if (proc_thor && proc_thor[0] == '1') {
        hssd_scenes = procthor_scenes;
    }
    
    std::vector<std::string> scene_paths;

    for (const auto &dir_entry :
            std::filesystem::directory_iterator(hssd_scenes)) {
        scene_paths.push_back(dir_entry.path());
    }

    if (cache_everything && std::stoi(cache_everything) == 1) {
        num_unique_scenes = scene_paths.size();
    }

    std::vector<std::string> render_asset_paths;
    render_asset_paths.resize((size_t)SimObjectDefault::NumObjects);

    render_asset_paths[(size_t)SimObjectDefault::Cube] =
        (std::filesystem::path(DATA_DIR) / "cube_render.obj").string();
    render_asset_paths[(size_t)SimObjectDefault::Wall] =
        (std::filesystem::path(DATA_DIR) / "wall_render.obj").string();
    render_asset_paths[(size_t)SimObjectDefault::Door] =
        (std::filesystem::path(DATA_DIR) / "wall_render.obj").string();
    render_asset_paths[(size_t)SimObjectDefault::Agent] =
        (std::filesystem::path(DATA_DIR) / "agent_render.obj").string();
    render_asset_paths[(size_t)SimObjectDefault::Button] =
        (std::filesystem::path(DATA_DIR) / "cube_render.obj").string();
    render_asset_paths[(size_t)SimObjectDefault::Plane] =
        (std::filesystem::path(DATA_DIR) / "plane.obj").string();
    render_asset_paths[(size_t)SimObjectDefault::Dust2] =
        (std::filesystem::path(DATA_DIR) / "cube_render.obj").string();

    float height_offset = 0.f;
    float scale = 10.f;

    // Generate random permutation of iota
    std::vector<int> random_indices(scene_paths.size());
    std::iota(random_indices.begin(), random_indices.end(), 0);

    auto rnd_dev = std::random_device {};
    auto rng = std::default_random_engine { rnd_dev() };

    const char *seed_str = getenv("MADRONA_SEED");
    if (seed_str) {
        printf("Using seed!\n");
        rng.seed(std::stoi(seed_str));
    }

    std::shuffle(random_indices.begin(), random_indices.end(), rng);

    // Get all the asset paths and push unique scene infos
    uint32_t num_loaded_scenes = 0;

    for (int i = first_unique_scene; i < num_unique_scenes; ++i) {
        int random_index = random_indices[i];
        printf("################ Loading scene with index %d #######################\n", random_index);

        std::string scene_path = scene_paths[random_index];

        HabitatJSON::Scene loaded_scene;

        //uncomment this for procthor
        if (proc_thor && proc_thor[0] == '1') {
            loaded_scene = HabitatJSON::procThorJSONLoad(
                    procthor_root,
                    procthor_obj_root,
                    scene_path);
        } else {
            loaded_scene = HabitatJSON::habitatJSONLoad(scene_path);
        }

        // Store the current imported instances offset
        uint32_t imported_instances_offset = 
            load_result.importedInstances.size();

        UniqueScene unique_scene_info = {
            .numInstances = 0,
            .instancesOffset = imported_instances_offset,
            .center = { 0.f, 0.f, 0.f }
        };

        float stage_angle = 0;
        if(loaded_scene.stageFront[0] == -1){
            stage_angle = -pi/2;
        }
        Quat stage_rot = Quat::angleAxis(pi_d2,{ 1.f, 0.f, 0.f }) *
                        Quat::angleAxis(stage_angle,{0,1,0});

        load_result.importedInstances.push_back({
            .position = stage_rot.rotateVec({ 0.f, 0.f, 0.f + height_offset }),
            .rotation = stage_rot,
            .scale = { scale, scale, scale },
            .objectID = (int32_t)render_asset_paths.size(),
        });

        render_asset_paths.push_back(loaded_scene.stagePath.string());

        std::unordered_map<std::string, uint32_t> loaded_gltfs;
        std::unordered_map<uint32_t, uint32_t> object_to_imported_instance;
        uint32_t num_center_contribs = 0;

        for (const HabitatJSON::AdditionalInstance &inst :
                loaded_scene.additionalInstances) {
            auto path_view = inst.gltfPath.string();
            auto extension_pos = path_view.rfind('.');
            assert(extension_pos != path_view.npos);
            auto extension = path_view.substr(extension_pos + 1);

            if (extension == "json") {
                continue;
            }

            auto [iter, insert_success] = loaded_gltfs.emplace(inst.gltfPath.string(), 
                    render_asset_paths.size());
            if (insert_success) {
                auto pos = Quat::angleAxis(pi_d2, { 1.f, 0.f, 0.f }).
                           rotateVec(Vector3{ inst.pos[0], inst.pos[1], 
                                              inst.pos[2] + height_offset });

                auto scale_vec = madrona::math::Diag3x3 {
                    inst.scale[0] * scale,
                    inst.scale[1] * scale,
                    inst.scale[2] * scale
                };
                
                ImportedInstance new_inst = {
                    .position = {pos.x * scale, pos.y * scale, pos.z * scale},
                    .rotation = Quat::angleAxis(pi_d2, { 1.f, 0.f, 0.f }) * 
                                Quat{ inst.rotation[0], inst.rotation[1],
                                      inst.rotation[2], inst.rotation[3] },
                    .scale = scale_vec,
                    .objectID = (int32_t)render_asset_paths.size(),
                };

                unique_scene_info.center += math::Vector3{
                    new_inst.position.x, new_inst.position.y, 0.f };
                num_center_contribs++;

                load_result.importedInstances.push_back(new_inst);
                render_asset_paths.push_back(inst.gltfPath.string());
            } else {
                // Push the instance to the instances array
                auto pos = Quat::angleAxis(pi_d2, { 1.f, 0.f, 0.f }).
                           rotateVec(Vector3{ inst.pos[0], inst.pos[1], 
                                              inst.pos[2] + height_offset });

                auto scale_vec = madrona::math::Diag3x3 {
                    inst.scale[0] * scale,
                    inst.scale[1] * scale,
                    inst.scale[2] * scale
                };

                ImportedInstance new_inst = {
                    .position = {pos.x * scale,pos.y * scale,pos.z * scale},
                    .rotation = Quat::angleAxis(pi_d2,{ 1.f, 0.f, 0.f }) *
                                Quat{ inst.rotation[0], inst.rotation[1],
                                      inst.rotation[2], inst.rotation[3] },
                    .scale = scale_vec,
                    .objectID = (int32_t)iter->second,
                };

                unique_scene_info.center += math::Vector3{
                    new_inst.position.x, new_inst.position.y, 0.f };
                num_center_contribs++;

                load_result.importedInstances.push_back(new_inst);
            }

            unique_scene_info.numInstances =
                load_result.importedInstances.size() - unique_scene_info.instancesOffset;
        }

        unique_scene_info.center = unique_scene_info.center / (float)num_center_contribs;

        printf("%f %f %f\n", unique_scene_info.center.x, unique_scene_info.center.y, unique_scene_info.center.z);

        load_result.uniqueSceneInfos.push_back(unique_scene_info);

        printf("Loaded %d render objects\n", (int)loaded_gltfs.size());

        num_loaded_scenes++;
    }

    printf("$$$$$$$$$$$$$$$$$$$$$$$ Loaded %d scenes\n $$$$$$$$$$$$$$$$$$$$$\n", num_loaded_scenes);

    std::vector<const char *> render_asset_cstrs;
    for (size_t i = 0; i < render_asset_paths.size(); i++) {
        render_asset_cstrs.push_back(render_asset_paths[i].c_str());
    }

    std::array<char, 1024> import_err;
    auto render_assets = imp::ImportedAssets::importFromDisk(
        render_asset_cstrs, Span<char>(import_err.data(), import_err.size()),
        true, true);

    if (cache_everything && std::stoi(cache_everything) == 1) {
        exit(0);
    }

    if (!render_assets.has_value()) {
        FATAL("Failed to load render assets: %s", import_err);
    }


    // Push the additional textures
    int32_t green_grid_tex_idx = render_assets->texture.size();
    int32_t smile_tex_idx = green_grid_tex_idx + 1;

    // render_assets->texture.push_back(imp::SourceTexture(green_grid_path.c_str()));
    // render_assets->texture.push_back(imp::SourceTexture(smile_path.c_str()));

    auto materials = std::to_array<imp::SourceMaterial>({
        { render::rgb8ToFloat(191, 108, 10), -1, 0.8f, 0.2f },
        { math::Vector4{0.4f, 0.4f, 0.4f, 0.0f}, -1, 0.8f, 0.2f,},
        { math::Vector4{1.f, 1.f, 1.f, 0.0f}, smile_tex_idx, 0.5f, 1.0f,},
        { render::rgb8ToFloat(230, 230, 230),   -1, 0.8f, 1.0f },
        { math::Vector4{0.5f, 0.3f, 0.3f, 0.0f},  green_grid_tex_idx, 0.8f, 0.2f,},
        { render::rgb8ToFloat(230, 20, 20),   -1, 0.8f, 1.0f },
        { render::rgb8ToFloat(230, 230, 20),   -1, 0.8f, 1.0f },
        { render::rgb8ToFloat(230, 230, 230),   -1, 0.8f, 1.0f },
    });

    uint32_t habitat_material = 7;

    render_assets->objects[(CountT)SimObjectDefault::Cube].meshes[0].materialIDX = 0;
    render_assets->objects[(CountT)SimObjectDefault::Wall].meshes[0].materialIDX = 1;
    render_assets->objects[(CountT)SimObjectDefault::Door].meshes[0].materialIDX = 5;
    render_assets->objects[(CountT)SimObjectDefault::Agent].meshes[0].materialIDX = 2;
    render_assets->objects[(CountT)SimObjectDefault::Agent].meshes[1].materialIDX = 3;
    render_assets->objects[(CountT)SimObjectDefault::Agent].meshes[2].materialIDX = 3;
    render_assets->objects[(CountT)SimObjectDefault::Button].meshes[0].materialIDX = 6;
    render_assets->objects[(CountT)SimObjectDefault::Plane].meshes[0].materialIDX = 4;

    if (render_mgr.has_value()) {
        printf("Rasterizer is loading assets\n");

        render_mgr->loadObjects(render_assets->objects, 
                render_assets->materials, 
                render_assets->texture);

        render_mgr->configureLighting({
            { true, math::Vector3{1.0f, -1.0f, -0.05f}, math::Vector3{1.0f, 1.0f, 1.0f} }
        });
    }

    return std::move(*render_assets);
}
#endif

static void loadPhysicsObjects(PhysicsLoader &loader)
{
    std::array<std::string, (size_t)SimObjectDefault::NumObjects - 1> asset_paths;
    asset_paths[(size_t)SimObjectDefault::Cube] =
        (std::filesystem::path(DATA_DIR) / "cube_collision.obj").string();
    asset_paths[(size_t)SimObjectDefault::Wall] =
        (std::filesystem::path(DATA_DIR) / "wall_collision.obj").string();
    asset_paths[(size_t)SimObjectDefault::Door] =
        (std::filesystem::path(DATA_DIR) / "wall_collision.obj").string();
    asset_paths[(size_t)SimObjectDefault::Agent] =
        (std::filesystem::path(DATA_DIR) / "agent_collision_simplified.obj").string();
    asset_paths[(size_t)SimObjectDefault::Button] =
        (std::filesystem::path(DATA_DIR) / "cube_collision.obj").string();
    asset_paths[(size_t)SimObjectDefault::Dust2] =
       (std::filesystem::path(DATA_DIR) / "cube_collision.obj").string();

    std::array<const char *, (size_t)SimObjectDefault::NumObjects - 1> asset_cstrs;
    for (size_t i = 0; i < asset_paths.size(); i++) {
        asset_cstrs[i] = asset_paths[i].c_str();
    }

    char import_err_buffer[4096];
    auto imported_src_hulls = imp::ImportedAssets::importFromDisk(
        asset_cstrs, import_err_buffer, true);

    if (!imported_src_hulls.has_value()) {
        FATAL("%s", import_err_buffer);
    }

    DynArray<imp::SourceMesh> src_convex_hulls(
        imported_src_hulls->objects.size());

    DynArray<DynArray<SourceCollisionPrimitive>> prim_arrays(0);
    HeapArray<SourceCollisionObject> src_objs(
        (CountT)SimObjectDefault::NumObjects);

    auto setupHull = [&](SimObjectDefault obj_id,
                         float inv_mass,
                         RigidBodyFrictionData friction) {
        auto meshes = imported_src_hulls->objects[(CountT)obj_id].meshes;
        DynArray<SourceCollisionPrimitive> prims(meshes.size());

        for (const imp::SourceMesh &mesh : meshes) {
            src_convex_hulls.push_back(mesh);
            prims.push_back({
                .type = CollisionPrimitive::Type::Hull,
                .hullInput = {
                    .hullIDX = uint32_t(src_convex_hulls.size() - 1),
                },
            });
        }

        prim_arrays.emplace_back(std::move(prims));

        src_objs[(CountT)obj_id] = SourceCollisionObject {
            .prims = Span<const SourceCollisionPrimitive>(prim_arrays.back()),
            .invMass = inv_mass,
            .friction = friction,
        };
    };

    setupHull(SimObjectDefault::Cube, 0.075f, {
        .muS = 0.5f,
        .muD = 0.75f,
    });

    setupHull(SimObjectDefault::Wall, 0.f, {
        .muS = 0.5f,
        .muD = 0.5f,
    });

    setupHull(SimObjectDefault::Door, 0.f, {
        .muS = 0.5f,
        .muD = 0.5f,
    });

    setupHull(SimObjectDefault::Agent, 1.f, {
        .muS = 0.5f,
        .muD = 0.5f,
    });

    setupHull(SimObjectDefault::Button, 1.f, {
        .muS = 0.5f,
        .muD = 0.5f,
    });

    setupHull(SimObjectDefault::Dust2, 1.f, {
        .muS = 0.5f,
        .muD = 0.5f,
    });

    SourceCollisionPrimitive plane_prim {
        .type = CollisionPrimitive::Type::Plane,
    };

    src_objs[(CountT)SimObjectDefault::Plane] = {
        .prims = Span<const SourceCollisionPrimitive>(&plane_prim, 1),
        .invMass = 0.f,
        .friction = {
            .muS = 0.5f,
            .muD = 0.5f,
        },
    };

    StackAlloc tmp_alloc;
    RigidBodyAssets rigid_body_assets;
    CountT num_rigid_body_data_bytes;
    void *rigid_body_data = RigidBodyAssets::processRigidBodyAssets(
        src_convex_hulls,
        src_objs,
        false,
        tmp_alloc,
        &rigid_body_assets,
        &num_rigid_body_data_bytes);

    if (rigid_body_data == nullptr) {
        FATAL("Invalid collision hull input");
    }

    // This is a bit hacky, but in order to make sure the agents
    // remain controllable by the policy, they are only allowed to
    // rotate around the Z axis (infinite inertia in x & y axes)
    rigid_body_assets.metadatas[
        (CountT)SimObjectDefault::Agent].mass.invInertiaTensor.x = 0.f;
    rigid_body_assets.metadatas[
        (CountT)SimObjectDefault::Agent].mass.invInertiaTensor.y = 0.f;

    loader.loadRigidBodies(rigid_body_assets);
    free(rigid_body_data);
}

Manager::Impl * Manager::Impl::init(
    const Manager::Config &mgr_cfg)
{
    Sim::Config sim_cfg;
    sim_cfg.autoReset = mgr_cfg.autoReset;
    sim_cfg.initRandKey = rand::initKey(mgr_cfg.randSeed);

    const char *num_agents_str = getenv("HIDESEEK_NUM_AGENTS");
    if (num_agents_str) {
        uint32_t num_agents = std::stoi(num_agents_str);
        sim_cfg.numAgents = num_agents;
    } else {
        sim_cfg.numAgents = 1;
    }

    std::array<std::string,1> collision_asset_paths;
    collision_asset_paths[0] =
        (std::filesystem::path(DATA_DIR) / "funky2.obj").string();
    std::array<const char *, 1> collision_asset_cstrs;
    for (size_t i = 0; i < collision_asset_paths.size(); i++) {
        collision_asset_cstrs[i] = collision_asset_paths[i].c_str();
    }
    std::array<char, 1024> import_err;
    auto collision_assets = imp::ImportedAssets::importFromDisk(
        collision_asset_cstrs, Span<char>(import_err.data(), import_err.size()));

    if (!collision_assets.has_value()) {
        FATAL("Failed to load collision meshes: %s", import_err);
    }

    auto* bvh = (MeshBVH*)malloc(sizeof(MeshBVH));
    StackAlloc tmp_alloc;
    CountT numBytes;
    MeshBVHBuilder::build(collision_assets->objects[0].meshes,
                             tmp_alloc,
                             bvh,
                             &numBytes);

    switch (mgr_cfg.execMode) {
    case ExecMode::CUDA: {
#ifdef MADRONA_CUDA_SUPPORT
        printf("Cuda init:\n");
        CUcontext cu_ctx = MWCudaExecutor::initCUDA(mgr_cfg.gpuID);
        printf("post Cuda init:\n");
        PhysicsLoader phys_loader(ExecMode::CUDA, 10);
        loadPhysicsObjects(phys_loader);

        ObjectManager *phys_obj_mgr = &phys_loader.getObjectManager();

        Optional<RenderGPUState> render_gpu_state =
            initRenderGPUState(mgr_cfg);

        Optional<render::RenderManager> render_mgr =
            initRenderManager(mgr_cfg, render_gpu_state);

        imp::ImportedAssets::GPUGeometryData gpu_imported_assets;
        std::vector<ImportedInstance> imported_instances;

        sim_cfg.mergeAll = false;

        const char *first_unique_scene_str = getenv("HSSD_FIRST_SCENE");
        const char *num_unique_scene_str = getenv("HSSD_NUM_SCENES");

        assert(first_unique_scene_str && num_unique_scene_str);

        printf("%d %d\n", std::stoi(first_unique_scene_str), std::stoi(num_unique_scene_str));

        LoadResult load_result = {};

        std::string green_grid_path = (std::filesystem::path(DATA_DIR) /
                                       "green_grid.png").string();

        std::string smile_path = (std::filesystem::path(DATA_DIR) /
                                  "smile.png").string();

        auto imported_assets = loadScenes(
                render_mgr, std::stoi(first_unique_scene_str),
                std::stoi(num_unique_scene_str),
                load_result,
                green_grid_path, smile_path);

        sim_cfg.importedInstances = (ImportedInstance *)cu::allocGPU(
                sizeof(ImportedInstance) *
                load_result.importedInstances.size());

        sim_cfg.numImportedInstances = load_result.importedInstances.size();
        // sim_cfg.numObjects = gpu_imported_assets.numBVHs;

        sim_cfg.numUniqueScenes = load_result.uniqueSceneInfos.size();
        sim_cfg.uniqueScenes = (UniqueScene *)cu::allocGPU(
                sizeof(UniqueScene) * load_result.uniqueSceneInfos.size());

        sim_cfg.numWorlds = mgr_cfg.numWorlds;

        REQ_CUDA(cudaMemcpy(sim_cfg.importedInstances, 
                    load_result.importedInstances.data(),
                    sizeof(ImportedInstance) *
                    load_result.importedInstances.size(),
                    cudaMemcpyHostToDevice));

        REQ_CUDA(cudaMemcpy(sim_cfg.uniqueScenes, 
                    load_result.uniqueSceneInfos.data(),
                    sizeof(UniqueScene) *
                    load_result.uniqueSceneInfos.size(),
                    cudaMemcpyHostToDevice));


        if (render_mgr.has_value()) {
            sim_cfg.renderBridge = render_mgr->bridge();
        } else {
            sim_cfg.renderBridge = nullptr;
        }

        HeapArray<Sim::WorldInit> world_inits(mgr_cfg.numWorlds);

        uint32_t raycast_output_resolution = mgr_cfg.raycastOutputResolution;

        // If the rasterizer is enabled, disable the raycaster
        if (mgr_cfg.enableBatchRenderer) {
            raycast_output_resolution = 0;
        }

        printf("Combine compile:\n");
        MWCudaExecutor gpu_exec({
            .worldInitPtr = world_inits.data(),
            .numWorldInitBytes = sizeof(Sim::WorldInit),
            .userConfigPtr = (void *)&sim_cfg,
            .numUserConfigBytes = sizeof(Sim::Config),
            .numWorldDataBytes = sizeof(Sim),
            .worldDataAlignment = alignof(Sim),
            .numWorlds = mgr_cfg.numWorlds,
            .numTaskGraphs = (uint32_t)TaskGraphID::NumTaskGraphs,
            .numExportedBuffers = (uint32_t)ExportID::NumExports, 
            .geometryData = &gpu_imported_assets,
            .materials = imported_assets.materials,
            .textures = imported_assets.texture,
            .raycastOutputResolution = raycast_output_resolution,
            .nearSphere = 3.f
        }, {
            { GPU_HIDESEEK_SRC_LIST },
            { GPU_HIDESEEK_COMPILE_FLAGS },
            CompileConfig::OptMode::LTO,
        }, {
            .importedAssets = &imported_assets,
            .renderResolution = raycast_output_resolution,
            .nearPlane = 0.1f,
            .farPlane = 1000.f
        }, cu_ctx);

        MWCudaLaunchGraph step_graph = gpu_exec.buildLaunchGraph(
                TaskGraphID::Step, false, "step");
        MWCudaLaunchGraph render_graph = gpu_exec.buildLaunchGraph(
                TaskGraphID::Render, !mgr_cfg.enableBatchRenderer,
                "render_sort");

        printf("Combine postcompile\n");
        WorldReset *world_reset_buffer = 
            (WorldReset *)gpu_exec.getExported((uint32_t)ExportID::Reset);

        Action *agent_actions_buffer = 
            (Action *)gpu_exec.getExported((uint32_t)ExportID::Action);
        fprintf(stderr,"About to exit\n");
        return new CUDAImpl {
            mgr_cfg,
            std::move(phys_loader),
            world_reset_buffer,
            agent_actions_buffer,
            std::move(render_gpu_state),
            std::move(render_mgr),
            std::move(gpu_exec),
            std::move(step_graph),
            std::move(render_graph)
        };
#else
        FATAL("Madrona was not compiled with CUDA support");
#endif
    } break;
    case ExecMode::CPU: {
    } break;
    default: MADRONA_UNREACHABLE();
    }
}

Manager::Manager(const Config &cfg)
    : impl_(Impl::init(cfg))
{
    // Currently, there is no way to populate the initial set of observations
    // without stepping the simulations in order to execute the taskgraph.
    // Therefore, after setup, we step all the simulations with a forced reset
    // that ensures the first real step will have valid observations at the
    // start of a fresh episode in order to compute actions.
    //
    // This will be improved in the future with support for multiple task
    // graphs, allowing a small task graph to be executed after initialization.
    const char *num_agents_str = getenv("HIDESEEK_NUM_AGENTS");
    if (num_agents_str) {
        uint32_t num_agents = std::stoi(num_agents_str);
        numAgents = num_agents;
    } else {
        numAgents = 1;
    }
    
    for (int32_t i = 0; i < (int32_t)cfg.numWorlds; i++) {
        triggerReset(i);
    }

    step();
}

Manager::~Manager() {}

void Manager::step()
{
    impl_->run();

    if (impl_->headlessMode) {
        if (impl_->cfg.enableBatchRenderer) {
            impl_->renderMgr->readECS();
        }
    } else {
        if (impl_->renderMgr.has_value()) {
            impl_->renderMgr->readECS();
        }
    }

    /*
#if defined(MADRONA_VIEWER)
    if (impl_->renderMgr.has_value()) {
#else
    if (impl_->cfg.enableBatchRenderer) {
#endif
        impl_->renderMgr->readECS();
    }
    */

    if (impl_->cfg.enableBatchRenderer) {
        impl_->renderMgr->batchRender();
    }
}

Tensor Manager::resetTensor() const
{
    return impl_->exportTensor(ExportID::Reset,
                               TensorElementType::Int32,
                               {
                                   impl_->cfg.numWorlds,
                                   1,
                               });
}

Tensor Manager::actionTensor() const
{
    return impl_->exportTensor(ExportID::Action, TensorElementType::Int32,
        {
            impl_->cfg.numWorlds,
            numAgents,
            4,
        });
}

Tensor Manager::rewardTensor() const
{
    return impl_->exportTensor(ExportID::Reward, TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   numAgents,
                                   1,
                               });
}

Tensor Manager::doneTensor() const
{
    return impl_->exportTensor(ExportID::Done, TensorElementType::Int32,
                               {
                                   impl_->cfg.numWorlds,
                                   numAgents,
                                   1,
                               });
}

Tensor Manager::selfObservationTensor() const
{
    return impl_->exportTensor(ExportID::SelfObservation,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   numAgents,
                                   8,
                               });
}

Tensor Manager::partnerObservationsTensor() const
{
    return impl_->exportTensor(ExportID::PartnerObservations,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   numAgents,
                                   numAgents - 1,
                                   3,
                               });
}

Tensor Manager::roomEntityObservationsTensor() const
{
    return impl_->exportTensor(ExportID::RoomEntityObservations,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   numAgents,
                                   consts::maxEntitiesPerRoom,
                                   3,
                               });
}

Tensor Manager::doorObservationTensor() const
{
    return impl_->exportTensor(ExportID::DoorObservation,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   numAgents,
                                   3,
                               });
}

Tensor Manager::lidarTensor() const
{
    return impl_->exportTensor(ExportID::Lidar, TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   numAgents,
                                   consts::numLidarSamples,
                                   2,
                               });
}

Tensor Manager::stepsRemainingTensor() const
{
    return impl_->exportTensor(ExportID::StepsRemaining,
                               TensorElementType::Int32,
                               {
                                   impl_->cfg.numWorlds,
                                   numAgents,
                                   1,
                               });
}

Tensor Manager::rgbTensor() const
{
    const uint8_t *rgb_ptr = impl_->renderMgr->batchRendererRGBOut();

    return Tensor((void*)rgb_ptr, TensorElementType::UInt8, {
        impl_->cfg.numWorlds,
        numAgents,
        impl_->cfg.batchRenderViewHeight,
        impl_->cfg.batchRenderViewWidth,
        4,
    }, impl_->cfg.gpuID);
}

Tensor Manager::depthTensor() const
{
    const float *depth_ptr = impl_->renderMgr->batchRendererDepthOut();

    return Tensor((void *)depth_ptr, TensorElementType::Float32, {
        impl_->cfg.numWorlds,
        numAgents,
        impl_->cfg.batchRenderViewHeight,
        impl_->cfg.batchRenderViewWidth,
        1,
    }, impl_->cfg.gpuID);
}

Tensor Manager::raycastTensor() const
{
    uint32_t pixels_per_view = impl_->raycastOutputResolution *
        impl_->raycastOutputResolution;
    return impl_->exportTensor(ExportID::Raycast,
                               TensorElementType::UInt8,
                               {
                                   impl_->cfg.numWorlds*numAgents,
                                   pixels_per_view * 3,
                               });
}

void Manager::triggerReset(int32_t world_idx)
{
    WorldReset reset {
        1,
    };

    auto *reset_ptr = impl_->worldResetBuffer + world_idx;

    if (impl_->cfg.execMode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
        cudaMemcpy(reset_ptr, &reset, sizeof(WorldReset),
                   cudaMemcpyHostToDevice);
#endif
    }  else {
        *reset_ptr = reset;
    }
}

void Manager::setAction(int32_t world_idx,
                        int32_t agent_idx,
                        int32_t move_amount,
                        int32_t move_angle,
                        int32_t rotate,
                        int32_t grab,
                        int32_t x,
                        int32_t y,
                        int32_t z,
                        int32_t rot,
                        int32_t vrot)
{
    Action action { 
        .moveAmount = move_amount,
        .moveAngle = move_angle,
        .rotate = rotate,
        .grab = grab,
        .x = x,
        .y = y,
        .z = z,
        .rot = rot,
        .vrot = vrot,
    };

    auto *action_ptr = impl_->agentActionsBuffer +
        world_idx * numAgents + agent_idx;

    if (impl_->cfg.execMode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
        cudaMemcpy(action_ptr, &action, sizeof(Action),
                   cudaMemcpyHostToDevice);
#endif
    } else {
        *action_ptr = action;
    }
}

render::RenderManager & Manager::getRenderManager()
{
    return *impl_->renderMgr;
}

}
