#include <madrona/mw_gpu_entry.hpp>

#include "sim.hpp"
#include "level_gen.hpp"
//#include "madrona/mesh_bvh2.hpp"

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;

namespace RenderingSystem = madrona::render::RenderingSystem;

namespace madEscape {


inline Quat eulerToQuat(float yaw,float pitch) {
    const double halfC = 3.1415926535 / 180;

    float ex = pitch;
    float ey = 0;
    float ez = yaw;
    float sx = sinf(ex * 0.5f);
    float cx = cosf(ex * 0.5f);
    float sy = sinf(ey * 0.5f);
    float cy = cosf(ey * 0.5f);
    float sz = sinf(ez * 0.5f);
    float cz = cosf(ez * 0.5f);

    ex = (float)(cy * sx * cz - sy * cx * sz);
    ey = (float)(sy * cx * cz + cy * sx * sz);
    ez = (float)(cy * cx * sz - sy * sx * cz);
    float w = (float)(cy * cx * cz + sy * sx * sz);

    Quat cur_rot = Quat{w, ex, ey, ez};
    return cur_rot;
}

// Register all the ECS components and archetypes that will be
// used in the simulation
void Sim::registerTypes(ECSRegistry &registry, const Config &cfg)
{
    base::registerTypes(registry);
    //phys::RigidBodyPhysicsSystem::registerTypes(registry);

    RenderingSystem::registerTypes(registry, cfg.renderBridge);

    registry.registerComponent<Action>();
    registry.registerComponent<SelfObservation>();
    registry.registerComponent<Reward>();
    registry.registerComponent<Done>();
    registry.registerComponent<GrabState>();
    registry.registerComponent<Progress>();
    registry.registerComponent<OtherAgents>();
    registry.registerComponent<PartnerObservations>();
    registry.registerComponent<RoomEntityObservations>();
    registry.registerComponent<DoorObservation>();
    registry.registerComponent<ButtonState>();
    registry.registerComponent<OpenState>();
    registry.registerComponent<DoorProperties>();
    registry.registerComponent<Lidar>();
    registry.registerComponent<RaycastObservation>();
    registry.registerComponent<StepsRemaining>();
    registry.registerComponent<EntityType>();

    registry.registerSingleton<WorldReset>();
    registry.registerSingleton<LevelState>();

    registry.registerArchetype<DetatchedCamera>();
    registry.registerComponent<AgentCamera>();
    registry.registerArchetype<Agent>();
    registry.registerArchetype<PhysicsEntity>();
    registry.registerArchetype<DoorEntity>();
    registry.registerArchetype<ButtonEntity>();
    registry.registerArchetype<DummyRenderable>();
    registry.registerArchetype<ImportedEntity>();



    registry.exportSingleton<WorldReset>(
        (uint32_t)ExportID::Reset);
    registry.exportColumn<Agent, Action>(
        (uint32_t)ExportID::Action);
    registry.exportColumn<Agent, SelfObservation>(
        (uint32_t)ExportID::SelfObservation);
    registry.exportColumn<Agent, PartnerObservations>(
        (uint32_t)ExportID::PartnerObservations);
    registry.exportColumn<Agent, RoomEntityObservations>(
        (uint32_t)ExportID::RoomEntityObservations);
    registry.exportColumn<Agent, DoorObservation>(
        (uint32_t)ExportID::DoorObservation);
    registry.exportColumn<Agent, Lidar>(
        (uint32_t)ExportID::Lidar);
    registry.exportColumn<Agent, StepsRemaining>(
        (uint32_t)ExportID::StepsRemaining);
    registry.exportColumn<Agent, Reward>(
        (uint32_t)ExportID::Reward);
    registry.exportColumn<Agent, Done>(
        (uint32_t)ExportID::Done);
    registry.exportColumn<render::RenderCameraArchetype, render::RenderOutput>(
        (uint32_t)ExportID::Raycast);
}

static inline void cleanupWorld(Engine &ctx)
{
    // Destroy current level entities
    LevelState &level = ctx.singleton<LevelState>();
    /*for (CountT i = 0; i < consts::numRooms; i++) {
        Room &room = level.rooms[i];
        for (CountT j = 0; j < consts::maxEntitiesPerRoom; j++) {
            if (room.entities[j] != Entity::none()) {
                ctx.destroyRenderableEntity(room.entities[j]);
            }
        }

        ctx.destroyRenderableEntity(room.walls[0]);
        ctx.destroyRenderableEntity(room.walls[1]);
        ctx.destroyRenderableEntity(room.door);
    }*/
}

static inline void initWorld(Engine &ctx)
{
    //phys::RigidBodyPhysicsSystem::reset(ctx);

    // Assign a new episode ID
    ctx.data().rng = RNG(rand::split_i(ctx.data().initRandKey,
        ctx.data().curWorldEpisode++, (uint32_t)ctx.worldID().idx));

    // Defined in src/level_gen.hpp / src/level_gen.cpp
    generateWorld(ctx);
}

// This system runs each frame and checks if the current episode is complete
// or if code external to the application has forced a reset by writing to the
// WorldReset singleton.
//
// If a reset is needed, cleanup the existing world and generate a new one.
inline void resetSystem(Engine &ctx, WorldReset &reset)
{
    int32_t should_reset = reset.reset;
    if (ctx.data().autoReset) {
        for (CountT i = 0; i < consts::numAgents; i++) {
            Entity agent = ctx.data().agents[i];
            Done done = ctx.get<Done>(agent);
            if (done.v) {
                should_reset = 1;
            }
        }
    }

    if (should_reset != 0) {
        reset.reset = 0;

        cleanupWorld(ctx);
        initWorld(ctx);
    }
}

// Translates discrete actions from the Action component to forces
// used by the physics simulation.
inline void movementSystem(Engine &ctx,
                           Action &action, 
                           Rotation &rot,
                           Position &pos,
                           AgentCamera& cam)
{
    Quat cur_rot = eulerToQuat(cam.yaw,0);
    int actionX = action.x - 1;
    int actionY = action.y - 1;
    Vector3 walkVec = cur_rot.rotateVec({(float)actionY,(float)actionX,0});
    walkVec = walkVec.length2() == 0 ? Vector3{0,0,0} : walkVec.normalize();
    walkVec *= 0.8f;

    Vector3 newVelocity = {0,0,0};
    newVelocity.x =  walkVec.x;
    newVelocity.y = walkVec.y;
    newVelocity.z = action.z - 1;

    cam.yaw += (action.rot-1)*consts::sensitivity;
    cam.yaw -= math::pi_m2 * std::floor((cam.yaw + math::pi) * (1. / math::pi_m2));

    cam.pitch += (action.vrot-1) * consts::sensitivity;
    cam.pitch = std::clamp(cam.pitch,-math::pi_d2,math::pi_d2);
    pos += newVelocity;

    ctx.get<Position>(cam.camera) = Vector3{ pos.x,pos.y,pos.z};
    ctx.get<Rotation>(cam.camera) = eulerToQuat(cam.yaw, cam.pitch);
    rot = eulerToQuat(cam.yaw, 0);

    //printf("%f,%f,%f\n",pos.x,pos.y,pos.z);
}

// Implements the grab action by casting a short ray in front of the agent
// and creating a joint constraint if a grabbable entity is hit.
inline void grabSystem(Engine &ctx,
                       Entity e,
                       Position pos,
                       Rotation rot,
                       Action action,
                       GrabState &grab)
{
    /*
    if (action.grab == 0) {
        return;
    }

    // if a grab is currently in progress, triggering the grab action
    // just releases the object
    if (grab.constraintEntity != Entity::none()) {
        ctx.destroyEntity(grab.constraintEntity);
        grab.constraintEntity = Entity::none();
        
        return;
    } 

    // Get the per-world BVH singleton component
    auto &bvh = ctx.singleton<broadphase::BVH>();
    float hit_t;
    Vector3 hit_normal;

    Vector3 ray_o = pos + 0.5f * math::up;
    Vector3 ray_d = rot.rotateVec(math::fwd);

    Entity grab_entity =
        bvh.traceRay(ray_o, ray_d, &hit_t, &hit_normal, 2.0f);

    if (grab_entity == Entity::none()) {
        return;
    }

    auto response_type = ctx.get<ResponseType>(grab_entity);
    if (response_type != ResponseType::Dynamic) {
        return;
    }

    auto entity_type = ctx.get<EntityType>(grab_entity);
    if (entity_type == EntityType::Agent) {
        return;
    }

    Entity constraint_entity = ctx.makeEntity<ConstraintData>();
    grab.constraintEntity = constraint_entity;

    Vector3 other_pos = ctx.get<Position>(grab_entity);
    Quat other_rot = ctx.get<Rotation>(grab_entity);

    Vector3 r1 = 1.25f * math::fwd + 0.5f * math::up;

    Vector3 hit_pos = ray_o + ray_d * hit_t;
    Vector3 r2 =
        other_rot.inv().rotateVec(hit_pos - other_pos);

    Quat attach1 = { 1, 0, 0, 0 };
    Quat attach2 = (other_rot.inv() * rot).normalize();

    float separation = hit_t - 1.25f;

    ctx.get<JointConstraint>(constraint_entity) = JointConstraint::setupFixed(
        e, grab_entity, attach1, attach2, r1, r2, separation);*/
}

// Animates the doors opening and closing based on OpenState
inline void setDoorPositionSystem(Engine &,
                                  Position &pos,
                                  OpenState &open_state)
{
    if (open_state.isOpen) {
        // Put underground
        if (pos.z > -4.5f) {
            pos.z += -consts::doorSpeed * consts::deltaT;
        }
    }
    else if (pos.z < 0.0f) {
        // Put back on surface
        pos.z += consts::doorSpeed * consts::deltaT;
    }
    
    if (pos.z >= 0.0f) {
        pos.z = 0.0f;
    }
}


// Checks if there is an entity standing on the button and updates
// ButtonState if so.
inline void buttonSystem(Engine &ctx,
                         Position pos,
                         ButtonState &state)
{
    AABB button_aabb {
        .pMin = pos + Vector3 { 
            -consts::buttonWidth / 2.f, 
            -consts::buttonWidth / 2.f,
            0.f,
        },
        .pMax = pos + Vector3 { 
            consts::buttonWidth / 2.f, 
            consts::buttonWidth / 2.f,
            0.25f
        },
    };

    bool button_pressed = false;
    /*RigidBodyPhysicsSystem::findEntitiesWithinAABB(
            ctx, button_aabb, [&](Entity) {
        button_pressed = true;
    });*/

    state.isPressed = button_pressed;
}

// Check if all the buttons linked to the door are pressed and open if so.
// Optionally, close the door if the buttons aren't pressed.
inline void doorOpenSystem(Engine &ctx,
                           OpenState &open_state,
                           const DoorProperties &props)
{
    bool all_pressed = true;
    for (int32_t i = 0; i < props.numButtons; i++) {
        Entity button = props.buttons[i];
        all_pressed = all_pressed && ctx.get<ButtonState>(button).isPressed;
    }

    if (all_pressed) {
        open_state.isOpen = true;
    } else if (!props.isPersistent) {
        open_state.isOpen = false;
    }
}

// Make the agents easier to control by zeroing out their velocity
// after each step.
/*
inline void agentZeroVelSystem(Engine &,
                               Velocity &vel,
                               Action &)
{
    vel.linear.x = 0;
    vel.linear.y = 0;
    vel.linear.z = fminf(vel.linear.z, 0);

    vel.angular = Vector3::zero();
}*/

static inline float distObs(float v)
{
    return v / consts::worldLength;
}

static inline float globalPosObs(float v)
{
    return v / consts::worldLength;
}

static inline float angleObs(float v)
{
    return v / math::pi;
}

// Translate xy delta to polar observations for learning.
static inline PolarObservation xyToPolar(Vector3 v)
{
    Vector2 xy { v.x, v.y };

    float r = xy.length();

    // Note that this is angle off y-forward
    float theta = atan2f(xy.x, xy.y);

    return PolarObservation {
        .r = distObs(r),
        .theta = angleObs(theta),
    };
}

static inline float encodeType(EntityType type)
{
    return (float)type / (float)EntityType::NumTypes;
}

static inline float computeZAngle(Quat q)
{
    float siny_cosp = 2.f * (q.w * q.z + q.x * q.y);
    float cosy_cosp = 1.f - 2.f * (q.y * q.y + q.z * q.z);
    return atan2f(siny_cosp, cosy_cosp);
}

// This system packages all the egocentric observations together 
// for the policy inputs.
inline void collectObservationsSystem(Engine &ctx,
                                      Position pos,
                                      Rotation rot,
                                      const Progress &progress,
                                      const GrabState &grab,
                                      const OtherAgents &other_agents,
                                      SelfObservation &self_obs,
                                      PartnerObservations &partner_obs,
                                      RoomEntityObservations &room_ent_obs,
                                      DoorObservation &door_obs)
{
    CountT cur_room_idx = CountT(pos.y / consts::roomLength);
    cur_room_idx = std::max(CountT(0), 
        std::min(consts::numRooms - 1, cur_room_idx));

    self_obs.roomX = pos.x / (consts::worldWidth / 2.f);
    self_obs.roomY = (pos.y - cur_room_idx * consts::roomLength) /
        consts::roomLength;
    self_obs.globalX = globalPosObs(pos.x);
    self_obs.globalY = globalPosObs(pos.y);
    self_obs.globalZ = globalPosObs(pos.z);
    self_obs.maxY = globalPosObs(progress.maxY);
    self_obs.theta = angleObs(computeZAngle(rot));
    self_obs.isGrabbing = grab.constraintEntity != Entity::none() ?
        1.f : 0.f;

    Quat to_view = rot.inv();

#pragma unroll
    for (CountT i = 0; i < consts::numAgents - 1; i++) {
        Entity other = other_agents.e[i];

        Vector3 other_pos = ctx.get<Position>(other);
        GrabState other_grab = ctx.get<GrabState>(other);
        Vector3 to_other = other_pos - pos;

        partner_obs.obs[i] = {
            .polar = xyToPolar(to_view.rotateVec(to_other)),
            .isGrabbing = other_grab.constraintEntity != Entity::none() ?
                1.f : 0.f,
        };
    }

    const LevelState &level = ctx.singleton<LevelState>();
    /*const Room &room = level.rooms[cur_room_idx];

    for (CountT i = 0; i < consts::maxEntitiesPerRoom; i++) {
        Entity entity = room.entities[i];

        EntityObservation ob;
        if (entity == Entity::none()) {
            ob.polar = { 0.f, 1.f };
            ob.encodedType = encodeType(EntityType::None);
        } else {
            Vector3 entity_pos = ctx.get<Position>(entity);
            EntityType entity_type = ctx.get<EntityType>(entity);

            Vector3 to_entity = entity_pos - pos;
            ob.polar = xyToPolar(to_view.rotateVec(to_entity));
            ob.encodedType = encodeType(entity_type);
        }

        room_ent_obs.obs[i] = ob;
    }

    Entity cur_door = room.door;
    Vector3 door_pos = ctx.get<Position>(cur_door);
    OpenState door_open_state = ctx.get<OpenState>(cur_door);

    door_obs.polar = xyToPolar(to_view.rotateVec(door_pos - pos));
    door_obs.isOpen = door_open_state.isOpen ? 1.f : 0.f;*/
}

// Launches consts::numLidarSamples per agent.
// This system is specially optimized in the GPU version:
// a warp of threads is dispatched for each invocation of the system
// and each thread in the warp traces one lidar ray for the agent.
inline void lidarSystem(Engine &ctx,
                        Entity e,
                        Lidar &lidar)
{
    /*
    Vector3 pos = ctx.get<Position>(e);
    Quat rot = ctx.get<Rotation>(e);
    auto &bvh = ctx.singleton<broadphase::BVH>();

    Vector3 agent_fwd = rot.rotateVec(math::fwd);
    Vector3 right = rot.rotateVec(math::right);

    auto traceRay = [&](int32_t idx) {
        float theta = 2.f * math::pi * (
            float(idx) / float(consts::numLidarSamples)) + math::pi / 2.f;
        float x = cosf(theta);
        float y = sinf(theta);

        Vector3 ray_dir = (x * right + y * agent_fwd).normalize();

        float hit_t;
        Vector3 hit_normal;
        Entity hit_entity =
            bvh.traceRay(pos + 0.5f * math::up, ray_dir, &hit_t,
                         &hit_normal, 200.f);

        if (hit_entity == Entity::none()) {
            lidar.samples[idx] = {
                .depth = 0.f,
                .encodedType = encodeType(EntityType::None),
            };
        } else {
            EntityType entity_type = ctx.get<EntityType>(hit_entity);

            lidar.samples[idx] = {
                .depth = distObs(hit_t),
                .encodedType = encodeType(entity_type),
            };
        }
    };


    // MADRONA_GPU_MODE guards GPU specific logic
#ifdef MADRONA_GPU_MODE
    // Can use standard cuda variables like threadIdx for 
    // warp level programming
    int32_t idx = threadIdx.x % 32;

    if (idx < consts::numLidarSamples) {
        traceRay(idx);
    }
#else
    for (CountT i = 0; i < consts::numLidarSamples; i++) {
        traceRay(i);
    }
#endif*/
}

inline void raycastSystem(Engine &ctx,
                        Entity e,
                        RaycastObservation &raycast,
                        AgentCamera& camera,
                        Position& position)
{
    /*printf("%d\n",ctx.worldID().idx);
    Vector3 pos = ctx.get<Position>(e) + Vector3{0,0, 0};
    Quat rot = eulerToQuat(camera.yaw,camera.pitch);

    Vector3 agent_fwd = rot.rotateVec(math::fwd);
    Vector3 right = rot.rotateVec(math::right);

    Vector3 ray_start = pos;
    Vector3 lookAt = eulerToQuat(camera.yaw, camera.pitch).rotateVec({ 0,1,0 });
    constexpr float theta = toRadians(90);
    const float h = tanf(theta/2);
    const auto viewport_height = 2 * h;
    const auto viewport_width = viewport_height;
    const auto forward = lookAt.normalize();
    auto u = cross({ 0,0,1 }, forward).normalize();

    u = eulerToQuat(camera.yaw, camera.pitch).rotateVec({ 1,0,0 });
    auto v = cross(forward, u).normalize();
    auto horizontal =  u *  viewport_width;
    auto vertical = v * viewport_height;
    auto lower_left_corner = ray_start - horizontal / 2 - vertical / 2 + forward;

    auto traceRay = [&](int32_t idx, int32_t subthread) {
        int pixelY = idx / consts::rayObservationWidth;
        int pixelX = idx % consts::rayObservationWidth;
        float v = ((float)pixelY) / consts::rayObservationHeight;
        float u = ((float)pixelX) / consts::rayObservationWidth;

        Vector3 ray_dir = lower_left_corner + u * horizontal + v*vertical - ray_start;
        ray_dir = ray_dir.normalize();

        float t;
        Vector3 normal = {0,0,0};
        //(madrona::phys2::MeshBVH*)(ctx.data().bvh)->traceRay(ray_start,ray_dir,&t,&normal);
        //bool hit = (ctx.data().bvh)->traceRay(ray_start,ray_dir,&t,&normal);
        Vector3 lightDir = Vector3{0.5,0.5,0.5};
        lightDir = lightDir.normalize();
        float lightness = normal.dot(lightDir);
        if(normal.length2() != 0){
            lightness += 0.005;
        }
        //if(idx == 0)
#ifdef MADRONA_GPU_MODE
       // if(ctx.worldID().idx ==0){
         //   printf("%d,%d,%d\n",threadIdx.x,pixelX,pixelY);
        //}
#endif
        //printf("%d,%f,%f,%f\n",ctx.worldID().idx,normal.x,normal.y,normal.z);

        if (hit && subthread == 0) {
            raycast.raycast[pixelX][pixelY][0] = (normal.x * 0.5f + 0.5f) * 255;
            raycast.raycast[pixelX][pixelY][1] = (normal.y * 0.5f + 0.5f) * 255;
            raycast.raycast[pixelX][pixelY][2] = (normal.z * 0.5f + 0.5f) * 255;
        }else if(subthread == 0){
            raycast.raycast[pixelX][pixelY][0] = 0;
            raycast.raycast[pixelX][pixelY][1] = 0;
            raycast.raycast[pixelX][pixelY][2] = 0;
        }
        //raycast.raycast[pixelX][pixelY][0] = lightness*255;
        //raycast.raycast[pixelX][pixelY][1] = lightness*255;
        //raycast.raycast[pixelX][pixelY][2] = lightness*255;
    };


#ifdef MADRONA_GPU_MODE
    int32_t idx = threadIdx.x;
    //4 threads per ray
    int32_t subgroup = idx / 4;
    //printf("dispatch %d,%d,%d,%d\n",ctx.worldID().idx,threadIdx.x,threadIdx.y,threadIdx.z);
    const int32_t mwgpu_warp_id = threadIdx.x / 32;
    const int32_t mwgpu_warp_lane = threadIdx.x % 32;

    if (idx < 256) {
        for(int32_t rays = 0;rays<16;rays++){
            traceRay(idx*16+rays,0);
        }
    }
#else
    for (CountT i = 0; i < consts::rayObservationWidth * consts::rayObservationHeight; i++) {
        traceRay(i,0);
    }
#endif*/
}

// Computes reward for each agent and keeps track of the max distance achieved
// so far through the challenge. Continuous reward is provided for any new
// distance achieved.
inline void rewardSystem(Engine &,
                         Position pos,
                         Progress &progress,
                         Reward &out_reward)
{
    // Just in case agents do something crazy, clamp total reward
    float reward_pos = fminf(pos.y, consts::worldLength * 2);

    float old_max_y = progress.maxY;

    float new_progress = reward_pos - old_max_y;

    float reward;
    if (new_progress > 0) {
        reward = new_progress * consts::rewardPerDist;
        progress.maxY = reward_pos;
    } else {
        reward = consts::slackReward;
    }

    out_reward.v = reward;
}

// Each agent gets a small bonus to it's reward if the other agent has
// progressed a similar distance, to encourage them to cooperate.
// This system reads the values of the Progress component written by
// rewardSystem for other agents, so it must run after.
inline void bonusRewardSystem(Engine &ctx,
                              OtherAgents &others,
                              Progress &progress,
                              Reward &reward)
{
    bool partners_close = true;
    for (CountT i = 0; i < consts::numAgents - 1; i++) {
        Entity other = others.e[i];
        Progress other_progress = ctx.get<Progress>(other);

        if (fabsf(other_progress.maxY - progress.maxY) > 2.f) {
            partners_close = false;
        }
    }

    if (partners_close && reward.v > 0.f) {
        reward.v *= 1.25f;
    }
}

// Keep track of the number of steps remaining in the episode and
// notify training that an episode has completed by
// setting done = 1 on the final step of the episode
inline void stepTrackerSystem(Engine &,
                              StepsRemaining &steps_remaining,
                              Done &done)
{
    int32_t num_remaining = --steps_remaining.t;
    if (num_remaining == consts::episodeLen - 1) {
        done.v = 0;
    } else if (num_remaining == 0) {
        done.v = 1;
    }

}

inline void testerSystem(Engine& ctx, render::BVHModel& r,render::InstanceData& id){
    //printf("testerwhat %p, %p, %d,%d  %f,%f,%f\n",ctx.data().bvhs,r.ptr,id.objectID,id.worldIDX, id.position.x,id.position.y,id.position.z);
}

// Helper function for sorting nodes in the taskgraph.
// Sorting is only supported / required on the GPU backend,
// since the CPU backend currently keeps separate tables for each world.
// This will likely change in the future with sorting required for both
// environments
#ifdef MADRONA_GPU_MODE
template <typename ArchetypeT>
TaskGraph::NodeID queueSortByWorld(TaskGraph::Builder &builder,
                                   Span<const TaskGraph::NodeID> deps)
{
    auto sort_sys =
        builder.addToGraph<SortArchetypeNode<ArchetypeT, WorldID>>(
            deps);
    auto post_sort_reset_tmp =
        builder.addToGraph<ResetTmpAllocNode>({sort_sys});

    return post_sort_reset_tmp;
}
#endif

// Build the task graph
void Sim::setupTasks(TaskGraphBuilder &builder, const Config &cfg)
{
    // Turn policy actions into movement
    auto move_sys = builder.addToGraph<ParallelForNode<Engine,
        movementSystem,
            Action,
            Rotation,
            Position,
            AgentCamera
        >>({});
    auto test_sys = builder.addToGraph<ParallelForNode<Engine,
        testerSystem,
            render::BVHModel,render::InstanceData
        >>({});
/*
    // Scripted door behavior
    auto set_door_pos_sys = builder.addToGraph<ParallelForNode<Engine,
        setDoorPositionSystem,
            Position,
            OpenState
        >>({move_sys});

    // Build BVH for broadphase / raycasting
    auto broadphase_setup_sys =
        phys::RigidBodyPhysicsSystem::setupBroadphaseTasks(builder, 
                                                           {set_door_pos_sys});

    // Grab action, post BVH build to allow raycasting
    auto grab_sys = builder.addToGraph<ParallelForNode<Engine,
        grabSystem,
            Entity,
            Position,
            Rotation,
            Action,
            GrabState
        >>({broadphase_setup_sys});

    // Physics collision detection and solver
    auto substep_sys = phys::RigidBodyPhysicsSystem::setupSubstepTasks(builder,
        {grab_sys}, consts::numPhysicsSubsteps);

    // Improve controllability of agents by setting their velocity to 0
    // after physics is done.
    auto agent_zero_vel = builder.addToGraph<ParallelForNode<Engine,
        agentZeroVelSystem, Velocity, Action>>(
            {substep_sys});

    // Finalize physics subsystem work
    auto phys_done = phys::RigidBodyPhysicsSystem::setupCleanupTasks(
        builder, {agent_zero_vel});

    // Check buttons
    auto button_sys = builder.addToGraph<ParallelForNode<Engine,
        buttonSystem,
            Position,
            ButtonState
        >>({phys_done});

    // Set door to start opening if button conditions are met
    auto door_open_sys = builder.addToGraph<ParallelForNode<Engine,
        doorOpenSystem,
            OpenState,
            DoorProperties
        >>({button_sys});

    // Compute initial reward now that physics has updated the world state
    auto reward_sys = builder.addToGraph<ParallelForNode<Engine,
         rewardSystem,
            Position,bonus_reward_
            Progress,
            Reward
        >>({move_sys});

    // Assign partner's reward
    auto bonus_reward_sys = builder.addToGraph<ParallelForNode<Engine,
         bonusRewardSystem,
            OtherAgents,
            Progress,
            Reward
        >>({reward_sys});
*/
    // Check if the episode is over
    auto done_sys = builder.addToGraph<ParallelForNode<Engine,
        stepTrackerSystem,
            StepsRemaining,
            Done
        >>({move_sys});

    // Conditionally reset the world if the episode is over
    auto reset_sys = builder.addToGraph<ParallelForNode<Engine,
        resetSystem,
            WorldReset
        >>({done_sys});

    auto clear_tmp = builder.addToGraph<ResetTmpAllocNode>({reset_sys});
    (void)clear_tmp;

#ifdef MADRONA_GPU_MODE
    // RecycleEntitiesNode is required on the GPU backend in order to reclaim
    // deleted entity IDs.
    auto recycle_sys = builder.addToGraph<RecycleEntitiesNode>({reset_sys});
    (void)recycle_sys;
#endif

    // This second BVH build is a limitation of the current taskgraph API.
    // It's only necessary if the world was reset, but we don't have a way
    // to conditionally queue taskgraph nodes yet.
    //auto post_reset_broadphase = phys::RigidBodyPhysicsSystem::setupBroadphaseTasks(
    //    builder, {reset_sys});

    // Finally, collect observations for the next step.
    auto collect_obs = builder.addToGraph<ParallelForNode<Engine,
        collectObservationsSystem,
            Position,
            Rotation,
            Progress,
            GrabState,
            OtherAgents,
            SelfObservation,
            PartnerObservations,
            RoomEntityObservations,
            DoorObservation
        >>({reset_sys});
#ifdef MADRONA_GPU_MODE
    auto raycast = builder.addToGraph<CustomParallelForNode<Engine,
        raycastSystem, 256, 1,
#else
    auto raycast = builder.addToGraph<ParallelForNode<Engine,
            raycastSystem,
#endif
            Entity,
            RaycastObservation,
            AgentCamera,
            Position
        >>({reset_sys});

    // The lidar system
/*#ifdef MADRONA_GPU_MODE
    // Note the use of CustomParallelForNode to create a taskgraph node
    // that launches a warp of threads (32) for each invocation (1).
    // The 32, 1 parameters could be changed to 32, 32 to create a system
    // that cooperatively processes 32 entities within a warp.
    auto lidar = builder.addToGraph<CustomParallelForNode<Engine,
        lidarSystem, 256, 1,
#else
    auto lidar = builder.addToGraph<ParallelForNode<Engine,
        lidarSystem,
#endif
            Entity,
            Lidar
        >>({post_reset_broadphase});
*/
    RenderingSystem::setupTasks(builder, {reset_sys});

#ifdef MADRONA_GPU_MODE
    // Sort entities, this could be conditional on reset like the second
    // BVH build above.
    auto sort_agents = queueSortByWorld<Agent>(
        builder, {collect_obs});
    auto sort_phys_objects = queueSortByWorld<PhysicsEntity>(
        builder, {sort_agents});
    auto sort_buttons = queueSortByWorld<ButtonEntity>(
        builder, {sort_phys_objects});
    auto sort_walls = queueSortByWorld<DoorEntity>(
        builder, {sort_buttons});
    (void)sort_walls;
#else
    (void)collect_obs;
#endif
}

Sim::Sim(Engine &ctx,
         const Config &cfg,
         const WorldInit &)
    : WorldBase(ctx)
{
    // Currently the physics system needs an upper bound on the number of
    // entities that will be stored in the BVH. We plan to fix this in
    // a future release.
    constexpr CountT max_total_entities = consts::numAgents +
        consts::numRooms * (consts::maxEntitiesPerRoom + 3) +
        4; // side walls + floor
    
    importedInstances = cfg.importedInstances;
    numImportedInstances = cfg.numImportedInstances;

    /*phys::RigidBodyPhysicsSystem::init(ctx, cfg.rigidBodyObjMgr,
        consts::deltaT, consts::numPhysicsSubsteps, -9.8f * math::up,
        max_total_entities, max_total_entities * max_total_entities / 2,
        consts::numAgents);
    */
    initRandKey = cfg.initRandKey;
    autoReset = cfg.autoReset;

    enableRender = cfg.renderBridge != nullptr;

    if (enableRender) {
        RenderingSystem::init(ctx, cfg.renderBridge);
    }

    curWorldEpisode = 0;
    bvhs = cfg.bvhs;

    // Creates agents, walls, etc.
    createPersistentEntities(ctx);

    // Generate initial world state
    initWorld(ctx);
}

// This declaration is needed for the GPU backend in order to generate the
// CUDA kernel for world initialization, which needs to be specialized to the
// application's world data type (Sim) and config and initialization types.
// On the CPU it is a no-op.
MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Sim::Config, Sim::WorldInit);

}
