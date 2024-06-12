#include <madrona/viz/viewer.hpp>
#include <madrona/render/render_mgr.hpp>
#include <madrona/window.hpp>

#include "sim.hpp"
#include "mgr.hpp"
#include "types.hpp"

#include <filesystem>
#include <fstream>
#include <imgui.h>

#include <stb_image_write.h>

using namespace madrona;
using namespace madrona::viz;

void transposeImage(char *output, 
                    const char *input,
                    uint32_t res,
                    uint32_t comp)
{
    for (uint32_t y = 0; y < res; ++y) {
        for (uint32_t x = 0; x < res; ++x) {
            output[3*(y + x * res) + 0] = input[3*(x + y * res) + 0];
            output[3*(y + x * res) + 1] = input[3*(x + y * res) + 1];
            output[3*(y + x * res) + 2] = input[3*(x + y * res) + 2];
        }
    }
}

static HeapArray<int32_t> readReplayLog(const char *path)
{
    std::ifstream replay_log(path, std::ios::binary);
    replay_log.seekg(0, std::ios::end);
    int64_t size = replay_log.tellg();
    replay_log.seekg(0, std::ios::beg);

    HeapArray<int32_t> log(size / sizeof(int32_t));

    replay_log.read((char *)log.data(), (size / sizeof(int32_t)) * sizeof(int32_t));

    return log;
}

int main(int argc, char *argv[])
{
    using namespace madEscape;

    constexpr int64_t num_views = 2;

    printf("Started in: \n");
    // Read command line arguments
    uint32_t num_worlds = 1;
    if (argc >= 2) {
        num_worlds = (uint32_t)atoi(argv[1]);
    }

    ExecMode exec_mode = ExecMode::CPU;
    if (argc >= 3) {
        if (!strcmp("--cpu", argv[2])) {
            exec_mode = ExecMode::CPU;
        } else if (!strcmp("--cuda", argv[2])) {
            exec_mode = ExecMode::CUDA;
        }
    }

    // Setup replay log
    const char *replay_log_path = nullptr;
    if (argc >= 4) {
        replay_log_path = argv[3];
    }

    auto replay_log = Optional<HeapArray<int32_t>>::none();
    uint32_t cur_replay_step = 0;
    uint32_t num_replay_steps = 0;
    if (replay_log_path != nullptr) {
        replay_log = readReplayLog(replay_log_path);
        num_replay_steps = replay_log->size() / (num_worlds * num_views * 4);
    }

    // Render mode 0 is no rendering
    // Render mode 1 is rasterization.
    // Render mode 2 is raycasting.
    auto *render_mode = getenv("MADRONA_RENDER_MODE");

    bool enable_batch_renderer =
#ifdef MADRONA_MACOS
        false;
#else
        render_mode[0] == '1';
#endif

    //WindowManager wm {WindowManager::Config{.enableRenderAPIValidation=true,.renderBackendSelect =
    //        render::APIBackendSelect::Auto}};
    WindowManager wm {};
    WindowHandle window = wm.makeWindow("Escape Room", 1080, 720);
    printf("Here\n");
    render::GPUHandle render_gpu = wm.initGPU(0, { window.get() });


    printf("premanage: \n");

    auto *resolution_str = getenv("MADRONA_RENDER_RESOLUTION");

    uint32_t raycast_output_resolution = std::stoi(resolution_str);

    // Create the simulation manager
    Manager mgr({
        .execMode = exec_mode,
        .gpuID = 0,
        .numWorlds = num_worlds,
        .randSeed = 5,
        .autoReset = replay_log.has_value(),
        .enableBatchRenderer = enable_batch_renderer,
        .batchRenderViewWidth = raycast_output_resolution,
        .batchRenderViewHeight = raycast_output_resolution,
        .extRenderAPI = wm.gpuAPIManager().backend(),
        .extRenderDev = render_gpu.device(),
        .raycastOutputResolution = raycast_output_resolution,
    });
    printf("postmanage: \n");

    float camera_move_speed = 10.f;

    math::Vector3 initial_camera_position = { 0, consts::worldLength / 2.f, 30 };

    math::Quat initial_camera_rotation =
        (math::Quat::angleAxis(-math::pi / 2.f, math::up) *
        math::Quat::angleAxis(-math::pi / 2.f, math::right)).normalize();


    // Create the viewer viewer
    viz::Viewer viewer(mgr.getRenderManager(), window.get(), {
        .numWorlds = num_worlds,
        .simTickRate = 120,
        .cameraMoveSpeed = camera_move_speed * 7.f,
        .cameraPosition = initial_camera_position,
        .cameraRotation = initial_camera_rotation,
    });

    // Replay step
    auto replayStep = [&]() {
        if (cur_replay_step == num_replay_steps - 1) {
            return true;
        }

        printf("Step: %u\n", cur_replay_step);

        for (uint32_t i = 0; i < num_worlds; i++) {
            for (uint32_t j = 0; j < num_views; j++) {
                uint32_t base_idx = 0;
                base_idx = 4 * (cur_replay_step * num_views * num_worlds +
                    i * num_views + j);

                int32_t move_amount = (*replay_log)[base_idx];
                int32_t move_angle = (*replay_log)[base_idx + 1];
                int32_t turn = (*replay_log)[base_idx + 2];
                int32_t g = (*replay_log)[base_idx + 3];

                printf("%d, %d: %d %d %d %d\n",
                       i, j, move_amount, move_angle, turn, g);
                mgr.setAction(i, j, move_amount, move_angle, turn, g,1,1,1,1,1);
            }
        }

        cur_replay_step++;

        return false;
    };

    // Printers
#if 0
    auto self_printer = mgr.selfObservationTensor().makePrinter();
    auto partner_printer = mgr.partnerObservationsTensor().makePrinter();
    auto room_ent_printer = mgr.roomEntityObservationsTensor().makePrinter();
    auto door_printer = mgr.doorObservationTensor().makePrinter();
    auto lidar_printer = mgr.lidarTensor().makePrinter();
    auto steps_remaining_printer = mgr.stepsRemainingTensor().makePrinter();
    auto reward_printer = mgr.rewardTensor().makePrinter();
#endif

    auto printObs = [&]() {
        /*printf("Self\n");
        self_printer.print();

        printf("Partner\n");
        partner_printer.print();

        printf("Room Entities\n");
        room_ent_printer.print();

        printf("Door\n");
        door_printer.print();

        printf("Lidar\n");
        lidar_printer.print();

        printf("Steps Remaining\n");
        steps_remaining_printer.print();

        printf("Reward\n");
        reward_printer.print();

        printf("\n");*/
    };



    // Main loop for the viewer viewer
    viewer.loop(
    [&mgr](CountT world_idx, const Viewer::UserInput &input)
    {
        // printf("new frame\n");

        using Key = Viewer::KeyboardKey;
        if (input.keyHit(Key::R)) {
            mgr.triggerReset(world_idx);
        }
    },
    [&mgr](CountT world_idx, CountT agent_idx,
           const Viewer::UserInput &input)
    {
        using Key = Viewer::KeyboardKey;

        int32_t x = 0;
        int32_t y = 0;
        int32_t r = 2;
        int32_t g = 0;

        bool shift_pressed = input.keyPressed(Key::Shift);

        if (input.keyPressed(Key::W)) {
            y += 1;
        }
        if (input.keyPressed(Key::S)) {
            y -= 1;
        }

        if (input.keyPressed(Key::D)) {
            x += 1;
        }
        if (input.keyPressed(Key::A)) {
            x -= 1;
        }

        if (input.keyPressed(Key::Q)) {
            r += shift_pressed ? 2 : 1;
        }
        if (input.keyPressed(Key::E)) {
            r -= shift_pressed ? 2 : 1;
        }

        if (input.keyHit(Key::G)) {
            g = 1;
        }

        int32_t move_amount;
        if (x == 0 && y == 0) {
            move_amount = 0;
        } else if (shift_pressed) {
            move_amount = consts::numMoveAmountBuckets - 1;
        } else {
            move_amount = 1;
        }

        int32_t move_angle;
        if (x == 0 && y == 1) {
            move_angle = 0;
        } else if (x == 1 && y == 1) {
            move_angle = 1;
        } else if (x == 1 && y == 0) {
            move_angle = 2;
        } else if (x == 1 && y == -1) {
            move_angle = 3;
        } else if (x == 0 && y == -1) {
            move_angle = 4;
        } else if (x == -1 && y == -1) {
            move_angle = 5;
        } else if (x == -1 && y == 0) {
            move_angle = 6;
        } else if (x == -1 && y == 1) {
            move_angle = 7;
        } else {
            move_angle = 0;
        }

        x = 1;
        if (input.keyPressed(Key::W)) {
            x = 2;
        }
        if (input.keyPressed(Key::S)) {
            x = 0;
        }

        y = 1;
        if (input.keyPressed(Key::D)) {
            y = 2;
        }
        if (input.keyPressed(Key::A)) {
            y = 0;
        }

        int rot=1;
        if (input.keyPressed(Key::Q)) {
            rot = 2;
        }
        if (input.keyPressed(Key::E)) {
            rot = 0;
        }

        int vrot = 1;
        if (input.keyPressed(Key::T)) {
            vrot = 2;
        }
        if (input.keyPressed(Key::F)) {
            vrot = 0;
        }

        int z = 1;
        if (input.keyPressed(Key::Space)) {
            z = 2;
        }

        if (input.keyPressed(Key::Shift)) {
            z = 0;
        }

        mgr.setAction(world_idx, agent_idx, move_amount, move_angle, r, g,x,y,z,rot,vrot);
    }, [&]() {
        if (replay_log.has_value()) {
            bool replay_finished = replayStep();

            if (replay_finished) {
                viewer.stopLoop();
            }
        }

        mgr.step();

        printObs();
    }, [&]() {
        {
            uint32_t num_image_x = 1;
            uint32_t num_image_y = 1;

            uint32_t num_images_total = num_image_x * num_image_y;

            unsigned char* print_ptr;
#ifdef MADRONA_CUDA_SUPPORT
            int64_t num_bytes = 3 * raycast_output_resolution * raycast_output_resolution * num_images_total;
            print_ptr = (unsigned char*)cu::allocReadback(num_bytes);
#else
            print_ptr = nullptr;
#endif

            char *raycast_tensor = (char *)(mgr.raycastTensor().devicePtr());

            uint32_t bytes_per_image = 3 * raycast_output_resolution * raycast_output_resolution;

            uint32_t image_idx = viewer.getCurrentWorldID() * 1 + 
                std::max(viewer.getCurrentViewID(), (CountT)0);

            uint32_t base_image_idx = num_images_total * (image_idx / num_images_total);

            raycast_tensor += image_idx * bytes_per_image;

            if(exec_mode == ExecMode::CUDA){
#ifdef MADRONA_CUDA_SUPPORT
                cudaMemcpy(print_ptr, raycast_tensor,
                        num_bytes,
                        cudaMemcpyDeviceToHost);
                raycast_tensor = (char *)print_ptr;
#endif
            }

            ImGui::Begin("Raycast");

            auto draw2 = ImGui::GetWindowDrawList();
            ImVec2 windowPos = ImGui::GetWindowPos();
            char *raycasters = raycast_tensor;

#if 0
            for (int i = 0; i < 100; ++i) {
                printf("%u ", (uint8_t)raycasters[i]);
            }
            printf("\n");
#endif

            int vertOff = 70;

            float pixScale = 1;
            int extentsX = (int)(pixScale * raycast_output_resolution);
            int extentsY = (int)(pixScale * raycast_output_resolution);

            for (int image_y = 0; image_y < num_image_y; ++image_y) {
                for (int image_x = 0; image_x < num_image_x; ++image_x) {
                    for (int i = 0; i < raycast_output_resolution; i++) {
                        for (int j = 0; j < raycast_output_resolution; j++) {
                            uint32_t linear_image_idx = image_x + image_y * num_image_x;

                            uint32_t linear_idx = 3 * 
                                (j + (i + linear_image_idx * raycast_output_resolution) * raycast_output_resolution);

                            auto realColor = IM_COL32(
                                    (uint8_t)raycasters[linear_idx + 0],
                                    (uint8_t)raycasters[linear_idx + 1],
                                    (uint8_t)raycasters[linear_idx + 2],
                                    255);

                            draw2->AddRectFilled(
                                    { ((i + image_x * raycast_output_resolution) * pixScale) + windowPos.x, 
                                    ((j + image_y * raycast_output_resolution) * pixScale) + windowPos.y + vertOff }, 
                                    { ((i + 1 + image_x * raycast_output_resolution) * pixScale) + windowPos.x,   
                                    ((j + image_y * raycast_output_resolution + 1) * pixScale) + +windowPos.y + vertOff },
                                    realColor, 0, 0);
                        }
                    }
                }
            }
            ImGui::End();
        }





        {
            uint32_t num_images_total = 1;

            unsigned char* print_ptr;
            int64_t num_bytes = 3 * raycast_output_resolution * raycast_output_resolution * num_images_total;
            print_ptr = (unsigned char*)cu::allocReadback(num_bytes);

            char *raycast_tensor = (char *)(mgr.raycastTensor().devicePtr());

            uint32_t bytes_per_image = 3 * raycast_output_resolution * raycast_output_resolution;
            uint32_t row_stride_bytes = 3 * raycast_output_resolution;

            uint32_t image_idx = 0;

            uint32_t base_image_idx = num_images_total * (image_idx / num_images_total);

            raycast_tensor += image_idx * bytes_per_image;

            if(exec_mode == ExecMode::CUDA){
                cudaMemcpy(print_ptr, raycast_tensor,
                        num_bytes,
                        cudaMemcpyDeviceToHost);
                raycast_tensor = (char *)print_ptr;
            }

            char *tmp_image_memory = (char *)malloc(bytes_per_image);

            char *image_memory = (char *)malloc(bytes_per_image * num_images_total);

            uint32_t num_images_y = 1;
            uint32_t num_images_x = num_images_total / num_images_y;

            uint32_t output_num_pixels_x = num_images_x * raycast_output_resolution;

            for (uint32_t image_y = 0; image_y < num_images_y; ++image_y) {
                for (uint32_t image_x = 0; image_x < num_images_x; ++image_x) {
                    uint32_t image_idx = image_x + image_y * num_images_x;

                    const char *input_image = raycast_tensor + image_idx * bytes_per_image;

                    transposeImage(tmp_image_memory, input_image, raycast_output_resolution, 3);

                    for (uint32_t row_idx = 0; row_idx < raycast_output_resolution; ++row_idx) {
                        const char *input_row = tmp_image_memory + row_idx * row_stride_bytes;

                        uint32_t output_pixel_x = image_x * raycast_output_resolution;
                        uint32_t output_pixel_y = image_y * raycast_output_resolution + row_idx;
                        char *output_row = image_memory + 3 * (output_pixel_x + output_pixel_y * output_num_pixels_x);

                        memcpy(output_row, input_row, 3 * raycast_output_resolution);
                    }
                }
            }

            std::string file_name = std::string("out") + std::to_string(0) + ".png";
            stbi_write_png(file_name.c_str(), raycast_output_resolution * num_images_x, num_images_y * raycast_output_resolution,
                    3, image_memory, 3 * num_images_x * raycast_output_resolution);

            free(image_memory);
        }
    });
}
