#include "mgr.hpp"
#include "types.hpp"

#include <cstdio>
#include <chrono>
#include <string>
#include <filesystem>
#include <fstream>
#include <random>
#include <iostream>

#include <madrona/heap_array.hpp>

using namespace madrona;

[[maybe_unused]] static void saveWorldActions(
    const HeapArray<int32_t> &action_store,
    int32_t total_num_steps,
    int32_t world_idx)
{
    const int32_t *world_base = action_store.data() + world_idx * total_num_steps * 2 * 3;

    std::ofstream f("/tmp/actions", std::ios::binary);
    f.write((char *)world_base,
            sizeof(uint32_t) * total_num_steps * 2 * 3);
}

int main(int argc, char *argv[])
{
    using namespace madEscape;

    if (argc < 4) {
        fprintf(stderr, "%s TYPE NUM_WORLDS NUM_STEPS [--rand-actions]\n", argv[0]);
        return -1;
    }
    std::string type(argv[1]);

    ExecMode exec_mode;
    if (type == "CPU") {
        exec_mode = ExecMode::CPU;
    } else if (type == "CUDA") {
        exec_mode = ExecMode::CUDA;
    } else {
        fprintf(stderr, "Invalid ExecMode\n");
        return -1;
    }

    uint64_t num_worlds = std::stoul(argv[2]);
    uint64_t num_steps = std::stoul(argv[3]);

    HeapArray<int32_t> action_store(
        num_worlds * 2 * num_steps * 3);

    bool rand_actions = false;
    if (argc >= 5) {
        if (std::string(argv[4]) == "--rand-actions") {
            rand_actions = true;
        }
    }

    int record = -1;
    if (argc >= 6) {
        record = atoi(argv[5]);
    }

    Manager mgr({
        .execMode = exec_mode,
        .gpuID = 0,
        .numWorlds = (uint32_t)num_worlds,
        .autoReset = false,
        .enableBatchRenderer = true,
    });

    std::random_device rd;
    std::mt19937 rand_gen(rd());
    std::uniform_int_distribution<int32_t> act_rand(0, 2);

    auto start = std::chrono::system_clock::now();

    unsigned char* print_ptr;
    if(record != -1) {
#ifdef MADRONA_CUDA_SUPPORT
        int64_t num_bytes = sizeof(RaycastObservation);
        print_ptr = (unsigned char*)cu::allocReadback(num_bytes);
#else
        print_ptr = nullptr;
#endif
    }


    for (CountT i = 0; i < (CountT)num_steps; i++) {
        if (rand_actions) {
            for (CountT j = 0; j < (CountT)num_worlds; j++) {
                for (CountT k = 0; k < 2; k++) {
                    int32_t x = act_rand(rand_gen);
                    int32_t y = act_rand(rand_gen);
                    int32_t r = act_rand(rand_gen);

                    mgr.setAction(j, k, x, y, r, 0,act_rand(rand_gen),1,1,2,1);
                    
                    int64_t base_idx = j * num_steps * 2 * 3 + i * 2 * 3 + k * 3;
                    action_store[base_idx] = x;
                    action_store[base_idx + 1] = y;
                    action_store[base_idx + 2] = r;
                }
            }
        }

        if(record != -1) {
            auto raycastTensor = (render::RenderOutput *) (mgr.raycastTensor().devicePtr());
            raycastTensor = raycastTensor + (record * 2) + (std::max(1, 0));
            if (exec_mode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
                cudaMemcpy(print_ptr, raycastTensor,
                   1 * sizeof(render::RenderOutput),
                   cudaMemcpyDeviceToHost);
                raycastTensor = (render::RenderOutput*)print_ptr;
                printf("%d\n",raycastTensor->output[0][0][0]);
#else

#endif

            }
            std::ofstream out("out_" + std::to_string(i) + ".ppm", std::ios::out | std::ios::binary);
            int width = 64;
            int height = 64;
            out.write("P3\n", 3);
            out.write("64 64\n", 6);
            out.write("255\n", 4);
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    //std::cout << std::to_string(raycastTensor->output[i][j][0])  << std::endl;
                    auto out2 = std::to_string(raycastTensor->output[i][j][0]) + " " +
                                std::to_string(raycastTensor->output[i][j][1]) + " " +
                                std::to_string(raycastTensor->output[i][j][2]) + "\n";
                    out.write(out2.c_str(), out2.size());
                }
            }
            out.close();
        }

        mgr.step();
    }

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    float fps = (double)num_steps * (double)num_worlds / elapsed.count();
    printf("FPS %f\n", fps);
}
