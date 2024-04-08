# MADRONA_TIMING_FILE=out.json MADRONA_MWGPU_KERNEL_CACHE=/tmp/luc_cache MADRONA_RENDER_MODE=2 MADRONA_MWGPU_FORCE_DEBUG=1 ./headless CUDA 512 10

import os

"""
render_modes = { "raster": 1, "raycast": 2 }
render_resolutions = [ 0, 1, 2 ]

for render_mode in render_modes:
    run_index = 0

    for res in render_resolutions:
        for num_worlds in [512, 1024, 2048]:
            render_mode_number = render_modes[render_mode]

            actual_res = 32 * (2 ** res)

            output_file_name = f"env3/out_{render_mode}_{num_worlds}_{actual_res}x{actual_res}.json"
            output_fps_file_name = f"out_{render_mode}_{num_worlds}_{actual_res}x{actual_res}_fps.txt";

            command = f"MADRONA_RENDER_RESOLUTION={res} MADRONA_FPS_FILE={output_fps_file_name} MADRONA_TIMING_FILE={output_file_name} MADRONA_MWGPU_KERNEL_CACHE=cache MADRONA_RENDER_MODE={render_mode_number} ./build/headless CUDA {num_worlds} 10"

            os.system(command)

            run_index = run_index + 1
"""

render_modes = { "raycast": 2 }
render_resolutions = [ 0, 1, 2 ]

for render_mode in render_modes:
    run_index = 0

    for res in render_resolutions:
        for num_worlds in [512, 1024, 2048]:
            render_mode_number = render_modes[render_mode]

            actual_res = 32 * (2 ** res)

            output_file_name = f"env3/out_{render_mode}_slowbuild_{num_worlds}_{actual_res}x{actual_res}.json"

            command = f"MADRONA_RENDER_RESOLUTION={res} MADRONA_TIMING_FILE={output_file_name} MADRONA_MWGPU_KERNEL_CACHE=cache MADRONA_RENDER_MODE={render_mode_number} ./build/headless CUDA {num_worlds} 10"

            os.system(command)

            run_index = run_index + 1
