# MADRONA_TIMING_FILE=out.json MADRONA_MWGPU_KERNEL_CACHE=/tmp/luc_cache MADRONA_RENDER_MODE=2 MADRONA_MWGPU_FORCE_DEBUG=1 ./headless CUDA 512 10

import os
from dataclasses import dataclass

RAST_NO = 1
RT_NO = 2

@dataclass
class RenderMode:
    name: str
    no: int
    track_split: int
    wide: int
    lbvh: int


# Configure things to profile
render_cfgs = [ 
    RenderMode(name="RastCull", no=RAST_NO, track_split=0, wide=0, lbvh=0)
]

render_resolutions = [ 0, 1, 2 ]

for env in ["0", "1", "2"]:
    env_name = f"env{env}"

    for render_cfg in render_cfgs:
        for res in render_resolutions:
            for num_worlds in [512, 1024, 2048]:
                actual_res = 32 * (2 ** res)

                output_file_name = "" 
                if render_cfg.track_split == 1:
                    output_file_name = f"{env_name}/out_{render_cfg.name}_{num_worlds}_{actual_res}x{actual_res}_split.json"
                else:
                    output_file_name = f"{env_name}/out_{render_cfg.name}_{num_worlds}_{actual_res}x{actual_res}.json"

                actual_num_worlds = num_worlds

                if render_cfg.track_split:
                    actual_num_worlds = num_worlds / 64

                command = f"MADRONA_MERGE_ALL=0 "
                command = command + f"MADRONA_LOADED_ENV={env} "
                command = command + f"MADRONA_TRACE_TEST=0 "
                command = command + f"MADRONA_TRACK_TRACE_SPLIT={render_cfg.track_split} "
                command = command + f"MADRONA_RENDER_RESOLUTION={res} "
                command = command + f"MADRONA_TIMING_FILE={output_file_name} "
                command = command + f"MADRONA_MWGPU_KERNEL_CACHE=cache "
                command = command + f"MADRONA_RENDER_MODE={render_cfg.no} "
                command = command + f"MADRONA_LBVH={render_cfg.lbvh} "
                command = command + f"MADRONA_WIDEN={render_cfg.wide} "
                command = command + f"./build/headless CUDA {actual_num_worlds} 32"

                print(command)
                os.system(command)





"""
render_modes = { "RaycastBinnedSAHWide": 2 }
render_resolutions = [ 0, 1, 2 ]

for env in ["0", "1", "2"]:
    env_name = f"env{env}"

    for render_mode in render_modes:
        run_index = 0

        for res in render_resolutions:
            for num_worlds in [512, 1024, 2048]:
                render_mode_number = render_modes[render_mode]

                actual_res = 32 * (2 ** res)

                output_file_name = f"{env_name}/out_{render_mode}_{num_worlds}_{actual_res}x{actual_res}.json"

                command = f"MADRONA_MERGE_ALL=0 MADRONA_LOADED_ENV={env} MADRONA_TRACE_TEST=0 MADRONA_TRACK_TRACE_SPLIT=0 MADRONA_RENDER_RESOLUTION={res} MADRONA_TIMING_FILE={output_file_name} MADRONA_MWGPU_KERNEL_CACHE=cache MADRONA_RENDER_MODE={render_mode_number} ./build/headless CUDA {num_worlds} 32"

                os.system(command)

                run_index = run_index + 1
"""
