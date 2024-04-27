# MADRONA_TRACK_TRACE_SPLIT=0
# HSSD_FIRST_SCENE=0 
# HSSD_NUM_SCENES=16 
# MADRONA_LBVH=0 
# MADRONA_WIDEN=1 
# MADRONA_MWGPU_FORCE_DEBUG=0 
# MADRONA_TIMING_FILE=timing.txt
# MADRONA_RENDER_RESOLUTION=1
# MADRONA_RENDER_MODE=2
# MADRONA_MWGPU_KERNEL_CACHE=cache 
# ./viewer 64 --cuda

import os
from dataclasses import dataclass

RAST_NO = 1
RT_NO = 2
NUM_STEPS = 32
SPLIT_TRACE_DIV = 16

RENDER_MODE_NAME = { RAST_NO: "Rast", RT_NO: "RT" }

HSSD_BASE_PATH = "madrona_escape_room"
HIDESEEK_BASE_PATH = "gpu_hideseek"
MJX_BASE_PATH = "madrona_mjx"

DO_TRACE_SPLIT = True

# Environment variables that all runs will set (although not all need)
@dataclass
class EnvironmentVars:
    track_trace_split: int
    first_scene_index: int
    num_scenes: int
    timing_file: str
    render_resolution: int
    render_mode: int
    cache_path: str

class EnvironmentGen:
    def __init__(self, vars: EnvironmentVars):
        self.env_vars = vars

    def generate_str(self):
        command = f"MADRONA_TRACK_TRACE_SPLIT={self.env_vars.track_trace_split} "
        command = command + f"HSSD_FIRST_SCENE={self.env_vars.first_scene_index} "
        command = command + f"HSSD_NUM_SCENES={self.env_vars.num_scenes} "
        command = command + f"MADRONA_LBVH={0} "
        command = command + f"MADRONA_WIDEN={1} "
        command = command + f"MADRONA_MWGPU_FORCE_DEBUG={0} "
        command = command + f"MADRONA_TIMING_FILE={self.env_vars.timing_file} "
        command = command + f"MADRONA_RENDER_RESOLUTION={self.env_vars.render_resolution} "
        command = command + f"MADRONA_RENDER_MODE={self.env_vars.render_mode} "
        command = command + f"MADRONA_MWGPU_KERNEL_CACHE={self.env_vars.cache_path} "

        return command



# Configurations of the various specific runs
@dataclass
class RunConfig:
    num_worlds: int
    num_steps: int
    base_path: str

class MJXRun:
    def __init__(self, run_cfg: RunConfig, env_vars: EnvironmentVars):
        self.run_cfg = run_cfg
        self.env_vars = env_vars

    def run(self):
        script_path = self.run_cfg.base_path + "/scripts/headless.py"

        env_gen = EnvironmentGen(self.env_vars)

        command = env_gen.generate_str()

        command = command + f"python {script_path} "
        command = command + f"--num-worlds {self.run_cfg.num_worlds} "
        command = command + f"--num-steps {self.run_cfg.num_steps} "
        command = command + f"--batch-render-view-width {self.env_vars.render_resolution} "
        command = command + f"--batch-render-view-height {self.env_vars.render_resolution}"

        print(command)
        os.system(command)

class HSSDRun:
    def __init__(self, run_cfg: RunConfig, env_vars: EnvironmentVars):
        self.run_cfg = run_cfg
        self.env_vars = env_vars

    def run(self):
        bin_path = self.run_cfg.base_path + f"/build/headless"

        env_gen = EnvironmentGen(self.env_vars)

        command = env_gen.generate_str()

        command = command + f"{bin_path} "
        command = command + f"CUDA "
        command = command + f"{self.run_cfg.num_worlds} "
        command = command + f"{self.run_cfg.num_steps} "

        print(command)
        os.system(command)

class HideseekRun:
    def __init__(self, run_cfg: RunConfig, env_vars: EnvironmentVars):
        self.run_cfg = run_cfg
        self.env_vars = env_vars

    def run(self):
        bin_path = self.run_cfg.base_path + f"/build/headless"

        env_gen = EnvironmentGen(self.env_vars)
        
        command = env_gen.generate_str()
        command = command + f"{bin_path} "
        command = command + f"CUDA "
        command = command + f"{self.run_cfg.num_worlds} "
        command = command + f"{self.run_cfg.num_steps}"

        print(command)
        os.system(command)


def do_hssd_run(render_mode, res, num_worlds, num_scenes):
    output_file_name = f"hssd/out_{RENDER_MODE_NAME[render_mode]}_{num_worlds}_{res}x{res}_{num_scenes}.json"

    env_vars = EnvironmentVars(
        track_trace_split=0,
        first_scene_index=0,
        num_scenes=num_scenes,
        timing_file=output_file_name,
        render_resolution=res,
        render_mode=render_mode,
        cache_path="hssd_cache"
    )

    run_cfg = RunConfig(
        num_worlds=num_worlds,
        num_steps=NUM_STEPS,
        base_path=HSSD_BASE_PATH
    )

    hssd_run = HSSDRun(run_cfg, env_vars)
    hssd_run.run()

    if render_mode == RT_NO and DO_TRACE_SPLIT:
        # Start a split run too
        split_output_file_name = f"hssd/out_{RENDER_MODE_NAME[render_mode]}_{num_worlds}_{res}x{res}_{num_scenes}_split.json"

        env_vars.track_trace_split = 1
        env_vars.timing_file = split_output_file_name
        run_cfg.num_worlds = run_cfg.num_worlds / SPLIT_TRACE_DIV

        split_hssd_run = HSSDRun(run_cfg, env_vars)
        split_hssd_run.run()

def do_hideseek_run(render_mode, res, num_worlds):
    output_file_name = f"hideseek/out_{RENDER_MODE_NAME[render_mode]}_{num_worlds}_{res}x{res}_1.json"

    env_vars = EnvironmentVars(
        track_trace_split=0,
        first_scene_index=0,
        num_scenes=1,
        timing_file=output_file_name,
        render_resolution=res,
        render_mode=render_mode,
        cache_path="hideseek_cache"
    )

    run_cfg = RunConfig(
        num_worlds=num_worlds,
        num_steps=NUM_STEPS,
        base_path=HIDESEEK_BASE_PATH
    )

    hideseek_run = HideseekRun(run_cfg, env_vars)
    hideseek_run.run()

    if render_mode == RT_NO and DO_TRACE_SPLIT:
        # Start a split run too
        split_output_file_name = f"hideseek/out_{RENDER_MODE_NAME[render_mode]}_{num_worlds}_{res}x{res}_1_split.json"

        env_vars.track_trace_split = 1
        env_vars.timing_file = split_output_file_name
        run_cfg.num_worlds = run_cfg.num_worlds / SPLIT_TRACE_DIV

        split_hideseek_run = HideseekRun(run_cfg, env_vars)
        split_hideseek_run.run()

def do_mjx_run(render_mode, res, num_worlds):
    output_file_name = f"mjx/out_{RENDER_MODE_NAME[render_mode]}_{num_worlds}_{res}x{res}_1.json"

    env_vars = EnvironmentVars(
        track_trace_split=0,
        first_scene_index=0,
        num_scenes=1,
        timing_file=output_file_name,
        render_resolution=res,
        render_mode=render_mode,
        cache_path="mjx_cache"
    )

    run_cfg = RunConfig(
        num_worlds=num_worlds,
        num_steps=NUM_STEPS,
        base_path=MJX_BASE_PATH
    )

    mjx_run = MJXRun(run_cfg, env_vars)
    mjx_run.run()

    if render_mode == RT_NO and DO_TRACE_SPLIT:
        # Start a split run too
        split_output_file_name = f"mjx/out_{RENDER_MODE_NAME[render_mode]}_{num_worlds}_{res}x{res}_1_split.json"

        env_vars.track_trace_split = 1
        env_vars.timing_file = split_output_file_name
        run_cfg.num_worlds = run_cfg.num_worlds / SPLIT_TRACE_DIV

        split_mjx_run = MJXRun(run_cfg, env_vars)
        split_mjx_run.run()


# Perform the runs
def main():
    render_modes_list = [ RT_NO ]
    render_resolutions_list = [ 64 ]
    num_worlds_list = [ 64 ]
    num_unique_scenes_list = [ 4 ]

    # Try this on hideseek environment
    for render_mode in render_modes_list:
        for render_resolution in render_resolutions_list:
            for num_worlds in num_worlds_list:
                do_hideseek_run(render_mode, render_resolution, num_worlds)

    # Try this on HSSD environment
    """
    for render_mode in render_modes_list:
        for render_resolution in render_resolutions_list:
            for num_worlds in num_worlds_list:
                for num_unique_scenes in num_unique_scenes_list:
                    do_hssd_run(render_mode, render_resolution, 
                            num_worlds, num_unique_scenes);
    """

if __name__ == "__main__":
    main()
