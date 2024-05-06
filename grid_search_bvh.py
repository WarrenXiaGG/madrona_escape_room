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
from os.path import expanduser
from dataclasses import dataclass
import glob
import json
import sys

RAST_NO = 1
RT_NO = 2
NUM_STEPS = 32
SPLIT_TRACE_DIV = 16

RENDER_MODE_NAME = { RAST_NO: "Rast", RT_NO: "RT" }

HSSD_BASE_PATH = "madrona_escape_room"
HIDESEEK_BASE_PATH = "gpu_hideseek"
MJX_BASE_PATH = "madrona_mjx"

KERNEL_CACHE_PATH = expanduser("~") + "/kernel_cache/cache"
BVH_CACHE_PATH = expanduser("~") + "/bvh_cache"
TEST_RUN_OUT = "~/test_outs"

NUM_SAMPLES = 2

DO_TRACE_SPLIT = False

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
    bvh_cache_path: str
    cache_all_bvh: int
    proc_thor: int
    seed: int

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
        command = command + f"MADRONA_BVH_CACHE_DIR={self.env_vars.bvh_cache_path} "
        command = command + f"MADRONA_PROC_THOR={self.env_vars.proc_thor} "
        command = command + f"MADRONA_SEED={self.env_vars.seed} "

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


def do_hssd_run(render_mode, res, num_worlds, num_scenes, bvh_config):
    total_time = 0
    for i in range(NUM_SAMPLES):
        config_name = '_'.join(map(str,list(bvh_config)))
        output_file_name = f"hssd/run{i}/out_{RENDER_MODE_NAME[render_mode]}_{num_worlds}_{res}x{res}_{num_scenes}_bvh_{config_name}.json"

        env_vars = EnvironmentVars(
            track_trace_split=0,
            first_scene_index=0,
            num_scenes=num_scenes,
            timing_file=output_file_name,
            render_resolution=res,
            render_mode=render_mode,
            cache_path=KERNEL_CACHE_PATH,
            bvh_cache_path=BVH_CACHE_PATH,
            cache_all_bvh=0,
            proc_thor=0,
            seed=i
        )

        run_cfg = RunConfig(
            num_worlds=num_worlds,
            num_steps=NUM_STEPS,
            base_path=HSSD_BASE_PATH
        )


        hssd_run = HSSDRun(run_cfg, env_vars)
        hssd_run.run()

        try:
            with open(output_file_name,"r") as f:
                new_data = json.load(f)
                total_time += new_data["avg_total_time"]
        except IOError:
            pass

    return total_time

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

    single_set_test = True

    if single_set_test:
        render_resolutions_list = [ 32 ]
        num_worlds_list = [ 1 ]
        num_unique_scenes_list = [ 1]

        num_leaf_options = [1,2]
        bvh_width_options = [2,4]
    else:
        render_resolutions_list = [32, 64, 128 ]
        num_worlds_list = [ 1024, 2048, 4096 ]
        num_unique_scenes_list = [ 1,4 ]

        num_leaf_options = [1,2,4]
        bvh_width_options = [2,4,8]

    metrics = {}

    env_config_labels = ["Resolution","Views","Unique Scenes"]
    bvh_param_labels = ["Leaf Size","Bottom Level Width"]

    saved_path = os.getcwd()

    for render_leaf in num_leaf_options:
        for bvh_width in bvh_width_options:
            bvh_config = (render_leaf, bvh_width)

            os.chdir(HSSD_BASE_PATH+"/build/")
            cmake_cmd = f"cmake -DMADRONA_BLAS_WIDTH={bvh_width} -DMADRONA_LEAF_WIDTH={render_leaf} .."
            os.system(cmake_cmd)
            make_cmd = "make -j 8"
            os.system(make_cmd)
            os.chdir(saved_path)

            #Clear Kernel and BVH Cache
            if os.path.exists(KERNEL_CACHE_PATH):
                os.remove(KERNEL_CACHE_PATH)

            files = glob.glob(BVH_CACHE_PATH + "/*")
            for f in files:
                os.remove(f)

            for render_resolution in render_resolutions_list:
                for num_worlds in num_worlds_list:
                    for num_unique_scenes in num_unique_scenes_list:
                        timeout = do_hssd_run(RT_NO, render_resolution,
                                num_worlds, num_unique_scenes, bvh_config);
                        env_config = (render_resolution,num_worlds,num_unique_scenes)
                        runs = metrics.get(env_config,[])
                        runs.append((timeout, bvh_config))
                        metrics[env_config] = runs

    print(metrics)

    for env_key in metrics:
        metrics[env_key] = sorted(metrics[env_key], key=(lambda x: x[0]))

    print("Sorted Metrics")
    print(metrics)

if __name__ == "__main__":
    main()
