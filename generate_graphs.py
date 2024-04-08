import os
import sys
import json
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class GraphData:
    """
    raster_time: float
    raster_cull_time: float
    raycast_time: float
    raycast_slowbuild_time: float
    """
    times: dict[str, float]

@dataclass
class ResolutionData:
    # One graph per view count
    graphs_data: dict[str, GraphData]

@dataclass
class EnvironmentData:
    # One set of graphs per resolution
    resolutions_data: dict[str, ResolutionData]

def main():
    cmd_args = sys.argv
    if len(cmd_args) < 2:
        print("usage: python generate_graphs.py [env_directory]")
        exit(-1)

    env_dir = cmd_args[1]

    stat_files = os.listdir(env_dir)

    # We are going to generate a graph per resolution per environment
    env_data = EnvironmentData(resolutions_data={})

    for file_name in stat_files:
        name_components = file_name.split('.')[0].split('_');

        render_type = name_components[1]
        num_views = name_components[2]
        resolution_str = name_components[3]

        with open(env_dir + '/' + file_name, 'r') as file:
            data = json.load(file)

            avg_total_time = data['avg_total_time']

            print(f"resolution: {resolution_str}; num_views: {num_views}; render_type={render_type}")

            if resolution_str not in env_data.resolutions_data:
                env_data.resolutions_data[resolution_str] = ResolutionData(graphs_data={})

            if num_views not in env_data.resolutions_data[resolution_str].graphs_data:
                env_data.resolutions_data[resolution_str].graphs_data[num_views] = GraphData(times={})

            env_data.resolutions_data[resolution_str].graphs_data[num_views].times[render_type] = avg_total_time


    # Now, we are going to generate all the graphs
    example = env_data.resolutions_data['32x32'].graphs_data['512'].times
    print(example)


if __name__ == "__main__":
    main()
