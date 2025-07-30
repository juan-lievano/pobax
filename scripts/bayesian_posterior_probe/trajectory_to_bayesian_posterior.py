# read our csv data
import pandas as pd

# read arguments from terminal
import sys 

# handle file paths
from pathlib import Path

import time

# to convert the literal strings in the csv to jax numpy arrays (gives ast.literal_eval method)
import ast

import jax.numpy as jnp

# to load npz file which contains the trajectories
import numpy as np

def extract_human_actions(observations, robot_actions):

    print(observations)
    print(type(observations))
    print(robot_actions)
    print(type(robot_actions))
    return 

def main(results_df : pd.DataFrame):

    for _, trajectory in results_df.iterrows():

        observations_string = trajectory['observation']
        robot_actions_string = trajectory['robot_action']

        parsed_observations = [jnp.array(obs) for obs in ast.literal_eval(observations_string)]
        parsed_robot_actions = [jnp.array(act) for act in ast.literal_eval(robot_actions_string)]

        extract_human_actions(observations = parsed_observations,
                              robot_actions = parsed_robot_actions)

        return
    
def load_trajectories_from_npz(filepath):
    """
    Load a list of trajectory dicts from a .npz file.

    Returns:
        List of trajectories, each a dict with keys:
            - 'observation': list of jnp arrays
            - 'hidden_rnn_state': list of jnp arrays
            - 'robot_action': list of ints
            - 'trajectory_id': int
    """
    data = np.load(filepath, allow_pickle=True)
    num_trajectories = len([k for k in data if k.endswith("_trajectory_id")])

    trajectories = []
    for i in range(num_trajectories):
        traj = {
            "observations": [jnp.array(obs) for obs in data[f"traj_{i}_observations"]],
            "hidden_rnn_states": [jnp.array(h) for h in data[f"traj_{i}_hidden_rnn_states"]],
            "robot_actions": data[f"traj_{i}_robot_actions"].tolist(),
            "trajectory_id": int(data[f"traj_{i}_trajectory_id"])
        }
        trajectories.append(traj)

    return trajectories

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python script.py <results.npz> \n" \
        "<results.npz> should be in a directory called results that exists in the same directory as this script.")
        sys.exit(1)

    results_filename = sys.argv[1]
    results_path = Path("results") / results_filename

    if not results_path.exists():
        print(f"Config file not found: {results_path}")
        sys.exit(1)

    # timer starts right before calling main
    start_time = time.time()

    # trajectories_array is a list of trajectories
    # a single trajectory is a dictionary with keys ['observations','hidden_rnn_states','robot_actions','trajectory_id']
    trajectories_array = load_trajectories_from_npz(filepath = results_path)

    print(trajectories_array[0])

    total_time = time.time() - start_time
    print(f"\nTotal main method time: {total_time:.2f} seconds")
