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

# makes it easy to load a config and extract goal set and distributions
from pobax.envs.jax.lightbulbs import LightBulbs

import json

import pandas as pd

import re

from datetime import datetime

def extract_human_actions(observations, robot_actions):

    '''
    if len(obs) == size, this will return size-1 human actions. 
    '''

    size = len(observations[0])

    human_actions = jnp.zeros(size, dtype='int32')

    for current_obs, next_obs, robot_action in zip(observations, observations[1:], robot_actions):

        # vector is only non-zero at index chosen by human
        # a value of 1 means human turned on that lightbulb
        # a value of -1 means human turned off that lightbulbs
        curr_human_action_vector = (next_obs - current_obs).at[robot_action].set(0)
        
        human_actions += curr_human_action_vector #TODO this only works if the human never reverses an action it did previously

    return human_actions

def bayesian_posterior_from_human_actions(human_actions : jnp.ndarray, goal_set : jnp.ndarray, current_goal_distribution:jnp.ndarray) -> jnp.ndarray: 

    """
    human_actions is an array that has a positive number in indices where human turned on a lighbulb and a negative number in indices where human turned off a lightbulb

    returns an array of length len(current_goal_distribution) specifying distribution over possible goals

    #TODO currently this is assuming a uniform distribution over goals
    """

    # find indices where goal_distribution is non-zero
    # loop through these non-zero indices
    # # check whether that goal is compatible with human actions 
    # # save the index in an array if yes

    non_zero_prob_goals_idxs = jnp.where(current_goal_distribution > 0)[0]
    compatible_goal_idxs = []

    for idx in non_zero_prob_goals_idxs: 

        goal = goal_set[idx]
        compatible = True # assume compatible until we find the opposite

        for lightbulb, h_action in enumerate(human_actions):

            # lightbulb is an index, h_action is either -1 0 or 1

            if h_action == 0: # we have no info about the current idx (human hasn't touched it)
                continue 

            elif h_action > 0 and goal[lightbulb] == 0: # human toggled this lightbulb on but this goal wants it off
                compatible = False
                break

            elif h_action < -1 and goal[lightbulb] == 1: # human toggled this lightbulbs off but this goal wants it on
                compatible = False
                break

        if compatible:
            compatible_goal_idxs.append(idx)

    def make_uniform_distribution(compatible_idxs, size):
        result = jnp.zeros(size)
        if len(compatible_idxs) == 0: 
            return result
        value = 1.0 / len(compatible_idxs)
        compatible_idxs = jnp.array(compatible_idxs) # this is necessary for the next line to work
        result = result.at[compatible_idxs].set(value)
        return result
    
    posterior_distribution = make_uniform_distribution(compatible_idxs=compatible_goal_idxs, size = len(goal_set))        

    return posterior_distribution

def main(trajectories_filepath):

    # load trajectories and metadata
    trajectories_array, trajectories_sampling_config = load_trajectories_and_env_config_from_npz(trajectories_filepath = trajectories_filepath)

    environment_config_filepath = trajectories_sampling_config['environment_config_path']

    #load goal set and prior distribution (assuming uniform for now)
    environment = LightBulbs(environment_config_filepath)
    goal_set = environment.goals
    current_bayes = environment.goal_distribution

    supervised_learning_data = {'rnn_hidden_state':[],
                                'bayes_posterior':[],
                                'trajectory_id': [],
                                'time_step' : []} # TODO I could allocate the size from the start cause I can calculate how many I will need

    for traj_id, trajectory in enumerate(trajectories_array):

        observations = trajectory['observations']
        robot_actions = trajectory['robot_actions']
        current_bayes = environment.goal_distribution

        for i in range(0,len(observations)):

            # extract_human_actions called on observations[:i+1] returns i human actions
            human_actions = extract_human_actions(observations= observations[:i+1], robot_actions=robot_actions[:i+1]) # TODO there's a lot of redundacy here

            # if we have i human actions we can calculate the bayes posterior at the start of time step i (0-indexed)
            current_bayes = bayesian_posterior_from_human_actions(human_actions=human_actions, goal_set=goal_set, current_goal_distribution = current_bayes) # TODO there's a lot of redundancy here

            # trajectory['rnn_hidden_states'][i] is the rnn step AT THE START of timestep i
            rnn_hidden_state = trajectory['hidden_rnn_states'][i]

            supervised_learning_data['rnn_hidden_state'].append(rnn_hidden_state)
            supervised_learning_data['bayes_posterior'].append(current_bayes)
            supervised_learning_data['trajectory_id'].append(traj_id) 
            supervised_learning_data['time_step'].append(i) 
            

    df = pd.DataFrame(supervised_learning_data)

    # this whole part is just to create the output filename
    env_config_path = trajectories_sampling_config["environment_config_path"]
    stem = Path(env_config_path).stem  # e.g., "lightbulbs_20_8_config"
    match = re.match(r"(lightbulbs.*?)_config", stem)
    env_name = match.group(1) if match else "lightbulbs_unknown"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # csv version

    study_name = trajectories_sampling_config["study_name"]
    output_dir = Path("supervised_learning_data")
    output_filename = f"{study_name}_{timestamp}.csv"
    output_path = output_dir / output_filename

    df.to_csv(output_path)
    print(f'results saved to {output_path}')
    
def load_trajectories_and_env_config_from_npz(trajectories_filepath):
    """
    Load a list of trajectory dicts and the environment config from a .npz file.

    Returns:
        - trajectories: List of trajectories, each a dict with keys:
            - 'observations': list of jnp arrays
            - 'hidden_rnn_states': list of jnp arrays
            - 'robot_actions': list of ints
            - 'trajectory_id': int
        - config_dict: environment config as a Python dictionary
    """
    data = np.load(trajectories_filepath, allow_pickle=True)
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

    # Load environment config
    config_str = str(data["environment_config_json"])
    config_dict = json.loads(config_str)

    return trajectories, config_dict

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
    main(trajectories_filepath= results_path)

    total_time = time.time() - start_time
    print(f"\nTotal main method time: {total_time:.2f} seconds")
