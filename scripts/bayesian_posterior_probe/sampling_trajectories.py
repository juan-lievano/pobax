# to see how long things take
import time

# for working with arguments provided int terminal
# # but i will just give the filename of a config JSON file which will have the other args
import json
import argparse
import sys
import jax
import jax.numpy as jnp

# needed for restoring model with robax
from pathlib import Path
import orbax.checkpoint

# needed to assert that action size and network_class make sense
from pobax.models import get_gymnax_network_fn

# needed because the model will travel through this environment
from pobax.envs.jax.lightbulbs import LightBulbs

# this is the type of network in which the weights will be loaded
from pobax.models.discrete import DiscreteActorCriticRNN

# this is the the class for the RNN part of the architecture 
from pobax.models import ScannedRNN

# to check if environment and model coincide
from pobax.models import get_gymnax_network_fn

# to save data as a file
import numpy as np
# import pandas as pd

def print_config(args):
    print("Config loaded:")
    for key, value in args.items():
        print(f"{key}: {value}")

# def load_model_and_environment(config_json):
#     """
#     Input:
#     config_json is a json file including keys: 
#     - checkpoint_directory_path
#     - environment_config_path

#     Asserts:
#     - Loaded model and loaded environment fit with eachother according to the configs and the get_gymnax_network_fn function

#     Returns:
#     - a loaded network of type DiscreteActorCriticRNN with the weights determined by the orbax checkpoint
#     - a LightBulbs environment 
#     """
#     # load model into discrete actor critic network 

#     ckpt_path = Path(config_json['checkpoint_directory_path']).resolve() # resolve cause it has to be absolute TODO this solution ok?
#     orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
#     restored = orbax_checkpointer.restore(ckpt_path)
#     train_state = restored['final_train_state']

def generate_single_trajectory(key : jax.random.PRNGKey, environment : LightBulbs, size : int, model : DiscreteActorCriticRNN, weights : dict, restored_orbax_checkpoint : dict, trajectory_id, max_length = 50):

    """
    for a trajectory,
    observations[i] = observation on time step i
    hidden_rnn_states[i] = rnn state that augmented observation at time step i
    robot_actions[i] = action that robot chose after seeing observations[i]
    trajectory_id = metadata to differentiate this trajectory from others in the database
    """

    # reset state and get initial observation
    sub, key = jax.random.split(key)
    observation, state = environment.reset_env(key = sub, params = None) #TODO Does params = None make sense?

    # manually create noop action to append to initial observation
    action_int = size


    args = restored_orbax_checkpoint['args']

    hidden_rnn_state = ScannedRNN.initialize_carry(batch_size=1, hidden_size=args['hidden_size']) #This is a float32 array with every value equal to 0. 

    done = False
    i = 0

    trajectory = {'observations' : [],
                  'hidden_rnn_states' : [],
                  'robot_actions' : [],
                  'trajectory_id' : trajectory_id} 
    
    while not done and i < max_length:

        # format prev_action_int as a vector
        prev_action_vector = jnp.zeros(size + 1).astype(dtype='int32')
        prev_action_vector.at[action_int].set(1) # this will always be noop
        concatenated_obs_action = jnp.concatenate([observation, prev_action_vector], axis = -1)

        concatenated_obs_action_batch = concatenated_obs_action[None, None, :] # changes shape from (41,) to (1,1,41) which we need

        # to the trajectory append the hidden state that augments the current observation
        trajectory['hidden_rnn_states'].append(hidden_rnn_state)

        # 1x1 array of dones
        done_batch = jnp.zeros((1, 1), dtype=bool)

        hidden_rnn_state, pi, dummy = model.apply(weights, hidden_rnn_state, (concatenated_obs_action_batch, done_batch))

        sub, key = jax.random.split(key)
        action = pi.sample(seed = sub)
        action_int = action[0,0]

        # to the trajectory append current observation and the action that the observation produced
        
        trajectory['observations'].append(observation)
        trajectory['robot_actions'].append(action_int)

        # step environment 
        sub, key = jax.random.split(key)

        observation, state, reward, done, dummy = environment.step_env(sub, state, action_int, environment.default_params) # this means that we won't ever see that last observation because when it is done it won't get into the while again
        # but that is probably fine

    return trajectory


def main(config_json : json):

    """
    config_json keys: 
    "checkpoint_directory_path": "/path/to/checkpoints",
    "max_run_time_minutes": 60,
    "trajectories_sample_size": 100,
    "environment_config_path":
    "random_seed": 0
    """

    # create environment
    environment = LightBulbs(config_json['environment_config_path'])

    network_class, action_size = get_gymnax_network_fn(environment, environment.default_params)

    assert network_class is DiscreteActorCriticRNN, f"Expected DiscreteActorCriticRNN, got {network_class}"
    assert action_size == environment.size + 1, f"Expected {environment.size + 1} possible actions, got {action_size}"

    # load model weights into DiscreteActorCriticRNN
    ckpt_path = Path(config_json['checkpoint_directory_path']).resolve()
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    restored = orbax_checkpointer.restore(ckpt_path)
    seed = config_json['random_seed']
    key = jax.random.key(seed)

    args = restored['args']

    model = DiscreteActorCriticRNN(
        action_dim    = action_size,
        double_critic = args['double_critic'],
        hidden_size   = args['hidden_size'],
        )
    
    ts_dict = jax.tree.map(lambda x: x[0, 0, 0, 0, 0, 0, 0], restored['final_train_state'])
    weights = ts_dict['params']

    max_run_time_seconds = int(config_json['max_run_time_minutes'])*60

    trajectories_sample_size = int(config_json['trajectories_sample_size'])

    trajectories = []
    trajectory_id = 0

    while trajectory_id < trajectories_sample_size: # TODO add max_run_time check
        sub, key = jax.random.split(key)
        traj = generate_single_trajectory(key = sub, 
                                      environment = environment,size = environment.size, model = model, 
                                      weights = weights, 
                                      trajectory_id = trajectory_id,restored_orbax_checkpoint = restored)
        trajectories.append(traj)
        trajectory_id += 1
    
    return trajectories

# def save_trajectories_to_npz(trajectories, filepath):
#     """
#     Save a list of trajectory dicts to a .npz file.

#     Each trajectory will be stored under a key like 'traj_0', 'traj_1', etc.
#     Each value will be a dict with NumPy arrays.
#     """
#     npz_data = {}

#     for i, traj in enumerate(trajectories):
#         npz_data[f"traj_{i}_observations"] = np.stack([np.array(obs) for obs in traj["observations"]])
#         npz_data[f"traj_{i}_hidden_rnn_states"] = np.stack([np.array(h) for h in traj["hidden_rnn_states"]])
#         npz_data[f"traj_{i}_robot_actions"] = np.array(traj["robot_actions"])
#         npz_data[f"traj_{i}_trajectory_id"] = np.array(traj["trajectory_id"])

#     np.savez_compressed(filepath, **npz_data)

def save_trajectories_to_npz(trajectories, filepath, env_config):
    """
    Save a list of trajectory dicts and the config dict to a .npz file.
    """
    npz_data = {}

    for i, traj in enumerate(trajectories):
        npz_data[f"traj_{i}_observations"] = np.stack([np.array(obs) for obs in traj["observations"]])
        npz_data[f"traj_{i}_hidden_rnn_states"] = np.stack([np.array(h) for h in traj["hidden_rnn_states"]])
        npz_data[f"traj_{i}_robot_actions"] = np.array(traj["robot_actions"])
        npz_data[f"traj_{i}_trajectory_id"] = np.array(traj["trajectory_id"])

    # Save environment config as JSON string
    config_str = json.dumps(env_config)
    npz_data["environment_config_json"] = np.array(config_str)

    np.savez_compressed(filepath, **npz_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample trajectories using a trained model")

    parser.add_argument("--config", required=True, type=str,
                    help="Path to JSON config file (e.g. configs/my_config.json)")
    args = parser.parse_args()

    config_filename = config_path = Path(args.config)

    config_path = Path("configs") / config_filename

    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = json.load(f)

    print_config(config)

    # timer starts right before calling main
    start_time = time.time()

    trajectories_array = main(config_json = config)


    # crate file name to save results
    study_name = config['study_name']
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{study_name}_{timestamp}.npz"
    filepath = Path("results") / filename
    
    save_trajectories_to_npz(trajectories=trajectories_array, filepath=filepath, env_config=config)

    total_time = time.time() - start_time
    print(f"\nTotal main method time: {total_time:.2f} seconds")














