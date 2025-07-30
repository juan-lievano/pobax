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

import pandas as pd

def print_config(args):
    print("Config loaded:")
    for key, value in args.items():
        print(f"{key}: {value}")

def load_model_and_environment(config_json):
    """
    Input:
    config_json is a json file including keys: 
    - checkpoint_directory_path
    - environment_config_path

    Asserts:
    - Loaded model and loaded environment fit with eachother according to the configs and the get_gymnax_network_fn function

    Returns:
    - a loaded network of type DiscreteActorCriticRNN with the weights determined by the orbax checkpoint
    - a LightBulbs environment 
    """
    # load model into discrete actor critic network 

    ckpt_path = Path(config_json['checkpoint_directory_path']).resolve() # resolve cause it has to be absolute TODO this solution ok?
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    restored = orbax_checkpointer.restore(ckpt_path)
    train_state = restored['final_train_state']

def generate_single_trajectory(key : jax.random.PRNGKey, environment : LightBulbs, size : int, model : DiscreteActorCriticRNN, weights : dict, restored_orbax_checkpoint : dict, trajectory_id, max_length = 50):

    # reset state and get initial observation
    sub, key = jax.random.split(key)
    observation, state = environment.reset_env(key = sub, params = None) #TODO Does params = None make sense?

    # manually create noop action to append to initial observation
    action_int = size


    args = restored_orbax_checkpoint['args']

    hidden_rnn_state = ScannedRNN.initialize_carry(batch_size=1, hidden_size=args['hidden_size']) #This is a float32 array with every value equal to 0. 

    done = False
    i = 0

    trajectory = {'observation' : [],
                  'hidden_rnn_state' : [],
                  'robot_action' : [],
                  'trajectory_id' : []} 
    
    while not done and i < max_length:

        # format prev_action_int as a vector
        prev_action_vector = jnp.zeros(size + 1).astype(dtype='int32')
        prev_action_vector.at[action_int].set(1) # this will always be noop
        concatenated_obs_action = jnp.concatenate([observation, prev_action_vector], axis = -1)

        concatenated_obs_action_batch = concatenated_obs_action[None, None, :] # changes shape from (41,) to (1,1,41) which we need

        # 1x1 array of dones
        done_batch = jnp.zeros((1, 1), dtype=bool)

        hidden_rnn_state, pi, dummy = model.apply(weights, hidden_rnn_state, (concatenated_obs_action_batch, done_batch))

        sub, key = jax.random.split(key)
        action = pi.sample(seed = sub)
        action_int = action[0,0]

        # add current observation and the hidden_rnn_state and action that it produced to the trajectory
        
        trajectory['observation'].append(observation)
        trajectory['hidden_rnn_state'].append(hidden_rnn_state)
        trajectory['robot_action'].append(action_int)
        trajectory['trajectory_id'].append(trajectory_id)

        # step environment 
        sub, key = jax.random.split(key)

        observation, state, reward, done, dummy = environment.step_env(key, state, action_int, environment.default_params) # this means that we won't ever see that last observation because when it is done it won't get into the while again
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
        traj = generate_single_trajectory(key = key, 
                                      environment = environment,size = environment.size, model = model, 
                                      weights = weights, 
                                      trajectory_id = trajectory_id,restored_orbax_checkpoint = restored)
        trajectories.append(traj)
        trajectory_id += 1
    
    return trajectories

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <config_filename.json> \n" \
        "<config_filename.json> should be in a directory called configs that exists in the same directory as this script.")
        sys.exit(1)

    config_filename = sys.argv[1]
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
    
    df = pd.DataFrame(trajectories_array)
    df.to_csv('results/test.csv')

    total_time = time.time() - start_time
    print(f"\nTotal main method time: {total_time:.2f} seconds")














