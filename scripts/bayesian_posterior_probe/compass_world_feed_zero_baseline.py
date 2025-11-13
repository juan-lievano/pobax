#!/usr/bin/env python3
import argparse
# import math
import time
t00 = time.time()
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple, Optional
import ast # this is to parse the list data that comes as strings from the csv

import jax
import jax.numpy as jnp
# from jax import lax
# from jax import random as jr
import numpy as np
import orbax.checkpoint
import pandas as pd

from pobax.envs.jax.compass_world import CompassWorld
from pobax.models import ScannedRNN, get_gymnax_network_fn
from pobax.models.discrete import DiscreteActorCriticRNN

from scripts.bayesian_posterior_probe.compass_world_trajectory_sampling import infer_expected_input_dim 


def main():
    t0 = time.time()

    p = argparse.ArgumentParser(description="TODO needs description")
    p.add_argument("--csv", type=str, required = True)
    p.add_argument("--checkpoint", type=str, required = True)
    p.add_argument("--grid_size", type = int, default = 8)
    args = p.parse_args()


    # load RNN checkpoint

    ckpt_path = Path(args.checkpoint).resolve()
    restored = orbax.checkpoint.PyTreeCheckpointer().restore(ckpt_path)
    hidden_size = int(restored["args"]["hidden_size"])
    double_critic = bool(restored["args"]["double_critic"])

    env = CompassWorld(size=args.grid_size)
    network_class, action_size = get_gymnax_network_fn(env, env.default_params)
    assert network_class is DiscreteActorCriticRNN, f"Expected DiscreteActorCriticRNN, got {network_class}"
    num_actions = int(action_size)
    
    model = DiscreteActorCriticRNN(
        action_dim=num_actions, double_critic=double_critic, hidden_size=hidden_size
    )
    ts_dict = jax.tree.map(lambda x: x[0, 0, 0, 0, 0, 0, 0], restored["final_train_state"])
    weights = ts_dict["params"]

    # check dimensions

    expected_input_dimension = infer_expected_input_dim(weights, hidden_size)

    tmp_obs0, _ = env.reset_env(jax.random.PRNGKey(0), params=None)
    obs_dim = int(tmp_obs0.shape[-1])

    assert obs_dim == expected_input_dimension, f"The RNN is expecting an input of size {expected_input_dimension} but the observations have dimension {obs_dim}"

    # load CSV

    df_raw = pd.read_csv(args.csv)
    df = df_raw.copy()
    
    # extract relevant columns
    # the relevant columns are the observation and the analytical belief state
    
    def parse_obs(s: str) -> np.ndarray:
        return np.array(ast.literal_eval(s), dtype=np.float32)

    def one_step_feed_zero(params, obs_vec):
        """Feed one observation with a zero hidden state."""
        network_input = jnp.asarray(obs_vec)[None, None, :]      # (1,1,obs_dim)
        done_flag     = jnp.zeros((1, 1), dtype=jnp.bool_)       # (1,1)
        h0            = jnp.zeros((1, hidden_size), jnp.float32) # (1,H)
        hidden_new, _, _ = model.apply(params, h0, (network_input, done_flag))
        return hidden_new[0]  # (H,)

    # one_step_zero_jit = jax.jit(lambda obs_vec: one_step_feed_zero(weights, obs_vec))

    # Loop through dataframe rows
    # h_out = np.empty((len(df), hidden_size), dtype=np.float32)
    # for i, s in enumerate(df["observation"]):
    #     obs_vec = parse_obs(s)
    #     h_out[i] = np.asarray(one_step_zero_jit(obs_vec))
    #
    # df["zero_fed_h_state"] = [h_out[i] for i in range(len(df))]
    obs_np = np.stack([np.array(ast.literal_eval(s), dtype=np.float32)
                   for s in df["observation"].to_numpy()], axis=0)  # (N, obs_dim)

    batched = jax.jit(jax.vmap(lambda o: one_step_feed_zero(weights, o), in_axes=0))

    h_out = np.asarray(batched(jnp.array(obs_np)))  # (N, hidden_size)

    df["zero_fed_h_state"] = [h_out[i] for i in range(h_out.shape[0])]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path("supervised_learning_data")
    out_root.mkdir(parents=True, exist_ok=True)
    ckpt_suffix = ckpt_path.name[-10:]
    n_traj = len(df['trajectory_id'].unique())
    out_name = f"compass_world_zero_fed_{n_traj}_trajs_model_{ckpt_suffix}_ts_{timestamp}.csv"
    out_path = out_root / out_name

    df.to_csv(out_path, index=False)

    print(f"\nSaved CSV to {out_path}")
    elapsed0 = time.time() - t0
    elapsed00 = time.time() - t00

    print(f"{elapsed0=:.2f}s, {elapsed00=:.2f}s")

if __name__ == '__main__':
    main()

