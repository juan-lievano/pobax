from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Tuple, Optional

import json
import chex
import jax
import jax.numpy as jnp
from jax import random, lax

import gymnax
from gymnax.environments.environment import Environment, EnvParams
from gymnax.environments import spaces

from definitions import ROOT_DIR


def half_dist_prob(dist: jnp.ndarray, max_dist: float, lb: float = 0.5) -> jnp.ndarray:
    """Probability that a 'check' returns the true label, in [0, 1]."""
    prob = (1.0 + jnp.power(2.0, -dist / max_dist)) * lb
    return jnp.clip(prob, 0.0, 1.0).astype(jnp.float32)


@chex.dataclass
class RockSampleState:
    position: chex.Array       # (2,) int32  [y, x]
    rock_morality: chex.Array  # (k,) int32  values in {0,1}


class RockSample(Environment):
    """
    RockSample (JAX/Gymnax).

    Observation shape: (2*size + k), dtype float32
      - one-hot y position (size)
      - one-hot x position (size)
      - k rock observation slots for this step only:
          0.0 : no reading this step
         -1.0 : sampled at this rock cell this step (marker)
          0.0/1.0 : reading from 'check rock i' (stochastic, distance-based)

    Actions: Discrete(k + 5)
      0: North, 1: East, 2: South, 3: West,
      4: Sample (consume rock at current cell if present; +/- reward),
      5..(k+4): Check rock i (no state change; produces an observation reading).

    Episode ends (done=True) when x == size-1 (rightmost column / exit).
    """
    # (dy, dx) for N,E,S,W
    direction_mapping = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=jnp.int32)

    def __init__(
        self,
        key: chex.PRNGKey,
        config_path: Path = Path(ROOT_DIR, "pobax", "envs", "configs", "rocksample_7_8_config.json"),
    ):
        with open(config_path) as f:
            config = json.load(f)

        self.config_path = config_path
        self.size: int = int(config["size"])
        self.k: int = int(config["rocks"])

        self.half_efficiency_distance: float = float(config["half_efficiency_distance"])
        self.bad_rock_reward: float = float(config["bad_rock_reward"])
        self.good_rock_reward: float = float(config["good_rock_reward"])
        self.exit_reward: float = float(config["exit_reward"])

        # Pre-sample rock positions (exclude exit column)
        self.rock_positions: chex.Array = self.generate_map(self.size, self.k, key)  # (k,2) int32

    # -------------------- Gymnax API --------------------

    def observation_space(self, env_params: EnvParams) -> spaces.Box:
        return spaces.Box(low=-1.0, high=1.0, shape=(2 * self.size + self.k,), dtype=jnp.float32)

    def action_space(self, env_params: EnvParams) -> spaces.Discrete:
        return spaces.Discrete(self.k + 5)

    @property
    def default_params(self) -> EnvParams:
        return EnvParams(max_steps_in_episode=1000)

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[EnvParams] = None
    ) -> Tuple[chex.Array, RockSampleState]:
        if params is None:
            params = self.default_params

        key_moral, key_pos = random.split(key)
        rock_morality = self.sample_morality(key_moral)  # (k,) int32
        agent_position = self.sample_positions(self.size, key_pos, 1)[0]  # (2,) int32

        state = RockSampleState(position=agent_position, rock_morality=rock_morality)
        obs = self._get_obs(state, jnp.int32(0), key).astype(jnp.float32)  # no check on reset
        return lax.stop_gradient(obs), lax.stop_gradient(state)

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: RockSampleState,
        action: int,
        params: EnvParams,
    ) -> Tuple[chex.Array, RockSampleState, jnp.ndarray, jnp.ndarray, dict]:
        if params is None:
            params = self.default_params

        a = jnp.asarray(action, dtype=jnp.int32)

        # --- SAMPLE transition (if action == 4) ---
        def _do_sample(_):
            ele = (self.rock_positions == state.position)           # (k,2) bool
            on_cell = jnp.all(ele, axis=-1).astype(jnp.int32)       # (k,) 1 where rock at agent cell
            new_rock_morality = jnp.maximum(state.rock_morality - on_cell, 0)

            all_rock_rews = (self.good_rock_reward * state.rock_morality.astype(jnp.float32) +
                             self.bad_rock_reward  * (1 - state.rock_morality).astype(jnp.float32))
            rew = jnp.dot(on_cell.astype(jnp.float32), all_rock_rews)  # scalar float32
            return new_rock_morality, rew

        rock_morality, samp_rew = lax.cond(a == 4, _do_sample, lambda _: (state.rock_morality, jnp.float32(0.0)), operand=None)

        # --- MOVE transition (if action in {0,1,2,3}) ---
        def _do_move(_):
            new_pos = state.position + self.direction_mapping[a % 4]
            pos_max = jnp.array([self.size - 1, self.size - 1], dtype=jnp.int32)
            return jnp.maximum(jnp.minimum(new_pos, pos_max), 0)

        position = lax.cond(a < 4, _do_move, lambda _: state.position, operand=None)

        # --- Termination & exit reward ---
        terminal = (position[1] == (self.size - 1))  # bool
        reward = samp_rew + terminal.astype(jnp.float32) * jnp.float32(self.exit_reward)

        next_state = RockSampleState(position=position, rock_morality=rock_morality)
        obs = self._get_obs(next_state, a, key).astype(jnp.float32)
        info = {}

        return (
            lax.stop_gradient(obs),
            lax.stop_gradient(next_state),
            reward.astype(jnp.float32),                 # scalar (0-dim) float32
            jnp.asarray(terminal, dtype=jnp.bool_),     # scalar (0-dim) bool
            info,
        )

    # -------------------- Internals --------------------

    @staticmethod
    def generate_map(size: int, k: int, key: chex.PRNGKey) -> chex.Array:
        """Pick k distinct rock positions, excluding the exit column (x == size-1)."""
        rows = jnp.arange(size, dtype=jnp.int32)
        cols = jnp.arange(size - 1, dtype=jnp.int32)  # exclude last column
        cand = jnp.dstack(jnp.meshgrid(rows, cols, indexing="ij")).reshape(-1, 2)  # (size*(size-1),2)
        idx = random.choice(key, cand.shape[0], shape=(k,), replace=False)
        return cand[idx].astype(jnp.int32)

    @staticmethod
    @partial(jax.jit, static_argnums=(0, 2))
    def sample_positions(size: int, key: chex.PRNGKey, n: int = 1) -> chex.Array:
        """Sample n valid positions uniformly, excluding exit column."""
        rows = jnp.arange(size, dtype=jnp.int32)
        cols = jnp.arange(size - 1, dtype=jnp.int32)
        cand = jnp.dstack(jnp.meshgrid(rows, cols, indexing="ij")).reshape(-1, 2)  # (size*(size-1),2)
        idx = random.choice(key, cand.shape[0], shape=(n,), replace=True)
        return cand[idx].astype(jnp.int32)

    @partial(jax.jit, static_argnums=(0,))
    def sample_morality(self, key: chex.PRNGKey) -> chex.Array:
        """Bernoulli(k) rock labels: 1=good, 0=bad (int32)."""
        return random.bernoulli(key, p=0.5, shape=(self.rock_positions.shape[0],)).astype(jnp.int32)

    @partial(jax.jit, static_argnums=(0,))
    def _get_obs(self, state: RockSampleState, action: jnp.ndarray, key: chex.PRNGKey) -> chex.Array:
        """Build observation: [onehot_y | onehot_x | rock_obs(k)], dtype int32, convert to float32 by caller."""
        # Position one-hots
        y_oh = jnp.zeros(self.size, dtype=jnp.int32).at[state.position[0]].set(1)
        x_oh = jnp.zeros(self.size, dtype=jnp.int32).at[state.position[1]].set(1)

        # Rock observation block (depends on action)
        def _check(_):
            rock_idx = (action - 5).astype(jnp.int32)
            dist = jnp.linalg.norm(state.position.astype(jnp.float32) -
                                   self.rock_positions[rock_idx].astype(jnp.float32), ord=2)
            prob = half_dist_prob(dist, self.half_efficiency_distance)  # float32 in [0,1]
            true_lab = state.rock_morality[rock_idx].astype(jnp.int32)
            choices = jnp.array([true_lab, 1 - true_lab], dtype=jnp.int32)
            probs = jnp.array([prob, 1.0 - prob], dtype=jnp.float32)
            reading = random.choice(key, choices, shape=(1,), p=probs)[0]  # int32 0/1
            return jnp.zeros(self.k, dtype=jnp.int32).at[rock_idx].set(reading)

        rock_obs = lax.cond(action > 4, _check, lambda _: jnp.zeros(self.k, dtype=jnp.int32), operand=None)

        # If sampling, put -1 at rock(s) on the current cell (marker)
        def _sample_marker(rocks_block):
            ele = (self.rock_positions == state.position)   # (k,2) bool
            on_cell = jnp.all(ele, axis=-1).astype(jnp.int32)  # (k,)
            return rocks_block * (1 - on_cell) + (-1) * on_cell

        rock_obs = lax.cond(action == 4, _sample_marker, lambda x: x, rock_obs)

        return jnp.concatenate([y_oh, x_oh, rock_obs], axis=0).astype(jnp.int32)
