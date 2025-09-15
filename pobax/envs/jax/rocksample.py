from __future__ import annotations

from functools import partial
from pathlib import Path
import json
from typing import Tuple, Optional

import chex
import jax
import jax.numpy as jnp
from jax import random, lax

import gymnax
from gymnax.environments.environment import Environment, EnvParams
from gymnax.environments import spaces

from definitions import ROOT_DIR


def half_dist_prob(dist: float, max_dist: float, lb: float = 0.5) -> jnp.ndarray:
    """Returns probability in [0, 1] that a 'check' reads the true rock label.
    For lb=0.5, the range is [0.5, 1.0]. Clamped for safety."""
    prob = (1.0 + jnp.power(2.0, -dist / max_dist)) * lb
    return jnp.clip(prob, 0.0, 1.0)


@chex.dataclass
class RockSampleState:
    position: chex.Array          # shape (2,), dtype int: [y, x]
    rock_morality: chex.Array     # shape (k,), dtype int in {0,1} (0=bad, 1=good)


class RockSample(Environment):
    """RockSample (JAX/Gymnax).

    Observation: length (2*size + k)
      - one-hot for y position (size)
      - one-hot for x position (size)
      - k rock observation slots:
          0  : no reading this step
          -1 : sampled at this rock's cell this step (special 'sample' marker)
          0/1: result from 'check rock i' (1 good, 0 bad)

    Actions: Discrete(k + 5)
      0: North, 1: East, 2: South, 3: West,
      4: Sample (if on a rock cell, receive +/- reward and set that rock to 0 = 'depleted')
      5..(k+4): Check rock i (no immediate env change; stochastic observation depending on distance)

    Episode ends (done=True) when agent reaches the rightmost column (x == size-1).
    """
    # (dy, dx) in order N,E,S,W
    direction_mapping = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=int)

    def __init__(
        self,
        key: chex.PRNGKey,
        config_path: Path = Path(ROOT_DIR, "pobax", "envs", "../configs", "rocksample_7_8_config.json"),
    ):
        # Load config
        with open(config_path) as f:
            config = json.load(f)

        self.config_path = config_path
        self.size: int = int(config["size"])
        self.k: int = int(config["rocks"])

        # Reward/transition hyperparams
        self.half_efficiency_distance: float = float(config["half_efficiency_distance"])
        self.bad_rock_reward: float = float(config["bad_rock_reward"])
        self.good_rock_reward: float = float(config["good_rock_reward"])
        self.exit_reward: float = float(config["exit_reward"])

        # Static map: choose k rock positions; exclude the rightmost column (exit column)
        self.rock_positions: chex.Array = self.generate_map(self.size, self.k, key)

    # -------------------- Gymnax API --------------------

    def observation_space(self, env_params: EnvParams) -> spaces.Box:
        # Values are in {-1, 0, 1} for the rock part, and {0,1} for one-hots.
        # Use a Box with bounds [-1, 1].
        return spaces.Box(low=-1, high=1, shape=(2 * self.size + self.k,), dtype=jnp.int32)

    def action_space(self, env_params: EnvParams) -> spaces.Discrete:
        return spaces.Discrete(self.k + 5)

    @property
    def default_params(self) -> EnvParams:
        # Only used if you integrate with other Gymnax utilities that expect it.
        return EnvParams(max_steps_in_episode=1000)

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[EnvParams] = None
    ) -> Tuple[chex.Array, RockSampleState]:
        """Reset to a random agent position (not on the exit column) and random rock moralities."""
        if params is None:
            params = self.default_params

        key_moral, key_pos = random.split(key)
        rock_morality = self.sample_morality(key_moral)
        agent_position = self.sample_positions(self.size, key_pos, 1)[0]

        state = RockSampleState(position=agent_position, rock_morality=rock_morality)
        # For reset, we use a non-check action (e.g., 0) so rock obs block is zeros
        obs = self._get_obs(state, 0, key)
        return lax.stop_gradient(obs), lax.stop_gradient(state)

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: RockSampleState,
        action: int,
        params: EnvParams,
    ) -> Tuple[chex.Array, RockSampleState, float, bool, dict]:
        """One environment step."""
        if params is None:
            params = self.default_params

        # --- SAMPLE transition (if action == 4) ---
        def _do_sample(_):
            # If on a rock cell, consume it and get reward according to its morality.
            ele = (self.rock_positions == state.position)                 # (k, 2)
            on_cell = jnp.all(ele, axis=-1).astype(int)                   # (k,)
            new_rock_morality = jnp.maximum(state.rock_morality - on_cell, 0)

            # Reward for sampling this step (dot with one-hot 'on_cell')
            all_rock_rews = self.good_rock_reward * state.rock_morality + \
                            self.bad_rock_reward * (1 - state.rock_morality)
            rew = jnp.dot(on_cell, all_rock_rews)
            return new_rock_morality, rew

        def _no_sample(_):
            return state.rock_morality, 0.0

        rock_morality, samp_rew = lax.cond(
            jnp.asarray(action) == 4, _do_sample, _no_sample, operand=None
        )

        # --- MOVE transition (if action in {0,1,2,3}) ---
        def _do_move(_):
            new_pos = state.position + self.direction_mapping[action % 4]
            pos_max = jnp.array([self.size - 1, self.size - 1])
            return jnp.maximum(jnp.minimum(new_pos, pos_max), 0)

        def _no_move(_):
            return state.position

        position = lax.cond(
            jnp.asarray(action) < 4, _do_move, _no_move, operand=None
        )

        # --- Termination & exit reward ---
        terminal = position[1] == (self.size - 1)
        reward = samp_rew + terminal.astype(jnp.float32) * self.exit_reward

        next_state = RockSampleState(position=position, rock_morality=rock_morality)
        obs = self._get_obs(next_state, action, key)
        info = {}

        return (
            lax.stop_gradient(obs),
            lax.stop_gradient(next_state),
            float(reward),
            bool(terminal),
            info,
        )

    # -------------------- Internals --------------------

    @staticmethod
    def generate_map(size: int, k: int, key: chex.PRNGKey) -> chex.Array:
        """Sample k distinct rock positions, excluding the exit column (x == size-1)."""
        rows = jnp.arange(size)
        cols = rows[:-1]  # exclude last column (exit)
        # Cartesian product grid -> (size*(size-1), 2)
        cand = jnp.dstack(jnp.meshgrid(rows, cols, indexing="ij")).reshape(-1, 2)
        idx = random.choice(key, cand.shape[0], shape=(k,), replace=False)
        return cand[idx].astype(int)

    @staticmethod
    @partial(jax.jit, static_argnums=(0, 2))
    def sample_positions(size: int, key: chex.PRNGKey, n: int = 1) -> chex.Array:
        """Sample n valid agent positions uniformly, excluding exit column."""
        rows = jnp.arange(size)
        cols = rows[:-1]
        cand = jnp.dstack(jnp.meshgrid(rows, cols, indexing="ij")).reshape(-1, 2)
        idx = random.choice(key, cand.shape[0], shape=(n,), replace=True)
        return cand[idx].astype(int)

    @partial(jax.jit, static_argnums=(0,))
    def sample_morality(self, key: chex.PRNGKey) -> chex.Array:
        """Bernoulli(k) morality: 1=good, 0=bad."""
        return random.bernoulli(key, p=0.5, shape=(self.rock_positions.shape[0],)).astype(int)

    @partial(jax.jit, static_argnums=(0,))
    def _get_obs(self, state: RockSampleState, action: int, key: chex.PRNGKey) -> chex.Array:
        """Build observation: [onehot_y | onehot_x | rock_obs(k)]."""
        # Position one-hots
        y_oh = jnp.zeros(self.size, dtype=int).at[state.position[0]].set(1)
        x_oh = jnp.zeros(self.size, dtype=int).at[state.position[1]].set(1)

        # Rock observation block (depends on action)
        def _check(_):
            # Which rock is being checked?
            rock_idx = (jnp.asarray(action) - 5).astype(int)
            # Distance-based correctness probability
            dist = jnp.linalg.norm(state.position - self.rock_positions[rock_idx], ord=2)
            prob = half_dist_prob(dist, self.half_efficiency_distance)

            # True label
            true_lab = state.rock_morality[rock_idx]
            choices = jnp.array([true_lab, 1 - true_lab])
            probs = jnp.array([prob, 1.0 - prob])
            reading = random.choice(key, choices, shape=(1,), p=probs)[0]  # 0/1

            # Place at index; others are 0 this step
            return jnp.zeros(self.k, dtype=int).at[rock_idx].set(reading)

        def _not_check(_):
            return jnp.zeros(self.k, dtype=int)

        rock_obs = lax.cond(
            jnp.asarray(action) > 4, _check, _not_check, operand=None
        )

        # If sampling, place a special -1 marker at current rock cell (if on any rock)
        def _sample_marker(rocks_block):
            ele = (self.rock_positions == state.position)    # (k,2)
            on_cell = jnp.all(ele, axis=-1)                  # (k,)
            # overwrite those indices with -1; else keep whatever is there (usually 0)
            return rocks_block * (1 - on_cell) + (-1) * on_cell

        rock_obs = lax.cond(jnp.asarray(action) == 4, _sample_marker, lambda x: x, rock_obs)

        return jnp.concatenate([y_oh, x_oh, rock_obs], axis=0).astype(jnp.int32)
