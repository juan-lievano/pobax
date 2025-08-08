from functools import partial

import chex
import gymnax
from gymnax.environments.environment import Environment, EnvParams
import jax
import jax.numpy as jnp
from jax import random
from typing import Tuple


def choose_goal2d(size: int, key: jax.random.key) -> jnp.ndarray:
    """
    Returns a 2D mask (size, size) with 1s along random horizontal, vertical,
    and any-offset diagonals of slopes +1 or -1.
    """
    # Initialize empty goal
    goal0 = jnp.zeros((size, size), dtype=jnp.int32)
    # Coordinates for diagonal masks
    i_idx, j_idx = jnp.indices((size, size))

    # Generate a random permutation of the 4 line types
    key, k_perm = random.split(key)
    perm = random.permutation(k_perm, jnp.arange(4, dtype=jnp.int32))  # unique types
    # Sample number of lines
    key, k_n = random.split(key)
    n_lines = random.randint(k_n, (), 0, 5)

    def body(i, goal):
        lt = perm[i]
        subk = random.fold_in(key, i)
        if lt == 0:
            r = random.randint(subk, (), 0, size)
            return goal.at[r, :].set(1)
        elif lt == 1:
            c = random.randint(subk, (), 0, size)
            return goal.at[:, c].set(1)
        elif lt == 2:
            b = random.randint(subk, (), -(size - 1), size)
            mask = (j_idx - i_idx) == b
            return goal.at[mask].set(1)
        else:
            k_val = random.randint(subk, (), 0, 2 * size)
            mask = (i_idx + j_idx) == k_val
            return goal.at[mask].set(1)

    # Use fori_loop to apply each line in a JIT-friendly way
    goal = jax.lax.fori_loop(0, n_lines, body, goal0)
    return goal


@chex.dataclass
class LightBulbs2DState:
    bulbs: chex.Array  # current 2D state of lightbulbs
    goal: chex.Array   # 2D goal mask
    h_action: int      # latest human action index, 0..n_actions-1 or n_actions for noop
    t: int             # step counter


class LightBulbs2D(Environment):
    """
    LightBulbs2D environment on a size x size grid.
    Robot toggles one bulb or does noop at flat index = n_actions.
    Human flips one mismatched bulb toward the goal or noop.
    """
    def __init__(self, size: int):
        self.size = size
        self.n_actions = size * size

    def observation_space(self, env_params: EnvParams):
        dim = self.n_actions + (self.n_actions + 1)
        return gymnax.environments.spaces.Box(0, 1, (dim,), jnp.int32)

    def action_space(self, env_params: EnvParams):
        # 0..n_actions-1 toggles, n_actions is noop
        return gymnax.environments.spaces.Discrete(self.n_actions + 1)

    @property
    def default_params(self) -> EnvParams:
        return EnvParams(max_steps_in_episode=1000)

    def get_obs(self, state: LightBulbs2DState) -> jnp.ndarray:
        bulbs_flat = state.bulbs.reshape(-1)
        one_hot = jax.nn.one_hot(state.h_action,
                                 self.n_actions + 1,
                                 dtype=jnp.int32)
        return jnp.concatenate([bulbs_flat, one_hot], axis=0)

    @staticmethod
    def human_policy(key: jax.random.key,
                     bulbs: jnp.ndarray,
                     goal: jnp.ndarray,
                     n_actions: int) -> Tuple[jnp.ndarray, int]:
        mask = bulbs != goal
        has_mismatch = jnp.any(mask)

        def flip(_):
            key2, subk = random.split(key)
            probs = mask.astype(jnp.float32)
            probs = probs / jnp.sum(probs)
            idx_flat = random.choice(subk, bulbs.size, p=probs.reshape(-1))
            i = idx_flat // bulbs.shape[1]
            j = idx_flat % bulbs.shape[1]
            new_bulbs = bulbs.at[i, j].set(goal[i, j])
            return new_bulbs, idx_flat

        new_bulbs, idx = jax.lax.cond(has_mismatch,
                                      flip,
                                      lambda _: (bulbs, n_actions),
                                      operand=None)
        return new_bulbs, idx

    @partial(jax.jit, static_argnums=(0,))
    def reset_env(self,
                  key: jax.random.key,
                  params: EnvParams) -> Tuple[jnp.ndarray, LightBulbs2DState]:
        key, subkey = random.split(key)
        bulbs = random.bernoulli(subkey,
                                 p=0.5,
                                 shape=(self.size, self.size)).astype(jnp.int32)
        # sample goal (no key returned)
        goal = choose_goal2d(self.size, subkey)
        state = LightBulbs2DState(bulbs=bulbs,
                                   goal=goal,
                                   h_action=self.n_actions,
                                   t=0)
        return self.get_obs(state), state

    def transition(self,
                   key: jax.random.key,
                   state: LightBulbs2DState,
                   action: int) -> Tuple[LightBulbs2DState, jnp.ndarray, bool]:
        def toggle_fn(args):
            b, idx = args
            i, j = divmod(idx, self.size)
            return b.at[i, j].set(1 - b[i, j])

        bulbs_after = jax.lax.cond(action == self.n_actions,
                                    lambda args: args[0],
                                    toggle_fn,
                                    (state.bulbs, action))
        key, subkey = random.split(key)
        bulbs_after_human, h_idx = self.human_policy(subkey,
                                                     bulbs_after,
                                                     state.goal,
                                                     self.n_actions)
        done = jnp.all(bulbs_after_human == state.goal)
        reward = jnp.where(done, 0, -1)
        next_state = LightBulbs2DState(bulbs=bulbs_after_human,
                                       goal=state.goal,
                                       h_action=int(h_idx),
                                       t=state.t + 1)
        return next_state, reward, done

    @partial(jax.jit, static_argnums=(0,))
    def step_env(self,
                 key: jax.random.key,
                 state: LightBulbs2DState,
                 action: int,
                 params: EnvParams):
        key, subkey = random.split(key)
        next_state, reward, done = self.transition(subkey, state, action)
        obs = self.get_obs(next_state)
        return obs, next_state, reward, done, {}
