from functools import partial
from typing import Tuple

import chex
import gymnax
from gymnax.environments.environment import Environment, EnvParams
import jax
import jax.numpy as jnp
from jax import random


@chex.dataclass
class SunAndMoonState:
    pos: chex.Array   # int32 in [0, N-1]
    t: chex.Array     # int32 step counter


class SunAndMoon(Environment):
    """
    Ring of N states with one uniquely observable 'Sun' state (index sun_idx),
    one fixed 'Moon' goal state opposite the sun (index (sun_idx + N//2) % N),
    and all other states aliased as identical 'hallway' observations.

    Actions: 0 = move clockwise, 1 = move counter-clockwise.
    With probability epsilon, the action is flipped (i.e., noise).

    Episodes last exactly H steps. Reward is 1 on the *final* step if the agent
    is exactly on the Moon; otherwise reward 0 (and 0 at all prior steps).
    """
    def __init__(self, n_states: int = 24, horizon: int = 100, epsilon: float = 0.1, sun_idx: int = 0):
        if n_states < 2:
            raise ValueError("n_states must be >= 2.")
        # 'Opposite' is well-defined when n_states is even; we still allow odd,
        # in which case 'moon' is floor-opposite.
        self.N = int(n_states)
        self.horizon = int(horizon)
        self.epsilon = float(epsilon)
        self.sun_idx = int(sun_idx) % self.N
        self.moon_idx = (self.sun_idx + (self.N // 2)) % self.N

    def observation_space(self, env_params: EnvParams):
        # One-hot over [sun, hallway]. Moon is aliased as hallway.
        return gymnax.environments.spaces.Box(0, 1, (2,), dtype=jnp.uint8)

    def action_space(self, env_params: EnvParams):
        # 0 = clockwise, 1 = counter-clockwise
        return gymnax.environments.spaces.Discrete(2)

    @property
    def default_params(self) -> EnvParams:
        return EnvParams(max_steps_in_episode=self.horizon)

    def _obs_from_pos(self, pos: jnp.ndarray) -> jnp.ndarray:
        is_sun = (pos == self.sun_idx)
        # [sun, hallway]; hallway is 1 whenever not sun
        return jnp.array([is_sun, jnp.logical_not(is_sun)], dtype=jnp.uint8)

    @partial(jax.jit, static_argnums=(0,))
    def reset_env(self, key, params: EnvParams) -> Tuple[jnp.ndarray, SunAndMoonState]:
        # Uniform start over all ring positions
        pos = random.randint(key, shape=(), minval=0, maxval=self.N, dtype=jnp.int32)
        state = SunAndMoonState(pos=pos, t=jnp.int32(0))
        obs = self._obs_from_pos(state.pos)
        return obs, state

    def _apply_action(self, key: chex.PRNGKey, pos: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        """
        action: int32 in {0 (CW), 1 (CCW)}
        With prob epsilon, flip the action.
        """
        # Flip = True with prob epsilon
        flip = random.bernoulli(key, p=self.epsilon)
        # Map to delta: 0 -> +1 (CW), 1 -> -1 (CCW)
        intended_delta = jnp.where(action == 0, 1, -1).astype(jnp.int32)
        # If flipped, multiply by -1
        delta = jnp.where(flip, -intended_delta, intended_delta)
        new_pos = jnp.mod(pos + delta, self.N).astype(jnp.int32)
        return new_pos

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self,
        key: chex.PRNGKey,
        state: SunAndMoonState,
        action: int,
        params: EnvParams,
    ):
        # Stochastic transition due to epsilon-noise
        key, k_act = random.split(key)
        next_pos = self._apply_action(k_act, state.pos, jnp.int32(action))

        next_t = state.t + jnp.int32(1)
        next_state = SunAndMoonState(pos=next_pos, t=next_t)

        # Done exactly at horizon
        done = (next_t >= jnp.int32(params.max_steps_in_episode))

        # Reward only on final step, and only if at moon
        at_moon = (next_pos == self.moon_idx)
        rew = jnp.where(jnp.logical_and(done, at_moon), 1, 0).astype(jnp.int32)

        obs = self._obs_from_pos(next_pos)
        info = {}
        return obs, next_state, rew, done, info
