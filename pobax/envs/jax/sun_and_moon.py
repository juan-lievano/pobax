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
    Ring of N states with one uniquely observable 'Sun' at index 0,
    and a fixed 'Moon' goal at index N//2. All other states are aliased
    as identical 'hallway' observations.

    Actions: 0 = clockwise, 1 = counter-clockwise.
    With probability epsilon, the action is flipped.

    Episodes last exactly H steps (H = self.horizon). Reward is 1 on the final
    step iff the agent is on the Moon; otherwise 0.

    Reset: start positions are sampled uniformly from even indices only.
    """
    def __init__(self, n_states: int = 12, horizon: int = 24, epsilon: float = 0.1):
        if n_states < 2:
            raise ValueError("n_states must be >= 2.")
        self.N = int(n_states)
        self.horizon = int(horizon)
        self.epsilon = float(epsilon)
        self.sun_idx = 0
        self.moon_idx = self.N // 2  # fixed opposite

    def observation_space(self, env_params: EnvParams):
        # One-hot over [sun, hallway]. Moon is aliased as hallway.
        return gymnax.environments.spaces.Box(0, 1, (2,))

    def action_space(self, env_params: EnvParams):
        # 0 = clockwise, 1 = counter-clockwise
        return gymnax.environments.spaces.Discrete(2)

    @property
    def default_params(self) -> EnvParams:
        # Kept for API compatibility, but step_env uses self.horizon directly.
        return EnvParams(max_steps_in_episode=self.horizon)

    def _obs_from_pos(self, pos: jnp.ndarray) -> jnp.ndarray:
        is_sun = (pos == self.sun_idx)
        # [sun, hallway]; hallway is 1 whenever not sun (moon is aliased)
        return jnp.array([is_sun, jnp.logical_not(is_sun)], dtype=jnp.uint8)

    @partial(jax.jit, static_argnums=(0,))
    def reset_env(self, key, params: EnvParams) -> Tuple[jnp.ndarray, SunAndMoonState]:
        """
        Start uniformly from even indices: {0, 2, 4, ..., N-2}.
        """
        half = self.N // 2  # number of even indices in [0, N)
        even_slot = random.randint(key, shape=(), minval=0, maxval=half, dtype=jnp.int32)
        pos = (even_slot * 2) % self.N
        state = SunAndMoonState(pos=pos, t=jnp.int32(0))
        obs = self._obs_from_pos(state.pos)
        return obs, state

    def _apply_action(self, key: chex.PRNGKey, pos: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        """
        action: {0 (clockwise), 1 (counter-clockwise)}
        With prob epsilon, flip the action.
        """
        flip = random.bernoulli(key, p=self.epsilon)
        intended_delta = jnp.where(action == 0, 1, -1).astype(jnp.int32)  # +1 or -1
        delta = jnp.where(flip, -intended_delta, intended_delta)
        new_pos = jnp.mod(pos + delta, self.N).astype(jnp.int32)
        return new_pos

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self,
        key: chex.PRNGKey,
        state: SunAndMoonState,
        action: int,
        params: EnvParams,  # kept for signature compatibility; not used for horizon
    ):
        key, k_act = random.split(key)
        next_pos = self._apply_action(k_act, state.pos, jnp.int32(action))
        next_t = state.t + jnp.int32(1)
        next_state = SunAndMoonState(pos=next_pos, t=next_t)

        done = (next_t >= jnp.int32(self.horizon))  # compare to self.horizon directly
        at_moon = (next_pos == self.moon_idx)
        rew = jnp.where(jnp.logical_and(done, at_moon), 1, 0).astype(jnp.int32)

        obs = self._obs_from_pos(next_pos)
        info = {}
        return obs, next_state, rew, done, info
