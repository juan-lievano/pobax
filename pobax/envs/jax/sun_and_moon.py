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
    """
        pos: Agent position on the ring, int32 in [0, N-1].
        t:   Time step counter (int32).
    """
    pos: chex.Array   # int32 in [0, N-1]
    t: chex.Array     # int32 step counter


class SunAndMoon(Environment):
    """
    Configuration
    -------------
    - Ring of N states (indices 0..N-1).
    - 'Sun' is uniquely observable at index 0.
    - 'Moon' (goal) is fixed at index N//2 (opposite side).
    - All non-sun, non-terminal states are aliased as "hallway".
    - Episodes last exactly `horizon` steps; the final observation is terminal (code 2).

    Actions
    -------
    - 0: clockwise
    - 1: counter-clockwise
    With probability `epsilon`, the chosen action is flipped.

    Observations/reward:

    The agent gets 1 reward if t == horizon - 1 and it is at the moon.
    The agent should see observation "2" if t == horizon.
    The episode ends if t == horizon + 1

    Reset
    -----
    - Start positions are sampled uniformly from even indices {0, 2, ..., N-2}.
    """

    def __init__(self, n_states: int = 12, horizon: int = 25, epsilon: float = 0.1):
        if n_states < 2:
            raise ValueError("n_states must be >= 2.")
        if horizon % 2 == 0:
            raise ValueError("horizon must be odd or game isn't solvable")
        self.N = int(n_states)
        self.horizon = int(horizon)
        self.epsilon = float(epsilon)
        self.sun_idx = 0
        self.moon_idx = self.N // 2  # fixed opposite

    def observation_space(self, env_params: EnvParams):
        # Wrap discrete {0,1,2} as Box((1,)) to satisfy wrappers/models.
        return gymnax.environments.spaces.Box(0, 1, (3,))


    def action_space(self, env_params: EnvParams):
        return gymnax.environments.spaces.Discrete(2)

    @property
    def default_params(self) -> EnvParams:
        return EnvParams(max_steps_in_episode=1000)  # it ends at horizon anyway

    def _obs_from_state(self, pos: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        is_terminal = (t == self.horizon)
        is_sun = (pos == self.sun_idx) & (~is_terminal)
        is_hallway = (~is_sun) & (~is_terminal)
        return jnp.array([is_sun, is_hallway, is_terminal], dtype=jnp.uint8)

    @partial(jax.jit, static_argnums=(0,))
    def reset_env(self, key, params: EnvParams) -> Tuple[jnp.ndarray, SunAndMoonState]:
        """Reset: uniformly sample an even index; t = 0; return obs(Box(1,)), state."""
        half = self.N // 2  # number of even indices
        even_slot = random.randint(key, shape=(), minval=0, maxval=half, dtype=jnp.int32)
        pos = (even_slot * 2) % self.N
        state = SunAndMoonState(pos=pos, t=jnp.int32(0))
        obs = self._obs_from_state(state.pos, state.t)
        return obs, state

    def _apply_action(self, key: chex.PRNGKey, pos: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        """Apply action with flip-noise epsilon; return next position (int32)."""
        flip = random.bernoulli(key, p=self.epsilon)
        intended_delta = jnp.where(action == 0, 1, -1).astype(jnp.int32)  # +1 or -1
        delta = jnp.where(flip, -intended_delta, intended_delta)
        return jnp.mod(pos + delta, self.N).astype(jnp.int32)

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self,
        key: chex.PRNGKey,
        state: SunAndMoonState,
        action: int,
        params: EnvParams,
    ):
        """One environment step with terminal obs at the boundary."""
        key, k_act = random.split(key)

        # Next state
        next_pos = self._apply_action(k_act, state.pos, jnp.int32(action))
        next_t = state.t + jnp.int32(1)
        next_state = SunAndMoonState(pos=next_pos, t=next_t)

        # Termination 
        done = (state.t == jnp.int32(self.horizon) + 1)

        # Reward
        reward = (state.t == (self.horizon - 1)) & (state.pos == self.moon_idx)

        # Observation (terminal code 2 when t >= horizon)
        obs = self._obs_from_state(next_pos, next_t)

        info = {}

        return obs, next_state, reward, done, info
