from functools import partial

import chex
import gymnax
from gymnax.environments.environment import Environment, EnvParams
import jax
import jax.numpy as jnp
from jax import random
from typing import Tuple


@chex.dataclass
class CompassWorldState:
    pos: chex.Array   # (y, x) in [1, size-2]
    dir: chex.Array   # 0=N,1=E,2=S,3=W
    t: chex.Array


class CompassWorld(Environment):
    def __init__(self, size: int = 8, random_start: bool = True):
        self.size = int(size)
        self.random_start = bool(random_start)
        self._dir_map = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=jnp.int32)
        self._state_min = jnp.array([1, 1], dtype=jnp.int32)
        self._state_max = jnp.array([self.size - 2, self.size - 2], dtype=jnp.int32)
        # goal in the middle of west wall
        y_mid = jnp.int32((self.size - 1) // 2)
        self._goal_pos = jnp.array([y_mid, 1], dtype=jnp.int32)
        self._goal_dir = jnp.int32(3) # wait why does the goal have a direction? (oh i think the goal "position" is where the robot has to be, not where the goal is really (which is in front of the robot)). 

    def observation_space(self, env_params: EnvParams):
        return gymnax.environments.spaces.Box(0, 1, (5,))

    def action_space(self, env_params: EnvParams):
        return gymnax.environments.spaces.Discrete(3)

    @property
    def default_params(self) -> EnvParams:
        return EnvParams(max_steps_in_episode=1000)

    def _obs_from_state(self, pos: jnp.ndarray, dir_: jnp.ndarray) -> jnp.ndarray:
        # one-hot over [N-wall, E-wall, S-wall, blue(W), green(W@(1,1))]
        n = (dir_ == 0) & (pos[0] == 1) # pos[0] is the row, pos[1] is the column, so this makes sense. (althogh it means the coordinates are (y,x) which is sort of annoying)
        e = (dir_ == 1) & (pos[1] == self.size - 2)
        s = (dir_ == 2) & (pos[0] == self.size - 2)
        w_border = (dir_ == 3) & (pos[1] == 1) # column with index 1, and facing west
        g = w_border & (pos[0] == self._goal_pos[0]) # in west border and in the goal's row
        w = w_border & (pos[0] != self._goal_pos[0]) # in west border and not in the goal's row
        return jnp.array([n, e, s, w, g], dtype=jnp.uint8)

    def _done(self, pos: jnp.ndarray, dir_: jnp.ndarray) -> jnp.ndarray:
        return jnp.logical_and(jnp.all(pos == self._goal_pos), dir_ == self._goal_dir)

    def _reward(self, done: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(done, 0, -1).astype(jnp.int32)

    def _forward(self, pos: jnp.ndarray, dir_: jnp.ndarray) -> jnp.ndarray:
        nxt = pos + self._dir_map[dir_]
        return jnp.clip(nxt, self._state_min, self._state_max)

    def _turn_right(self, dir_: jnp.ndarray) -> jnp.ndarray:
        return (dir_ + 1) % 4

    def _turn_left(self, dir_: jnp.ndarray) -> jnp.ndarray:
        return (dir_ - 1) % 4

    @partial(jax.jit, static_argnums=(0,))
    def reset_env(self, key: chex.PRNGKey, params: EnvParams) -> Tuple[jnp.ndarray, CompassWorldState]:
        if not self.random_start:
            yx = jnp.array([3, 3], dtype=jnp.int32)
            dir_ = jnp.int32(0)
        else:
            key, k1, k2, k3 = random.split(key, 4)
            y = random.randint(k1, (), 1, self.size - 1, dtype=jnp.int32)
            x = random.randint(k2, (), 1, self.size - 1, dtype=jnp.int32)
            d = random.randint(k3, (), 0, 4, dtype=jnp.int32)
            # avoid starting exactly at goal state; adjust dir if so
            same_pos = jnp.all(jnp.array([y, x], jnp.int32) == self._goal_pos)
            dir_ = jnp.where(jnp.logical_and(same_pos, d == self._goal_dir), (d + 1) % 4, d)
            yx = jnp.array([y, x], dtype=jnp.int32)

        state = CompassWorldState(pos=yx, dir=dir_, t=jnp.int32(0))
        obs = self._obs_from_state(state.pos, state.dir)
        return obs, state

    def transition(self, state: CompassWorldState, action: jnp.ndarray) -> CompassWorldState:
        pos, dir_ = state.pos, state.dir
        pos2 = jax.lax.cond(action == 0, lambda _: self._forward(pos, dir_), lambda _: pos, operand=None)
        dir2 = jax.lax.switch(
            jnp.clip(action, 0, 2),
            [
                lambda: dir_,
                lambda: self._turn_right(dir_),
                lambda: self._turn_left(dir_),
            ],
        )
        return CompassWorldState(pos=pos2, dir=dir2, t=state.t + 1)

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self,
        key: chex.PRNGKey,
        state: CompassWorldState,
        action: int,
        params: EnvParams,
    ):
        next_state = self.transition(state, jnp.int32(action))
        done = self._done(next_state.pos, next_state.dir)
        rew = self._reward(done)
        obs = self._obs_from_state(next_state.pos, next_state.dir)
        info = {}
        return obs, next_state, rew, done, info
