from functools import partial

import chex
import gymnax
from gymnax.environments.environment import Environment, EnvParams
import jax
import jax.numpy as jnp
from jax import random
from typing import Tuple


import jax, jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnames='size')
def choose_goal2d(size: int, key):
    i_idx, j_idx = jnp.indices((size, size))

    key, k_draw, k_r, k_c, k_a, k_d = jax.random.split(key, 6)
    draws = jax.random.bernoulli(k_draw, 0.75, (4,)) #0.75 instead of 0.5 to have 3 lines in expectaction (0.5 felt like it would be all 0s too often)

    r  = jax.random.randint(k_r, (), 0, size) # to choose a random row
    c  = jax.random.randint(k_c, (), 0, size) # to choose a random column
    a  = jax.random.randint(k_a, (), -(size - 1), size)   # choose random anti-diagonal offset
    d  = jax.random.randint(k_d, (), 0, 2 * size - 1)     # choose random diagonal offset

    masks = jnp.stack((
        (i_idx == r),
        (j_idx == c),
        ((j_idx - i_idx) == a),
        ((i_idx + j_idx) == d),
    ), axis=0)

    goal = jnp.any(masks & draws[:, None, None], axis=0)
    return goal.astype(jnp.int32)


@chex.dataclass
class LightBulbs2DState:
    bulbs: chex.Array  # current 2D state of lightbulbs.
    goal: chex.Array   # 2D goal mask
    h_action: int      # 1-hot 2D tensor with latest human action. 
    t: int             # step counter


class LightBulbs2D(Environment):
    """
    LightBulbs2D environment on a size x size grid.
    Robot toggles one bulb or does noop at flat index = n_actions.
    Human flips one mismatched bulb toward the goal or noop.
    """
    def __init__(self, dim: int):
        """
        dim is the side length of the 2D matrix
        """
        self.dim = dim

    def observation_space(self, env_params: EnvParams):
        """
        Box dimension is (dim, dim, 2). 
        That is, two layers of a dim x dim box.
        Filtering for third coordinate == 0 we get the bulb array.
        Filtering for third coordinate == 1 we get the human action 1-hot box.
        """
        dim = self.dim
        return gymnax.environments.spaces.Box(0, 1,(dim,dim,2), jnp.int32)  

    def action_space(self, env_params: EnvParams):
        """
        dim * dim + 1 actions because we have a noop
        """
        dim = self.dim
        n_actions = dim * dim + 1
        return gymnax.environments.spaces.Discrete(n_actions)

    @property
    def default_params(self) -> EnvParams:
        return EnvParams(max_steps_in_episode=1000)

    def get_obs(self, state: LightBulbs2DState) -> jnp.ndarray:
        """
        """
        # bulbs and h_action are (dim, dim) binary ararys
        bulbs = state.bulbs.astype(jnp.int32) 
        h_action = state.h_action.astype(jnp.int32)  # 2D one‐hot

        obs = jnp.stack((bulbs, h_action), axis=-1)  # (dim, dim, 2)

        # Note : The bulbs array is given by obs[:, :, 0]),
        # the human action one-hot is given by obs[:, :, 1])
        return obs

    # def human_policy(key_in: jax.Array,
    #              bulbs: jnp.ndarray,
    #              goal: jnp.ndarray):
        
    #     mismatches = bulbs != goal
    #     has_mismatch = jnp.any(mismatches)

    #     dim = bulbs.shape[0]
    #     noop_index = dim * dim  # dim*dim

    #     def flip_branch(operand):
    #         key_in, bulbs, goal, mismatches = operand
    #         subkey, key_out = jax.random.split(key_in)

    #         # adds independent noice to all of the indices with missmatches
    #         # assings -infity to the indices with matches
    #         # then takes a max value
    #         # so in the end it chose a random mismatch
    #         # this is the "gumbel-max" trick
    #         gumbel_noise = jax.random.gumbel(subkey, bulbs.shape)
    #         masked_scores = jnp.where(mismatches, gumbel_noise, -jnp.inf)

    #         # we flatten to take the max as described in the gumbel-max trick above
    #         flat_index = jnp.argmax(masked_scores.reshape(-1))

    #         # conver flat index to 2D coords
    #         i = flat_index // dim
    #         j = flat_index %  dim

    #         # updates bulbs array

    #         bulbs_updated = bulbs_updated = bulbs.at[i, j].set(bulbs[i, j] ^ 1) # xor (^) with "1" will simply toggle the value

    #         return bulbs_updated, flat_index, key_out

    #     def noop_branch(operand):
    #         key_in, bulbs, goal, mismatches = operand
    #         return bulbs, noop_index, key_in

    #     bulbs_next, chosen_index, key_out = jax.lax.cond(
    #         has_mismatch,
    #         flip_branch,
    #         noop_branch,
    #         operand=(key_in, bulbs, goal, mismatches),
    #     )
    #     return bulbs_next, chosen_index, key_out


    @partial(jax.jit, static_argnums=(0,))
    def reset_env(self,
                key: jax.Array,
                params: EnvParams):
        dim = self.dim

        # Split keys: one for bulbs, one for goal
        key, k_bulbs, k_goal = random.split(key, 3)

        # Random initial bulbs (0/1) on a dim x dim grid
        bulbs = random.bernoulli(k_bulbs, p=0.5, shape=(dim, dim)).astype(jnp.int32)

        # Sample a goal mask with your JIT-able generator
        goal = choose_goal2d(dim, k_goal).astype(jnp.int32)

        # No previous human action at reset → 2D zero plane
        h_action = jnp.zeros((dim, dim), dtype=jnp.int32)

        state = LightBulbs2DState(
            bulbs=bulbs,
            goal=goal,
            h_action=h_action,
            t=0
        )

        # get_obs should stack (bulbs, h_action) along channel axis → (dim, dim, 2)
        obs = self.get_obs(state)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step_env(self,
                key: jax.Array,
                state: LightBulbs2DState,
                action: jnp.int32,
                params: EnvParams):
        
        dim  = self.dim
        noop = dim * dim

        key, k_human = random.split(key)

        # robot toggle function (this is only called if robot didn't noop)
        def toggle_fn(args):
            bulbs, idx = args
            i, j = divmod(idx, dim)
            return bulbs.at[i, j].set(bulbs[i, j] ^ 1)

        # bulbs after robot toggle
        bulbs_after = jax.lax.cond(
            action == noop,
            lambda args: args[0],
            toggle_fn,
            operand=(state.bulbs, action),
        )

        # human toggle
        mismatches   = bulbs_after != state.goal
        has_mismatch = jnp.any(mismatches)

        def human_flip(_): # takes a dummy input because jax.lax expects to be able to feed an input
            """
            The "gumbel-max" trick is used to select a random index to fix.
            It goes like this:
            1. adds independent noice to all of the indices with missmatches
            2. assings -infity to the indices with matches
            3. then takes a max value (in the flattened array)
            4. so in the end it chose a random mismatch
            """
            g = random.gumbel(k_human, bulbs_after.shape) 
            scores = jnp.where(mismatches, g, -jnp.inf)
            h_flat = jnp.argmax(scores)                     # flat index
            i, j   = jnp.unravel_index(h_flat, (dim, dim))
            bulbs_h = bulbs_after.at[i, j].set(bulbs_after[i, j] ^ 1)
            return bulbs_h, h_flat

        bulbs_after_h, h_flat = jax.lax.cond(
            has_mismatch,
            human_flip,
            lambda _: (bulbs_after, noop),
            operand=None,
        )

        # 3) Build human 2D one-hot (all zeros if noop)
        def build_one_hot(_):
            i, j = jnp.unravel_index(h_flat, (dim, dim))
            return jnp.zeros((dim, dim), jnp.int32).at[i, j].set(1)

        h_action_plane = jax.lax.cond(
            h_flat == noop,
            lambda _: jnp.zeros((dim, dim), jnp.int32),
            build_one_hot,
            operand=None,
        )

        # 4) Done / reward
        done   = jnp.all(bulbs_after_h == state.goal)
        reward = jnp.where(done, 0, -1)

        next_state = LightBulbs2DState(
            bulbs=bulbs_after_h,
            goal=state.goal,
            h_action=h_action_plane,
            t=state.t + 1,
        )

        obs = self.get_obs(next_state)
        return obs, next_state, reward, done, {}
