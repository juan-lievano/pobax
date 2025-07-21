from functools import partial

import chex
import gymnax
from gymnax.environments.environment import Environment, EnvParams
import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from typing import Tuple
import json



@chex.dataclass
class LightBulbsState:
    bulbs: chex.Array # current state of lightbulbs array.
    goal: chex.Array # The goal array. 
    t: chex.Array # step counter. 


class LightBulbs(Environment):

    def __init__(self,
                 config_path: str, 
                 ): 
        """
        size: How many light bulbs in the array.
        config_path: path to json with keys "goals" and "distribution" containing array of possible goals and likelihood of choosing each goal. 
        """
        
        self.size, self.goals, self.goal_distribution, self.robot_noop = self._load_config(config_path)

    def _load_config(self, path: str):

        with open(path) as f:
            data = json.load(f)

        size = data["size"]
        goals = jnp.asarray(data["goals"], jnp.int32)
        dist  = jnp.asarray(data["distribution"], jnp.float32)
        robot_noop  = data["robot_noop"]

        if goals.shape[1] != size:
            raise ValueError(f"goal length {goals.shape[1]} != size {self.size}")
        if dist.shape[0] != goals.shape[0]:
            raise ValueError("distribution length must match number of goals")
        if not isinstance(robot_noop, bool):
            raise TypeError("'robot_noop' must be a JSON boolean (true/false)")

        return size, goals, dist / jnp.sum(dist), robot_noop




    def observation_space(self, env_params: EnvParams):
        """
        An observation is a binary array of size `self.size`.
        So we will use Box with low = 0 and high = 1 and dimensions (size,).
        """
        return gymnax.environments.spaces.Box(0, 1, (self.size,))
        

    def action_space(self, env_params: EnvParams): #TODO Why do the action and observations spaces take EnvParams as inputs?
        """
        An action is a choice of index to toggle.
        If robot_noop == True, we will allow the robot to pass its turn, which means we have an extra action. 
        `Noop` action is the choosing the integer `size` (which is out of bounds to be an index). 
        """
        if self.robot_noop: 
            return gymnax.environments.spaces.Discrete(self.size+1)
        else:
            return gymnax.environments.spaces.Discrete(self.size)

    @property
    def default_params(self) -> EnvParams:
        return EnvParams(max_steps_in_episode=1000)

    def get_obs(self, state: LightBulbsState) -> jnp.ndarray:
        '''
        The robot only oberves the state of the bulbs array.

        Returns:
        Current state of bulbs array, i.e. `LightBulbState.bulbs`
        '''
        # TODO it might learn better if it sees the human action. 
        return state.bulbs
    
    @staticmethod
    def human_policy(key: jax.random.key, state: jnp.ndarray, goal: jnp.ndarray) -> jnp.ndarray:
        """
        Flip exactly one randomly-chosen bit where `state` and `goal` differ.
        If they already match, return `state` unchanged.
        #TODO
        """
        mask = state != goal         
        has_mismatch = jnp.any(mask)

        def _flip(_):
            # Build a probability vector over *all* indices: 1 for mismatches, 0 otherwise.
            probs = mask.astype(jnp.float32)
            probs /= jnp.sum(probs) 
            idx = random.choice(key, state.shape[0], p=probs)
            return state.at[idx].set(goal[idx])

        # If no mismatches, just return `state`
        new_state = jax.lax.cond(has_mismatch,
                                _flip,                 # true branch
                                lambda _: state,       # false branch
                                operand=None)

        return new_state


    @partial(jax.jit, static_argnums=(0,))
    def reset_env(self, key: chex.PRNGKey, params: EnvParams) -> Tuple[jnp.ndarray, LightBulbsState]:
        """
        Choose a random starting array and call it `bulbs`.
        Use self.goal_set and self.goal_distribution to choose a goal array and call it `goal`.
        Returns:
        Tuple of
        1. first observation
        2. LightBulbsState with the 3 attributes above and timestep counter set at 0.
        """

        def _random_binary_array(key, size):
            return jax.random.bernoulli(key, p=0.5, shape=(size,)).astype(jnp.int32)
        
        def _sample_theta(key, distribution):
            return jax.random.choice(key, distribution.shape[0], p=distribution)

        key, subkey = jax.random.split(key)
        bulbs = _random_binary_array(subkey, self.size)

        key, subkey = jax.random.split(key)
        theta = _sample_theta(subkey, self.goal_distribution)

        goal = self.goals[theta]

        state = LightBulbsState(bulbs = bulbs,
                                goal = goal,
                                t = 0)
        
        return self.get_obs(state), state
    
    def transition(self, key: chex.PRNGKey, state: LightBulbsState, action: int):
        """
        Steps:
        1. Toggle the bulb at the given action index.
        2. Have the human policy toggle one bit toward the goal.
        3. Check if the resulting bulb state matches the goal.
        4. Compute reward.
        """

        # 1. robot move / no-op
        def _noop(bulbs_and_idx):
            bulbs, _ = bulbs_and_idx
            return bulbs

        def _toggle(bulbs_and_idx):
            bulbs, idx = bulbs_and_idx
            return bulbs.at[idx].set(1 - bulbs[idx])

        bulbs_after_robot = jax.lax.cond(
            action == self.size, 
            _noop,
            _toggle,
            (state.bulbs, action)
        )
        

        # 2. Human responds: flip one mismatching bit
        key, subkey = jax.random.split(key)
        bulbs_after_human = self.human_policy(subkey, bulbs_after_robot, state.goal)

        # 3. Check for goal match
        done = jnp.all(bulbs_after_human == state.goal)

        # 4. Reward
        reward = jnp.where(done, 0, -1)

        # 5. Increment timestep
        next_t = state.t + 1

        # 6. Create new state
        next_state = LightBulbsState(
            bulbs=bulbs_after_human,
            goal=state.goal,
            t=next_t
        )

        return next_state, reward, done

    @partial(jax.jit, static_argnums=(0,))
    def step_env(self,
                 key: chex.PRNGKey, # TODO don't really understand if this key is necessary. 
                 state: LightBulbsState,
                 action: int,
                 params: EnvParams):
    
        key, subkey = jax.random.split(key) #TODO I'm worried that I might be using the same key time and time again.
        next_state, reward, done = self.transition(subkey, state, action) 
        next_observation = next_state.bulbs
        return next_observation, next_state, reward, done, {}

