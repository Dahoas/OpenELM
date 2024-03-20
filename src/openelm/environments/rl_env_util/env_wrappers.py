"""
Note: There is a Wrapper class provided in gynasium: 
https://gymnasium.farama.org/_modules/gymnasium/core/#Env
"""
import gymnasium as gym
import numpy as np
from minigrid.core.world_object import Door, Ball, Key


class BaseWrapper:
    def __init__(self, env):
        self.env = env

    def step(self, action):
        return self.env(action)
    
    def reset(self, seed):
        return self.env.reset(seed=seed)
    
    def render(self):
        self.env.render()


class MinigridBlockedUnlockPickupWrapper(BaseWrapper):
    def __init__(self, env):
        super(MinigridBlockedUnlockPickupWrapper, self).__init__(env)
        self.has_key = False
        self.has_ball = False
        self.door_open = False

    def get_grid_types(self):
        return [type(obj) for obj in self.grid]
    
    def get_has_key(self):
        self.has_key = self.has_key or Key not in self.get_grid_types()
        return self.has_key
    
    def get_has_ball(self):
        self.has_ball = self.has_ball or Ball not in self.get_grid_types()
        return self.has_ball
    
    def get_door_open(self):
        self.door_open = self.door_open or self.door.is_open
        return self.door_open

    def step(self, action):
        """
        We will shape the reward by giving a reward of
            +0.1 if any object is picked up
            +0.2 if door is opened
        """
        prev_has_key = self.get_has_key()
        prev_has_ball = self.get_has_ball()
        prev_door_open = self.get_door_open()

        observation, reward, terminated, _, _ = self.env.step(action)
        cur_has_key = self.get_has_key()
        cur_has_ball = self.get_has_ball()
        cur_door_open = self.get_door_open()

        if not prev_has_ball and cur_has_ball:
            reward += 0.1
        elif not prev_has_key and cur_has_key:
            reward += 0.1
        elif not prev_door_open and cur_door_open:
            reward += 0.2

        return observation, reward, terminated, _, _
    
    def reset(self, seed):
        observation, _ = self.env.reset(seed=seed)
        self.cur_observation = observation
        self.has_key = False
        self.has_ball = False
        self.door_open = False

        self.grid = self.env.grid.grid
        self.door = None
        for obj in self.grid:
            if type(obj) is Door:
                self.door = obj
        assert self.door is not None

        return observation, _
    

class MinigridUnlockPickupWrapper(BaseWrapper):
    def __init__(self, env):
        super(MinigridUnlockPickupWrapper, self).__init__(env)
        self.has_key = False
        self.door_open = False
        self.gone_through_door = False

    def get_grid_types(self):
        return [type(obj) for obj in self.grid]
    
    def get_has_key(self):
        self.has_key = self.has_key or Key not in self.get_grid_types()
        return self.has_key
    
    def get_door_open(self):
        self.door_open = self.door_open or self.door.is_open
        return self.door_open
    
    def get_through_door(self):
        self.gone_through_door = self.gone_through_door or self.cur_observation["image"][3][6][0] == 4
        return self.gone_through_door
    
    def postprocess_obs(self, observation):
        inv = [5] if Key not in self.get_grid_types() else []
        observation = {"agent": observation, "inv": inv}
        return observation

    def step(self, action):
        """
        We will shape the reward by giving a reward of
            +0.1 if any object is picked up
            +0.2 if door is opened
            +0.1 if gone through door
        """
        prev_has_key = self.get_has_key()
        prev_door_open = self.get_door_open()
        prev_through_door = self.get_through_door()

        observation, reward, terminated, _, _ = self.env.step(action)
        self.cur_observation = observation
        cur_has_key = self.get_has_key()
        cur_door_open = self.get_door_open()
        cur_through_door = self.get_through_door()

        if not prev_has_key and cur_has_key:
            reward += 0.1
        elif not prev_door_open and cur_door_open:
            reward += 0.2
        elif not prev_through_door and cur_through_door:
            reward += 0.1
            
        observation = self.postprocess_obs(observation)
        return observation, reward, terminated, _, _
    
    def reset(self, seed):
        observation, _ = self.env.reset(seed=seed)
        self.cur_observation = observation
        self.has_key = False
        self.has_ball = False
        self.door_open = False
        self.gone_through_door = False

        self.grid = self.env.grid.grid
        self.door = None
        for obj in self.grid:
            if type(obj) is Door:
                self.door = obj
        assert self.door is not None

        observation = self.postprocess_obs(observation)
        return observation, _


def get_wrapped_env(rl_env_name, render_mode):
    if "wrapped" in rl_env_name:
        rl_env_name = rl_env_name.replace("-wrapped", "")
        env = gym.make(rl_env_name, render_mode=render_mode)
        if rl_env_name == "MiniGrid-BlockedUnlockPickup-v0":
            return MinigridBlockedUnlockPickupWrapper(env)
        elif rl_env_name == "MiniGrid-UnlockPickup-v0":
            return MinigridUnlockPickupWrapper(env)
        else:
            raise ValueError(f"No wrapper found for {rl_env_name}!!!")
    else:
        return gym.make(rl_env_name, render_mode=render_mode)