# -*- coding: utf-8 -*-
"""Functions for Genius environment.

- Author: Furkan Arslan
- Contact: furkan.arslan@ozu.edu.tr
"""
import gym
from gym import spaces
import numpy as np

_LOWEST_VALUE = np.float32(0.0)
_HIGHEST_VALUE = np.float32(1.0)


def _get_observation_space() -> spaces.Box:
    high = np.concatenate(
        [
            np.array([_HIGHEST_VALUE]),
            np.array([_HIGHEST_VALUE]),
            np.array([_HIGHEST_VALUE]),
        ]
    )
    low = np.concatenate(
        [
            np.array([_LOWEST_VALUE]),
            np.array([_LOWEST_VALUE]),
            np.array([_LOWEST_VALUE]),
        ]
    )

    return spaces.Box(low=low, high=high)


class GeniusEnv(gym.Env):
    def reset(self):
        pass

    def render(self, mode="human"):
        pass

    def step(self, action):
        pass

    def __init__(self):
        self.action_space = spaces.Box(
            low=_LOWEST_VALUE, high=_HIGHEST_VALUE, shape=(1,)
        )
        self.observation_space = _get_observation_space()
