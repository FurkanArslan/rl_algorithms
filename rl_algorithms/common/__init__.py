import logging

from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id="Genius-v0", entry_point="rl_algorithms.common.env:GeniusEnv",
)
