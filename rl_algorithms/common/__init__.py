from gym.envs.registration import register

register(
    id="Genius-v1", entry_point="rl_algorithms.common.env:GeniusEnv",
)

register(
    id="GeniusContinuous-v1", entry_point="rl_algorithms.common.env:GeniusContinuesEnv",
)
