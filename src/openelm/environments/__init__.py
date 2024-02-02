from openelm.environments.base import BaseEnvironment, Genotype

__all__ = [
    "Genotype",
    "BaseEnvironment",
]

import gymnasium as gym
# Register the environment
gym.register(
    id='chess',
    entry_point='openelm.environments.chess_env:ChessEnv',  # Replace 'your_module' with the actual module where `ChessEnv` is defined
)
