import gymnasium as gym
import numpy as np
import time
from functools import reduce
from typing import Optional, List, Any
import json
import cProfile
import argparse

from openelm.environments.rl_env import ELMRLEnv, Program
from openelm.configs import RLEnvConfig
import multiprocessing as mp
from openelm.environments.rl_envs.chess_env import ChessEnv

def make_env(env):
    env = ChessEnv()
    env.reset(seed=42)

procs = []
for i in range(2):
    p = mp.Process(target=make_env, args=())
    p.start()
    procs.append(p)

for i in range(2):
    procs[i].join()
