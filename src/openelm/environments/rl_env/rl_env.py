import json
import re
import warnings
from typing import Optional, Union

import numpy as np

from openelm.configs import P3ProblemEnvConfig, P3ProbSolEnvConfig
from openelm.environments.base import BaseEnvironment, Genotype, Phenotype
from openelm.mutation_model import MutationModel
from openelm.sandbox.server.sandbox_codex_execute import ExecResult
from openelm.utils.code_eval import pass_at_k, pool_exec_processes, type_check


class PolicyGenotype(Genotype):
    def __init__(self, 
                 policy_str: str,):
        """
        Genotype for policy parameterized by python coe.
        Args:
            policy_str: the policy in python code as a string
        """
        self.policy_str = policy_str

    def to_phenotype(self) -> Phenotype:
        raise ValueError("Undecided phenotype for python policies")


class ELMRLEnv(BaseEnvironment[PolicyGenotype]):
    def __init__(self,
                 config,
                 mutation_model,
                 env,
                 game_description,
                 state_description,):
        """
        The objective is to generate well-performing python policies 
        for the given environment.
        + config: the config file path or dict
        + mutation_model: model used to mutate existing python policies
        + env: given RL environments
        + game_description: description of game
        + state_description: description of state
        """
        self.config = config
        self.mutation_model = mutation_model
        self.env = env

    def get_rng_state(self) -> Optional[np.random._generator.Generator]:
        warnings.warn("WARNING: rng state not used in this environment")
        return None

    def set_rng_state(self, rng_state: Optional[np.random._generator.Generator]):
        warnings.warn("WARNING: rng state not used in this environment")
        pass

    def random(self) -> list[P3Solution]:
        program_list = [self.construct_prompt() for _ in range(self.config.batch_size)]
        new_solutions = self.generate_programs(program_list)
        return new_solutions
    
    def mutate(self, sol_list: list[P3Solution]) -> list[P3Solution]:
        sols = [s.program_str for s in sol_list]
        program_list = list(map(self.construct_prompt, sols))
        new_sols = self.generate_programs(program_list)
        return new_sols

    def fitness(self, sol: P3Solution) -> float:
        """
        If passing the solution to the problem returns True, fitness is 1.0
            else -np.inf
        """
        result = self.evaluate_solution(sol)

        if result is True:
            return 1.0
        else:
            return -np.inf
        
    def construct_prompt(self):
        pass