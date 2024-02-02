"""
Implementation of FunSearch using https://github.com/google-deepmind/funsearch in ELM
"""
import os
from typing import Optional, List
import dataclasses
import time
import torch
import pathlib
import json
from copy import deepcopy

import numpy as np

from openelm.configs import QDConfig
from openelm.environments import BaseEnvironment, Genotype

from torch.nn.functional import softmax


######## Datastructures ########

@dataclasses.dataclass
class Program(Genotype):
  """
  Class defining program samples held in Database
  + src: source code
  """
  src: str
  island_id: Optional[int] = None

  def __str__(self) -> str:
     return self.src
  
  def to_phenotype(self) -> None:
     return None


@dataclasses.dataclass(frozen=True)
class DatabaseConfig:
  """Configuration of a ProgramsDatabase.

  Attributes:
    + functions_per_prompt: Number of previous programs to include in prompts.
    + num_islands: Number of islands to maintain as a diversity mechanism.
    + reset_period: How often (in steps) the weakest islands should be reset.
    + cluster_sampling_temperature_init: Initial temperature for softmax sampling
        of clusters within an island.
    + cluster_sampling_temperature_period: Period of linear decay of the cluster
        sampling temperature.
  """
  functions_per_prompt: int = 1
  num_islands: int = 10
  reset_period: int = 100
  cluster_sampling_temperature_init: float = 0.1
  cluster_sampling_temperature_period: int = 30_000


class Database:
  """A collection of programs, organized as islands."""

  def __init__(
      self,
      config: DatabaseConfig,
      log_dir: str,
  ) -> None:
    self.config: DatabaseConfig = config
    # Set up logging path
    self.log_dir = pathlib.Path(log_dir).mkdir(exist_ok=True, parents=True)
    self.log_file = os.path.join(log_dir, "database.jsonl")
    # Initialize empty islands.
    self.islands: list[Island] = []
    for _ in range(config.num_islands):
      self.islands.append(
          Island(config.functions_per_prompt,
                 config.cluster_sampling_temperature_init,
                 config.cluster_sampling_temperature_period,))

    self.tot_programs = 0

  def log(self, 
          program: Program, 
          fitness: float,
          island_ids: List[int],):
    """Log program and fitness"""
    with open(self.log_file, "a") as f:
        obj = dict(
           src=program.src,
           fitness=fitness,
           islands=island_ids,           
        )
        json.dump(obj, f)
        f.write("\n")

  def add(
      self,
      program: Program,
      fitness: float,
      island_ids: List[int],
  ) -> None:
    """Registers `program` in the database."""
    # In an asynchronous implementation we should consider the possibility of
    # registering a program on an island that had been reset after the prompt
    # was generated. Leaving that out here for simplicity.

    # First log the program
    self.log(program, fitness, island_ids)
    # Then add the program to each island in the island_ids list
    for island_id in island_ids:
        island_program = deepcopy(program)
        island_program.island_id = island_id
        self.islands[island_id].add(island_program, fitness)

    self.tot_programs += 1
    # Check whether it is time to reset an island.
    if (self.tot_programs + 1) % self.config.reset_period == 0:
      self.reset_islands()

  def reset_islands(self) -> None:
    """Resets the weaker half of islands."""
    # We sort best scores after adding minor noise to break ties.
    best_score_per_island = [island.best_sample.fitness for island in self.islands]
    indices_sorted_by_score: np.ndarray = np.argsort(best_score_per_island + np.random.randn(len(best_score_per_island)) * 1e-6)
    num_islands_to_reset = self.config.num_islands // 2
    reset_islands_ids = indices_sorted_by_score[:num_islands_to_reset]
    keep_islands_ids = indices_sorted_by_score[num_islands_to_reset:]
    for island_id in reset_islands_ids:
      island_id = int(island_id)
      # Reset island by:
      # First, dumping all existing samples
      # Second, sampling surviving island and adding best program to reset
      self.islands[island_id] = Island(
          self.config.functions_per_prompt,
          self.config.cluster_sampling_temperature_init,
          self.config.cluster_sampling_temperature_period)
      founder_island_id = np.random.choice(keep_islands_ids)
      island = self.islands[founder_island_id]
      founder, founder_fitness = island.best_sample.program, island.best_sample.fitness
      island_program = deepcopy(founder)
      island_program.island_id = island_id
      self.islands[island_id].add(island_program, founder_fitness)

  def sample_programs(self):
     """Sample an island and then island programs."""
     island = np.random.choice(self.islands)
     programs = island.sample_programs()
     return programs


class Island:
  """A sub-population of the programs database."""

  @dataclasses.dataclass
  class Sample:
     program: Program
     fitness: float

  def __init__(
      self,
      functions_per_prompt: int,
      cluster_sampling_temperature_init: float,
      cluster_sampling_temperature_period: int,
  ) -> None:
    self.functions_per_prompt = functions_per_prompt
    self.cluster_sampling_temperature_init = cluster_sampling_temperature_init
    self.cluster_sampling_temperature_period = cluster_sampling_temperature_period
    
    self.samples: List[Island.Sample] = []
    self.num_programs = 0
    self.best_sample = None

  def add(
      self,
      program: Program,
      fitness: float,
  ) -> None:
    """Stores a program on this island, in its appropriate cluster."""
    assert program.island_id is not None
    sample = Island.Sample(program, fitness)
    self.samples.append(sample)
    self.num_programs += 1
    if self.best_sample is None or \
    fitness > self.best_sample.fitness:
       self.best_sample = sample

  def sample_programs(self) -> List[Program]:
    """Constructs a prompt containing functions from this island."""

    # Convert fitnesses to probabilities using softmax with temperature schedule.
    period = self.cluster_sampling_temperature_period
    temperature = self.cluster_sampling_temperature_init * (
        1 - (self.num_programs % period) / period)
    logits = np.array([sample.fitness for sample in self.samples]) / temperature
    assert np.all(np.isfinite(logits))
    probabilities = softmax(torch.tensor(logits, dtype=torch.float32), dim=0).numpy()

    # At the beginning of an experiment when we have few clusters, place fewer
    # programs into the prompt.
    #functions_per_prompt = min(len(self._clusters), self._functions_per_prompt)

    idx = np.random.choice(self.num_programs, size=self.functions_per_prompt, p=probabilities)
    chosen_samples = [self.samples[i].program for i in idx]
    return chosen_samples


######## Algorithm ########

class FunSearch:
    """
    Class implementing FunSearch algorithm: 
    https://www.nature.com/articles/s41586-023-06924-6
    """

    def __init__(
        self,
        env,
        config: QDConfig,
    ):
        """
        Class implementing FunSearch algorithm: 
        https://www.nature.com/articles/s41586-023-06924-6

        Args:
            env (BaseEnvironment): The environment to evaluate solutions in. This
            should be a subclass of `BaseEnvironment`, and should implement
            methods to generate random solutions, mutate existing solutions,
            and evaluate solutions for their fitness in the environment.
            config (QDConfig): The configuration for the algorithm.
            init_databse (Databse, optional): A databse to use for the algorithm. If not passed,
            a new databse will be created. Defaults to None.
        """
        self.env: BaseEnvironment = env
        self.config: QDConfig = config
        self.save_history = self.config.save_history
        self.save_snapshot_interval = self.config.save_snapshot_interval
        self.start_step = 0
        self.save_np_rng_state = self.config.save_np_rng_state
        self.load_np_rng_state = self.config.load_np_rng_state
        self.rng = np.random.default_rng(self.config.seed)
        self.rng_generators = None

        self.database = Database(DatabaseConfig(**config.database_config), log_dir=config.output_dir)
        self.stats = dict(
           step=0,
           best_step=0,
           best_fitness=-np.inf,
           best_program="",
           eval_runtimes=[],
           fitness_runtimes=[],
        )
        self.stats_log_file = os.path.join(config.output_dir, "stats.jsonl")
        print("Loading finished!")

    def random_selection(self):
       return self.database.sample_programs()

    def search(self, 
               init_steps: int, 
               total_steps: int, 
               atol: float = 0.0) -> str:
        """
        Run the genetic algorithm.

        Args:
            initsteps (int): Number of initial random solutions to generate.
            totalsteps (int): Total number of steps to run the algorithm for,
                including initial steps.
            atol (float, optional): Tolerance for how close the best performing
                solution has to be to the maximum possible fitness before the
                search stops early. Defaults to 1.

        Returns:
            str: A string representation of the best perfoming solution. The
                best performing solution object can be accessed via the
                `current_max_genome` class attribute.
        """
        total_steps = int(total_steps)
        for n_steps in range(total_steps):
            if n_steps < init_steps:
                # Initialise by generating initsteps random solutions
                new_individuals: List[Program] = self.env.random()
                # Each initial program spreads to all islandsd
                island_ids_list = [list(range(len(self.database.islands))) for _ in new_individuals]
            else:
                # Randomly select a batch of individuals
                batch: List[List[Program]] = [self.random_selection() for _ in range(self.env.batch_size)]
                # Mutate
                new_individuals: List[Program] = self.env.mutate(batch)
                # Assign island ids equal to ids of prompt exemplars
                island_ids_list = [[programs[0].island_id] for programs in batch]

            for individual, island_ids in zip(new_individuals, island_ids_list):
                # Evaluate fitness
                res = self.env.fitness(individual)
                fitness = res["fitness"]
                if np.isinf(fitness):
                    continue
                self.database.add(individual, fitness, island_ids=island_ids)
                # Update stats
                self.stats["eval_runtimes"] += res["eval_runtimes"]
                self.stats["fitness_runtimes"].append(sum(res["eval_runtimes"]))
                if fitness > self.stats["best_fitness"]:
                   self.stats["best_fitness"] = fitness
                   self.stats["best_program"] = individual.src
                   self.stats["best_step"] = n_steps

            self.stats["step"] = n_steps
            if n_steps % self.config.log_stats_steps == 0 and n_steps > 1:
               stats = deepcopy(self.stats)
               eval_runtimes = stats.pop("eval_runtimes")
               stats["eval_runtime_avg"] = np.mean(eval_runtimes)
               stats["eval_runtime_std"] = np.std(eval_runtimes)
               fitness_runtimes = stats.pop("fitness_runtimes")
               stats["fitness_runtime_avg"] = np.mean(fitness_runtimes)
               stats["fitness_runtime_std"] = np.std(fitness_runtimes)
               print(json.dumps(self.stats, indent=2))
               with open(self.stats_log_file, "a+") as f:
                  json.dump(self.stats, f)
            