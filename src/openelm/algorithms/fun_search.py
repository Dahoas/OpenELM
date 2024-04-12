"""
Implementation of FunSearch using https://github.com/google-deepmind/funsearch in ELM
"""
import os
from typing import List, Union, Optional
import dataclasses
import time
import torch
import pathlib
import json
from copy import deepcopy

import numpy as np

from openelm.configs import QDConfig
from openelm.environments import BaseEnvironment, Genotype
from openelm.mutation_models.rl_mutation_model import PromptMode, MutationMode

from torch.nn.functional import softmax


######## Datastructures ########

@dataclasses.dataclass
class Program(Genotype):
  """
  Class defining program samples held in Database
  + src: source code
  + fitness: evaluated fitness of policy in environment
  + trajectory_path: path to file containing policy interaction data
  + island_id: island id, or list of island ids, that the program belongs to
  """
  src: str
  fitness: float
  trajectory_path: str
  island_id: Union[int, List[int]]
  critique: Optional[str] = None

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
          program: Program,):
    """Log program and fitness"""
    with open(self.log_file, "a") as f:
        obj = dict(
           src=program.src,
           fitness=program.fitness,
           island_id=program.island_id,
           trajectory_path=program.trajectory_path,           
        )
        json.dump(obj, f)
        f.write("\n")

  def add(
      self,
      program: Program,
  ) -> None:
    """Registers `program` in the database."""
    # In an asynchronous implementation we should consider the possibility of
    # registering a program on an island that had been reset after the prompt
    # was generated. Leaving that out here for simplicity.
    fitness = program.fitness
    island_ids = program.island_id

    # First log the program
    self.log(program)
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
        # Load seed policies if available
        if self.config.seed_policies_dir is not None:
            path = pathlib.Path(self.config.seed_policies_dir)
            print(f"Loading {self.config.seed_policies_dir} as initial policies...")
            if path.is_file():
                assert ".jsonl" in path.name
                with open(path, "r") as f:
                    samples = f.readlines()
                print(f"Found {len(samples)} samples.")
                samples = [json.loads(sample) for sample in samples]
                for sample in samples:
                    program = Program(src=src, 
                                      fitness=sample["fitness"], 
                                      trajectory_path=sample["trajectory_path"], 
                                      island_id=sample["island_id"],)
                    program.trajectories = sample.get("trajectories")
                    self.database.add(program)
                    # Update stats
                    res = dict(fitness=fitness,
                               eval_runtimes=[],
                               fitness_runtime=0,)
                    self.update_stats(self.start_step, program, res)
                    self.start_step += 1
            else:
                seed_files = path.glob("*")
                for seed_file in seed_files:
                    with open(seed_file, "r") as f:
                        src = "\n".join(f.readlines())
                    t = time.time()
                    res = self.env.fitness(src)
                    res["fitness_runtime"] = time.time() - t
                    island_ids = list(range(len(self.database.islands)))
                    program = Program(src=src,
                                      fitness=res["fitness"],
                                      trajectory_path=res["trajectory_path"],
                                      island_id=island_ids,)
                    self.database.add(program)
                    # Update stats
                    self.update_stats(self.start_step, program, res)
                    self.start_step += 1
        print(f"Loading finished! Starting on step {self.start_step}.")

    def random_selection(self):
       return self.database.sample_programs()

    def choose_prompt_mode(self, step):
        if step % self.config.analysis_steps == 0:
            return PromptMode.ANALYZER, MutationMode.FEEDBACK
        else:
            p = {
                MutationMode.SAMPLING: 1/4,
                MutationMode.MUTATION: 3/4,
                MutationMode.FEEDBACK: 0,
            }
            p = np.array(list(p.values()))
            p = p / sum(p)
            mutation_mode = np.random.choice(list(MutationMode), p=p)
            return PromptMode.DESIGNER, mutation_mode

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
        for n_steps in range(self.start_step, total_steps):
            if n_steps < init_steps:
                # Initialise by generating initsteps random solutions
                new_individuals: List[str] = self.env.random()
                # Each initial program spreads to all islandsd
                island_ids_list = [list(range(len(self.database.islands))) for _ in new_individuals]
            else:
                # Randomly select a batch of individuals
                batch: List[List[Program]] = [self.random_selection() for _ in range(self.env.batch_size)]
                # Next select a mutation mode
                prompt_mode, mutation_mode = self.choose_mode(n_steps)
                # Mutate
                new_individuals: List[str] = self.env.mutate(batch, prompt_mode, mutation_mode)
                # Assign island ids equal to ids of prompt exemplars
                island_ids_list = [[programs[0].island_id] for programs in batch]

            for individual, island_ids in zip(new_individuals, island_ids_list):
                # Evaluate fitness
                t = time.time()
                res = self.env.fitness(individual)
                res["fitness_runtime"] = time.time() - t
                fitness = res["fitness"]
                program = Program(src=individual,
                                  fitness=fitness,
                                  trajectory_path=res["trajectory_path"],
                                  island_id=island_ids,)
                if np.isinf(fitness):
                    continue
                self.database.add(program)
                # Update stats
                self.update_stats(n_steps, individual, res)

    def update_stats(self, n_steps, individual, res):
        fitness = res["fitness"]
        self.stats["eval_runtimes"] += res["eval_runtimes"]
        self.stats["fitness_runtimes"].append(res["fitness_runtime"])
        if fitness > self.stats["best_fitness"]:
            self.stats["best_fitness"] = fitness
            self.stats["best_program"] = individual
            self.stats["best_step"] = n_steps
        self.stats["step"] = n_steps
        if n_steps % self.config.log_stats_steps == 0:
           stats = deepcopy(self.stats)
           eval_runtimes = stats.pop("eval_runtimes")
           stats["eval_runtime_avg"] = np.mean(eval_runtimes)
           stats["eval_runtime_std"] = np.std(eval_runtimes)
           fitness_runtimes = stats.pop("fitness_runtimes")
           stats["fitness_runtime_avg"] = np.mean(fitness_runtimes)
           stats["fitness_runtime_std"] = np.std(fitness_runtimes)
           print(json.dumps(stats, indent=2))
           with open(self.stats_log_file, "a+") as f:
              json.dump(stats, f)
              f.write("\n")
            
