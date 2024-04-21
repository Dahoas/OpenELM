"""
Implementation of FunSearch using https://github.com/google-deepmind/funsearch in ELM
"""
import os
from typing import List, Union, Optional
import dataclasses
from dataclasses import asdict
import time
import torch
import pathlib
import json
from copy import deepcopy, copy
from random import shuffle

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
        obj = asdict(program)
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
    island_ids = program.island_id if type(program.island_id) is list else [program.island_id]

    # First log the program
    self.log(program)
    # Then add the program to each island in the island_ids list
    for island_id in island_ids:
        island_program = deepcopy(program)
        island_program.island_id = island_id
        self.islands[island_id].add(island_program)

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
      self.islands[island_id].add(island_program)

  def sample_programs(self, requires_critique=False):
     """Sample an island and then island programs."""
     islands = copy(self.islands)
     shuffle(islands)
     for island in islands:
        programs = island.sample_programs(requires_critique=requires_critique)
        if None not in programs: break
     return programs

  def filtered_best_sample(self, filters: List[Program], fitness_threshold=0.0):
    # Randomly sort islands
    islands = copy(self.islands)
    shuffle(islands)
    for island in islands:
        samples = sorted(island.samples, key=lambda sample: sample.fitness, reverse=True)
        for sample in samples:
            # Only return program if 
            # - it is not in the filter 
            # - not yet critiqued
            # - has fitness above threshold
            if sample not in filters \
            and sample.critique is None \
            and sample.fitness >= fitness_threshold:
                return sample
    # No valid samples found
    return None

  def remove(self, program):
    island = self.islands[program.island_id]
    island.samples = [sample for sample in island.samples if sample != program]


class Island:
  """A sub-population of the programs database."""

  def __init__(
      self,
      functions_per_prompt: int,
      cluster_sampling_temperature_init: float,
      cluster_sampling_temperature_period: int,
  ) -> None:
    self.functions_per_prompt = functions_per_prompt
    self.cluster_sampling_temperature_init = cluster_sampling_temperature_init
    self.cluster_sampling_temperature_period = cluster_sampling_temperature_period
    
    self.samples: List[Program] = []
    self.best_sample = None

  def add(
      self,
      program: Program,
  ) -> None:
    """Stores a program on this island, in its appropriate cluster."""
    assert program.island_id is not None
    self.samples.append(program)
    if self.best_sample is None or \
    program.fitness > self.best_sample.fitness:
       self.best_sample = program

  @property
  def num_programs(self):
    return len(self.samples)

  def sample_programs(self, requires_critique=False) -> List[Optional[Program]]:
    """Constructs a prompt containing functions from this island."""
    if requires_critique:
        sample_programs = [program for program in self.samples if program.critique is not None]
    else:
        sample_programs = self.samples

    if len(sample_programs) == 0:
        return [None for _ in range(self.functions_per_prompt)]

    # Convert fitnesses to probabilities using softmax with temperature schedule.
    period = self.cluster_sampling_temperature_period
    temperature = self.cluster_sampling_temperature_init * (
        1 - (len(sample_programs) % period) / period)
    logits = np.array([sample.fitness for sample in sample_programs]) / temperature
    assert np.all(np.isfinite(logits))
    probabilities = softmax(torch.tensor(logits, dtype=torch.float32), dim=0).numpy()

    # At the beginning of an experiment when we have few clusters, place fewer
    # programs into the prompt.
    #functions_per_prompt = min(len(self._clusters), self._functions_per_prompt)

    chosen_samples = np.random.choice(sample_programs, size=self.functions_per_prompt, p=probabilities)
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
            # Loading in jsonl
            path = pathlib.Path(self.config.seed_policies_dir)
            print(f"Loading {self.config.seed_policies_dir} as initial policies...")
            assert ".jsonl" in path.name
            with open(path, "r") as f:
                samples = f.readlines()
            samples = [json.loads(sample) for sample in samples]
            print(f"Found {len(samples)} samples.")
            # Extracting programs
            for sample in samples:
                assert "src" in sample
                if "fitness" not in sample or "trajectory_path" not in sample:
                    res = self.env.fitness(sample["src"])
                    sample["fitness"] = res["fitness"]
                    sample["trajectory_path"] = res["trajectory_path"]
                if "island_id" not in sample:
                    sample["island_id"] = list(range(len(self.database.islands)))
                program = Program(src=sample["src"],
                                  fitness=sample["fitness"],
                                  trajectory_path=sample["trajectory_path"],
                                  island_id=sample["island_id"],)
                # If program has a critique, first remove pre-existing instance without critique
                if sample.get("critique") is not None:
                    self.database.remove(program)
                    program.critique = sample["critique"]
                self.database.add(program)
                # Update stats
                res = dict(fitness=sample["fitness"],
                           eval_runtimes=[],
                           fitness_runtime=0,)
                self.update_stats(self.start_step, program, res)
                self.start_step += 1
        print(f"Loading finished! Starting on step {self.start_step}.")

    def random_selection(self):
       return self.database.sample_programs()

    def choose_prompt_mode(self, step):
        if step % self.config.analysis_steps == 0:
            return PromptMode.ANALYZER, MutationMode.CRITIQUE
        else:
            # Check if critique is available for any policy
            if step > self.config.analysis_steps:
                p = {
                    MutationMode.UNCONDITIONAL: 0,
                    MutationMode.CONDITIONAL: 1/4,
                    MutationMode.CRITIQUE: 3/4,  # Select a sample with a critique
                }
            # Otherwise simply sample conditionally
            else:
                p = {
                    MutationMode.UNCONDITIONAL: 0,
                    MutationMode.CONDITIONAL: 1,
                    MutationMode.CRITIQUE: 0,  # Select a sample with a critique
                }
            p = np.array(list(p.values()))
            p = p / sum(p)
            mutation_mode = np.random.choice(list(MutationMode), p=p)
            return PromptMode.DESIGNER, mutation_mode

    def select_batch(self, prompt_mode, mutation_mode):
        if prompt_mode == PromptMode.ANALYZER:
            # If prompt mode is analyzer then want to choose best performing programs without a critique
            batch = []
            for _ in range(self.env.batch_size):
                # Randomly select an island and then select the best program
                sample = self.database.filtered_best_sample(batch)
                assert sample is not None
                batch.append(sample)
            batch = [[sample] for sample in batch]
            return batch
        elif mutation_mode == MutationMode.CRITIQUE:
            # If mutation mode is critique then want to pick a program that has been critiqued and use that for mutation
            batch = []
            for _ in range(self.env.batch_size):
                programs = self.database.sample_programs(requires_critique=True)
                assert None not in programs
                batch.append(programs)
            return batch
        else:
            return [self.random_selection() for _ in range(self.env.batch_size)]

    def update_database(self, 
                        batch: List[List[Program]], 
                        outputs: list[dict], 
                        prompt_mode: PromptMode, 
                        mutation_mode: MutationMode,):
        if prompt_mode == PromptMode.ANALYZER:
            for inp, out in zip(batch, outputs):
                inp = inp[0]
                self.database.remove(inp)
                inp.critique = out["critique"]
                self.database.add(inp)


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
                new_individuals: List[dict] = self.env.random()
                new_individuals = [sample["src"] for sample in new_individuals]
                # Each initial program spreads to all islandsd
                island_ids_list = [list(range(len(self.database.islands))) for _ in new_individuals]
            else:
                # Select a mutation mode
                prompt_mode, mutation_mode = self.choose_prompt_mode(n_steps)
                # Randomly select a batch of individuals based on mutation mode
                batch: List[List[Program]] = self.select_batch(prompt_mode, mutation_mode)
                # Mutate
                outputs: List[dict] = self.env.mutate(batch, prompt_mode, mutation_mode)
                # Update database if given feedback from policy analyzer
                self.update_database(batch, outputs, prompt_mode, mutation_mode)
                # Retrieve src of new generations
                new_individuals: List[str] = [sample["src"] for sample in outputs]
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
                self.update_stats(n_steps, program, res)

    def update_stats(self, n_steps, individual, res):
        fitness = res["fitness"]
        self.stats["eval_runtimes"] += res["eval_runtimes"]
        self.stats["fitness_runtimes"].append(res["fitness_runtime"])
        if fitness > self.stats["best_fitness"]:
            self.stats["best_fitness"] = fitness
            self.stats["best_program"] = individual.src
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
            
