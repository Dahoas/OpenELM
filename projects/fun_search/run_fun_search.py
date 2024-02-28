"""
This module gives an example of how to run the main ELM class.

It uses the hydra library to load the config from the config dataclasses in
configs.py.

This config file demonstrates an example of running ELM with the Sodarace
environment, a 2D physics-based environment in which robots specified by
Python dictionaries are evolved over.

"""
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from openelm import ELM

from openelm.configs import ModelConfig, FunSearchConfig, RLEnvConfig, FitnessCurriculum
from rl_env_descriptions import envs


@hydra.main(
    config_name="elmconfig",
)
def main(config):
    rl_env_name = "chess"
    config.output_dir = HydraConfig.get().runtime.output_dir
    config.model = ModelConfig(model_type="gptquery",
                               model_path="gpt-3.5-turbo-1106",
                               gen_max_len=4096,
                               temp=1.0,
                               batch_size=1,)
    seeds = "/storage/home/hcoda1/6/ahavrilla3/p-wliao60-0/alex/repos/OpenELM/logs/elm/24-02-08_16:01/database.jsonl"
    seeds = "/storage/home/hcoda1/6/ahavrilla3/p-wliao60-0/alex/repos/OpenELM/init_policies/chess/"
    config.qd = FunSearchConfig(seed_policies_dir=seeds)
    curriculum = [{"stockfish_depth": i} for i in range(1, 21)]
    fitness_curriculum = FitnessCurriculum(num_eval_rollouts=20,
                                           curriculum=curriculum,)
    config.env = RLEnvConfig(rl_env_name=rl_env_name,
                             task_type="value",
                             task_description=envs[rl_env_name]["task_description"],
                             observation_description=envs[rl_env_name]["observation_description"],
                             action_description=envs[rl_env_name]["action_description"],
                             reward_description=envs[rl_env_name]["reward_description"],
                             action_exemplar=envs[rl_env_name]["action_exemplar"],
                             fitness_curriculum=fitness_curriculum,)

    print("----------------- Config ---------------")
    print(OmegaConf.to_yaml(config))
    print("-----------------  End -----------------")
    config = OmegaConf.to_object(config)

    elm = ELM(config)
    elm.run(init_steps=config.qd.init_steps, 
            total_steps=config.qd.total_steps)


if __name__ == "__main__":
    main()
