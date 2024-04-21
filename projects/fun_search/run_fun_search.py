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

from openelm.configs import RLEnvModelConfig, FunSearchConfig, RLEnvConfig, FitnessCurriculum
from openelm.environments.rl_env_util.rl_env_descriptions import envs

import os


@hydra.main(
    config_name="elmconfig",
)
def main(config):
    rl_env_name = "MiniGrid-UnlockPickup-v0-wrapped"
    config.output_dir = HydraConfig.get().runtime.output_dir
    config.model = RLEnvModelConfig(model_type="gptquery",
                                    designer_model_path="gpt-3.5-turbo-0125", #"gpt-4-0125-preview",  # gpt-3.5-turbo-0125, # claude-3-haiku-20240307
                                    analyzer_model_path="gpt-4-turbo",
                                    designer_temp=1.1,
                                    analyzer_temp=0.3,
                                    gen_max_len=4096,
                                    batch_size=1,
                                    model_path="",)
    total_steps = 250
    init_steps = 1
    analysis_steps = 500
    seed_policies_dir = "/storage/home/hcoda1/6/ahavrilla3/p-wliao60/alex/repos/OpenELM/projects/fun_search/init_policies/door_key/jsonls/init.jsonl"
    database_config = dict(functions_per_prompt=1,
                           num_islands=5,
                           reset_period=50,
                           cluster_sampling_temperature_init=0.1,
                           cluster_sampling_temperature_period=30_000,)
    config.qd = FunSearchConfig(total_steps=total_steps, 
                                init_steps=init_steps,
                                analysis_steps=analysis_steps,
                                database_config=database_config,
                                seed_policies_dir=seed_policies_dir,)
    num_eval_rollouts = 10
    horizon = 300
    curriculum = [dict() for _ in range(num_eval_rollouts)]
    fitness_curriculum = FitnessCurriculum(num_eval_rollouts=num_eval_rollouts,
                                           curriculum=curriculum,)
    rl_env_name_t = rl_env_name
    rl_env_name = rl_env_name.replace("-wrapped", "")
    config.env = RLEnvConfig(rl_env_name=rl_env_name_t,
                             task_type="policy",
                             task_description=envs[rl_env_name]["task_description"],
                             observation_description=envs[rl_env_name]["observation_description"],
                             action_description=envs[rl_env_name]["action_description"],
                             reward_description=envs[rl_env_name]["reward_description"],
                             action_exemplar=envs[rl_env_name]["action_exemplar"],
                             fitness_curriculum=fitness_curriculum,
                             api_description="",
                             api_list=[],
                             horizon=horizon,)

    print("----------------- Config ---------------")
    print(OmegaConf.to_yaml(config))
    print("-----------------  End -----------------")
    config = OmegaConf.to_object(config)

    elm = ELM(config)
    elm.run(init_steps=config.qd.init_steps, 
            total_steps=config.qd.total_steps)


if __name__ == "__main__":
    main()
