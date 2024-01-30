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

from openelm.configs import ModelConfig, FunSearchConfig, RLEnvConfig
from rl_env_descriptions import envs


@hydra.main(
    config_name="elmconfig",
)
def main(config):
    rl_env_name = "MiniGrid-UnlockPickup-v0-wrapped"
    config.output_dir = HydraConfig.get().runtime.output_dir
    config.model = ModelConfig(model_type="gptquery",
                               model_path="gpt-3.5-turbo-1106",#"palm/chat-bison",#"gpt-4-1106-preview",#"gpt-3.5-turbo-1106",
                               gen_max_len=4096,
                               temp=1.0,
                               batch_size=1,)
    config.qd = FunSearchConfig()  # seed_policies_dir="/mnt/c/Users/alexd/Projects/Research/OpenELM/seed_policies/"
    unwrapped_rl_env_name = rl_env_name.replace("-wrapped", "")
    config.env = RLEnvConfig(rl_env_name=rl_env_name,
                             task_description=envs[unwrapped_rl_env_name]["task_description"],
                             observation_description=envs[unwrapped_rl_env_name]["observation_description"],
                             action_description=envs[unwrapped_rl_env_name]["action_description"],
                             reward_description=envs[unwrapped_rl_env_name]["reward_description"],
                             action_exemplar=envs[unwrapped_rl_env_name]["action_exemplar"],
                             api_description=envs[unwrapped_rl_env_name]["api_description"],
                             api_list=envs[unwrapped_rl_env_name]["api_list"],)

    print("----------------- Config ---------------")
    print(OmegaConf.to_yaml(config))
    print("-----------------  End -----------------")
    config = OmegaConf.to_object(config)

    elm = ELM(config)
    elm.run(init_steps=config.qd.init_steps, 
            total_steps=config.qd.total_steps)


if __name__ == "__main__":
    main()
