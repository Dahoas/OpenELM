import os
import re
from enum import Enum
from gptquery.gpt import GPT
from gptquery import dict_to_jsonl, jsonl_to_dict
from typing import List

from openelm.mutation_model import MutationModel
from openelm.mutation_models.inference_pipelines import CritiquePipeline
from openelm.configs import RLEnvModelConfig
from openelm.mutation_models.prompts import designer_prompts


class PromptMode(Enum):
    DESIGNER = 0
    ANALYZER = 1

    @classmethod
    def from_string(cls, name):
        name_to_val = {val.name: val for val in cls}
        if name_to_val.get(name.upper(), None):
            return name_to_val[name.upper()]
        else:
            raise ValueError(f"Unknown name: {name}!!!")


class MutationMode(Enum):
    UNCONDITIONAL = 0
    CONDITIONAL = 1
    CRITIQUE = 2

    @classmethod
    def from_string(cls, name):
        name_to_val = {val.name: val for val in cls}
        if name_to_val.get(name.upper(), None):
            return name_to_val[name.upper()]
        else:
            raise ValueError(f"Unknown name: {name}!!!")


class PolicyDesigner:
    def __init__(self, config: RLEnvModelConfig):
        self.config = config
        logging_path = os.path.join(config.logging_path, "designer.jsonl")
        self.llm = GPT(model_name=config.designer_model_path,
                       temperature=config.designer_temp,
                       max_num_tokens=config.gen_max_len,
                       mb_size=config.batch_size,
                       task_prompt_text="{prompt}",
                       logging_path=logging_path,)
        self.is_complete_keyword = "<DONE>"

    def extract_src(self, response: str):
        try:
            response = response.replace(self.is_complete_keyword, "")
            return re.findall(r"```python\n([^`]*)```", response)[-1]
        except IndexError:
            return ""

    def construct_input(self, batch: list, mutation_mode: MutationMode):
        for sample in batch:
            env_description = designer_prompts["env_description"].format(**sample)
            unconditional_prompt = designer_prompts["unconditional_prompt"].format(env_description=env_description, is_complete_keyword=self.is_complete_keyword)
            if mutation_mode == MutationMode.UNCONDITIONAL:
                sample["prompt"] = unconditional_prompt
            else:
                conditional_prompt = designer_prompts["conditional_prompt"].format(unconditional_prompt=unconditional_prompt, policy=sample["policy"])
                if mutation_mode == MutationMode.CONDITIONAL:
                    sample["prompt"] = conditional_prompt
                elif mutation_mode == MutationMode.CRITIQUE:
                    assert sample["critique"] is not None
                    sample["prompt"] = designer_prompts["critique_prompt"].format(conditional_prompt=conditional_prompt, critique=sample["critique"])
        return batch

    def __call__(self, batch: list, mutation_mode: MutationMode):
        batch = self.construct_input(batch, mutation_mode)
        out_batch = self.llm(batch, is_complete_keyword=self.is_complete_keyword)
        responses = [sample["response"] for sample in out_batch]
        policies = [self.extract_src(response) for response in responses]
        return [{"src": policy} for policy in policies]


class PolicyAnalyzer:
    def __init__(self, config: RLEnvModelConfig):
        self.config = config
        logging_path = os.path.join(config.logging_path, "analyzer.jsonl")
        self.critique_pipeline = CritiquePipeline(logging_path)

    def construct_input(self, batch):
        for sample in batch:
            sample["env_description"] = designer_prompts["env_description"].format(**sample)
        return batch

    def __call__(self, batch: list):
        # Need to put batch in dict format
        batch = self.construct_input(batch)
        dict_batch = jsonl_to_dict(batch)
        out_batch = dict_to_jsonl(self.critique_pipeline(dict_batch))
        critiques = [sample.get("critique", "") for sample in out_batch]
        return [{"critique": critique} for critique in critiques]


class RLEnvModel(MutationModel):
    def __init__(self, config: RLEnvModelConfig):
        self.config = config
        
        self.policy_designer = PolicyDesigner(config)
        self.policy_analyzer = PolicyAnalyzer(config)

    def generate_programs(self, 
                 sample_dicts: List[dict[str, str]],
                 prompt_mode: PromptMode,
                 mutation_mode: MutationMode) -> list[dict]:
        self(sample_dicts, prompt_mode, mutation_mode)

    def __call__(self, 
                 sample_dicts: List[dict[str, str]],
                 prompt_mode: PromptMode,
                 mutation_mode: MutationMode) -> list[dict]:
        """
        Generates programs using prompts in prompt_dicts.
        + prompt_dicts: List of dicts with "prompt" field
        """
        if prompt_mode == PromptMode.DESIGNER:
            return self.policy_designer(sample_dicts, mutation_mode)
        elif prompt_mode == PromptMode.ANALYZER:
            critiques = self.policy_analyzer(sample_dicts)
            sample_dicts = [{**sample_dict, **critique} for sample_dict, critique in zip(sample_dicts, critiques)]
            policies = self.policy_designer(sample_dicts, MutationMode.CRITIQUE)
            outputs = [{**policy, **critique} for policy, critique in zip(policies, critiques)]
            return outputs