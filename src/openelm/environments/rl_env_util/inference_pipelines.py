from typing import List, Dict, Any, Optional
import re
import json
import os

from gptquery import GPT
from gptquery.datatypes import PipelineState
from gptquery.pipelines import InferenceComponent, InferencePipeline, PipelineState
from gptquery.composer import InferenceComposer
from gptquery import dict_to_jsonl, jsonl_to_dict

from openelm.environments.rl_env_util.reports import compute_reports
from openelm.environments.rl_env_util.rl_env_descriptions import envs as env_descriptions


class InitialAnalysis(InferenceComponent):
    def __init__(self, name, logging_path):
        super().__init__(name)
        self.analysis_prompt = """\
You are responsible for critiquing a decision policy to solve the following task:\n\
{env_description}\n\n\
Here is an example policy:\n\n\
{policy}\n\n\
The above policy has been rolled out in the environment and a dataset of trajectories has been collected.
You must now analyze this dataset to better understand the policy's behavior and environment dynamics.
This is done by using reporting functions which capture parts of the policy's behavior and the environmental dynamics.\n\n\
Report:\n\n\
{report_results}\n\n\
Based on this report form ONE specific, concrete hypothesis about how the policy may be failing. \
Enclose this hypothesis within the gats <hypothesis>...</hypothesis>\n\n\
If you feel absolutely confident the hypothesis is confirmed by the existing report print CONFIRMED. \
Then write a specific, concise critique enclosed in <critique>...</critique> tags explaining what is wrong and a specific, concrete strategy for how it might fixed. \
If you are not confident then do not write a critique and print UNCERTAIN.\
"""
        logging_path = os.path.join(logging_path, "policy_analyzer.jsonl")
        self.gpt = GPT(model_name="gpt-4-turbo",
                       task_prompt_text=self.analysis_prompt,
                       logging_path=logging_path,)

    def __call__(self, data, start, end):
        jsonl_batch = dict_to_jsonl(data)
        jsonl_batch = self.gpt(jsonl_batch, output_key="analysis_response")
        for sample in jsonl_batch:
            analysis_response = sample.pop("analysis_response")
            sample["hypothesis_confirmed"] = "CONFIRMED" in analysis_response
            if sample["hypothesis_confirmed"]:
                sample["critique"] = re.findall(r"<critique>([\s\S]*)</critique>", analysis_response)[0]
            else:
                sample["hypothesis"] = re.findall(r"<hypothesis>([\s\S]*)</hypothesis>", analysis_response)[0]
        data = jsonl_to_dict(jsonl_batch)
        return data, start
    

def analysis_stopping_criteria(sample):
    return sample.get("hypothesis_confirmed", False)
    

class ProposeReport(InferenceComponent):
    def __init__(self, name, logging_path):
        super().__init__(name)
        self.propose_prompt = """\
You are responsible for critiquing a decision policy to solve the following task:\n\
{env_description}\n\n\
Here is an example policy:\n\n\
{policy}\n\n\
The above policy has been rolled out in the environment and a dataset of trajectories has been collected.
You must now analyze this dataset to better understand the policy's behavior and environment dynamics.
This is done by using reporting functions which capture parts of the policy's behavior and the environmental dynamics.
Note: len(trajectory["observations"]) == len(trajectory["actions"]) + 1 because the observation after 
the last action is also saved.\n\n\
You generated the following hypothesis:\n\n\
{hypothesis}\n\n\
To test your hypothesis you must use a report function which aggregates information about 
policy rollouts in the environment to better understand policy behavior and environmental dynamics. 
To do so you must design your own report function.

New report functions are of form "REPORT_NAME(observation, action, next_observation, memory: dict) -> dict.
- observation: observation at current timestep
- action: policy's action after seeing observation
- next_observation: observation of state after taking action
- memory: dict, a persistent object passed between calls to "REPORT_NAME" over a single trajectory. Can be used to store information
for report computation over multiple timesteps. memory["final_step"] = False until the end of the trajectory at which point it is True.
Each report should include a "description" variable describing the report. 
After the final time-step of the trajectory memory["result"]: Union[float, dict[str, float]] should either contain the 
final result of the report for the trajectory or a dictionary of final results for sub-statistics
memory["description"]: str should contain a detailed description of the report. "REPORT_NAME" should return the memory dict.
The report across all trajectories is computed by taking the mean for each trajectory.
The report should be contained in a single code block.

Advice for designing good reports:
- You should never assume an action results in the expected environmental change. Instead you should use observation and next_observation to check what change (if any) the action produced.
- If you are reporting a ratio of two numbers you should also report the numerator and denominator.\
"""
        logging_path = os.path.join(logging_path, "policy_analyzer.jsonl")
        self.gpt = GPT(model_name="gpt-4-turbo",
                       task_prompt_text=self.propose_prompt,
                       logging_path=logging_path,)
        
    def extract_report(self, response):
        report = re.findall(r"```python([\s\S]*)```", response)[0]
        # Get function name
        report_name = re.findall(r"def (\S*)\(", report)[0]
        # Load python object
        res = dict()
        exec(report, res)
        report = res[report_name]
        return {report_name: report}

    def __call__(self, data, start, end):
        jsonl_batch = dict_to_jsonl(data)
        jsonl_batch = self.gpt(jsonl_batch, output_key="report_response")
        for sample in jsonl_batch:
            report_response = sample.pop("report_response")
            report = self.extract_report(report_response)
            sample["report_list"] = {**sample.get("report_list", {}), **report}
        data = jsonl_to_dict(jsonl_batch)
        return data, start


class ComputeReport(InferenceComponent):
    def __init__(self, name):
        super().__init__(name)

    def load_trajectories(self, trajectory_path):
        with open(trajectory_path, "r") as f:
            trajectories = json.load(f)
        return trajectories

    def __call__(self, data, start, end):
        jsonl_batch = dict_to_jsonl(data)
        for sample in jsonl_batch:
            trajectory_path = sample["trajectory_path"]
            trajectories = self.load_trajectories(trajectory_path)
            report_list = sample.get("report_list", {})
            sample["report_results"] = compute_reports(trajectories=trajectories, extra_reports=report_list)
        data = jsonl_to_dict(jsonl_batch)
        return data, start


class AnalyzeReport(InferenceComponent):
    def __init__(self, name, logging_path):
        super().__init__(name)
        self.analysis_prompt = """\
You are responsible for critiquing a decision policy to solve the following task:\n\
{env_description}\n\n\
Here is an example policy:\n\n\
{policy}\n\n\
The above policy has been rolled out in the environment and a dataset of trajectories has been collected.
You must now analyze this dataset to better understand the policy's behavior and environment dynamics.
This is done by using reporting functions which capture parts of the policy's behavior and the environmental dynamics.\n\n\
You have formed the following hypothesis:\n\
{hypothesis}\n\n\
To test this hypothesis you have evaluated the policy using the following report:\n\
{report_results}\n\n\
If you feel absolutely confident the hypothesis is confirmed by the existing report print CONFIRMED. \
Then write a specific, concise critique enclosed in <critique>...</critique> tags explaining what is wrong and a specific, concrete strategy for how it might fixed. \
If you feel the hypothesis may still be correct but needs more testing then print UNCERTAIN. \
If the report refutes the hypothesis print REFUTED. Then generate a new concrete, specific hypothesis based on the existing reports. \
Enclose this hypothesis within the gats <hypothesis>...</hypothesis>\n\n\
"""
        logging_path = os.path.join(logging_path, "policy_analyzer.jsonl")
        self.gpt = GPT(model_name="gpt-4-turbo",
                       task_prompt_text=self.analysis_prompt,
                       logging_path=logging_path,)

    def __call__(self, data, start, end):
        jsonl_batch = dict_to_jsonl(data)
        jsonl_batch = self.gpt(jsonl_batch, output_key="analysis_response")
        for sample in jsonl_batch:
            analysis_response = sample.pop("analysis_response")
            sample["hypothesis_confirmed"] = "CONFIRMED" in analysis_response
            if sample["hypothesis_confirmed"]:
                sample["critique"] = re.findall(r"<critique>([\s\S]*)</critique>", analysis_response)[0]
            elif "REFUTED":
                # Update with new hypothesis
                sample["hypothesis"] = re.findall(r"<hypothesis>([\s\S]*)</hypothesis>", analysis_response)[0]
        data = jsonl_to_dict(jsonl_batch)
        return data, start


class CritiquePipeline:
    def __init__(self, logging_path):
        self.report_pipeline = InferencePipeline(name="gpt4_report_pipeline", 
                                       max_steps=3,  # Iterate analysis pipeline a max of three times
                                       inference_components=[ProposeReport(name="propose_report", logging_path=logging_path),
                                                             ComputeReport(name="compute_report"),
                                                             AnalyzeReport(name="analyze_report", logging_path=logging_path),],
                                        stopping_criteria=analysis_stopping_criteria,)
        self.critique_pipeline = InferenceComposer(InferencePipeline(name="gpt4_critique",
                                                       max_steps=1,  # Wrapper pipeline
                                                       inference_components=[ComputeReport(name="initial_compute_report"),
                                                                             InitialAnalysis(name="initial_analysis", logging_path=logging_path),
                                                                             self.report_pipeline]))
        
    def __call__(self, data):
        return self.critique_pipeline(data)


def test_critique_pipeline():
    """
    Pipeline input sample should have:
    {
        "env_description": str,
        "policy": str,
        "trajectory_path": str,
    }
    """
    env = "MiniGrid-UnlockPickup-v0"
    env_description = """\
{task_description}\n\n\
The observation space is defined formally as: 
{observation_description}\n\n\
The action space is defined formally as:
{action_description}\n\n\
The rewards are defined formally as:
{reward_description}\n\n\
""".format(**env_descriptions[env])
    policy_path = "/mnt/c/Users/Alex/Desktop/OpenELM/projects/fun_search/init_policies/door_key/offline_analysis/policy_2.py"
    with open(policy_path, "r") as f:
        policy = "\n".join(f.readlines())
    trajectory_path = "/mnt/c/Users/Alex/Desktop/OpenELM/projects/fun_search/logs/trajectories/policy_2.json"
    data = {
            "env_description": [env_description],
            "policy": [policy],
            "trajectory_path": [trajectory_path],
           }
    critique_pipeline = CritiquePipeline(logging_path="")
    data = critique_pipeline(data)
    print(data["critique"][0])


if __name__ == "__main__":
    test_critique_pipeline()