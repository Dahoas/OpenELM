prompts = {
    "env_prompt": """\
You are responsible for designing a decision policy to solve the following task: 
{task_description}\n\n\
You will write a python `Policy()`, which should be initializable without any parameters from the user, object which has two methods:
- `def act(observation)` which takes in an observation and returns an action.
- `update(observation, action, reward, next_observation)` which takes in the current observation, \
chosen action, reward, and next_observation and updates any persistent memory/state between observations. \
You should never assume the actions you take. `update` is a good place to test your understanding of the world and record the results.
The observation space is defined formally as: 
{observation_description}\n\n\
The action space is defined formally as:
{action_description}\n\n\
The rewards are defined formally as:
{reward_description}\n\n\n""",
    "impl_prompt": """\
You are allowed to use any python library you want but should not assume access \
to any other external resources (such as models with downloadable weights) unless otherwise specified. \
In particular you can assume access to the following APIs: \
{api_description}\n\n\
You are encouraged to be as creative as possible, do not simply copy one of the exemplars if given. \
Your policy should also be robust to adversity. If it finds itself getting stuck, repeating the same moves, \
it should try something new. Exploration is encouraged, but keep in mind the goal and the action preconditions.\
Make sure when you are engaging in exploration, it is diverse and incentivizing discovery of new states.\n\
Thank carefully about the order of steps you propose. Executing the right steps in the wrong order will be costly\n\
All code should be written in a single, large code block. \
When you are finished with your response you should write {is_complete_keyword} at the very end outside any code blocks.
""",
    "policy_prompt":"""\
Here is an example policy:\n\n\
```python\n{policy}```\n\n\
Average Policy Return: {ret}\n\n\
Num Policy Rollouts: {rollouts}\n\n\
Max rollout length: {max_steps}\n\n\
""",
    "metrics_prompt": """\
The above policy has been rolled out in the environment and a dataset of trajectories has been collected. \
This datastructure has the form:\n\
trajectories: list[dict]
    trajectory: dict
        "observations" -> list[OBSERVATION_DATATYPE]
        "actions" -> list[ACTION_DATATYPE]
        "rewards" -> list[float]
Write a class `Metrics` which analyzes statistics from over the agent's execution \
to understand its performance in the environement. This is also a good way to test your own understanding of the environment's dynamics. \
Metrics should implement a function `metrics(trajectories: list[dict]) -> dict[str, float]` which takes in trajectories \
and outputs a dictionatry of metrics used to understand policy behavior and world dynamics. \
When you implement a metric you should be very careful your implementation is correctly \
measuring what you want.
"""
}