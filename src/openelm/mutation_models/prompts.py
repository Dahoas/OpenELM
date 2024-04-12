

designer_prompts = dict(
    env_description="""\
Task: {task_description}\n\n\
The observation space is defined formally as:\n\
{observation_description}\n\n\
The action space is defined formally as:\n\
{action_description}\n\n\
Reward description:\n\
{reward_description}\
""",

    unconditional_prompt="""\
You are responsible for designing a decision policy to solve an RL task.\
You will write a python `Policy()`, which should be initializable without any parameters from the user, object which has two methods:
- `def act(observation)` which takes in an observation and returns an action.
- `update(observation, action, reward, next_observation)` which takes in the current observation, \
chosen action, reward, and next_observation and updates any persistent memory/state between observations. \
You should never assume the actions you take. `update` is a good place to test your understanding of the world and record the results.\n\
Note: You should not assume any exploration outside of what is learned during the agent's single rollout in \
the environment. This means you should not rely on Q-learning, etc.\
You are allowed to use any python library you want but should not assume access \
to any other external resources (such as models with downloadable weights) unless otherwise specified.\n\n\
{env_description}\n\n\
You should only write the Policy class and nothing else. \
You are encouraged to be as creative as possible, do not simply copy one of the exemplars if given. \
Your policy should also be robust to adversity. If it finds itself getting stuck, repeating the same moves, \
it should try something new. Think carefully about the order of steps you propose.\n\
All code should be written in a single, large code block. \
When you are finished with your response you should write {is_complete_keyword} at the very end outside any code blocks.\
""",

    conditional_prompt="""\
{unconditional_prompt}\n\n\
Example Policy:\n\n\
{policy}\
""",

    critique_prompt="""\
{conditional_prompt}\n\n\
Policy critique:\n\n\
{critique}\
""",
)


analyzer_prompts = dict(
    initial_analysis_prompt="""\
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
""",

    propose_report_prompt="""\
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
""",

    analysis_prompt="""\
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
""",
)