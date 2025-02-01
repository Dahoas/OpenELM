designer_prompts = dict(
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
All code should be written in a single, large python code block. \
When you are finished with your response you should write {is_complete_keyword} at the very end outside any code blocks.\
""",
)