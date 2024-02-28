# Notes

Stockfish takes the most time by far
- (perhaps I really should not be using it to model opponent behavior?)

20 seconds for a 100 move game
- stockfish is not the issue anymore
- main culprits are deepcopy and pushing moves
    - actually it's mostly deepcopy

Playing with stockfish level one and surface level value functions takes 7 seconds for 30 moves (15 rounds)

I don't think mcts properly deals with situations where you arrive at the same state from two different action sequences

mcts is perhaps much less effective in environments with lots of stochasticity
- but it also seems like you can change an environment with stochasticity into a determinsitc one by treating the 
stochasticity as controlled by some opponent (kind of like in chess)
- I guess it is not quite the same actually, since sometimes stochasticity is honest to god random and there's no opponent
- two main options:
    - closed loop: even in these cases it's recommended you just incorporate these evenets directly into your tree as "chance nodes"
        - better with lots of computation
    - open loop: instead of nodes representing states they now represent sequences of actions
        - better with more limited computation

<https://ai.stackexchange.com/questions/22914/how-to-run-a-monte-carlo-tree-search-mcts-for-stochastic-environment>

## 2/12

Next steps.
- chess just doesn't seem to be a good domain unfortunately. Too slow to evaluate and a trivial value function seems to work fairly well
    - also just doing more search seems like the best option
- still want to implement recursive code construction and feedback prompting
- just don't have a good idea of a compelling domain

What is a task that can be compactly represented in code which we can try optimizing for?
- neural network training is a good choice
- maybe something kaggle related is another good choice?
- how did yue's thing work/get past low level action issues? Maybe just need to use gpt4?
    - but even gpt4 was struggling
