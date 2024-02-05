# Notes

Stockfish takes the most time by far
- (perhaps I really should not be using it to model opponent behavior?)

20 seconds for a 100 move game
- stockfish is not the issue anymore
- main culprits are deepcopy and pushing moves
    - actually it's mostly deepcopy

Playing with stockfish level one and surface level value functions takes 7 seconds for 30 moves (15 rounds)