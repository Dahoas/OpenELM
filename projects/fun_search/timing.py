import chess
from time import time
import json
from copy import deepcopy
import numpy as np


def time_trial(f, iters):
    ts = []
    for _ in range(iters):
        t = time()
        f()
        t = time() - t
        ts.append(t) 
    return np.mean(ts)

board = chess.Board()

def f1():
    board = chess.Board()

def f2():
    global board
    board = deepcopy(board)

res = dict()
res["init"] = time_trial(f1, iters=1000)
res["deepcopy"] = time_trial(f2, iters=1000)
print(json.dumps(res, indent=2))