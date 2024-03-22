import numpy as np
import json
import torch


def get_image_target(name: str) -> np.ndarray:
    if name == "circle":
        target = np.zeros((32, 32, 3))
        for y in range(32):
            for x in range(32):
                if (y - 16) ** 2 + (x - 16) ** 2 <= 100:  # a radius-10 circle
                    target[y, x] = np.array([255, 255, 0])
    else:
        raise NotImplementedError(f"Image target {name} not implemented")
    return target


IMAGE_SEED: str = """
def draw():
\tpic = np.zeros((32, 32, 3))
\tfor x in range(2, 30):
\t\tfor y in range(2, 30):
\t\t\tpic[x, y] = np.array([0, 0, 255])
\treturn pic
"""

NULL_SEED: str = ""


def _convert(v):
    # Convert numpy objects
    numpy_dtypes = [np.int32, np.int64, np.float32, np.float64]
    if type(v) is np.ndarray:
        v = v.tolist()
    elif type(v) in numpy_dtypes:
        v = float(v)
    # Convert tensors
    elif type(v) is torch.tensor:
        v = v.tolist()
    return v


def _serialize(d):
    """
    Serialize all values of 'd' recursively.
    """
    if type(d) is dict:     
        for k, v in d.items():
            d[k] = _serialize(v)
    elif type(d) is list:
        for i in range(len(d)):
            d[i] = _serialize(d[i])
    else:
        d = _convert(d)
    return d


def robust_dump_json(d: dict, file_name: str):
    """
    Dump dict 'd' to file 'file_name' but first serialize values of 'd'.
    """
    d = _serialize(d)
    with open(file_name, "w") as f:
        json.dump(d, f)