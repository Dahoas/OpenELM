from typing import Union
import numpy as np
from collections import defaultdict


######## reports.py util ########

def flatten_dict(d: dict):
    """
    Flatten dict
    """
    new_d = dict()
    for k, v in d.items():
        #assert type(k) is str
        if type(v) is dict:
            new_v = flatten_dict(v)
            for kp, vp in new_v.items():
                new_d[f"{k}/{kp}"] = vp
        else:
            new_d[k] = v
    return new_d


def recursive_convert_lists_to_nparry(d: Union[dict, list]):
    primitive_types = [bool, int, float, str]
    if type(d) is list:
        if len(d) == 0:
            return d
        elif type(d[0]) in primitive_types:
            return np.array(d)
        else:
            d = [recursive_convert_lists_to_nparry(e) for e in d]
            if type(d[0]) is np.ndarray:
                d = np.stack(d, axis=0)
            return d
    elif type(d) is dict:
        for k, v in d.items():
            d[k] = recursive_convert_lists_to_nparry(v)
        return d
    elif type(d) in primitive_types:
        if type(d) is str:
            return d
        else:
            return np.array(d)
    else:
        raise ValueError(f"Unsupported type {type(d)}!!!\n\
                         Val: {d}")
    

def print_dict(d):
    for k, v in d.items():
        print(f"{k}: {v}")


def recursive_avg(d):
    if type(d[0]) in [bool, int, float]:
        return np.mean(d)
    elif type(d[0]) is dict:
        d_new = defaultdict(list)
        # Then group by key
        for d_in in d:
            # First flatten dict
            d_in = flatten_dict(d_in)
            for k, v in d_in.items():
                d_new[k].append(v)
        for k, v in d_new.items():
            d_new[k] = np.mean(v)
        return d_new
    else:
        raise ValueError(f"Unsupported type!!!")