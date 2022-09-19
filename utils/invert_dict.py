from typing import Dict, List, TypeVar, Iterable

K = TypeVar("K")
V = TypeVar("V")

def invert_dict(d: Dict[K, Iterable[V]]) -> Dict[V, List[K]]:
    """
    Inverts a dictionary
    Eg: {"dog":["bark", "roof"]} -> {"bark":["dog"], "roof":["dog"]}
    """
    d_out = dict()
    for k, vs in d.items():
        for v in vs:
            if v not in d_out.keys():
                d_out[v] = []
            d_out[v].append(k)
    return d_out
