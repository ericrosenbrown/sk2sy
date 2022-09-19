from typing import List, Callable, Dict

def partition_by_function(arr: List, part_function: Callable) -> Dict:
    """
    Partitions the values of arr based on the result of passing them to part_function
    """
    d = dict()
    for x in arr:
        k = part_function(x)
        if k not in d.keys():
            d[k] = []
        d[k].append(x)
    return d
