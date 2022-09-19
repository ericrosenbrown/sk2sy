from typing import Tuple
from abc import ABC

class Domain(ABC):
    def step(self, action: str) -> Tuple[str, float, bool]:
        """Returns new state, reward, and whether this is done"""
        pass