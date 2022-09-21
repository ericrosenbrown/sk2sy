from typing import Tuple
from abc import ABC
from sk2sy.classes import Action, Reward, State

class Domain(ABC):
    def step(self, action: Action) -> list[Action, State, State]:
        """Returns new state, reward, and whether this is done"""
        pass