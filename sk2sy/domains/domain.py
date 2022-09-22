from typing import Tuple, Optional
from abc import ABC
from sk2sy.classes import Action, Reward, State

class Domain(ABC):
    def step(self, action: Action) -> list[Action, State, State]:
        """Returns new state, reward, and whether this is done"""
        pass
    
    @property
    def state_var_names(self) -> Optional[str]:
        return None