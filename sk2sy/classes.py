from __future__ import annotations
from typing import List, NewType
from dataclasses import dataclass
import numpy as np


StateVar = NewType("StateVar", int)
# Factor = NewType("Factor", np.ndarray)
State = NewType("State", tuple[float,...])
States = NewType("States", np.ndarray)
Action = NewType("Action", str)
Reward = NewType("Reward", float)

@dataclass(frozen=True)
class Transition:
    start_state: State
    action: Action
    reward: Reward
    end_state: State



@dataclass(frozen=True)
class Factor:
    nm: str
    state_vars: tuple[StateVar,...]
