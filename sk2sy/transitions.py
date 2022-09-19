from typing import List
from dataclasses import dataclass

@dataclass(frozen=True)
class Transition:
    start_state: List
    action: str
    reward: float
    end_state: List
    