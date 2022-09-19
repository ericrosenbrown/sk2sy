from typing import List
from dataclasses import dataclass

@dataclass(frozen=True)
class Transition:
    start_state: List
    action: str
    end_state: List
    reward: float
    