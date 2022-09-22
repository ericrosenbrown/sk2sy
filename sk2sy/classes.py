from __future__ import annotations
from abc import ABC
from typing import List, NewType, Any, Optional, TypeVar, Iterable, Protocol
from dataclasses import dataclass
import numpy as np
from functools import reduce

class Classifier(Protocol):
    # TODO include shape in typehint
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        ...

class ClassifierFactory(Protocol):
    def fit(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        ...




StateVar = NewType("StateVar", int)
# Factor = NewType("Factor", np.ndarray)
State = NewType("State", tuple[float,...])
States = NewType("States", np.ndarray)
Action = NewType("Action", str)
Reward = NewType("Reward", float)

# @dataclass(frozen=True)
# class State:
#     s: tuple[float,...]

# class Classifier(ABC):
#     def predict(self, x):
#         pass

#     def fit


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
    state_var_names: Optional[tuple[str,...]] = None

@dataclass(frozen=True)
class Symbol:
    predictor: Classifier
    factors: tuple[Factor,...]

    @property
    def state_vars(self) -> tuple[state_vars,...]:
        svs = []
        for f in self.factors:
            svs.extend(f.state_vars)
        return tuple(sorted(list(set(svs))))
    
    @property
    def state_var_names(self) -> Optional[tuple[str,...]]:
        # If any factor is missing state_var_names, returns None
        svs = []
        for f in self.factors:
            if f.state_var_names is None:
                return None
            else:
                svs.extend(f.state_var_names)
        return tuple(sorted(list(set(svs))))
