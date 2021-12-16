"""Contains base class for the critical state observer."""
from abc import abstractmethod

import grid2op
import numpy as np
from grid2op.Observation import CompleteObservation
from gym import spaces

from maze.core.annotations import unused


class BaseCriticalStateObserver:
    """The critical state observer base class to decide whether a state is critical or not. That is whether a
    given state requires an interaction or not."""

    critical_state_observation_space = spaces.Dict(
        {"critical_state": spaces.Box(low=np.float32(0), high=np.float32(1), shape=(1,),
                                      dtype=np.float32)})

    def __init__(self, grid2op_env: grid2op.Environment.Environment):
        unused(grid2op_env)
        self._latest_state_hash = None
        self._latest_critical_state = None

    def is_state_critical(self, state: CompleteObservation) -> bool:
        """Decide whether the given state represents a critical situation or not.

        :param state: A state of the grid2op environment do be judged.
        :return: A decision, whether the given state is critical or not.
        """
        # compute current state hash
        state_hash = self._compute_state_hash(state)

        # update critical state hash and status if required
        if state_hash != self._latest_state_hash:
            self._latest_critical_state = self._is_state_critical(state)
            self._latest_state_hash = state_hash

        return self._latest_critical_state

    def __call__(self, state: CompleteObservation) -> bool:
        """Short hand method, that wraps the is_state_critical method.

        :param state: A state of the grid2op environment do be judged.
        :return: A decision, whether the given state is critical or not.
        """
        return self.is_state_critical(state=state)

    @abstractmethod
    def _is_state_critical(self, state: CompleteObservation) -> bool:
        """Decide whether the given state represents a critical situation or not.
        (This is the place where the actual implementation lives.)

        :param state: A state of the grid2op environment do be judged.
        :return: A decision, whether the given state is critical or not.
        """

    @abstractmethod
    def clone_from(self, critical_state_observer: 'BaseCriticalStateObserver') -> None:
        """Reset the critical state observer to the state of the provided critical state observer.

        :param critical_state_observer: The critical state observer to clone from.
        """
        self._latest_state_hash = critical_state_observer._latest_state_hash
        self._latest_critical_state = critical_state_observer._latest_critical_state

    @classmethod
    def _compute_state_hash(cls, state: CompleteObservation) -> str:
        """Returns a hash of the current environment.

        :return: The hash value.
        """
        return str(state.get_time_stamp())
