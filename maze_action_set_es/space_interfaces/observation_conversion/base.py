"""Contains the base class for state to observation interfaces."""
from abc import ABC
from typing import Union, Optional, Dict

import grid2op
import gym
import numpy as np
from grid2op.Observation import CompleteObservation
from gym import spaces
from omegaconf import DictConfig

from maze_action_set_es.critical_state_observer.base_critical_state_observer import BaseCriticalStateObserver
from maze.core.env.observation_conversion import ObservationConversionInterface
from maze.core.utils.factory import Factory


class BaseObservationConversion(ObservationConversionInterface, ABC):
    """Object representing an observation.
    For more information consider: https://grid2op.readthedocs.io/en/latest/space.html

    :param grid2op_env: The grid2op environment.
    :param critical_state_observer: An optional critical state observer to decide whether a given state needs an
        agent interaction or not.
    """

    def __init__(self, grid2op_env: grid2op.Environment.Environment,
                 critical_state_observer: Optional[Union[BaseCriticalStateObserver, DictConfig]]):
        self.observation_space = grid2op_env.observation_space

        self.critical_state_observer = None if critical_state_observer is None else \
            Factory(BaseCriticalStateObserver).instantiate(critical_state_observer, grid2op_env=grid2op_env)

        self._force_critical = False

    def is_critical_state(self, state: CompleteObservation) -> bool:
        """Check whether the current state is critical if applicable.

        :param state: The state to be checked.
        :return: A decision if the given state is considered to be critical.
        """
        return True if self.critical_state_observer is None or self._force_critical \
            else self.critical_state_observer(state)

    def set_force_critical(self, force_critical: bool) -> None:
        """If set to True, all next states will be forced to be critical states if applicable.

        :param force_critical: Specify whether states should be forced to be critical or not.
        """
        self._force_critical = force_critical

    def update_observation_space_for_critical_state_observer(self, observation_space: spaces.Dict) -> gym.spaces.Dict:
        """Update the given observation space by adding the critical state observation space if applicable.

        :param observation_space: The current observation spaces to be updated.
        :return: The updated observation spaces with added critical state if applicable.
        """
        if self.critical_state_observer is not None:
            observation_space.spaces.update(self.critical_state_observer.critical_state_observation_space.spaces)
        return observation_space

    def update_observation_for_critical_state_observer(self, observation: Dict[str, np.ndarray],
                                                       is_critical_state: bool) -> Dict[str, np.ndarray]:
        """Update the given observation by adding the critical state observation if applicable.

        :param observation: The current observation to be updated.
        :param is_critical_state: Whether the current state corresponding to the observation is 'critical'.
        :return: The updated observation with added critical state if applicable.
        """
        if self.critical_state_observer is not None:
            observation.update({'critical_state': np.asarray([is_critical_state], dtype=np.float32)})

        return observation

    def clone_from(self, observation_conversion: 'BaseObservationConversion') -> None:
        """Reset the observation conversion to the state of the provided observation conversion.

        :param observation_conversion: The observation conversion to clone from.
        """
        self._force_critical = observation_conversion._force_critical
        if self.critical_state_observer is not None:
            self.critical_state_observer.clone_from(observation_conversion.critical_state_observer)
