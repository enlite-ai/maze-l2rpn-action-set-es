""" Contains the l2rpn BaseActionConversion """
import numpy as np
from abc import ABC, abstractmethod
from grid2op.Action import ActionSpace, PlayableAction
from grid2op.Observation import CompleteObservation
from gym import spaces
from typing import Any

from maze.core.env.action_conversion import ActionConversionInterface


class BaseActionConversion(ActionConversionInterface, ABC):
    """Interface specifying the conversion of space to actual environment actions.

    :param action_space: The grid2op action space.
    """

    def __init__(self, action_space: ActionSpace):
        self.action_space = action_space

        self.n_gen = self.action_space.n_gen
        self.n_load = self.action_space.n_load
        self.n_sub = self.action_space.n_sub
        self.n_line = self.action_space.n_line
        self.max_buses = 2
        self.max_links = np.amax(self.action_space.sub_info)

        # generator specific values
        self.max_ramp_down = -self.action_space.gen_max_ramp_down
        self.max_ramp_up = self.action_space.gen_max_ramp_up

    def space_to_maze(self, action: Any, state: CompleteObservation) -> PlayableAction:
        """Converts agent action to environment action.

        :param action: gym space object to parse.
        :param state: the environment state.
        :return: action object.
        """
        raise NotImplementedError

    @abstractmethod
    def maze_to_space(self, action: PlayableAction) -> Any:
        """Converts environment to space action.
        """

    @abstractmethod
    def space(self) -> spaces.Space:
        """Returns respective gym action space.

        :return: Gym action space.
        """
