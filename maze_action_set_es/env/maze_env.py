"""Contains the MazeEnv env for grid2op."""
from copy import deepcopy
from typing import Union

from maze_action_set_es.env.core_env import Grid2OpCoreEnvironment, L2RPNSeedInfo
from maze_action_set_es.space_interfaces.observation_conversion.base import BaseObservationConversion
from maze.core.annotations import override
from maze.core.env.action_conversion import ActionConversionInterface
from maze.core.env.maze_env import MazeEnv
from maze.core.utils.factory import Factory, CollectionOfConfigType


class Grid2OpEnvironment(MazeEnv):
    """Maze environment for the grid2op.

    :param core_env: Core environment or dictionary of core environment parameters.
    :param action_conversion: A dictionary with policy names as keys, containing either
                              * action to action interface implementations
                              * or a config dictionary specifying the interface instance to construct
                                via the registration system.
    :param observation_conversion: A dictionary with policy names as keys, containing either
                                   * state to observation interface implementations
                                   * or a config dictionary specifying the interface instance to construct
                                     via the registration system.
    """

    def __init__(self,
                 core_env: Union[Grid2OpCoreEnvironment, dict],
                 action_conversion: CollectionOfConfigType,
                 observation_conversion: CollectionOfConfigType):
        core_env = Factory(Grid2OpCoreEnvironment).instantiate(core_env)

        action_conversion_dict = Factory(
            base_type=ActionConversionInterface).instantiate_collection(action_conversion,
                                                                        grid2op_env=core_env.wrapped_env)
        observation_conversion_dict = Factory(
            base_type=BaseObservationConversion).instantiate_collection(observation_conversion,
                                                                        grid2op_env=core_env.wrapped_env)

        super().__init__(
            core_env,
            action_conversion_dict,
            observation_conversion_dict
        )

    @override(MazeEnv)
    def seed(self, seed: Union[L2RPNSeedInfo, int]):
        """Apply action to action transformation to the action replays in the replay info."""
        if isinstance(seed, L2RPNSeedInfo):
            seed = deepcopy(seed)
            self.core_env.seed(seed)

    @override(MazeEnv)
    def clone_from(self, env: MazeEnv) -> None:
        """Reset the maze env to the state of the provided env.

        Note, that it also clones the CoreEnv and its member variables including environment context.

        :param env: The environment to clone from.
        """
        raise NotImplementedError
