""" Implements a wrapper that resets the environment close to a blackout. """
import copy
from typing import Sequence, Union

import numpy as np
from maze.core.annotations import override
from maze.core.env.maze_env import MazeEnv
from maze.core.env.observation_conversion import ObservationType
from maze.core.env.structured_env import StructuredEnv
from maze.core.utils.factory import Factory, ConfigType
from maze.core.wrappers.wrapper import Wrapper

from maze_action_set_es.env.core_env import L2RPNSeedInfo
from maze_action_set_es.env.maze_env import Grid2OpEnvironment


class RandomChronicsWrapper(Wrapper[Grid2OpEnvironment]):
    """A wrapper that resets the environment to a random chronics.

    :param env: Environment to wrap.
    :param seeds: Environment seeds to train on.
    """

    def __init__(self,
                 env: MazeEnv,
                 seeds: Union[Sequence[L2RPNSeedInfo], ConfigType]):
        super().__init__(env)
        self._seeds = list(Factory(base_type=Sequence).instantiate(seeds))
        self._wrapper_rng = np.random.RandomState()
        self._current_idx = 0

    @override(StructuredEnv)
    def seed(self, seed: Union[L2RPNSeedInfo, int]) -> None:
        """Apply seed to wrappers rng, and pass the seed forward to the env
        """

        # Convert to int, if passed as L2RPNSeedInfo
        numeric_seed = seed.random_seed if isinstance(seed, L2RPNSeedInfo) else seed

        # Create new random state for sampling the random steps
        self._wrapper_rng = np.random.RandomState(numeric_seed)

        return self.env.seed(seed)

    def reset(self) -> ObservationType:
        """Randomly resets the environment either to the beginning at step 0 or close to the step where a noop action
        sequence would cause a blackout.

        :return: The initial observation.
        """
        current_seed = self._wrapper_rng.choice(self._seeds)
        self._current_idx += 1
        current_seed.fast_forward = 0
        current_seed.actions = []
        self.env.seed(current_seed)
        return self.env.reset()

    def clone_from(self, env: 'RandomChronicsWrapper') -> None:
        """Reset this gym environment to the given state by creating a deep copy of the `env.state` instance variable"""
        self._seeds = copy.deepcopy(env._seeds)
        self._wrapper_rng = copy.deepcopy(env._wrapper_rng)

        self.env.clone_from(env)
