"""Implements a random policy that works with L2RPNSeedInfo."""
from typing import Union

import numpy as np
from maze.core.agent.random_policy import RandomPolicy
from maze.core.annotations import override
from maze.core.utils.seeding import MazeSeeding

from maze_action_set_es.env.core_env import L2RPNSeedInfo


class L2RPNRandomPolicy(RandomPolicy):
    """Patches l2rpn seeding for random policy
    """

    @override(RandomPolicy)
    def seed(self, seed: Union[L2RPNSeedInfo, int]) -> None:
        """Seed the policy by setting the action space seeds."""

        # Convert to int, if passed as L2RPNSeedInfo.
        if isinstance(seed, L2RPNSeedInfo):
            seed = seed.random_seed

        rng = np.random.RandomState(seed)
        for key, action_space in self.action_spaces_dict.items():
            action_space.seed(MazeSeeding.generate_seed_from_random_state(rng))
