"""Generate seeds for the L2RPN environment."""
from typing import List

import numpy as np

from maze_action_set_es.env.core_env import L2RPNSeedInfo


def seeds_for_power_grid(power_grid: str) -> List[L2RPNSeedInfo]:
    """Return the seeds for the given power grid.

    :param power_grid: The name of the power grid.
    :return: All seeds for the power grid.
    """
    if power_grid == 'rte_case14_realistic':
        return seeds_rte_case14_realistic()
    else:
        raise Exception(f'Seed method could not be found for given power_grid: {power_grid}')


def seeds_rte_case14_realistic() -> List[L2RPNSeedInfo]:
    """Generate seeds for rte_case14_realistic"""
    initial_seeds = [
        L2RPNSeedInfo(env_index=0, chronic_id=i, fast_forward=0, actions=tuple(), random_seed=random_seed)
        for i in range(1000) for random_seed in [1234, 2345, 3456, 4567, 5678]]
    np.random.shuffle(initial_seeds)

    return initial_seeds
