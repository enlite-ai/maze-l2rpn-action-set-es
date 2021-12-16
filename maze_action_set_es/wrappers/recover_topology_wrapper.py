""" Implements a topology recovery wrapper. """
from typing import Dict, Any, Tuple, Optional, List

import numpy as np

from grid2op.Observation import CompleteObservation
from maze.core.annotations import override
from maze.core.env.maze_env import MazeEnv
from maze.core.wrappers.wrapper import ObservationWrapper, Wrapper

from maze_action_set_es.env.maze_env import Grid2OpEnvironment


class RecoverTopologyWrapper(Wrapper[Grid2OpEnvironment]):
    """An wrapper that resets the grid to the original state once a critical state is resolved
    and all power lines are active again.

    :param env: Environment to wrap.
    :param recover_if_critical: Recover also if in a critical state.
    :param max_rho: Recover only if max(rho) is below max_rho.
    :param recovery_topology: Recovery topology configurations for substations.
                              If not specified bus one is assumed for all links.
    """

    def __init__(self,
                 env: MazeEnv,
                 recover_if_critical: bool,
                 max_rho: Optional[float],
                 recovery_topology: Optional[Dict[int, List[int]]]):
        super().__init__(env)

        self._recover_if_critical = recover_if_critical
        self._max_rho = max_rho
        self._recovery_topology = recovery_topology
        self._prepare_recovery_topology()

    def has_recovered(self) -> bool:
        """Returns true if recovery topology hase been reached.

        :return: Recovery status.
        """
        state = self.env.get_maze_state()
        for sub_id in range(self.env.wrapped_env.n_sub):
            if not np.all(state.sub_topology(sub_id) == self._recovery_topology[sub_id]):
                return False
        return True

    def recovery_possible(self) -> bool:
        """Checks if recovery is possible.

        :return: True if recovery is possible.
        """
        state = self.env.get_maze_state()

        if np.any(state.line_status == 0):
            return False
        elif state.rho.max() > self._max_rho:
            return False
        elif self.env.observation_conversion.is_critical_state(state) and not self._recover_if_critical:
            return False

        for sub_id in range(self.env.wrapped_env.n_sub):
            if not np.all(state.sub_topology(sub_id) == self._recovery_topology[sub_id]) and \
                    state.time_before_cooldown_sub[sub_id] > 0:
                return False

        return True

    @override(ObservationWrapper)
    def step(self, action) -> Tuple[Any, Any, bool, Dict[Any, Any]]:
        """Intercept ``ObservationWrapper.step`` and map observation."""

        # get grid2op state object
        state: CompleteObservation = self.get_maze_state()

        # find recovery path
        if self.recovery_possible() and not self.has_recovered():

            # collect actions to take
            actions_to_take = []
            for sub_id in range(state.n_sub):
                if not np.allclose(state.sub_topology(sub_id), self._recovery_topology[sub_id]):
                    action_dict = dict()
                    action_dict["set_bus"] = {"substations_id": [(sub_id, self._recovery_topology[sub_id])]}
                    actions_to_take.append(action_dict)

            # select next action
            least_rho = np.inf
            for action_dict in actions_to_take:
                playable_action = self.wrapped_env.action_space(action_dict)
                sim_state, _, done, _ = state.simulate(playable_action, time_step=1)
                max_rho = sim_state.rho.max()
                if not done and max_rho < self._max_rho and max_rho < least_rho:
                    action = playable_action
                    least_rho = max_rho

        # take actual step
        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, info

    @override(ObservationWrapper)
    def reset(self) -> Any:
        """Intercept ``ObservationWrapper.reset`` and map observation."""
        obs = self.env.reset()
        return obs

    def clone_from(self, env: 'RecoverTopologyWrapper') -> None:
        """Reset this gym environment to the given state by creating a deep copy of the `env.state` instance variable"""

        self._recover_if_critical = env._recover_if_critical
        self._max_rho = env._max_rho

        assert isinstance(env, MazeEnv)
        self.env.clone_from(env)

    def _prepare_recovery_topology(self) -> None:
        """Prepare recovery topology configuration on substation level.
        """
        if self._recovery_topology is None:
            self._recovery_topology = dict()
        else:
            self._recovery_topology = dict(self._recovery_topology)

        for sub_id in range(self.env.wrapped_env.n_sub):
            n_links = self.env.wrapped_env.sub_info[sub_id]

            if sub_id not in self._recovery_topology:
                self._recovery_topology[sub_id] = [1] * n_links
            else:
                assert len(self._recovery_topology[sub_id]) == n_links
                for bus in self._recovery_topology[sub_id]:
                    assert bus in [1, 2]
