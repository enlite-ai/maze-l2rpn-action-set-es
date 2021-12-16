"""Contains a class wrapping grid2op's built in simulation functionality."""
from typing import Dict, Tuple

from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation


class Grid2OpSimulator:
    """Class wrapping grid2op's built in simulation functionality.

    :param observation: Observation to simulate on.
    """

    def __init__(self,
                 observation: CompleteObservation):
        self._obs = observation.copy()

    def simulate(self, action: PlayableAction) -> Tuple[CompleteObservation, float, bool, Dict]:
        """Take one step simulation.

        :param action: The action to simulate.
        :return: The step return quadruple.
        """
        # Take simulation step
        sim_obs, sim_rew, sim_done, sim_info = self._obs.simulate(action, time_step=1)

        # fix observation
        sim_obs.time_next_maintenance = self._obs.time_next_maintenance.copy()
        sim_obs.time_next_maintenance[sim_obs.time_next_maintenance > 0] -= 1

        sim_obs.duration_next_maintenance = self._obs.duration_next_maintenance.copy()
        sim_obs.duration_next_maintenance[self._obs.time_next_maintenance == 0] -= 1

        sim_obs.time_before_cooldown_line[self._obs.time_before_cooldown_line > 0] = \
            self._obs.time_before_cooldown_line[self._obs.time_before_cooldown_line > 0] - 1
        sim_obs.time_next_maintenance[self._obs.time_next_maintenance >= 0] = \
            self._obs.time_next_maintenance[self._obs.time_next_maintenance >= 0] - 1

        return sim_obs, sim_rew, sim_done, sim_info
