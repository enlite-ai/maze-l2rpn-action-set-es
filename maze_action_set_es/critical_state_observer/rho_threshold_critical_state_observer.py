"""Critical state observer that also check the max-rho-delta to the last observation"""
import grid2op
import numpy as np
from grid2op.Observation import CompleteObservation

from maze_action_set_es.critical_state_observer.base_critical_state_observer import BaseCriticalStateObserver


class RhoThresholdCriticalStateObserver(BaseCriticalStateObserver):
    """An critical state observer that focuses solely on the maximal rho of the given state.

    :param grid2op_env: The grid2op environment.
    :param max_rho: The maximum line capacity be for considered a critical state.
    """

    def __init__(self, grid2op_env: grid2op.Environment.Environment, max_rho: float):
        super().__init__(grid2op_env=grid2op_env)
        self._max_rho = max_rho

    def is_state_critical(self, state: CompleteObservation) -> bool:
        """Checks whether the maximum rho of the given state is over the defined threshold or not.

        :param state: The state to be checked.
        :return: True if the max rho is greater than the threshold, false otherwise.
        """
        if np.any(np.isnan(state.rho)) or np.any(state.rho > self._max_rho):
            return True
        else:
            return False

    def clone_from(self, critical_state_observer: 'RhoThresholdCriticalStateObserver') -> None:
        """Reset the critical state observer to the state of the provided critical state observer.

        :param critical_state_observer: The critical state observer to clone from.
        """
