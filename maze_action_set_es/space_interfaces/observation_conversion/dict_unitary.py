"""Contains a state to observation interface converting the state into a dictionary of feature blocks.
Adopted from: https://github.com/PaddlePaddle/PARL/blob/f508bc6085420431b504441c7ff129e64826603e/benchmark/torch/NeurIPS2020-Learning-to-Run-a-Power-Network-Challenge/track1/utils.py#L69
"""
from typing import Dict, Optional, Union

import grid2op
import numpy as np
from grid2op.Observation import CompleteObservation
from gym import spaces
from maze.core.annotations import override
from maze.core.env.observation_conversion import ObservationConversionInterface
from omegaconf import DictConfig

from maze_action_set_es.critical_state_observer.base_critical_state_observer import BaseCriticalStateObserver
from maze_action_set_es.space_interfaces.observation_conversion.base import BaseObservationConversion


class ObservationConversion(BaseObservationConversion):
    """Object representing an observation.
    For more information consider: https://grid2op.readthedocs.io/en/latest/space.html

    :param grid2op_env: The grid2op environment.
    :param critical_state_observer: An optional critical state observer to decide whether a given state needs an
                                    agent interaction or not.
    """

    def __init__(self,
                 grid2op_env: grid2op.Environment,
                 critical_state_observer: Optional[Union[BaseCriticalStateObserver, DictConfig]]):
        super().__init__(grid2op_env, critical_state_observer=critical_state_observer)

        self._n_features_normalize = 2 * grid2op_env.n_load + 2 * grid2op_env.n_gen
        self._n_features_ready = grid2op_env.n_line + 2

    @override(ObservationConversionInterface)
    def maze_to_space(self, state: CompleteObservation) -> Dict[str, np.ndarray]:
        """Converts Maze state to space observation.
        For more information consider: https://grid2op.readthedocs.io/en/latest/observation.html#objectives

        :param state: The state returned by the powergrid env step.
        :return: The resulting dictionary observation.
        """
        # check if state is critical
        is_critical_state = self.is_critical_state(state=state)

        dict_obs = state.to_dict()

        # prepare load and generator features
        loads = []
        for key in ['q', 'v']:
            loads.append(dict_obs['loads'][key])
        loads = np.concatenate(loads)

        prods = []
        for key in ['q', 'v']:
            prods.append(dict_obs['prods'][key])
        prods = np.concatenate(prods)

        features_normalize = np.concatenate([loads, prods])
        features_normalize = np.array([features_normalize[i] for i in range(len(features_normalize))])

        # prepare some additional features
        rho = dict_obs['rho']
        time_info = np.array([state.month - 1, state.hour_of_day])
        features_ready = np.concatenate([rho, time_info]).astype(np.float32)

        dict_observation = {"features_normalize": features_normalize,
                            "features_ready": features_ready}

        return self.update_observation_for_critical_state_observer(dict_observation,
                                                                   is_critical_state=is_critical_state)

    @override(ObservationConversionInterface)
    def space_to_maze(self, observation: dict) -> CompleteObservation:
        """Converts space observation to Maze state.
        (This is most like not possible for most observation observation_conversion)
        """
        raise NotImplementedError

    @override(ObservationConversionInterface)
    def space(self) -> spaces.Dict:
        """Return the observation space shape based on the given params.
        """
        float_max = np.finfo(np.float32).max
        float_min = np.finfo(np.float32).min

        return self.update_observation_space_for_critical_state_observer(spaces.Dict({
            "features_normalize": spaces.Box(dtype=np.float32, shape=(self._n_features_normalize,),
                                             low=float_min, high=float_max),
            "features_ready": spaces.Box(dtype=np.float32, shape=(self._n_features_ready,),
                                         low=float_min, high=float_max)
        }))
