""" Contains a flattened dictionary l2rpn ActionConversion """
import copy
import os
from typing import Dict, Optional

import grid2op
import numpy as np
from grid2op.Action import TopologyAndDispatchAction, PlayableAction
from grid2op.Converter import IdToAct
from grid2op.Observation.CompleteObservation import CompleteObservation
from gym import spaces
from maze.core.annotations import override
from maze.core.env.action_conversion import ActionConversionInterface

from maze_action_set_es.space_interfaces.action_conversion.base import BaseActionConversion


class ActionConversion(BaseActionConversion):
    """Interface specifying the conversion of agent actions to actual environment executions.

    :param grid2op_env: The grid2op environment.
    :param action_selection_vector: List of actions that should be kept from the original action space.
                                    (Helpful for exploration space reduction)
    :param action_selection_vector_dump: Path to an .npy file holding the action_selection_vector.
    :param set_line_status: Include set line status actions.
    :param change_line_status: Include change line status actions.
    :param set_topo_vect: Include set topology actions.
    :param change_bus_vect: Include change bus actions actions.
    :param redispatch: Include redispatch actions.
    :param curtail: Include curtail actions.
    :param storage: Include storage actions.
    """

    def __init__(self,
                 grid2op_env: grid2op.Environment,
                 action_selection_vector: Optional[np.ndarray],
                 action_selection_vector_dump: Optional[str],
                 set_line_status: bool,
                 change_line_status: bool,
                 set_topo_vect: bool,
                 change_bus_vect: bool,
                 redispatch: bool,
                 curtail: bool,
                 storage: bool):
        super().__init__(grid2op_env.action_space)
        self.id_to_act = IdToAct(self.action_space)
        self.id_to_act.init_converter(set_line_status=set_line_status,
                                      change_line_status=change_line_status,
                                      set_topo_vect=set_topo_vect,
                                      change_bus_vect=change_bus_vect,
                                      redispatch=redispatch,
                                      curtail=curtail,
                                      storage=storage)

        # load masking vector if dump file was provided
        if action_selection_vector_dump:
            assert not action_selection_vector,\
                "Either provide an already loaded action action selection vector or a dump file; not both!"
            assert os.path.exists(action_selection_vector_dump)
            action_selection_vector = np.load(action_selection_vector_dump)

        # prepare masking vector
        if action_selection_vector is None:
            action_selection_vector = np.asarray(range(self.id_to_act.n), dtype=np.int)

        # pre-compute required mappings
        self._num_active_actions = len(action_selection_vector)

        self._space_id_to_converter_id = dict()
        for i, converter_id in enumerate(action_selection_vector):
            self._space_id_to_converter_id[i] = converter_id

    @override(BaseActionConversion)
    def space_to_maze(self,
                      action: Dict[str, int],
                      state: Optional[CompleteObservation]) -> TopologyAndDispatchAction:
        """Converts space to environment action.

        :param action: gym space object to parse.
        :param state: the environment state.
        :return: action object.
        """

        # Bypass checks if action is already a PlayableAction
        if isinstance(action, PlayableAction):
            return action

        assert 'action' in action

        # special treatment if action is passed as an array
        action_id = action['action']
        if isinstance(action_id, np.ndarray):
            if action_id.shape == ():
                action_id = int(action_id)
        assert isinstance(action_id, (int, np.int32, np.int, np.int64))

        # convert action id to actual action applicable to the grid
        converter_action_id = self._space_id_to_converter_id[action_id]
        action_obj = self.id_to_act.convert_act(converter_action_id)

        # automatically turn power lines on again
        for line_idx in np.nonzero(state.line_status == 0)[0]:
            if state.time_before_cooldown_line[line_idx] == 0:
                action_obj = copy.deepcopy(action_obj)
                action_obj.update({"set_line_status": [(line_idx, 1)]})
                break

        return action_obj

    @override(BaseActionConversion)
    def maze_to_space(self,
                      execution: TopologyAndDispatchAction) -> Dict[str, int]:
        """Converts environment to space action.

        :param: action: the environment action to convert.
        :return: the dictionary action.
        """
        raise NotImplementedError

    @override(BaseActionConversion)
    def space(self) -> spaces.Dict:
        """Returns the respective gym action space.

        :return: Gym action space.
        """
        return spaces.Dict({"action": spaces.Discrete(n=self._num_active_actions)})

    @override(ActionConversionInterface)
    def noop_action(self) -> Dict[str, int]:
        """Return the noop action"""
        return {"action": 0}
