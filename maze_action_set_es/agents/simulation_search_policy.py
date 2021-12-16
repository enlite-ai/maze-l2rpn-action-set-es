"""Contains a simulation based search policy for the unitary aciton conversion."""
from copy import copy
from typing import Union, Optional, Tuple, Sequence, Dict, List

import numpy as np
import torch
from grid2op.Observation import CompleteObservation
from maze.core.agent.policy import Policy
from maze.core.agent.torch_model import TorchModel
from maze.core.agent.torch_policy import TorchPolicy
from maze.core.annotations import override
from maze.core.env.action_conversion import ActionType
from maze.core.env.observation_conversion import ObservationType
from maze.core.env.structured_env import ActorID
from maze.core.utils.factory import ConfigType, Factory

from maze_action_set_es.env.core_env import L2RPNSeedInfo
from maze_action_set_es.env.maze_env import Grid2OpEnvironment
from maze_action_set_es.space_interfaces.action_conversion.dict_unitary import ActionConversion


class SimulationSearchPolicy(Policy, TorchModel):
    """Policy that (1) simulates the top k action candidates predicted by the policy network and (2) selects the one
    with minimum risk (max(rho)) for execution.

    :param simulated_env: Configuration to instantiate a helper environment for action conversion.
    :param torch_policy: Policy to sample action candidates with.
    :param top_k_candidates: Number of action candidates to consider for simulation.
    """

    def __init__(self,
                 simulated_env: ConfigType,
                 torch_policy: TorchPolicy,
                 top_k_candidates: int,
                 ):
        self.torch_policy = Factory(base_type=TorchPolicy).instantiate(torch_policy)
        super().__init__(self.torch_policy.device)

        self._top_k_candidates = top_k_candidates

        self.rng = np.random.RandomState(None)

        self.simulated_env_config = simulated_env
        self._action_conversion = self._init_action_conversion()

    @override(Policy)
    def seed(self, seed: Union[L2RPNSeedInfo, int]) -> None:
        """Seed the policy."""

    @override(Policy)
    def needs_state(self) -> bool:
        """This policy requires the state object to compute the action."""
        return True

    @override(Policy)
    def needs_env(self) -> bool:
        """This policy does not require the env object to compute the action."""
        return False

    @override(Policy)
    def compute_action(self,
                       observation: ObservationType,
                       maze_state: Optional[CompleteObservation],
                       env: Optional[Grid2OpEnvironment],
                       actor_id: Optional[ActorID] = None,
                       deterministic: bool = False) -> ActionType:
        """Samples a random action.
        """

        # compute top k action candidates
        actions, scores = self.compute_top_action_candidates(observation,
                                                             num_candidates=self._top_k_candidates,
                                                             maze_state=maze_state,
                                                             env=env,
                                                             actor_id=actor_id)

        # simulate risk of top k candidates
        best_action, lowest_risk = None, np.inf
        for action in actions:
            # convert agent action to grid2op action
            maze_action = self.action_conversion.space_to_maze(action, state=maze_state)
            # simulation next step with action candidate
            obs_simulate, reward_simulate, done_simulate, info_simulate = maze_state.simulate(maze_action)
            # estimate risk of resulting state and keep best one
            risk = self._risk(state=obs_simulate, done=done_simulate)
            if risk < lowest_risk:
                lowest_risk = risk
                best_action = action
            # even in cases where the risk is np.inf, we need to take a valid action
            if best_action is None:
                best_action = action

        return best_action

    @override(Policy)
    def compute_top_action_candidates(self,
                                      observation: ObservationType,
                                      num_candidates: int,
                                      maze_state: Optional[CompleteObservation],
                                      env: Optional[Grid2OpEnvironment],
                                      actor_id: Union[str, int] = None) -> Tuple[Sequence[ActionType], Sequence[float]]:
        """
        Implementation of :py:attr:`~maze.core.agent.policy.Policy.compute_top_action_candidates`.
        """

        # just forward call to internal torch policy
        actions, probs = self.torch_policy.compute_top_action_candidates(
            observation=observation,
            num_candidates=num_candidates,
            maze_state=maze_state,
            env=env,
            actor_id=actor_id
        )
        return actions, probs

    @classmethod
    def _risk(cls, state: CompleteObservation, done: bool) -> float:
        """Risk function for a given state.

        :param state: The state to compute the risk value for.
        :return: Risk of provided state.
        """
        risk = np.inf if done else state.rho.max()
        return risk

    @override(TorchModel)
    def parameters(self) -> List[torch.Tensor]:
        """Forward the method call to the wrapped TorchPolicy"""
        return self.torch_policy.parameters()

    @override(TorchModel)
    def eval(self) -> None:
        """Forward the method call to the wrapped TorchPolicy"""
        self.torch_policy.eval()

    @override(TorchModel)
    def train(self) -> None:
        """Forward the method call to the wrapped TorchPolicy"""
        self.torch_policy.train()

    @override(TorchModel)
    def to(self, device: str) -> None:
        """Forward the method call to the wrapped TorchPolicy"""
        self.torch_policy.to(device)

    @override(TorchModel)
    def state_dict(self) -> Dict:
        """Forward the method call to the wrapped TorchPolicy"""
        return self.torch_policy.state_dict()

    @override(TorchModel)
    def load_state_dict(self, state_dict: Dict) -> None:
        """Forward the method call to the wrapped TorchPolicy"""
        self.torch_policy.load_state_dict(state_dict)

    @property
    def action_conversion(self):
        """Lazily initialize action_conversion if not available"""
        if not self._action_conversion:
            self._action_conversion = self._init_action_conversion()

        return self._action_conversion

    def _init_action_conversion(self) -> ActionConversion:
        """Build the action conversion through Grid2Op env instantiation."""
        env = Factory(base_type=Grid2OpEnvironment).instantiate(self.simulated_env_config)
        assert isinstance(env, Grid2OpEnvironment)
        assert isinstance(env.action_conversion, ActionConversion)
        return env.action_conversion

    def __getstate__(self) -> dict:
        """Skip action_conversion during pickling"""
        obj_dict = copy(self.__dict__)
        obj_dict["_action_conversion"] = None
        return obj_dict
