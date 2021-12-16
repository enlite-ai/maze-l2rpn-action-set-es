"""Contains a brute force simulation search policy."""
from typing import Union, Optional, Tuple, Sequence, Callable

import numpy as np
from grid2op.Observation import CompleteObservation
from maze.core.agent.policy import Policy
from maze.core.annotations import override
from maze.core.env.action_conversion import ActionType
from maze.core.env.observation_conversion import ObservationType
from maze.core.env.simulated_env_mixin import SimulatedEnvMixin
from maze.core.env.structured_env import ActorID
from maze.core.utils.factory import ConfigType, Factory

from maze_action_set_es.env.core_env import L2RPNSeedInfo
from maze_action_set_es.env.maze_env import Grid2OpEnvironment
from maze_action_set_es.space_interfaces.action_conversion.dict_unitary import ActionConversion


class BruteForcePolicy(Policy):
    """A brute force search policy that simulates all possible actions and
    selects the one with minimum risk (max(rho)) for execution.

    :param simulated_env: Configuration to instantiate a helper environment for action conversion.
    :param max_simulations_per_step: The maximum number actions to try in each step.
    """

    def __init__(self,
                 simulated_env: Union[SimulatedEnvMixin, Callable[[], SimulatedEnvMixin], ConfigType],
                 max_simulations_per_step: Optional[int]
                 ):
        super().__init__()

        self.rng = np.random.RandomState(None)

        # initialize env required for action conversion
        env = Factory(base_type=Grid2OpEnvironment).instantiate(simulated_env)
        assert isinstance(env, Grid2OpEnvironment)

        self.action_conversion = env.action_conversion
        assert isinstance(self.action_conversion, ActionConversion)
        self._n_actions = self.action_conversion.space()["action"].n

        # initialize the corresponding torch policy producing action candidates
        self._max_simulations_per_step = max_simulations_per_step

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
        num_candidates = self._max_simulations_per_step if self._max_simulations_per_step else self._n_actions
        actions, scores = self.compute_top_action_candidates(observation,
                                                             num_candidates=num_candidates,
                                                             maze_state=maze_state,
                                                             env=env,
                                                             actor_id=actor_id)

        # get index of action with lowest risk
        best_action_idx = int(np.argmax(np.asarray(scores)))
        best_action = actions[best_action_idx]

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

        # prepare action candidates
        action_candidates = range(self._n_actions)
        if self._max_simulations_per_step:
            action_candidates = np.random.choice(action_candidates, self._max_simulations_per_step)

        # iterate action candidates
        actions, risks = [], []
        for action_id in action_candidates:
            action = {"action": int(action_id)}

            # convert agent action to grid2op action
            maze_action = self.action_conversion.space_to_maze(action, state=maze_state)
            # simulation next step with action candidate
            obs_simulate, reward_simulate, done_simulate, info_simulate = maze_state.simulate(maze_action)
            # estimate risk of resulting state and keep best one
            risk = self._risk(state=obs_simulate, done=done_simulate)

            # book keeping
            actions.append(action)
            risks.append(risk)

        # keep only top actions and convert risk to score
        sorted_idxs = np.argsort(risks)
        actions = [actions[i] for i in sorted_idxs[:num_candidates]]
        scores = [-risks[i] for i in sorted_idxs[:num_candidates]]

        return actions, scores

    @classmethod
    def _risk(cls, state: CompleteObservation, done: bool) -> float:
        """Risk function for a given state.

        :param state: The state to compute the risk value for.
        :return: Risk of provided state.
        """
        risk = np.inf if done else state.rho.max()
        return risk
