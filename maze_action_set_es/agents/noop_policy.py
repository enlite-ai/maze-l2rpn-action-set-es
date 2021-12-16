"""Contains a noop policy."""
from typing import Union, Optional, Tuple, Sequence

import numpy as np
from grid2op.Observation import CompleteObservation
from maze.core.agent.policy import Policy
from maze.core.annotations import override
from maze.core.env.action_conversion import ActionType
from maze.core.env.observation_conversion import ObservationType
from maze.core.env.structured_env import ActorID

from maze_action_set_es.env.core_env import L2RPNSeedInfo
from maze_action_set_es.env.maze_env import Grid2OpEnvironment


class NoopPolicy(Policy):
    """ A noop policy.
    """

    def __init__(self):
        super().__init__()
        self.rng = np.random.RandomState(None)

    @override(Policy)
    def seed(self, seed: Union[L2RPNSeedInfo, int]) -> None:
        """Seed the policy."""

    @override(Policy)
    def needs_state(self) -> bool:
        """This policy requires the state object to compute the action."""
        return False

    @override(Policy)
    def needs_env(self) -> bool:
        """This policy does not require the env object to compute the action."""
        return True

    @override(Policy)
    def compute_action(self,
                       observation: ObservationType,
                       maze_state: Optional[CompleteObservation],
                       env: Optional[Grid2OpEnvironment],
                       actor_id: Optional[ActorID] = None,
                       deterministic: bool = False) -> ActionType:
        """Deterministically returns the noop action.
        """
        noop_action = env.action_conversion.noop_action()
        return noop_action

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
        raise NotImplementedError
