""" Implements a critical state trainer observation wrapper. """
import copy
import os
from abc import ABC
from functools import partial
from typing import Dict, Any, Tuple, Optional, Union, List

import numpy as np

from maze_action_set_es.env.maze_env import Grid2OpEnvironment
from maze.core.annotations import override
from maze.core.env.action_conversion import ActionType
from maze.core.env.maze_action import MazeActionType
from maze.core.env.maze_env import MazeEnv
from maze.core.env.maze_state import MazeStateType
from maze.core.env.observation_conversion import ObservationType
from maze.core.env.simulated_env_mixin import SimulatedEnvMixin
from maze.core.log_stats.event_decorators import define_epoch_stats, define_episode_stats, define_step_stats
from maze.core.wrappers.wrapper import ObservationWrapper, Wrapper

quantile_10 = partial(np.quantile, q=0.1)
quantile_25 = partial(np.quantile, q=0.25)


class CriticalStateEvents(ABC):
    """
    Event topic class with logging statistics based only on observations, therefore applicable to any valid
    reinforcement learning environment.
    """

    @define_epoch_stats(quantile_10, input_name="sum", output_name="quantile_10")
    @define_epoch_stats(quantile_25, input_name="sum", output_name="quantile_25")
    @define_epoch_stats(max, input_name="sum", output_name="max")
    @define_epoch_stats(min, input_name="sum", output_name="min")
    @define_epoch_stats(np.mean, input_name="sum", output_name="mean")
    @define_epoch_stats(np.median, input_name="sum", output_name="median")
    @define_epoch_stats(np.std, input_name="sum", output_name="std")
    @define_episode_stats(sum, output_name="sum")
    @define_step_stats(None)
    def steps(self, n_steps: int):
        """Total number of steps taken."""

    @define_epoch_stats(np.mean, input_name="sum", output_name="mean")
    @define_epoch_stats(np.median, input_name="sum", output_name="median")
    @define_episode_stats(sum, output_name="sum")
    @define_step_stats(None)
    def total_critical_states(self, n_states: float):
        """Total number of critical states."""

    @define_epoch_stats(np.mean, input_name="sum", output_name="mean")
    @define_epoch_stats(np.median, input_name="sum", output_name="median")
    @define_episode_stats(sum, output_name="sum")
    @define_step_stats(None)
    def dead_in_safe_state(self, value: float):
        """If the env encounters a blackout in a safe state."""

    @define_epoch_stats(max, input_name="sum", output_name="max")
    @define_epoch_stats(min, input_name="sum", output_name="min")
    @define_epoch_stats(np.mean, input_name="sum", output_name="mean")
    @define_epoch_stats(np.median, input_name="sum", output_name="median")
    @define_episode_stats(sum, output_name="sum")
    @define_step_stats(None)
    def steps_since_safe_state(self, n_time: float):
        """Time steps since the last safe state."""

    @define_epoch_stats(max, input_name="sum", output_name="max")
    @define_epoch_stats(min, input_name="sum", output_name="min")
    @define_epoch_stats(np.mean, input_name="sum", output_name="mean")
    @define_epoch_stats(np.median, input_name="sum", output_name="median")
    @define_episode_stats(sum, output_name="sum")
    @define_step_stats(None)
    def max_rho_before_death(self, max_rho: float):
        """Maximum rho on a powerline in step before blackout."""

    @define_epoch_stats(max, input_name="sum", output_name="max")
    @define_epoch_stats(min, input_name="sum", output_name="min")
    @define_epoch_stats(np.mean, input_name="sum", output_name="mean")
    @define_epoch_stats(np.median, input_name="sum", output_name="median")
    @define_episode_stats(sum, output_name="sum")
    @define_step_stats(None)
    def percent_critical_states(self, value: float):
        """Percent of critical states w.r.t. total number of states."""


class EnvDoneInResetException(Exception):
    """Exception raised if the env is already done in a reset"""
    pass


class CriticalStateTrainerWrapper(Wrapper[Grid2OpEnvironment]):
    """An wrapper skipping all steps but those having observation["critical_state"]=True.

    :param env: Environment to wrap.
    :param do_skip: If True none critical steps are skipped.
    :param raise_exception_on_impossible_seed: Specify whether to raise an exception on impossible seeds. (needed for
        mcts workers).
    """

    def __init__(self, env: MazeEnv, do_skip: bool, raise_exception_on_impossible_seed: bool):
        super().__init__(env)

        # event related
        self._step_events = self.core_env.context.event_service.create_event_topic(CriticalStateEvents)
        self._internal_steps = 0
        self._internal_rewards = []
        self._last_safe_state_steps = -1
        self._num_critical_states = 0
        self._prev_critical_state = 0

        self._do_skip = do_skip
        self._raise_exception_on_impossible_seed = raise_exception_on_impossible_seed
        self._max_env_steps = env.wrapped_env.chronics_handler.max_episode_duration() - 2

    @override(ObservationWrapper)
    def step(self, action: ActionType) -> Tuple[Any, Any, bool, Dict[Any, Any]]:
        """Intercept ``ObservationWrapper.step`` and map observation."""
        observation, reward, done, info = self._step_and_increase_values(action)
        self._check_observation(observation)

        total_reward = reward
        self._internal_rewards = [reward]
        while self._do_skip and not observation["critical_state"] and not done:
            action = self.action_conversion.noop_action()
            observation, reward, done, info = self._step_and_increase_values(action)
            total_reward += reward
            self._internal_rewards.append(reward)

        # report total number of steps
        total_steps = self._internal_steps
        info['n_steps'] = total_steps

        return observation, total_reward, done, info

    @override(ObservationWrapper)
    def reset(self) -> Any:
        """Intercept ``ObservationWrapper.reset`` and map observation."""
        self._reset_values()

        # reset underlying env
        obs, done = None, True
        while done:
            obs, done = self._try_reset()

        return obs

    def get_internal_steps(self) -> Union[int, None]:
        """
        Return internal steps taken to reach current node.
        """
        return self._internal_steps

    def get_internal_rewards(self) -> List[float]:
        """
        Return internal steps taken to reach current node.
        """
        return self._internal_rewards

    @override(SimulatedEnvMixin)
    def clone_from(self, env: 'CriticalStateTrainerWrapper') -> None:
        """Reset this gym environment to the given state by creating a deep copy of the `env.state` instance variable"""
        self._internal_steps = env._internal_steps
        self._internal_rewards = copy.deepcopy(env._internal_rewards)
        self._last_safe_state_steps = env._last_safe_state_steps
        self._num_critical_states = env._num_critical_states
        self._prev_critical_state = env._prev_critical_state
        self._do_skip = env._do_skip
        self._raise_exception_on_impossible_seed = env._raise_exception_on_impossible_seed

        self.env.clone_from(env)

    @override(Wrapper)
    def get_observation_and_action_dicts(self, maze_state: Optional[MazeStateType],
                                         maze_action: Optional[MazeActionType], first_step_in_episode: bool) -> Tuple[
            Optional[Dict[Union[int, str], Any]], Optional[Dict[Union[int, str], Any]]]:
        """Implementation of :class:`~maze.core.env.structured_env_spaces_mixin.StructuredEnvSpacesMixin` interface.
        """
        raise NotImplementedError

    def _step_and_increase_values(self, action: ActionType) -> Tuple[ObservationType, float, bool, dict]:
        """Preform an env step and increase the internal values."""
        last_rho = self.get_maze_state().rho.copy()
        obs, reward, done, info = self.env.step(action)
        self._internal_steps += 1

        # event logging and book keeping
        if done:
            # compute overall step count
            total_steps = self._internal_steps

            # check for end of episode
            end_of_episode_done = total_steps >= self._max_env_steps

            # fire events
            self._step_events.steps(n_steps=total_steps)
            self._step_events.steps_since_safe_state(n_time=self._internal_steps - self._last_safe_state_steps - 1)
            self._step_events.percent_critical_states(value=self._num_critical_states / float(total_steps))
            self._step_events.total_critical_states(n_states=self._num_critical_states)

            # log maximum rho encountered in previous step before done
            if last_rho is not None and not end_of_episode_done:
                self._step_events.max_rho_before_death(max_rho=last_rho.max())

            # check if done in safe state
            dead_in_safe_state = (self._prev_critical_state == 0) and not end_of_episode_done
            self._step_events.dead_in_safe_state(value=dead_in_safe_state)

        # remember last non-critical state
        elif not obs['critical_state']:
            self._last_safe_state_steps = self._internal_steps
        # count critical states
        elif obs['critical_state']:
            self._num_critical_states += 1

        self._prev_critical_state = obs['critical_state'][0]

        return obs, reward, done, info

    def _try_reset(self) -> Tuple[Optional[ObservationType], bool]:
        """Try to reset environment.

        :return: Tuple holding the first observation and the done flag.
        """
        obs = self.env.reset()
        self._reset_values()
        if not obs['critical_state']:
            self._last_safe_state_steps = self._internal_steps

        done = False
        while self._do_skip and not obs["critical_state"] and not done:
            action = self.action_conversion.noop_action()
            obs, reward, done, info = self._step_and_increase_values(action)
            # Only collect internal rewards in the regular step function!

        if done:
            print_txt = f"Env already done in reset after {self._internal_steps} steps in worker: {os.getpid()}"
            print(print_txt)
            # Raise an exception which should be caught by the mcts worker. This is necessary since, the reset alone
            #  would not iterate to the next seed during an alpha zero run.
            if self._raise_exception_on_impossible_seed:
                raise EnvDoneInResetException(print_txt)

        return obs, done

    def _reset_values(self) -> None:
        """Reset internal values, and statistics."""
        self._internal_steps = 0
        self._last_safe_state_steps = -1
        self._num_critical_states = 0
        self._prev_critical_state = 0

    @classmethod
    def _check_observation(cls, observation: ObservationType) -> None:
        """Check if critical_state flag is present in observation dict.

        :param observation: The observation to check.
        """
        assert "critical_state" in observation, \
            "Make sure that the boolean key critical_state is present in your observation dict" \
            "(e.g. via the CriticalStateObserverWrapper)!"
