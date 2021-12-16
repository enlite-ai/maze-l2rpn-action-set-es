"""L2RPN rendering classes."""
from typing import Optional, Union

import grid2op
import numpy as np
from grid2op.Observation import CompleteObservation
from grid2op.PlotGrid import PlotMatplot

from maze.core.annotations import override
from maze.core.env.maze_action import MazeActionType
from maze.core.log_events.step_event_log import StepEventLog
from maze.core.rendering.renderer import Renderer


class L2RPNRenderer(Renderer):
    """Maze-compatible wrapper around the Grid2Op renderer."""

    def __init__(self, observation_space: grid2op.Observation.ObservationSpace):
        self.plotter = PlotMatplot(observation_space)

    @override(Renderer)
    def render(self, maze_state: CompleteObservation, maze_action: Optional[MazeActionType], events: StepEventLog,
               **kwargs) -> None:
        """Render the current state using the Grid2Op matplotlib renderer."""
        self.plotter.plot_obs(maze_state, **kwargs)

    def render_line_ids(self, state: CompleteObservation):
        """Render the grid map, annotating lines with their IDs."""
        self.plotter.plot_info(line_values=list(np.arange(state.n_line)))


class L2RPNRendererFromRawData(L2RPNRenderer):
    """Renderer for rendering states represented as numpy arrays.

    On render, first converts the given numpy array into Grid2Op state (CompleteObservation), then renders it.

    Make sure that the states that you are loading correspond to the env you passed to the renderer on init.

    :param env: Grid2Op env (or its string identifier) that we will be rendering.
    """

    def __init__(self, env: Union[str, grid2op.Environment.Environment]):
        if isinstance(env, str):
            env = grid2op.make(env)

        super().__init__(env.observation_space)
        self._obs = env.reset()

    @override(L2RPNRenderer)
    def render(self, state: np.ndarray, execution: Optional[MazeActionType], events: StepEventLog, **kwargs) -> None:
        """Render the given state from the provided numpy array."""
        self._obs.from_vect(state)
        super().render(self._obs, execution, events, **kwargs)

    @override(L2RPNRenderer)
    def render_line_ids(self, state: CompleteObservation):
        """Render grid map along with IDs of individual lines.

        :param state: State represented as a numpy array.
        """
        self._obs.from_vect(state)
        super().render_line_ids(self._obs)
