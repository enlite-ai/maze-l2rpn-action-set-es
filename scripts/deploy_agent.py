"""Shows how to deploy a trained agent."""
import grid2op
import lightsim2grid
from maze_action_set_es.agents.simulation_search_policy import SimulationSearchPolicy
from maze.core.agent_deployment.agent_deployment import AgentDeployment
from maze.core.utils.config_utils import read_hydra_config, EnvFactory
from maze.core.utils.factory import Factory

from maze_action_set_es.utils import SwitchWorkingDirectory

# set the path to your training output directory
INPUT_DIR = '<path-to-training-output>'

# Parse Hydra config
hydra_overrides = {'policy': 'simulation_search_policy'}
cfg = read_hydra_config(config_module="maze_action_set_es.conf",
                        config_name="l2rpn_rollout", **hydra_overrides)

# Instantiate SimulationSearchPolicy from Hydra
with SwitchWorkingDirectory(target_dir=INPUT_DIR):
    policy = Factory(SimulationSearchPolicy).instantiate(cfg.policy)
    # Env used for action and observation conversion and wrapper stack
    deployment_env = EnvFactory(cfg.env, cfg.wrappers if "wrappers" in cfg else {})()

# Init agent deployment
agent_deployment = AgentDeployment(policy=policy, env=deployment_env)

# Simulate an external production environment that does not use Maze
external_env = grid2op.make("rte_case14_realistic", backend=lightsim2grid.LightSimBackend())

# Run interaction loop until done=True
maze_state = external_env.reset()
reward, done, info, step_count = 0, False, {}, 0
while not done:
    # Query the agent deployment for maze action, then step the environment with it
    maze_action = agent_deployment.act(maze_state, reward, done, info)
    maze_state, reward, done, info = external_env.step(maze_action)
    step_count += 1

print(f"Agent survived {step_count} steps!")
agent_deployment.close(maze_state, reward, done, info)
