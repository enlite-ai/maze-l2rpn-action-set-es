# A Re-Implementation of "Action Set Based Policy Optimization for Safe Power Grid Management" with Maze

The ["Learning to run a power network" (L2RPN)](https://l2rpn.chalearn.org/) challenge is a series of competitions
organized by RTE, the French Transmition System Operator with the aim to test the potential of
reinforcement learning (RL) to control electrical power transmission. The challenge is motivated by the fact that
existing methods are not adequate for real-time network operations on short temporal horizons in a reasonable compute
time. Also, power networks are facing a steadily growing share of renewable energy, requiring faster responses. This
raises the need for highly robust and adaptive power grid controllers.

This repository contains a baseline re-implementation of the method described in

Bo Zhou, Hongsheng Zeng, Yuecheng Liu, Kejiao Li, Fan Wang, Hao Tian (2021),
[Action Set Based Policy Optimization for Safe Power Grid Management](https://arxiv.org/abs/2106.15200).

The code in this repository builds on the [RL framework Maze](https://github.com/enlite-ai/maze).

### Overview
* [Installation and Dataset Preparation](#section-installation)
* [Agent Training, Rollout and Deployment](#section-train)
  * [Unitary Action Space Reduction](#sub-section-action-set-reduction)
  * [Training](#sub-section-training)
  * [Evaluation](#sub-section-evaluation)
  * [Deployment](#sub-section-deployment)
* [About the RL Framework Maze](#section-maze)

<a name="section-installation"></a>
## Installation and Dataset Preparation

Install all dependencies:
```shell
conda env create -f environment.yml
conda activate maze_action_set_es
```

**Note**: Per default PyTorch is installed via conda with CPU support only which is sufficient for ES training.
However, if you would like to use other trainers with this conda env
make sure to [install the appropriated PyTorch version with GPU support](https://pytorch.org/get-started/locally/).

The examples below are based on the *rte_case14_realistic* dataset.
Per default it is not shipped with a *difficulty_levels.json* file.
The command below downloads the dataset if not yet present and automatically adds a *difficulty_levels.json* file.

```shell
python scripts/prepare_data.py
```

<a name="section-train"></a>
## Agent Training, Evaluation and Deployment

<a name="sub-section-action-set-reduction"></a>
### Unitary Action Space Reduction (Optional)

To allow for more efficient exploration during training
you can reduce the unitary action space in a pre-processing step.

First, run some rollouts with the brute_force_search_policy (or actually any other policy).

```shell
maze-run -cn l2rpn_rollout +experiment=collect_action_candidates runner.n_processes=<p> runner.n_episodes=<e>
```

This will create a trajectory dump at a path similar to:

```Output directory: <root-directory>/outputs/2021-12-14/08-55-18/space_records```

Next, run the command below to extract the top 50 most often selected actions from the recorded trajectory dumps.

```shell
python scripts/top_action_selection.py --trajectory_data <path-to-output>/space_records \
--keep_k 50 --dump_file top_50_actions.npy
```

The top actions will be stored to a numpy file which you can pass as an argument to the
```maze_action_set_es/conf/env/unitary.yaml```

**Warning:** If you do not provide the reduced action set the entire unitary action set will be used for training
which is not a good idea for larger power grids due to the massive exploration space.

```python
# see: maze_action_set_es/conf/env/unitary.yaml
action_conversion:
  - _target_: maze_action_set_es.space_interfaces.action_conversion.dict_unitary.ActionConversion
    action_selection_vector: ~
    # pass reduced action set here
    action_selection_vector_dump: <absolute-path-to>/top_50_actions.npy
    set_line_status: false
    change_line_status: false
    set_topo_vect: true
    change_bus_vect: false
    redispatch: false
    curtail: false
    storage: false
```

<a name="sub-section-training"></a>
### Training

To train an agent in a locally distributed setting (single compute node), run:

```shell
maze-run -cn l2rpn_train +experiment=es_rte14_local \
runner.normalization_samples=1000 runner.n_train_workers=<num-distributed-workers>
```

* With ```runner.n_train_workers``` you can set the number of parallel ES processes collecting trajectories.
* For additional configuration options see ```maze_action_set_es/conf/experiment/es_rte14_local.yaml```.

* The results of this training run will be dumped to:
  ```Output directory: <root-directory>/outputs/unitary-default-es-local/2021-12-14_09-26-188216```
* To watch the training progress in Tensorboard, run: ```tensorboard --logdir outputs/```

<a name="sub-section-evaluation"></a>
### Evaluation

Below you find a few examples to evaluate different policies:

**Noop Policy (Baseline)**:

```shell
maze-run -cn l2rpn_rollout +experiment=rollout_rte14 policy=noop_policy wrappers=no_obs_norm
```

**Trained ES Policy (Plain argmax-Policy, no Simulation)**:

```shell
maze-run -cn l2rpn_rollout +experiment=rollout_rte14 policy=torch_policy \
input_dir=<path-to-training-output-directory>
```

**Trained ES Policy (Simulation Search Policy, 15 Candidates)**:

```shell
maze-run -cn l2rpn_rollout +experiment=rollout_rte14 policy=simulation_search_policy policy.top_k_candidates=15 \
input_dir=<path-to-training-output-directory>
```

<a name="sub-section-deployment"></a>
### Deployment

Finally, the code snippet below shows how you can execute a trained agent directly from Python
(e.g., in a challenge submission script).

(see runnable version in ```scripts/deploy_agent.py```)

```python
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
```

<a name="section-maze"></a>
## About the RL Framework Maze

![Banner](https://github.com/enlite-ai/maze/raw/main/docs/source/logos/main_logo.png)

[Maze](https://github.com/enlite-ai/maze) is an application-oriented deep reinforcement learning (RL) framework, addressing real-world decision problems.
Our vision is to cover the complete development life-cycle of RL applications, ranging from simulation engineering to agent development, training and deployment.

If you encounter a bug, miss a feature or have a question that the [documentation](https://maze-rl.readthedocs.io/) doesn't answer: We are happy to assist you! Report an [issue](https://github.com/enlite-ai/maze/issues) or start a discussion on [GitHub](https://github.com/enlite-ai/maze/discussions) or [StackOverflow](https://stackoverflow.com/questions/tagged/maze-rl).
