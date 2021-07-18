# ORSuite Contribution Guide

ORSuite is a collection of environments, agents, and instrumentation, aimed at providing researchers in computer science and operations research reinforcement learning implementation of various problems and models arising in operations research.  In general, reinforcement learning aims to model agents interacting with an unknown environment, where the agents are given rewards by the environment based on the actions they take.  The goal then, is to develop agents for different environments that maximize their long term cumulative reward.  As such, the ORSuite package contains three main components
- agents
- environments
- instrumentation

Our agents, contained in `or_suite/agents/` are implemented specifically for a variety of models in operations research.  Moreover, we also include some black-box RL algorithms for continuous spaces in `or_suite/agents/rl/`.  The environments, contained in `or_suite/envs/` are subclasses of OpenAI Gym environments, using the same action and observation spaces which makes it relatively easy to design agents that will interact with a given environment.  Lastly we provide instrumentation, `or_suite/experiments/` which outlines finite horizon experiments collecting data about a given agent interacting with an environment.

In this guide we outline how to contribute to the ORSuite package by adding either an additional agent or environment.

## Environment Contribution Guide

Our environments, contained in `or_suite/envs/` are implemented as a subclass of an OpenAI gym environment.  For more details on creating a new environment see the guide [here](https://stable-baselines.readthedocs.io/en/master/guide/custom_env.html).  When adding an environment we require the following file structure:

```
or_suite/envs/new_env_name:
    -> __init__.py
    -> env_name.py
    -> env_name_readme.ipynb
```

The `__init__.py` file should simply import the environment class from `env_name.py`.  The `env_name_readme.ipynb` is an optional jupyter notebook outlining the environment model, including specifying the state space, action space, transitions and rewards, and describing the parameters the user can use to configure the environment file.  Lastly, the `env_name.py` should implement the environment as a subclass of the openAI gym environment.  In particular, it should have the following structure:

- The environment needs to be a class which is inherited from `gym.Env`.
- In `__init__` you need to create two variables with fixed names and types.  The first, `self.action_space` outlines the space of actions the agent can play in.  The second, `self.observation_space` indicates the states of the environment.  Both of these need to be one of Gym's special class, called `space`, which outlines potential and common state and action spaces in reinforcement learning environments.  This is outlined more in the guide later.  We also typically have the `__init__` function take as input `config`, a dictionary outlining parameters for the environment.
- The `reset` function which returns a value that is within `self.observation_space`. This is the function that will re-start the environment, say, at the start of a game.
- The `step` function has one input parameter, `action` which is contained within `self.action_space`.  This function then dictates the step of the model, requiring a return of a 4-tuple in the following order:
    - `state`, a member of `self.observation_space`
    - `reward`, a number that informs the agent about the immediate consequences of its action
    - `done`, a boolean, value that is true if the environment reached an endpoint
    - `info`, a dictionary outlining the potential side information available to the algorithm

Once the environment file structure has been made, in order to include the environment in the package we additionally require:
- Specify default parameter values for the configuration dictionary of the environment, which is saved in `or_suite/envs/env_configs.py`
- Register the environment by modifying `or_suite/envs/__init__.py` to include a link to the class along with a name
- Modify `or_suite/envs/__init__.py` to import the new environment folder.

### Gym Spaces

Every environment in ORSuite consists of an action and observation space that are both [OpenAI Gym space](https://github.com/openai/gym/tree/master/gym/spaces) objects. The action space is all possible actions that can be chosen by the agent, and the observation space is all possible states that the environment can be in. Knowing what kind of spaces are used by the environment your agent is trying to interact with will allow you to write an agent that can effectively communicate with the environment.

Both the observation space and the action space will be of one of these types:

`box`: an n-dimensional continuous feature space with an upper and lower bound for each dimension

`dict`: a dictionary of simpler spaces and labels for those spaces

`discrete`: a discrete space over n integers { 0, 1, ..., n-1 }

`multi_binary`: a binary space of size n

`multi_discrete`: allows for multiple discrete spaces with a different number of actions in each

`tuple`: a tuple space is a tuple of simpler spaces

## Agent Contribution Guide

Our agents, contained in `or_suite/agents/` are implemented as a subclass of an agent as outlined in `or_suite/agents/agent.py`.  When adding an agent for a specific environment we require the following file structure:

```
or_suite/agents/env_name:
    -> __init__.py
    -> agent_name.py
```

Or, if the agent is more general purpose it should be added under `or_suite/agents/RL`.

The `__init__.py` file should simply import the agent class from `agent_name.py`.  The `env_name_readme.ipynb` is an optional jupyter notebook outlining the environment model, including specifying the state space, action space, transitions and rewards, and describing the parameters the user can use to configure the environment file.  Lastly, the `agent_name.py` should implement the agent as a subclass of the Agent environment.  In particular, it should have the following structure:

- `__init__(config)`: initializes any necessary information for the agent (such as episode length, or information about the structure of the environment) stored in the `config` dictionary.

- `update_obs(obs, action, reward, newObs, timestep, info)`: updates any information needed by the agent using the information passed in.
    * `obs`: the state of the system at the previous timestep
    * `action`: the most recent action chosen by the agent
    * `reward`: the reward received by the agent for their most recent action
    * `newObs`: the state of the system at the current timestep
    * `timestep`: the current timestep
    * `info`: a dictionary potentially containing additional information, see specific environment for details
    

- `update_policy(self, h)`: updates internal policy based upon records, where h is the timestep

- `pick_action(state, step)`: given the current state of the environment and timestep, `pick_action` chooses and returns an action from the action space.

Once the file is set-up, ensure that the new folder or code is added to the import statements within `or_suite/agents/__init__.py`.