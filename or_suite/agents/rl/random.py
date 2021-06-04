from .. import Agent

'''

Implementation of a randomized algorithm which employs a policy which samples uniformly at random from the action space

'''


class randomAgent(Agent):
    """Randomized RL Algorithm

    Implements the randomized RL algorithm - selection an action uniformly at random from the action space.  In particular,
    the algorithm stores an internal copy of the environment's action space and samples uniformly at random from it.

    """


    def __init__(self):
        pass


    def reset(self):
        pass

    def update_config(self, env, config = None):
        """Updates configuration file for the agent

        Updates the stored environment to sample uniformly from.

        Args:
            env: an openAI gym environment
            config: an (optional) dictionary containing parameters for the environment
        """

        self.environment = env
        pass

    def update_obs(self, obs, action, reward, newObs, timestep, info):
        pass

    def update_policy(self, h):
        pass

    def pick_action(self, obs, h):
        """Selects an action for the algorithm.

        Args:
            obs: a state for the environment
            h: timestep

        Returns:
            An action sampled uniformly at random from the environment's action space.
        """
        return self.environment.action_space.sample()
