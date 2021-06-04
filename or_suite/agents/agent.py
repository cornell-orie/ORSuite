'''
All agents should inherit from the Agent class.
'''


class Agent(object):

    def __init__(self):
        pass

    def update_config(self, env, config):
        ''' Update agent information based on the config__file'''
        pass

    def update_obs(self, obs, action, reward, newObs, info):
        '''Add observation to records'''

    def update_policy(self, h):
        '''Update internal policy based upon records'''

    def pick_action(self, obs):
        '''Select an action based upon the observation'''