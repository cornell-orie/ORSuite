'''
All agents should inherit from the Agent class.
'''


class Agent(object):

    def __init__(self):
        pass

    def reset(self):
        pass

    def update_config(self, env, config):
        ''' Update agent information based on the config__file'''
        self.config = config
        return
        
    def update_parameters(self, param):
        return

    def update_obs(self, obs, action, reward, newObs, timestep, info):
        '''Add observation to records'''

    def update_policy(self, h):
        '''Update internal policy based upon records'''

    def pick_action(self, obs, h):
        '''Select an action based upon the observation'''
