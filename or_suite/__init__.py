from gym.envs.registration import register
import or_suite.envs.ambulance
import or_suite.envs.resource_allocation
import or_suite.envs.finite_armed_bandit
import or_suite.envs.vaccine_allotment
import or_suite.envs.ridesharing

from or_suite.envs.env_configs import *
from or_suite.envs.vaccine_allotment import dynamics_model_4groups


# Ambulance Environments

register(id = 'Ambulance-v0',
    entry_point ='or_suite.envs.ambulance.ambulance_metric:AmbulanceEnvironment'
)

register(id = 'Ambulance-v1',
    entry_point ='or_suite.envs.ambulance.ambulance_graph:AmbulanceGraphEnvironment'
)

# Resource Allocation Environments

register(id = 'Resource-v0',
    entry_point = 'or_suite.envs.resource_allocation.resource_allocation:ResourceAllocationEnvironment'
)

# Finite Armed Bandit

register(id = 'Bandit-v0',
    entry_point = 'or_suite.envs.finite_armed_bandit.finite_bandit:FiniteBanditEnvironment'
)

# Vaccine Allotment Environments

register(id = 'Vaccine-v0',
         entry_point = 'or_suite.envs.vaccine_allotment.vacc_4groups:VaccineEnvironment'
)

# Ridesharing

register(id = 'Rideshare-v0',
        entry_point = 'or_suite.envs.ridesharing.rideshare_graph:RideshareGraphEnvironment'
)

# Oil Problem

register(id = 'Oil-v0',
        entry_point = 'or_suite.envs.oil_discovery.oil_problem:OilEnvironment'        
)
