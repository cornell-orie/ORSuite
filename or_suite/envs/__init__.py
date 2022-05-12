from gym.envs.registration import register
import or_suite.envs.ambulance
import or_suite.envs.resource_allocation
import or_suite.envs.finite_armed_bandit
import or_suite.envs.vaccine_allotment
import or_suite.envs.ridesharing
import or_suite.envs.inventory_control_multiple_suppliers
import or_suite.envs.airline_revenue_management

from or_suite.envs.env_configs import *
from or_suite.envs.vaccine_allotment import dynamics_model_4groups


# Ambulance Environments

register(id='Ambulance-v0',
         entry_point='or_suite.envs.ambulance.ambulance_metric:AmbulanceEnvironment'
         )

register(id='Ambulance-v1',
         entry_point='or_suite.envs.ambulance.ambulance_graph:AmbulanceGraphEnvironment'
         )

# Resource Allocation Environments

register(id='Resource-v0',
         entry_point='or_suite.envs.resource_allocation.resource_allocation:ResourceAllocationEnvironment'
         )

register(id='Resource-v1',
         entry_point='or_suite.envs.resource_allocation.resource_allocation_discrete:DiscreteResourceAllocationEnvironment'
         )


# Finite Armed Bandit

register(id='Bandit-v0',
         entry_point='or_suite.envs.finite_armed_bandit.finite_bandit:FiniteBanditEnvironment'
         )

# Vaccine Allotment Environments

register(id='Vaccine-v0',
         entry_point='or_suite.envs.vaccine_allotment.vacc_4groups:VaccineEnvironment'
         )

# Ridesharing

register(id='Rideshare-v0',
         entry_point='or_suite.envs.ridesharing.rideshare_graph:RideshareGraphEnvironment'
         )
register(id='Rideshare-v1',
         entry_point='or_suite.envs.ridesharing.rideshare_graph_traveltime:RideshareGraphEnvironment'
         )

# Oil Problem

register(id='Oil-v0',
         entry_point='or_suite.envs.oil_discovery.oil_problem:OilEnvironment'
         )

# Inventory Control With Multiple Suppliers
register(id='MultipleSuppliers-v0',
         entry_point='or_suite.envs.inventory_control_multiple_suppliers.multiple_suppliers_env:DualSourcingEnvironment'
         )


# Airline Problem
register(id='Airline-v0',
         entry_point='or_suite.envs.airline_revenue_management.airline_env:AirlineRevenueEnvironment'
         )
