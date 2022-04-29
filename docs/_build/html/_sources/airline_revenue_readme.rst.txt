Revenue Management
==================

Description
-----------

Online revenue management (also known as online stochastic bin packing)
considers managing different available resources consumed by different
classes of customers in order to maximize revenue. In this environment,
we model multiple types of resources with some initial availability. At
each iteration, the algorithm designer decides in the current time step
whether or not to accept customers from a given class. One customer of a
given class comes and arrives to the system, if the agent decides to
fulfill their request, they utilize some amount of the different
resources and provide an amount of revenue. At a high level, then, the
goal of the agent is to chose which types of customers to accept at
every iteration in order to maximize the total revenue. This requires
planning for the scarce resources and ensuring that the agent does not
allocate to individuals who will exhaust the remaining resources.

Model Assumptions
-----------------

-  Customers who are denied are not allowed to purchase resources later
   even if those resources are available. This did not extend to
   customer classes, though. A customer may be allowed to purchase
   resources even if another customer of the same class was denied at an
   earlier (or later) timestep.

Environment
-----------

Dynamics
~~~~~~~~

State Space
^^^^^^^^^^^

The state space is the set of all possible available seats for every
flight into and out of each location up to the full capacities. :math:`S = [0, B_1] \times [0, B_2] \times ... \times [0, B_k]` where :math:`B_i` is the maximum availability
of resource type :math:`i`and :math:`k` is the number of resource types.

Action Space
^^^^^^^^^^^^

The action space is all possible binary vectors of length :math:`n` which
tells you whether a customer class is accepted or declined by the
company, where n is the number of customer classes. :math:`A = {{0, 1}}^n`

Reward
^^^^^^

The one-step reward is the revenue gained from selling resources to the
customer class that arrives. If resources are not sold (because the
customer is denied or the resources desired are not available), then the
reward is zero.

Transitions
^^^^^^^^^^^

Given an arrival :math:`P_t` of type :math:`j_t \in [n]` or :math:`\emptyset`: 
   * if :math:`\emptyset` then :math:`S_{t+1} = S_t` with :math:`reward = 0`, indicating that no arrivals occured and so the agent receives no revenue 
   * if :math:`j_t`: 
  
     * if :math:`a_{j_t} = 0` (i.e. algorithm refuses to allocate to that type of customer) then :math:`S_{t+1} = S_t` with :math:`reward = 0`
     * if :math:`a_{j_t} = 1` and :math:`S_t - A_{j_t}^T ≥ 0` (i.e. budget for resources to satisfy the request) then :math:`S_{t + 1} = S_t - A_{j_t}^T` with :math:`reward = f_{j_t}`
  
At each time step a customer may or may not arrive. If no customer
arrives, then the next state is the same as the current state and the
reward is zero. If a customer does arrive they can either be accepted or
rejected according to the action taken for the timestep (the action is
determined before the arrival of the customer). If the customer is
rejected, the next state is the same as the current state and the reward
is zero. If the customer is accepted, the resources that they wish to
purchase may or may not be available. If they are not available, then
the next state is the same as the current state and the reward is zero.
If they are available, then the resources purchased are subtracted from
the current number available for the next state and the reward is the
value determined when initializing the environment for the class of
customer that arrived.

Configuration Parameters
^^^^^^^^^^^^^^^^^^^^^^^^

-  ``epLen``: The int number of time steps to run the experiment for.
-  ``f``: The float array representing the revenue per class.
-  ``A``: The 2-D float array representing the resource consumption.
-  ``starting_state``: The float array representing the number of
   available resources of each type.
-  ``P``: The float array representing the distribution over arrivals.

Heuristic Agents
----------------
