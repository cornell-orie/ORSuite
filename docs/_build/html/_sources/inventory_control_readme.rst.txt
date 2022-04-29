Inventory Control with Lead Times and Multiple Suppliers
========================================================

Description
-----------

One potential application of reinforcement learning involves ordering
supplies with mutliple suppliers having various lead times and costs in
order to meet a changing demand. Lead time in inventory management is
the lapse in time between when an order is placed to replenish inventory
and when the order is received. This affects the amount of stock a
supplier needs to hold at any point in time. Moreover, due to having
multiple suppliers, at every stage the supplier is faced with a decision
on how much to order from each supplier, noting that more costly
suppliers might have to be used to replenish the inventory from a
shorter lead time.

The inventory control model addresses this by modeling an environment
where there are multiplie suppliers with different costs and lead times.
Orders must be placed with these suppliers to have an on-hand inventory
to meet a changing demand. However, both having supplies on backorder
and holding unused inventory have associated costs. The goal of the
agent is to choose the amount to order from each supplier to maximize
the revenue earned.

At each time step, an order is placed to each supplier. If previous
orders have waited for the length of their supplier’s lead time, then
these orders will become part of the on-hand inventory. The demand is
then randomly chosen from a user-selected distribution and is subtracted
from the on-hand inventory. If the on-hand inventory would become less
than zero, than items are considered to be on backorder which decreases
the reward. The demand is subtracted from the on-hand inventory to
calculate on-hand inventory for the start of the next time step. A
remaining inventory (a positive nonzero number) at the end of this
calculation negatively influences the reward proportional to the holding
costs. There are two ways that the inventory can be setup for the
environment. The first allows negative inventory to be accumulated. In
this case the on-hand inventory is offset by adding the value of the
maximum inventory. This is done so that the observation space can be
properly represented using AI Gym. This allows for backorder costs to be
calculated if the inventory were to go become negative. The second way
does not allow for inventory to become negative. Backorders are still
calculated and they still negatively influence reward, but the inventory
is reset to 0 for the next timestep after the reward calculation. The
inventory is not offset by any number in this version of the
environment.

Model Assumptions
-----------------

-  Backorders are not retroactively fulfilled. If a high demand would
   cause inventory to become negative, this unfulfilled demand is not
   met later when there may be some inventory being held at the end of a
   timestep.

Environment
-----------

Dynamics
~~~~~~~~

State Space
^^^^^^^^^^^

The state space is
:math:`S = [0,\text{Max-Order}]^{L_1} \times [0,\text{Max-Order}]^{L_2} \times ... \times [0,\text{Max-Order}]^{L_N} \times I`
where :math:`N` is the number of suppliers and
:math:`[0,\text{Max-Order}]^{L_i}` represents a list of integers between
zero and the max order amount, maxorder (specified in the
configuration), with the length of the lead time of supplier :math:`i`.
This represents how many timesteps back each order is from being added
to the inventory. :math:`I` represents the current on-hand inventory. To
represent a timestep, an order will be moved up an index in the array
unless it is added to the inventory, in which case it is removed from
the array. Each supplier has their own set of indices in the array that
represent its lead times. Each index in the list (except for $ I $) has
a maximum value of the max_order parameter.

If negative inventory is allowed, the last index, the on-hand inventory,
is offset by adding the maximum inventory value to it. It is in the
range :math:`[0, 2 * maxinventory]` This is done so that a negative
value of the on-hand inventory can be temporarily kept to use in reward
calculations for backorders and so that the observation space can be
represented properly. Before this value is used in any calculations, the
value of the max inventory is subtracted so that the true value of the
inventory is used. Otherwise if negative inventory is not allowed, the
on-hand inventory must be in the range of :math:`[0,maxinventory]` and
directly corresponds to the current inventory.

Action Space
^^^^^^^^^^^^

The action space is :math:`A = [0,\text{Max-Order}]^N` where N is the
number of suppliers. This represents the amount to order from each
supplier for the current timestep. The order amount cannot be greater
than the max_order paramter (set in the initialization of the
environment).

Reward
^^^^^^

The reward is
:math:`R = - (Order + holdcost \times max(0,I) + backordercost \times max(0, -I))`
where :math:`Order = \sum_{i = 1}^{N} c_i \times a_i` and represents the
sum of the amount most recently ordered from each supplier, :math:`a_i`,
multiplied by the appropriate ordering cost, :math:`c_i`.
:math:`holdcost` represents the holding cost for excess inventory, and
:math:`backordercost` represents the backorder cost for when the
inventory would become negative.

Transitions
^^^^^^^^^^^

At each timestep, orders are placed into each supplier for a certain
amount of resources. These orders are processed and will add to the
on-hand inventory once the lead time for the appropriate supplier has
passed. The time that has passed for each order is trakced using the
state at each timestep. If any lead times have passed, the ordered
amount is added to the on-hand inventory. Then, the randomly chosen
demand is subtracted from the on-hand inventory. If the demand is higher
than the current inventory, then the inventory does become negative for
the next state. The reward is then calculated proportional to the
revenue earned from meeting the demand, but is inversely proportional to
the amount that is backordered (the difference between the inventory and
demand). If the demand is lower than the current inventory, the
inventory remains positive for the next state. The reward is still
proportional to the revenue earned from meeting the demand, but is
inversely proportional to the amount of inventory left over multiplied
by the holding costs.

Configuration Paramters
^^^^^^^^^^^^^^^^^^^^^^^

-  lead_times: array of ints representing the lead times of each
   supplier
-  demand_dist: The random number sampled from the given distribution to
   be used to calculate the demand
-  supplier_costs: array of ints representing the costs of each supplier
-  hold_cost: The int holding cost.
-  backorder_cost: The backorder holding cost.
-  max_inventory: The maximum value (int) that can be held in inventory
-  max_order: The maximum value (int) that can be ordered from each
   supplier
-  epLen: The int number of time steps to run the experiment for.
-  starting_state: An int list containing enough indices for the sum of
   all the lead times, plus an additional index for the initial on-hand
   inventory.
-  neg_inventory: A bool that says whether the on-hand inventory can be
   negative or not.

Heuristic Agents
----------------

Random Agent
~~~~~~~~~~~~

This agent randomly samples from the action space. For this environment,
the amount ordered from each supplier is an integer from
:math:`[0, maxorder]`. ### Base Surge Agent (TBS) The base surge agent
has 2 parameters, :math:`r` and :math:`S`. Each action is expressed as
:math:`[r,[orderamount]]`. :math:`r` is a vector of the order amounts
for all suppliers except the one with the greatest lead time. :math:`S`
represents the “order up to amount”. orderamount is calculated by
calculating :math:`S - I` where :math:`I` is the current on-hand
inventory. This value is then made 0 if it is negative or is reduced to
the :math:`maxorder` if it is greater. This order amount is used for the
supplier with the greatest lead time.
