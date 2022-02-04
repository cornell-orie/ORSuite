The Oil Discovery Problem
=========================

Description
-----------

This problem, adaptved from
`here <https://www.pnas.org/content/109/3/764>`__ is a continuous
variant of the “Grid World” environment. It comprises of an agent
surveying a d-dimensional map in search of hidden “oil deposits”. The
world is endowed with an unknown survey function which encodes the
probability of observing oil at that specific location. For agents to
move to a new location they pay a cost proportional to the distance
moved, and surveying the land produces noisy estimates of the true value
of that location. In addition, due to varying terrain the true location
the agent moves to is perturbed as a function of the state and action.

``oil_problem.py`` is a :math:`d`-dimensional reinforcement learning
environment in the space :math:`X = [0, 1]^d`. The action space
:math:`A = [0,1]^d` corresponding to the ability to attempt to move to
any desired location within the state space. On top of that, there is a
corresponding reward function :math:`f_h(x,a)` for the reward for moving
the agent to that location. Moving also causes an additional cost
:math:`\alpha d(x,a)` scaling with respect to the distance moved.

Dynamics
--------

State Space
~~~~~~~~~~~

The state space for the line environment is :math:`S = X^d` where
:math:`X = [0, 1]` and there are :math:`d` dimensions.

Action space
~~~~~~~~~~~~

The agent chooses a location to move to, and so the action space is also
:math:`A = X^d` where :math:`X = [0,1]` and there are :math:`d`
dimensions.

Reward
~~~~~~

The reward is
:math:`\text{oil prob}(s, a, h) - \alpha \sum_i |s_i - a_i|` where
:math:`s` is the previous state of the system, :math:`a` is the action
chosen by the user, :math:`\text{oil prob}` is a user specified reward
function, and :math:`\alpha` dictates the cost tradeoff for movement.
Clearly when :math:`\alpha = 0` then the optimal policy is to just take
the action that maximizes the resulting oil probability function.

The :math:`\alpha` parameter though more generally allows the user to
control how much to penalize the agent for moving.

Transitions
~~~~~~~~~~~

Given an initial state at the start of the iteration :math:`s`, an
action chosen by the user :math:`a`, the next state will be
:math:`\begin{align*}  s_{new} = a + \text{Normal}(0, \sigma(s,a,h)) \end{align*}`
where :math:`\sigma(s,a,h)` is a user-specified function corresponding
to the variance in movement.

Environment
-----------

Metric
~~~~~~

``reset``

Returns the environment to its original state.

``step(action)``

Takes an action from the agent and returns the state of the system.

Returns:

-  ``state``: A list containing the new location of the agent

-  ``reward``: The reward associated with the most recent action and
   event

-  ``pContinue``:

-  ``info``: empty

``render``

Currently unimplemented

``close``

Currently unimplemented

Init parameters for the line ambulance environment, passed in using a
dictionary named CONFIG

-  ``epLen``: the length of each episode

-  ``dim``: the dimension of the problem

-  ``alpha``: a float :math:`\in [0,1]` that controls the proportional
   difference between the cost to move

-  ``oil_prob``: a function corresponding to the reward for moving to a
   new location

-  ``noise_variance``: a function corresponding to the variance for
   movement

-  ``starting_state``: an element in :math:`[0,1]^{dim}`

Heuristic Agents
----------------

There are no currently implemented heuristic agents for this
environment.
