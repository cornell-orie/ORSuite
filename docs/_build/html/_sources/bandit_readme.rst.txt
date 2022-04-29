The Multi-Armed Bandit Problem
==============================

Description
-----------

The Multi-Armed Bandit Problem (MAB, or often called K or N-armed bandit
problems) is a problem where a fixed set of limied resources must be
allocated between competing choices in a way that maximizes their
expected gain, when the underlying rewards is not known at the start of
learning. This is a classic reinforcement learning problem that
exemplifies the exploration-exploitation tradeoff dilema. The crucial
tradeoff the algorithm faces at each trial is between “exploitation” of
the arm that has the highest expected payoff and “exploration” to get
more information about the expected payoffs of the other arms. The
trade-off between exploration and exploitation is also faced in machine
learning.

Dynamics
--------

State Space
~~~~~~~~~~~

The state space is represented as :math:`X = [K]^T` where :math:`K` is
the number of arms and :math:`T` is the number of timesteps. Each
component represents the number of times the arm has been pulled up to
the current iteration.

Action space
~~~~~~~~~~~~

The action space is :math:`[K]` representing the index of the arm
selected at that time instant.

Reward
~~~~~~

The reward is calculated via :math:`r(x,a)` taken as a random sample
from a specified distribution :math:`\mu(a)`.

Transitions
~~~~~~~~~~~

From state :math:`x` having taking action :math:`a` the agent
transitions to a new state :math:`x'` where :math:`x'[a]` is incremented
by one to denote the increment that the arm :math:`a` has been selected
an extra time.

Environment
-----------

Line
~~~~

``reset``

Returns the environment to its original state.

``step(action)``

Takes an action from the agent and returns the state of the system after
the next arrival. \* ``action``: the index of the selected arm

Returns:

-  ``state``: The number of times each arm has been selected so far

-  ``reward``: The reward drawn from the distribution specified by the
   given action.

-  ``pContinue``:

-  ``info``: Empty

``render``

Currently unimplemented

``close``

Currently unimplemented

Heuristic Agents
----------------

We currently have no heuristic algorithms implemented for this
environment.
