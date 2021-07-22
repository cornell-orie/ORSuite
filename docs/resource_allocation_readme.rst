Resource Allocation
===================

Description
~~~~~~~~~~~

The Food Bank of the Southern Tier (FBST) is a member of Feeding
America, focused on providing food security for people with limited
financial resources, and serves six counties and nearly 4,000 square
miles in the New York. Under normal operations (non COVID times), the
Mobile Food Pantry program is among the main activities of the FBST. The
goal of the service is to make nutritious and healthy food more
accessible to people in underserved communities. Even in areas where
other agencies provide assistance, clients may not always have access to
food due to limited public transportation options, or because those
agencies are only open hours or days per work.

The Mobile Food Pantry provides food directly to clients at various
distribution locations. In 2019 they serviced 70 regular sites, with 722
visits across all of them. When the truck arrives at a distribution
site, volunteers lay out the food on tables. The clients can then shop,
choosing items that they need. A typical mobile food pantry visit lasts
two hours and provides 200 to 250 families with nutritious food to help
them make ends meet. The goal of this project is to understand: *What is
a fair allocation here, and how can it be computed*?

In *offline* problems where demands for all locations are known to the
principal, there are many well-studied notions of fair allocation of
limited resources. A relevant notion in our context is that a fair
allocation is one satisfying three desiderata: \* *Pareto-efficiency*:
for any location to benefit, another must be hurt \* *Envy-freeness*: no
location prefers an allocation received by another \* *Proportionality*:
each location prefers the allocation received versus equal allocation

This definition draws its importance from the fact that in many
allocation settings it is known to be achievable. In particular, when
goods are divisible, then for a large class of utility functions, an
allocation satisfying both is easily computed (via a convex program) by
maximizing the Nash Social Welfare (NSW) objective with respect to the
allocation.

Many practical settings, however, operate more akin to the FBST mobile
food pantry, in that the principal makes allocation decisions *online*,
and has the additional question of *scheduling* which location to
provide food drop-offs and when. This project will take on extending the
notion of fairness to these resource allocation problems, and assessing
the efficacy of Reinforcement Learning algorithms on giving a fair
allocation online.

Dynamics
~~~~~~~~

-  Consider a principle that wants to find an allocation of :math:`K`
   commodities with initial budget :math:`B` and set of possible types
   :math:`\Theta` over :math:`n` locations each with an endowment
   distribution of :math:`\mathcal{F}_i`. We consider the MDP formed by
   this sequential allocation problem a 5-tuple
   :math:`(\mathcal{S}, \mathcal{A}, \mathcal{T}, R, \mathcal{H})`

   -  Our state space
      :math:`\mathcal{S} := \{(b,N)|b \in \mathbb{R}_+^{k},N \in \mathbb{R}_+^{|\Theta|}\}`
      where :math:`b` is a vector of all remaining budget for commodity
      :math:`k`, and :math:`N` is a vector of the number of people of
      each type present. Our initial state :math:`S_0 = (B,N_0)`, where
      :math:`B` is the full pre-planned budget and
      :math:`N_0 \sim \mathcal{F}_0`
   -  Actions correspond to the allocation we make to agent :math:`i`.
      We will split this allocation across all the types, leading to a
      matrix of allocation vectors. Formally the action-space in state
      :math:`i` is defined as
      :math:`A_i := \{X \in \mathbb{R}_+^{|\Theta| \times K}|\sum_{\theta \in \Theta}N_{\theta}X_{\theta} \leq b\}`
      where :math:`b` is the current budget vector.
   -  Our reward-space :math:`R` is the Nash Social Welfare: while in
      state :math:`s` and taking action :math:`a`, we have
      :math:`R(s,a) = \sum_{\theta \in \Theta} N_\theta \log u(X_{\theta},\theta) \rangle`
      where
      :math:`u: \mathbb{R}^{|\Theta| \times K} \times \mathbb{R}^k \to \mathbb{R}_+`
      is a utility function for the agents.
   -  Transitions. Given state :math:`(b,N_i) \in \mathcal{S}` and
      action :math:`X \in \mathcal{A}`. we have our new state
      :math:`s_{i+1} = (b-\sum_{\theta}N_{\theta}X_{\theta},N_{i+1})`
      where :math:`N_{i+1} \sim \mathcal{F}_{i+1}`
   -  Each episode will have the same number of steps as there are
      locations. Thus :math:`\mathcal{H}=n`

Model Assumptions
~~~~~~~~~~~~~~~~~

-  We also primarily focus on utility functions that are linear, where
   the latent individual type is characterized by a vector of
   preferences over each of the different resources. Here again, we
   believe that our techniques extend to more general homothetic
   preferences, but for ease of notation (and given the richness of
   linear utilities), choose to focus on these.

-  We assume that our resources aredivisible, in that allocations can
   take values inRK+.In our particular regimes of interest where we
   scale the number of rounds and budgets, this is easyto relax to
   integer allocations with vanishing loss in performance.

-  The assumption that latent types are finite is common in
   decision-making settings, as in practice, the set of possible types
   is approximated from historical data.

-  One limiting assumption is that in the online setting, the principal
   only knows the number of individuals from one location at a time. In
   reality the principal could have some additional information about
   future locations, e.g. via calling ahead, that could be incorporated
   in decidingan allocation. Our algorithmic approach naturally
   incorporates such additional information.

Environment
~~~~~~~~~~~

``reset``

Returns the environment to it’s original state.

``step(action)``

Takes in the action from the agent and returns the state of the system
of the next arrival.

-  ``action``: A matrix :math:`X \in \mathbb{R}^{|\Theta| \times K}`
   where each row is a :math:`K`-dimensional vector denoting how much of
   each commodity is given to each type. This information is encoded in
   AIGym’s ``spaces.Box(...)`` feature

Returns:

-  ``state``: A tuple :math:`(b,N)` where :math:`b \in \mathbb{R}^K` is
   a vector denoting remaining budget and
   :math:`N \in \mathbb{R}^{|\Theta|}` is a vector denoting the number
   of people of each type :math:`\theta` that appear

-  ``reward``: the reward is currently defined as the Nash Social
   Welfare: While in state :math:`s` and taking action :math:`a`, we
   have
   :math:`R(s,a) = \sum_{\theta \in \Theta} N_\theta \log u(X_{\theta},\theta) \rangle`
   where :math:`u: \mathbb{R}^k \times \mathbb{R}^k \to \mathbb{R}_+` is
   the individual utility function

-  ``pContinue``: information on whether or not the episode should
   continue

-  ``info``: a dictionary containing the type of the newest location

   -  Ex. ``{'type': np.array([1,2,3])}`` if the newest location has
      type of :math:`[1,2,3]`

``render``

Currently unimplemented

``close``

Currently unimplemented

``make_resource_allocationEnvMDP(config)``

Creates an instance of the environment according a ``config`` dictionary

-  ``K``: number of commodities available

-  ``num_agents``: number of locations to visit

-  ``weight_matrix``: A matrix
   :math:`W \in \mathbb{R}^{|\Theta| \times K}` where each row denotes a
   possible type

-  ``init_budget``: vector :math:`B \in \mathbb{R}^K` indicating the
   initial budget for all commodities

-  ``type_dist``: a function
   :math:`\mathcal{F}: \mathbb{N} \rightarrow \Delta(\mathbb{R}^{|\Theta|})`
   where :math:`\mathcal{F}_i` gives the distribution of the number of
   people of each type at location :math:`i`

-  ``u``: the utility function
   :math:`u: \mathbb{R}^{|\Theta| \times K} \times \mathbb{R}^{|\Theta|} \to \mathbb{R}_+`
   where
   :math:`u(X,N) = \sum_{\theta \in \Theta} N_\theta \log \langle X_{\theta},W_{\theta} \rangle`
   where :math:`W` is our previously defined weight matrix

-  ``starting_state``: a tuple of ``(init_budget, type_dist(0))``

Heuristic Agents
~~~~~~~~~~~~~~~~

Equal Allocation
^^^^^^^^^^^^^^^^

On a high level, the equal allocation agent will subdivide the initial
budget equally among all :math:`n` locations. Each location-specific
allocation will be further subdivided (so as to create the matrix of
allocation) by relative proportion of the types present at location
:math:`i`. More formally, our allocation
:math:`X_{i,\theta} \in \mathbb{R}^k` is defined as

.. math::


   X_{i,\theta} = B\left( \frac{\mathbb{E}\left[N_{i,\theta}\right]}{\sum_{i,\theta}\mathbb{E}\left[N_{i,\theta}\right]}\right) 
