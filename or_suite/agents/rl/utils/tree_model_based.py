import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
''' Implementation of a tree structured used in the Adaptive Discretization Algorithm'''




import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl

from or_suite.agents.rl.utils.bounds_utils import bounds_contains, split_bounds
from or_suite.agents.rl.utils.tree import Node, Tree




class MBNode(Node):

    """
    Node representing an l-infinity ball in R^d, that points
    to sub-balls (defined via node children).
    Stores a value for the q_estimate, a number of visits, and 
    
        **** rewards and transition probability to a list of other nodes. ***


    This class is used to represent (and store data about)
    a tuple (state, action, stage) = (x, a, h).


    Attributes:
        bounds : numpy.ndarray
            Bounds of each dimension [ [x0, y0], [x1, y1], ..., [xd, yd] ],
            representing the cartesian product in R^d:
            [x0, y0] X [x1, y1] X ... X [xd, yd]
        depth: int
            Node depth, root is at depth 0.
        qVal : double, default: 0
            Initial node Q value
        num_visits : int, default = 0
            Number of visits to the node.
    """



    def __init__(self, bounds, depth, qVal, rEst, pEst, num_visits):

        self.dim = len(bounds)
        self.radius = (bounds[:, 1] - bounds[:, 0]).max() / 2.0
        assert self.radius > 0.0

        self.bounds = bounds
        self.depth = depth
        self.qVal = qVal
        self.rEst = rEst
        self.pEst = pEst
        self.num_visits = num_visits

        self.children = []


    # Splits a node
    def split_node(self, inherit_flag = True, value = 1):

        child_bounds = split_bounds(self.bounds)
        for bounds in child_bounds:
            if inherit_flag:  # updates estimates based on whether we are inheriting estimates or not
                self.children.append(
                    MBNode(bounds, self.depth+1, self.qVal, self.rEst, self.pEst.copy(), self.num_visits)
                )
            else:
                self.children.append(
                    MBNode(bounds, self.depth+1, value, 0, [0 for _ in range(len(self.pEst))], 0)
                )

        return self.children





class MBTree(Tree):

    """
    Tree representing a collection of l-infinity ball in R^d, that points
    to sub-balls (defined via node children).
    Stores a hierarchical collections of nodes with value for the q_estimate, a number of visits, and 



    Attributes:
        dim : int
            Dimension of the space of R^d.
        head: (Node)
            Pointer to the first node in the hierarchical partition
        epLen: (int)
            Number of episodes (used for initializing estimates for Q Values)
    """


    # Defines a tree by the number of steps for the initialization
    def __init__(self, epLen, state_dim, action_dim):
        self.dim = state_dim+action_dim  # total dimension of state and action space
        self.epLen = epLen
        self.state_dim = state_dim  # stores state space dimension separately


        # initializes head of the tree
        bounds = np.asarray([[0.0,1.0] for _ in range(self.dim)])

        self.head = MBNode(bounds, 0, epLen, 0, [0.0], 0)

        # initializes state leaves of the tree and their value estimates used in the model based algorithm
        self.state_leaves = [[0.5 for _ in range(self.state_dim)]]
        self.leaves = [self.head]
        self.vEst = [self.epLen]


    def get_leaves(self):
        return self.leaves

    def tr_split_node(self, node, timestep = 0, inherit_flag = True, value = 1, previous_tree = None):
        """
        Splits a node, while simultaneously updating the estimate of the transition kernels for all nodes if needed.

        Args:
            node: MBNode to split
            inherit_flag: (bool) boolean of whether to inherit estimates of not
            value: (float) default qVal estimate
        """
        # Splits a node and updates the list of leaves
        self.leaves.remove(node)
        children = node.split_node(inherit_flag, value)
        self.leaves = self.leaves + children


        # Determines if we also need to adjust the state_leaves and carry those
        # estimates down as well

        # Gets one of their state value
        child_1_bounds = children[0].bounds
        child_1_radius = (child_1_bounds[:, 1] - child_1_bounds[:, 0]).max() / 2.0
        child_1_state = child_1_bounds[:self.state_dim, 0] + child_1_radius


        if np.min(np.abs(np.asarray(self.state_leaves) - child_1_state)) >= child_1_radius: # determines if the children are at a finer granularity

            # gets state portion of the value of the current node
            node_radius = (node.bounds[:, 1] - node.bounds[:, 0]).max() / 2.0
            node_state = node.bounds[:self.state_dim, 0] + node_radius

            # location of node in the larger state_leaves list
            parent_index = np.argmin(np.max(np.abs(np.asarray(self.state_leaves) - node_state), axis=1))

            parent_vEst = self.vEst[parent_index]


            # pops their estimate
            self.state_leaves.pop(parent_index)
            self.vEst.pop(parent_index)

            # keeps track of the number added for redistributing the transition kernel estimate
            num_add = 0
            for child in node.children:
                child_radius = (child.bounds[:,1] - child.bounds[:,0]).max() / 2.0
                child_state = child.bounds[:self.state_dim, 0] + child_radius # gets the state portion of the node


                # determines if this child state has been added before
                if len(self.state_leaves) == 0 or np.min(np.max(np.abs(np.asarray(self.state_leaves) - child_state), axis=1)) > 0: 
                    num_add += 1 
                    self.state_leaves.append(child_state)
                    self.vEst.append(parent_vEst) # updates estimates based on the parent

            # updates the transition distribution for all leaves in the previous tree
            if timestep >= 1:
                previous_tree.update_transitions_after_split(parent_index, num_add)

        return children


    def update_transitions_after_split(self, parent_index, num_add):
        """
            Helper function in order to update the transition estimates after a split.
            Args:
                parent_index: location in the list where the parent node was
                num_children: the numer of new nodes that were added for redistributing transition kernel estimate

        """

        for node in self.leaves:
            pEst_parent = node.pEst[parent_index]
            node.pEst.pop(parent_index)

            for _ in range(num_add):
                node.pEst.append(pEst_parent / num_add)