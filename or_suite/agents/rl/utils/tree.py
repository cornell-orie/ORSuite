import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl

from or_suite.agents.rl.utils.bounds_utils import bounds_contains, split_bounds





class Node():

    """
    Node representing an l-infinity ball in R^d, that points
    to sub-balls (defined via node children).
    Stores a value for the q_estimate, a number of visits, and 
    
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
        children: (list)
            List of children for the node
    """



    def __init__(self, bounds, depth, qVal, num_visits):
        """
        Initialization for a node.


        Args:
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

        self.dim = len(bounds)  # updates dimension of the ball

        self.radius = (bounds[:, 1] - bounds[:, 0]).max() / 2.0  # calculates its radius

        assert self.radius > 0.0, "Error: radius of a ball should be strictly positive"

        self.bounds = bounds
        self.depth = depth
        self.qVal = qVal
        self.num_visits = num_visits

        self.children = []  # list of children for a ball


    def is_leaf(self):
        return len(self.children) == None

    def contains(self, state):
        return bounds_contains(self.bounds, state)



    # Splits a node
    def split_node(self, inherit_flag = True, value = 1):
        """
        Splits a node across all of the dimensions


        Args:
            inherit_flag: (bool) boolean of whether to intialize estimates of children to that of parent
            value: default qValue to inherit if inherit_flag is false
        """


        child_bounds = split_bounds(self.bounds)  # splits the bounds of the ball

        for bounds in child_bounds:  # adds a child for each of the split bounds, initializing their values appropriately
            if inherit_flag:
                self.children.append(
                    Node(bounds, self.depth+1, self.qVal, self.num_visits)
                )
            else:
                self.children.append(
                    Node(bounds, self.depth+1, value, 0)
                )

        return self.children




class Tree():

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
    def __init__(self, epLen, dim):
        """
            Initializes the values passed
        """

        self.dim = dim

        bounds = np.asarray([[0.0,1.0] for _ in range(dim)])

        self.head = Node(bounds, 0, epLen, 0)
        self.epLen = epLen
        
    # Returns the head of the tree
    def get_head(self):
        return self.head

    # Returns the maximum Q value across all nodes in the tree
    def get_max(self, node = None, root = True):
        if root:
            node = self.head

        if len(node.children) == 0:
            return node.qVal
        else:
            return np.max([self.get_max(child, False) for child in node.children])

    # Returns the minimum Q value across all nodes in the tree
    def get_min(self, node = None, root = True):
        if root:
            node = self.head


        if len(node.children) == 0:
            return node.qVal
        else:
            return np.min([self.get_min(child, False) for child in node.children])

    # TODO: Might need to make some edits to this
    def plot(self, figname = 'tree plot', colormap_name = 'cool', max_value = 10, node=None, root=True,):
        if root:
            assert self.dim == 2, "Plot only available for 2-dimensional spaces."
        
        if node.is_leaf():
            x0, x1 = node.bounds[0, :]
            y0, y1 = node.bounds[1, :]
            colormap_fn = plt.get_cmap(colormap_name)
            color = colormap_fn(node.qVal / max_value)
            rectangle = plt.Rectangle((x0, y0), x1-x0, y1-y0, ec='black', color=color)
            plt.gca().add_patch(rectangle)
            plt.axis('scaled')
        else:
            for cc in node.children:
                self.plot(max_value = max_value, colormap_name = colormap_name, node=cc, root=False)


    # Recursive method which gets number of nodes across the tree
    def get_num_balls(self, node = None, root = True):
        if root:
            node = self.head

        num_balls = 1
        for child in node.children:
            num_balls += self.get_num_balls(child, False)
        return num_balls

    def get_active_ball(self, state, node = None, root = True):
        """
            Gets the active ball for a given state, i.e., the node in the tree containing the state with the largest Q Value

            Args:
                state: np.array corresponding to a state
                node: Current node we are searching for max value over children of
                root: indicator that we are at the root, and should start calculating from the head of the tree

            Returns:
                best_node: the node corresponding to the largest q value containing the state
                best_qVal: the value of the best node
                
            TODO: Fix to only iterate over leaves? Might improve computational complexity
        """


        if root:
            node = self.head

        if len(node.children) == 0:
            return node, node.qVal
        
        else:
            best_qVal = (-1)*np.inf

            for child in node.children:
                if child.contains(state):
                    nn, nn_qVal = self.get_active_ball(state, child, False)
                    if nn_qVal >= best_qVal:
                        best_node, best_qVal = nn, nn_qVal
            return best_node, best_qVal

