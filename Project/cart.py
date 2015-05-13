import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import *
from random import *
from copy import deepcopy as copy
from scipy.misc import comb
from scipy.stats import norm
from scipy.stats import t as t_dist
from sklearn.linear_model import *
from sklearn.cross_validation import StratifiedKFold

# node for tree
class DecisionTree(object):
    def __init__(self, leaf=False, var=None, value=None, parent=None, x=None,
            y=None, min_size=5):
        self.leaf = leaf
        self.var = var
        self.value = value
        self.parent = parent
        self.x = x
        self.y = y
        self.min_size = min_size
        
        self.l_child = None
        self.r_child = None

    # function to get all the leaves of a tree
    def get_leaves(self):
        
        # check if tree is a leaf
        if self.leaf:
            return None

        # initialize iterating node, sets
        inode = self
        visited = set()
        leaves = set()

        # cycle through all nodes until arriving at root
        while inode != None:
            if inode.leaf:
                visited.add(inode)
                leaves.add(inode)
                inode = inode.parent
            elif inode.l_child not in visited:
                inode = inode.l_child
            elif inode.r_child not in visited:
                inode = inode.r_child
            else:
                visited.add(inode)
                inode = inode.parent
        
        # return a list of leaves
        return list(leaves)

    # function to split a node's data by dimension d and specified split
    def split_node_data(self, d, split):
        
        x1 = self.x[self.x[:,d] > split]
        y1 = self.y[self.x[:,d] > split]
        x2 = self.x[self.x[:,d] <= split]
        y2 = self.y[self.x[:,d] <= split]

        return x1,y1,x2,y2

    # splits a node and creates children
    def split_node(self):

        self.leaf = False

        # separate data
        xL, yL, xR, yR = self.split_node_data(self.var, self.value)
        
        # get majority rule values
        lval = float(sum(yL) > len(yL)/2)
        rval = float(sum(yR) > len(yR)/2)

        # create children
        self.l_child = DecisionTree(x=xL, y=yL, leaf=True, parent=self, 
                value=lval, min_size=self.min_size)
        self.r_child = DecisionTree(x=xR, y=yR, leaf=True, parent=self, 
                value=rval, min_size=self.min_size)
       
        return

    # function to obtain the error in a split
    def split_error(self, d, split):

        # get partitioned data
        xL, yL, xR, yR = self.split_node_data(d, split)

        # get probabilities of each class
        p1_L = float(sum(yL)) / len(yL)
        p0_L = 1 - p1_L
        p1_R = float(sum(yR)) / len(yR)
        p0_R = 1 - p1_R

        # compute entropies
        '''
        if p0_L == 0 or p1_L == 0:
            ent_L = 0
        else:
            ent_L = - (p0_L * log(p0_L) + p1_L * log(p1_L))
        if p0_R == 0 or p1_R == 0:
            ent_R = 0
        else:
            ent_R = - (p0_R * log(p0_R) + p1_R * log(p1_R))
        '''

        # compute gini indexes
        gini_L = 1 - p0_L**2 - p1_L**2
        gini_R = 1 - p0_R**2 - p1_R**2

        # assign impurity values
        imp_L = gini_L
        imp_R = gini_R

        # compute split impurity
        imp_split = (imp_L * len(yL) + imp_R * len(yR)) / len(self.y)
        
        return imp_split

    # decide classifiers for X-data
    def predict(self, X):
        
        # get length, initialize y prediction array
        N = len(X)
        yp = np.zeros(N)

        # predict every point X_i
        for i in range(N):

            # go through tree unitl we arrive at a leaf
            node = self
            while not node.leaf:
                if X[i][node.var] > node.value:
                    node = node.l_child
                else:
                    node = node.r_child
            
            # take the value of the leaf
            yp[i] = node.value

        return yp

    # define function for creating a treedef build_tree(tree):
    def build_tree(self):

        # get dimensions of data
        N = self.x.shape[0]
        D = self.x.shape[1]

        # check to ensure node is large enough
        if N < self.min_size:
            return

        # check to ensure node is not pure
        if sum(self.y) == 0 or sum(self.y) == len(self.y):
            return
        
        # intialize best splits
        best_split_dim = None
        best_split_val = None
        best_split_error = float('inf')
        
        # find best single split of the data
        for d in xrange(D):
            
            # get a sorted list of all unique points of x in d dimension
            unique_pts = set()
            for i in xrange(N):
                unique_pts.add(self.x[i,d])
            unique_pts = sorted(unique_pts)
        
            # form all splits
            for k in xrange(len(unique_pts)-1):
                
                # define split half way in between unique points
                split = (unique_pts[k] + unique_pts[k+1]) / 2

                # test the effictiveness of each split
                error = self.split_error(d, split)
                
                # check leaf sizes
                xL, yL, xR, yR = self.split_node_data(d, split)
                too_small = len(yL) < self.min_size or len(yR) < self.min_size

                # check to see if split is best and within size constraints
                if error <= best_split_error and not too_small:
                    best_split_dim = d
                    best_split_val = split
                    best_split_error = error
        
        # check if no split available
        if best_split_val == None:
            return

        # set parent node parameters
        self.var = best_split_dim
        self.value = best_split_val

        # create children
        self.split_node()

        # create sub-trees
        self.l_child.build_tree()
        self.r_child.build_tree()
        return

    # fit the decision tree to user inputted X and Y data
    def fit(self, Xdata, Ydata):

        self.x = Xdata
        self.y = Ydata
        self.build_tree()

    # function to get all leaves
    def get_leaves(self):

        # initlaize sets, start at parent node
        visited = set()
        leaves = set()
        inode = self

        # cycle through all nodes until arriving back at the parent
        while inode != None:
            
            # check if current node is a leaf
            if inode.leaf:
                visited.add(inode)
                leaves.add(inode)
                inode = inode.parent

            # check if left child not visited
            elif inode.l_child not in visited:
                inode = inode.l_child
            
            # check if right child not visited
            elif inode.r_child not in visited:
                inode = inode.r_child

            # otherwise, move up to the parent
            else:
                visited.add(inode)
                inode = inode.parent
        
        return list(leaves)

    # converts a parent node into a majority-rule leaf
    def make_leaf(self):
        
        # check if already a leaf
        if self.leaf:
            return

        # convert to leaf
        self.leaf = True
        self.l_child = None
        self.r_child = None

        # assign majority-rule value
        self.value = float(sum(self.y) > len(self.y)/2)

        return

    # returns the generalized classification error of the tree with complexity
    # penalty alpha
    def gen_error(self, alpha):
        n_nodes = len(self.get_leaves())
        return sum(abs(self.predict(self.x) - self.y)) + alpha * n_nodes

    # prune tree with requested critera
    def pruned_tree(self, alpha):

        # create a copy of the tree
        tree = copy(self)

        # get a list of nodes subject to pruning
        nodes = set()
        leaves = tree.get_leaves()
        for leaf in leaves:
            nodes.add(leaf.parent)

        # calculate the current generalized classification error
        best_error = tree.gen_error(alpha)

        # print a list of nodes subject to pruning
        while len(nodes) != 0:
            
            # pick off a node from the set
            node = nodes.pop()

            if node.l_child.leaf and node.r_child.leaf:

                # prune node
                split_val = node.value
                node.make_leaf()

                # decide whether to re-split
                split_error = tree.gen_error(alpha)
                if split_error > best_error:
                    node.value = split_val
                    node.split_node()
                else:
                    best_error = split_error
                    if node.parent != None:
                        nodes.add(node.parent)

        return tree

    # exports descrition of tree to a string dot format
    def dot_export(self, var_names=None, cat_names=None, cat_vars=None):

        # initialize dot string and counter
        dot = "digraph Tree {\n"
        counter = 0
        
        # initialize lists
        current = []
        neighbors = [self]

        while len(neighbors) > 0:

            current = neighbors
            neighbors = []

            for node in current:

                node.count = counter

                dot += str(counter) + ' [label="'

                # check if node is a leaf
                if node.leaf:

                    # default to printing class number
                    if cat_names == None:
                        dot += "Value = " + str(int(node.value))
                        dot += '\\n'

                    # use category name if specified
                    else:
                        dot += cat_names[int(node.value)]
                        dot += '\\n'
                
                # otherwise print splitting condition
                else:

                    # determine if variable is categorical
                    print_cat = False
                    if cat_vars != None:
                        if cat_vars[node.var]:
                            print_cat = True

                    # if categorical variable, print categroy name
                    if print_cat:
                        dot += var_names[node.var] + '?\\n'

                    # otherwise, print split point
                    else:

                        # default to stating Xvector poistion
                        if var_names == None:
                            var = 'X[' + str(node.var) + ']'
                        
                        # use variable names instead if specified
                        else:
                            var = var_names[node.var]
                        
                        # add splitting condition rounded to 2 decimal places
                        dot += var
                        dot += ' > ' + str(round(node.value,2))
                        dot += '\\n'
                    
                    
                # record the number of samples
                dot += 'samples = ' + str(len(node.y)) + '", shape="box"];'
                dot += '\n'

                # draw arrows
                if counter > 0:
                    
                    dot += str(node.parent.count) + ' -> ' + str(counter)
                    if node.parent.l_child == node:
                        dot += ' [label="True"]'
                    elif node.parent.r_child == node:
                        dot += ' [label="False"]'
                    dot += ';\n'
                
                # add children to neighbors
                if node.l_child != None:
                    neighbors.append(node.l_child)
                if node.r_child != None:
                    neighbors.append(node.r_child)

                # increment counter
                counter += 1

        dot += '}'
        return dot

