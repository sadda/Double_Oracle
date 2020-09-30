#!/usr/bin/python
# -*- coding: utf-8 -*-

#developed for Python 3.8

import numpy as np
import scipy.optimize as sp


#####################################
#  BOUNDING BOXES
#####################################

class Space:
    def getCube(self):
        raise "not implemented yet"
    def getRandomPoint(self):
        raise "not implemented yet"


class HyperBlock(Space):
    def __init__(self, bounds):
        self.n = len(bounds)
        self.bounds = bounds

    def getCube(self):
        return self.bounds.T

    def getRandomPoint(self):
        return np.array( self.bounds @ np.array([-1,1]) * np.random.rand(self.n) + self.bounds[:,0] )


class ProbBlock(HyperBlock):
    def __init__(self, n):
        self.n = n
        
    def getCube(self):
        return np.eye(self.n)

    def getRandomPoint(self):
        arr =  np.array( self.bounds @ np.array([-1,1]) * np.random.rand(self.n) + self.bounds[:,0] )
        return arr / sum(arr)

    
#####################################
#  DEFINITION OF GAMES
#####################################

"""
Each game needs to have the following attributes:

COMPULSORY ATTRIBUTES    
- u: utility function
- X: bounding box for player 1
- Y: bounding box for player 2

USER-DEFINED FUNCTION FOR MORE COMPLICATED GAMES
- init_algorithm: returns initialization (one or multiple points) for the algorithm
- get_x_response: computes the best response of player 1
- get_y_response: computes the best response of player 2
"""

class Game:
    def __init__(self, X, Y, u, init_type="random"):
        if not u is None:
            self.u = u
        self.X = X
        self.Y = Y
        if init_type == "random":
            self.get_init_xy = self.get_init_xy_random
        elif init_type == "bounds":
            self.get_init_xy = self.get_init_xy_bounds
        elif init_type is None:
            pass
        else:
            raise "init_type not defined"
    
    def init_algorithm(self):
        # Initializes the initial values either via get_init_xy()
        # Functions get_init_xy_random() and get_init_xy_bounds() are predefined
        xs, ys = self.get_init_xy()
        p = np.ones(len(xs))/len(xs)
        q = np.ones(len(ys))/len(ys)   
        return xs, p, ys, q

    def get_init_xy_random(self):
        # Initializes via a random point
        xs = np.array([self.X.getRandomPoint()])
        ys = np.array([self.Y.getRandomPoint()])
        return xs, ys
    
    def get_init_xy_bounds(self):
        # Initializes via a box
        xs = np.array(self.X.getCube())
        ys = np.array(self.Y.getCube())
        return xs, ys    
        
    def get_xy_response(self, xs, p, ys, q):
        # Get best x and y responses
        x, x_val = self.get_x_response(ys, q)
        y, y_val = self.get_y_response(xs, p)
        return x, y, x_val, y_val
    
    def get_x_response(self, ys, q):
        # Get best x response.
        # For one dimension, it solves via a discretization (global guarantee)
        # For higher dimension, it solves via fminbound (no global guarantee)
        if not isinstance(self.X, HyperBlock):
            raise "Only works for hyperblocks. Please write your own best response function."
        if self.X.n == 1:
            n_test = 10000        
            x_test = np.linspace(self.X.bounds[:,0], self.X.bounds[:,1], n_test)
            f_test = [mixed_utility_function_x(x, ys, q, self.u) for x in x_test]
            ii     = np.argmin(f_test)
            x      = x_test[ii]
        else:
            x = sp.fminbound(mixed_utility_function_x, self.X.bounds[:,0], self.X.bounds[:,1], (ys, q, self.u))
        x_val = -mixed_utility_function_x(x, ys, q, self.u)
        return x, x_val
        
    def get_y_response(self, xs, p):
        # Get best y response.
        # For one dimension, it solves via a discretization (global guarantee)
        # For higher dimension, it solves via fminbound (no global guarantee)
        if not isinstance(self.Y, HyperBlock):
            raise "Only works for hyperblocks. Please write your own best response function."
        if self.Y.n == 1:
            n_test = 10000
            y_test = np.linspace(self.Y.bounds[:,0], self.Y.bounds[:,1], n_test)
            f_test = [mixed_utility_function_y(y, xs, p, self.u) for y in y_test]
            ii     = np.argmin(f_test)
            y      = y_test[ii]
        else:
            y = sp.fminbound(mixed_utility_function_y, self.Y.bounds[:,0], self.Y.bounds[:,1], (xs, p, self.u))
        y_val = mixed_utility_function_y(y, xs, p, self.u)
        return y, y_val
    
    def compute_matrix(self, xs, ys):
        # Computes the utility matrix of the reduced game
        matrix = np.zeros( (len(xs), len(ys)) )
        for i in range( len(xs) ):
            for j in range( len(ys) ):
                matrix[i,j] = self.u(xs[i], ys[j])
        return matrix

    def optimal_mixed_strategy(self, matrix, player='a', lp_solver="simplex"):
        # Computes the best strategy on the reduced game
        if player == 'a':
            matrix = matrix.transpose()
        height, width = matrix.shape
        # [1 0 0 0 ... 0]
        function_vector = np.insert( np.zeros(width), 0, 1)
        # [-1 | A]
        boundary_matrix = np.insert(matrix, 0, values=-1, axis=1)
        # [0 1 1 ... 1]
        eq_matrix = np.array([np.insert(np.ones(width), 0, values=0, axis=0)])
        # [ [-inf,inf], [0,inf]...[0,inf]]
        bnds = np.ones([width+1,2]) * np.array([0, np.inf]) 
        bnds[0] = np.array([-np.inf, np.inf])
        # {options} added on behalf what the functions itself demanded in stdout
        if player == 'a': #maximizing player
            ret = sp.linprog( -function_vector, -boundary_matrix, np.zeros(height), eq_matrix, np.array([1]), bnds, method=lp_solver, options={'autoscale': True, 'sym_pos':False, 'maxiter':1e4})
        else:             #minimizing player
            ret = sp.linprog( function_vector, boundary_matrix, np.zeros(height), eq_matrix, np.array([1]), bnds, method=lp_solver, options={'autoscale': True, 'sym_pos':False, 'maxiter':1e4})
        if ret['success'] is not True:
            raise "DID NOT FIND EQUILIBRIUM!"
       
        x = ret['x'][1:]
        return x


#####################################
#  DOUBLE ORACLE ALGORITHM
#####################################


def double_oracle(game, maxitr, method="DO", epsilon=1e-6):
    # Runs the double oracle algorithm
    np.set_printoptions(edgeitems=30, linewidth=1000, formatter=dict(float=lambda x: "%.9g" % x))

    # Initialize the algorithm
    xs, p, ys, q = game.init_algorithm()
    upper_bounds = []
    lower_bounds = []
    
    for itr in range(maxitr):
        # Find best pure response
        x, y, x_opt_val, y_opt_val = game.get_xy_response(xs, p, ys, q)
        upper_bounds.append( x_opt_val )
        lower_bounds.append( y_opt_val )   

        if method == "DO":
            # Add the best responses if they are already not found
            if not already_exists(xs, x):        
                xs = np.insert(xs, 0, values=x, axis=0)
            if not already_exists(ys, y):        
                ys = np.insert(ys, 0, values=y, axis=0)
            
            # Recompute the matrix of the reduced game
            matrix = game.compute_matrix(xs, ys)
            
            # Find the best strategies on the reduced game
            p = game.optimal_mixed_strategy(matrix, player='a', lp_solver="interior-point")
            q = game.optimal_mixed_strategy(matrix, player='b', lp_solver="interior-point")
        elif method == "FP":
            # Reweight the probabiltiies
            coef = 1/(itr+1)
            p = p * ( (itr)*coef)
            q = q * ( (itr)*coef)
            if already_exists(xs, x):
                p[np.where(xs==x)[0]] += coef
            else:
                xs = np.insert(xs, 0, values=x, axis=0)
                p  = np.insert(p, 0, values=coef, axis=0)
            if already_exists(ys, y):
                q[np.where(ys==y)[0]] += coef
            else:
                ys = np.insert(ys, 0, values=y, axis=0)
                q  = np.insert(q, 0, values=coef, axis=0)                        
                    
        print("Iter =", itr, "Upper - lower estimate =", x_opt_val - y_opt_val)                
        
        # Check convergence for termination
        if upper_bounds[-1] - lower_bounds[-1] < epsilon:
            break
        
    return np.flip(xs.T).T, np.flip(p), np.flip(ys.T).T, np.flip(q), lower_bounds, upper_bounds


def mixed_utility_function_y(y, xs, p, u):
    # Computed the utility function of player 2.
    # Sorry for the hack with converting array with one element to scalar.
    val =  p @ u(xs, y)
    if isinstance(val, np.ndarray):
        if len(val) > 1:
            raise "something wrong"
        else:
            val = val[0]
    return val

def mixed_utility_function_x(x, ys, q, u):
    # Computed the utility function of player 1.
    # Sorry for the hack with converting array with one element to scalar.
    val = -q @ u(x, ys)
    if isinstance(val, np.ndarray):
        if len(val) > 1:
            raise "something wrong"
        else:
            val = val[0]
    return val

def already_exists(xs, x):
    # Check if a strategy already exists
    exists = False
    for i in range(len(xs)):
        if all(xs[i] == x):
            exists = True
            break
    return exists
 
def reduce_strategies(xs, p, ys, q, epsilon=1e-8):
    # Remove strategies with small probabilities
    ii = p >= epsilon
    xs = xs[ii]
    p  = p[ii]
    p  = p / sum(p)
    jj = q >= epsilon
    ys = ys[jj]
    q  = q[jj]
    q  = q / sum(q)
    return xs, p, ys, q
        


