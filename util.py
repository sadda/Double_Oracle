#!/usr/bin/python
# -*- coding: utf-8 -*-

#developed for Python 3.8

import numpy as np
import scipy.optimize as sp
import matplotlib.pyplot as plt

steps = 1e4
#METHOD = 'Nelder-Mead' #! unbounded optimization
METHOD = 'TNC'
#METHOD = 'L-BFGS-B'
#METHOD = 'trust-constr'
#METHOD = 'SLSQP'

def optimal_mixed_strategy(matrix, player='a', lp_solver="simplex"):
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
        print("DID NOT FIND EQUILIBRIUM!")
        raise "DID NOT FIND EQUILIBRIUM!"
        #exit()
   
    x = ret['x'][1:]
    return x


def mixed_utility_function_y(y, xs, distribution, u):
    return distribution @ u(xs, np.array([y]))

def mixed_utility_function_x(x, ys, distribution, u):
    return -distribution @ u(np.array([x]), ys)
    
def optimal_response(player, xs, distribution, optimizationStrategy, u, X, prevx):
    bnds = X.getCube()
    if optimizationStrategy == "fminbound":
        if player == 'a':
            return sp.fminbound(mixed_utility_function_x, X.getCube()[0][0], X.getCube()[0][1], (xs, distribution, u), xtol=1e-15 )
        else:
            return sp.fminbound(mixed_utility_function_y, X.getCube()[0][0], X.getCube()[0][1], (xs, distribution, u), xtol=1e-15 )
    elif optimizationStrategy == "previous":
        if player == 'a':
            ret = sp.minimize(mixed_utility_function_x, prevx , method=METHOD, bounds=X.getCube(), args=(xs, distribution, u) )
            return ret['x']
        else:
            ret = sp.minimize(mixed_utility_function_y, prevx , method=METHOD, bounds=X.getCube(), args=(xs, distribution, u) )
            return ret['x']
    elif optimizationStrategy == "random":
        if player == 'a':
            ret = sp.minimize(mixed_utility_function_x, X.getRandomPoint() , method=METHOD, bounds=X.getCube(), args=(xs, distribution, u) )
            return ret['x']
        else:
            ret = sp.minimize(mixed_utility_function_y, X.getRandomPoint() , method=METHOD, bounds=X.getCube(), args=(xs, distribution, u) )
            return ret['x']
    elif optimizationStrategy == "discrete-best":
            val = discrete_optimum(player, u, xs, distribution, X.getCube()[0], steps, "min", "min")
            if player == 'a':
                ret = sp.minimize(mixed_utility_function_x, val , method=METHOD, bounds=X.getCube(), args=(xs, distribution, u), tol=1e-15 )
            else:
                ret = sp.minimize(mixed_utility_function_y, val , method=METHOD, bounds=X.getCube(), args=(xs, distribution, u), tol=1e-15 )
            return ret['x']
    elif optimizationStrategy == "discrete-worst":
            val = discrete_optimum(player, u, xs, distribution, X.getCube()[0], steps, "min", "max")
            if player == 'a':
                ret = sp.minimize(mixed_utility_function_x, val , method=METHOD, bounds=X.getCube(), args=(xs, distribution, u), tol=1e-15 )
            else:
                ret = sp.minimize(mixed_utility_function_y, val , method=METHOD, bounds=X.getCube(), args=(xs, distribution, u), tol=1e-15 )            
            return ret['x']
    elif optimizationStrategy == "discrete-random":
            val = discrete_optimum(player, u, xs, distribution, X.getCube()[0], steps, "min", "random")
            if player == 'a':
                ret = sp.minimize(mixed_utility_function_x, val , method=METHOD, bounds=X.getCube(), args=(xs, distribution, u), tol=1e-15 )
            else:
                ret = sp.minimize(mixed_utility_function_y, val , method=METHOD, bounds=X.getCube(), args=(xs, distribution, u), tol=1e-15 )            
            return ret['x']
    else:
        print("unsupported optimizationStrategy")
        exit()

def compact_strategies(qs, distribution, decimals):
    if decimals > 0:
        qs = np.round(qs, decimals)
    qs_reduced, qs_sum_indices = np.unique(qs, return_inverse=True, axis=0)
    red_distribution = []
    for i in range(len(qs_reduced)):
        temp = np.sum( distribution * (qs_sum_indices==i) )
        red_distribution.append(temp)
    return qs_reduced, np.array(red_distribution)

def discrete_optimum(player, u, xs, distribution, bounds, steps, optType, choice):
    if player == 'a':
        discrete_map = discretize_x_utility_newaxis(u, xs, distribution, bounds, steps)
    else:
        discrete_map = discretize_y_utility_newaxis(u, xs, distribution, bounds, steps)
    optima, idx = find_optima(discrete_map, optType)
    idx = pick_optimum_index(optima, idx, choice)
    return index_to_x(idx, bounds, steps)

def find_optima( m , oType = "max"):
    if oType == "max":
        bools = np.logical_and( np.greater_equal( m[1:-1], m[0:-2] ),  np.greater_equal( m[1:-1], m[2:] ) ) 
        bools = np.append(bools, m[-1] >= m[-2] )
        bools = np.insert(bools, 0, m[0] >= m[1])
    else:
        bools = np.logical_and( np.less_equal( m[1:-1], m[0:-2] ),  np.less_equal( m[1:-1], m[2:] ) ) 
        bools = np.append(bools, m[-1] <= m[-2] )
        bools = np.insert(bools, 0, m[0] <= m[1]) 

    optima = m[bools]
    idx = np.where( bools )[0]
    return optima, idx
    
def pick_optimum_index(optima, idx, oType = "max"):
    if oType == "max":
        opt_idx = int( np.argmax(optima, axis=0) )
        return idx[opt_idx]
    elif oType == "rand":
        return np.random.choice(idx)
    else:
        opt_idx = int( np.argmin(optima, axis=0) )
        return idx[opt_idx]

def index_to_x(index, bounds, steps):
    rng = abs(bounds[0] - bounds[1] )
    return index/steps * rng + bounds[0]

def discretize_x_utility(u, ys, distribution, bnds, steps):
    cart = np.array(np.meshgrid(ys, np.linspace(bnds[0], bnds[1], steps) )).T.reshape(-1,2)
    return -(np.array(distribution) @ u( cart[:,1], cart[:,0] ).T.reshape( len(ys) ,-1) )

def discretize_y_utility(u, xs, distribution, bnds, steps):
    cart = np.array(np.meshgrid(xs, np.linspace(bnds[0], bnds[1], steps) )).T.reshape(-1,2)
    return (np.array(distribution) @ u( cart[:,0], cart[:,1] ).T.reshape( len(xs) ,-1) )

def discretize_x_utility_newaxis(u, ys, distribution, bnds, steps):
    xs = np.linspace(bnds[0], bnds[1], steps)[:, np.newaxis]
    return -u(xs, ys.T) @ distribution.T

def discretize_y_utility_newaxis(u, xs, distribution, bnds, steps):
    ys = np.linspace(bnds[0], bnds[1], steps)[:, np.newaxis]
    return u(xs.T, ys) @ distribution