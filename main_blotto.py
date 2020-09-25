#!/usr/bin/python
# -*- coding: utf-8 -*-

#developed for Python 3.8

import numpy as np
import scipy.optimize as sp
import pickle
import time
import os


from util import *
from games import *
from blotto import *


def init_algorithm(game, init_type="bounds"):
    if init_type == "random":
        xs = np.array([game.X.getRandomPoint()])
        ys = np.array([game.Y.getRandomPoint()])
        p  = np.ones(len(xs))/len(xs)
        q  = np.ones(len(ys))/len(ys)    
    elif init_type == "bounds":
        xs = np.eye(game.n)
        ys = np.eye(game.n)
        p  = np.ones(len(xs))/len(xs)
        q  = np.ones(len(ys))/len(ys)    
    elif init_type == "uniform":
        assert(game.n==3)
        
        r1 = np.arange(0., 1., game.c)
        r2 = np.arange(1., 0., -game.c)
        r  = np.unique(np.round(np.concatenate((r1, r2)), 10))
        
        xs = np.zeros((0,game.n))
        ys = np.zeros((0,game.n))
        for i in range(len(r)):
            for j in range(len(r)):
                if r[i] + r[j] <= 1:
                    x  = np.array([r[i], r[j], 1-r[i]-r[j]])
                    xs = np.insert(xs, 0, values=x, axis=0)
                    ys = np.insert(ys, 0, values=x, axis=0)

        matrix = compute_matrix(xs, ys, game)
        p = optimal_mixed_strategy(matrix, player='a', lp_solver="interior-point")
        q = optimal_mixed_strategy(matrix, player='b', lp_solver="interior-point")
        
        xs, p, ys, q = reduce_strategies(xs, p, ys, q, reduce_type="small_prob")
    
    return xs, p, ys, q


def reduce_strategies(xs, p, ys, q, reduce_type="none", epsilon=1e-8):
    if reduce_type == "small_prob":
        ii = p >= epsilon
        xs = xs[ii]
        p  = p[ii]
        jj = q >= epsilon
        ys = ys[jj]
        q  = q[jj]
    elif reduce_type == "first_equal":
        for i in range(len(xs)-1):
            if np.linalg.norm(xs[0,:] - xs[i+1,:]) <= epsilon:
                xs = xs[1:]
                p  = p[1:]
                break
        for i in range(len(ys)-1):
            if np.linalg.norm(ys[0,:] - ys[i+1,:]) <= epsilon:
                ys = ys[1:]
                q  = q[1:]
                break
    return xs, p, ys, q
        

def compute_matrix(xs, ys, game):
    matrix = np.zeros( (len(xs), len(ys)) )
    for i in range( len(xs) ):
        for j in range( len(ys) ):
            matrix[i,j] = blotto_util(xs[i], ys[j], game.a, game.c)
    return matrix
            
            
def solve(game, maxitr, printout=False, init_type="bounds"):
    #Modify numpy array printing
    np.set_printoptions(edgeitems=30, linewidth=1000, formatter=dict(float=lambda x: "%.9g" % x))

    xs, p, ys, q = init_algorithm(game, init_type=init_type)
    
    #containers for values of the game
    upper_bounds = []
    lower_bounds = []
    
    for itr in range(maxitr):
        #find best pure response
        x, x_opt_val = blotto_x_response(q, ys, game.a, game.c)
        y, y_opt_val = blotto_y_response(p, xs, game.a, game.c)
        
        upper_bounds.append( x_opt_val )
        lower_bounds.append( y_opt_val )   
                
        xs = np.insert(xs, 0, values=x, axis=0)
        ys = np.insert(ys, 0, values=y, axis=0)
        
        xs, p, ys, q = reduce_strategies(xs, p, ys, q, reduce_type="first_equal")
        
        matrix = compute_matrix(xs, ys, game)
        
        #find best mixed strategy
        p = optimal_mixed_strategy(matrix, player='a', lp_solver="interior-point")
        q = optimal_mixed_strategy(matrix, player='b', lp_solver="interior-point")

        print("Iter =", itr, "Upper - lower estimate =", x_opt_val - y_opt_val)                
        if(printout):
            print("x response = ", x)
            print("y response = ", y)
            print("xs = ", xs)
            print("ys = ", ys)
            print("------------------------------------------------------------------------")
            print("Mixed x strategy: ", p)
            print("Mixed y strategy: ", q)
        
        #check convergence
        if upper_bounds[-1] - lower_bounds[-1] < game.epsilon:
            break
        
    if(printout):
        print("Mixed x strategy: ", p)
        print("xs", xs)
        print("Mixed y strategy: ", q)
        print("ys", ys)
        print("lower bounds", lower_bounds)
        print("upper bounds", upper_bounds)

    return np.flip(xs.T).T, np.flip(p), np.flip(ys.T).T, np.flip(q), lower_bounds, upper_bounds



n_all = [3]
c_all = [1/4, 1/8]
iter_type_all = ["bounds", "uniform"]
a_type_all = ["equal"]


#n_all = [10]
#c_all = [1/16]
#iter_type_all = ["bounds"]
#a_type_all = ["unequal"]


comb_all = np.array(np.meshgrid(n_all, c_all, iter_type_all, a_type_all)).T.reshape(-1,4)




maxitr = 100

dir_name = "Results"
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

for i in range(len(comb_all)):

    n         = int(comb_all[i,0])
    c         = float(comb_all[i,1])
    init_type = comb_all[i,2]
    a_type    = comb_all[i,3]
    if a_type == "equal":
        a = np.ones(n)
    elif a_type == "unequal":
        a = np.array(range(3, 3+n))
        
    game = Blotto( BlottoBlock( np.vstack((np.zeros(n), np.ones(n))).T ), BlottoBlock( np.vstack((np.zeros(n), np.ones(n))).T ), a, c, 1e-6, "blotto")
    
    time1 = time.time()
    xs_full, p_full, ys_full, q_full, lower_bounds, upper_bounds = solve(game, maxitr, init_type=init_type)
    time2 = time.time()
    elapsed_time = time2 - time1
    xs, p, ys, q = reduce_strategies(xs_full, p_full, ys_full, q_full, reduce_type="small_prob")
    
    file_name = os.path.join(dir_name, "Res" + "_" + str(n) + "_" + str(c) + "_" + init_type + "_" + a_type)
        
    with open(file_name + ".pkl", 'wb') as f:
        pickle.dump([game, xs, p, ys, q, xs_full, p_full, ys_full, q_full, lower_bounds, upper_bounds, elapsed_time, init_type], f)
    





