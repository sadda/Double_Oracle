#!/usr/bin/python
# -*- coding: utf-8 -*-

#developed for Python 3.8

from mip import *
import numpy as np

def l(z, c):
    #print("z")
    #print(z)
    #print( np.maximum(z, np.zeros_like(z)) )
    #print( np.maximum(1/c*(z+c), np.zeros_like(z) ) - np.maximum(1/c*(z-c), np.zeros_like(z) ) -1 )
    return np.maximum(1/c*(z+c), np.zeros_like(z) ) - np.maximum(1/c*(z-c), np.zeros_like(z) ) -1

def blotto_util(x, y, a, c):
    #print("blotto util")
    #print("x", x)
    #print("y", y)
    z = x-y
    vals = l(z, c)
    utils = vals @ a
    return utils

def blotto_x_util(x, ys, distribution, a, c):
    zs = x - ys
    vals = l(zs, c)
    utils = vals @ a
    return utils @ distribution

def blotto_y_util(y, xs, distribution, a, c):
    #print("y", y)
    #print("xs", xs)
    #print("dist", distribution)
    #print("a", a)
    #print("c", c)
    zs = xs - y
    #print("zs", zs)
    vals = l(zs, c)
    #print("vals", vals)
    utils = vals @ a
    #print("utils", utils)
    return utils @ distribution

def blotto_x_response(q, y, a, c):
    assert(c <= 0.5)    

    n = len(a)
    k = len(q)    
    
    M1_l = 1/c - 1
    M1_u = 1/c + 1
    M2_l = 1/c + 1
    M2_u = 1/c - 1    
    
    model = Model()
    model.verbose = 0
    
    Vn = range(n)
    Vk = range(k)

    x  = [ model.add_var(var_type=CONTINUOUS, name="x", lb=0.) for i in Vn ]
    s1 = [[ model.add_var(var_type=CONTINUOUS, name="s1", lb=0.) for i in Vn ] for j in Vk] #s1[j,i]
    s2 = [[ model.add_var(var_type=CONTINUOUS, name="s2", lb=0.) for i in Vn ] for j in Vk]
    z1 = [[ model.add_var(var_type=BINARY, name="z1") for i in Vn ] for j in Vk]
    z2 = [[ model.add_var(var_type=BINARY, name="z2") for i in Vn ] for j in Vk] 

    model += xsum( x[i] for i in Vn ) == 1
    
    for i in Vn:
        for j in Vk:
            model += s1[j][i] >= 1/c*(x[i] - y[j,i] + c)    
            model += s1[j][i] <= 1/c*(x[i] - y[j,i] + c) + M1_l*(1-z1[j][i])
            model += s1[j][i] <= M1_u*z1[j][i]

    for i in Vn:
        for j in Vk:
            model += s2[j][i] >= 1/c*(x[i] - y[j,i] - c)    
            model += s2[j][i] <= 1/c*(x[i] - y[j,i] - c) + M2_l*(1-z2[j][i])
            model += s2[j][i] <= M2_u*z2[j][i]
        
    model.objective = maximize(xsum( q[j]*xsum( a[i]*(s1[j][i]-s2[j][i]-1) for i in Vn ) for j in Vk))
    
    status = model.optimize()
    
    x_br = [ v.x for v in model.vars ][:n]
    # print(x_br)    
    return np.array(x_br), model.objective_value

def blotto_y_response(p, x, a, c):
        
    assert(c <= 0.5)    

    n = len(a)
    k = len(p)    

    M1_l = 1/c - 1
    M1_u = 1/c + 1
    M2_l = 1/c + 1
    M2_u = 1/c - 1    
    
    model = Model()
    model.verbose = 0

    Vn = range(n)
    Vk = range(k)

    y  = [ model.add_var(var_type=CONTINUOUS, name="y", lb=0.) for i in Vn ]
    s1 = [[ model.add_var(var_type=CONTINUOUS, name="s1", lb=0.) for i in Vn ] for j in Vk] #s1[j,i]
    s2 = [[ model.add_var(var_type=CONTINUOUS, name="s2", lb=0.) for i in Vn ] for j in Vk]
    z1 = [[ model.add_var(var_type=BINARY, name="z1") for i in Vn ] for j in Vk]
    z2 = [[ model.add_var(var_type=BINARY, name="z2") for i in Vn ] for j in Vk] 

    model += xsum( y[i] for i in Vn ) == 1
    
    for i in Vn:
        for j in Vk:
            model += s1[j][i] >= 1/c*(x[j][i] - y[i] + c)    
            model += s1[j][i] <= 1/c*(x[j][i] - y[i] + c) + M1_l*(1-z1[j][i])
            model += s1[j][i] <= M1_u*z1[j][i]

    for i in Vn:
        for j in Vk:
            model += s2[j][i] >= 1/c*(x[j][i] - y[i] - c)    
            model += s2[j][i] <= 1/c*(x[j][i] - y[i] - c) + M2_l*(1-z2[j][i])
            model += s2[j][i] <= M2_u*z2[j][i]
        
    model.objective = minimize(xsum( p[j]*xsum( a[i]*(s1[j][i]-s2[j][i]-1) for i in Vn ) for j in Vk))
    
    status = model.optimize()
    
    x_br = [ v.x for v in model.vars ][:n]
    # print(x_br)    
    return np.array(x_br), model.objective_value