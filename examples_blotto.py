#!/usr/bin/python
# -*- coding: utf-8 -*-

#developed for Python 3.8

import numpy as np

from mip import *
from double_oracle import *

# Defines the continuous Colonel Blotto game
# More information in the accompanied paper

class Blotto(Game):
    def __init__(self, X, Y, a, c, init_type="bounds"):
        assert(c <= 0.5)  
        
        self.a = a
        self.c = c
        self.n = len(a)
        
        if init_type == "uniform":
            self.init_algorithm = self.init_algorithm_uniform
            super().__init__(X, Y, None, init_type=None)
        else:
            super().__init__(X, Y, None, init_type=init_type)

        
    def u(self, x, y):
        # The utility function
        z    = x-y
        vals = self.l(z)
        return vals @ self.a

    def l(self, z):
        # The loss function on one battlefield
        c = self.c
        return np.maximum(1/c*(z+c), np.zeros_like(z) ) - np.maximum(1/c*(z-c), np.zeros_like(z) ) - 1

    def init_algorithm_uniform(self):
        # Possible initialization on a grid
        assert(self.n==3)
        
        r1 = np.arange(0., 1., self.c)
        r2 = np.arange(1., 0., -self.c)
        r  = np.unique(np.round(np.concatenate((r1, r2)), 10))
        
        xs = np.zeros((0,self.n))
        ys = np.zeros((0,self.n))
        for i in range(len(r)):
            for j in range(len(r)):
                if r[i] + r[j] <= 1:
                    x  = np.array([r[i], r[j], 1-r[i]-r[j]])
                    xs = np.insert(xs, 0, values=x, axis=0)
                    ys = np.insert(ys, 0, values=x, axis=0)
    
        matrix = self.compute_matrix(xs, ys)
        p = self.optimal_mixed_strategy(matrix, player='a', lp_solver="interior-point")
        q = self.optimal_mixed_strategy(matrix, player='b', lp_solver="interior-point")
        
        xs, p, ys, q = reduce_strategies(xs, p, ys, q)
        
        return xs, p, ys, q
        
    def get_x_response(self, ys, q):
        # Defines the computation of the best response of player 1
        c = self.c
        n = self.n
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
                model += s1[j][i] >= 1/c*(x[i] - ys[j,i] + c)    
                model += s1[j][i] <= 1/c*(x[i] - ys[j,i] + c) + M1_l*(1-z1[j][i])
                model += s1[j][i] <= M1_u*z1[j][i]
    
        for i in Vn:
            for j in Vk:
                model += s2[j][i] >= 1/c*(x[i] - ys[j,i] - c)    
                model += s2[j][i] <= 1/c*(x[i] - ys[j,i] - c) + M2_l*(1-z2[j][i])
                model += s2[j][i] <= M2_u*z2[j][i]
            
        model.objective = maximize(xsum( q[j]*xsum( self.a[i]*(s1[j][i]-s2[j][i]-1) for i in Vn ) for j in Vk))
        
        status = model.optimize()
        
        x_br = [ v.x for v in model.vars ][:n]
        return np.array(x_br), model.objective_value
            
    
    def get_y_response(self, xs, p):
        # Defines the computation of the best response of player 2
        c = self.c
        n = self.n
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
                model += s1[j][i] >= 1/c*(xs[j][i] - y[i] + c)    
                model += s1[j][i] <= 1/c*(xs[j][i] - y[i] + c) + M1_l*(1-z1[j][i])
                model += s1[j][i] <= M1_u*z1[j][i]
    
        for i in Vn:
            for j in Vk:
                model += s2[j][i] >= 1/c*(xs[j][i] - y[i] - c)    
                model += s2[j][i] <= 1/c*(xs[j][i] - y[i] - c) + M2_l*(1-z2[j][i])
                model += s2[j][i] <= M2_u*z2[j][i]
            
        model.objective = minimize(xsum( p[j]*xsum( self.a[i]*(s1[j][i]-s2[j][i]-1) for i in Vn ) for j in Vk))
        
        status = model.optimize()
        
        x_br = [ v.x for v in model.vars ][:n]  
        return np.array(x_br), model.objective_value

