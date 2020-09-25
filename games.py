#!/usr/bin/python
# -*- coding: utf-8 -*-

#developed for Python 3.8

import numpy as np
import scipy.optimize as sp
import matplotlib.pyplot as plt

################### C L A S S E S ######################
class Space:
    def getCube(self):
        raise "not implemented yet"
    def getRandomPoint(self):
        raise "not implemented yet"

class HyperBlock(Space):
    def __init__(self, bounds):
        self.dimension = len(bounds)
        self.bounds = bounds

    def getCube(self):
        return self.bounds

    def getRandomPoint(self):
        return np.array( self.bounds @ np.array([-1,1]) * np.random.rand(self.dimension) + self.bounds[:,0] )

class BlottoBlock(HyperBlock):
    def getRandomPoint(self):
        arr =  np.array( self.bounds @ np.array([-1,1]) * np.random.rand(self.dimension) + self.bounds[:,0] )
        return arr / sum(arr)

class HyperDelta(Space):
    def __init__(self, dimension):
        self.dimension = dimension
    def getCube(self):
        return self.dimension

class Game:
    def __init__(self, X, Y, u, epsilon, label):
        self.X = X
        self.Y = Y
        self.u = u
        self.epsilon = epsilon
        self.label = label

class Blotto(Game):
    def __init__(self, X, Y, a, c, epsilon, label):
        self.X = X
        self.Y = Y
        self.a = a
        self.c = c
        self.n = len(a)
        self.epsilon = epsilon
        self.label = label
    

############### I N S T A N C E S #################

game1 = Game(HyperBlock( np.array([[-1,1]]) ), HyperBlock( np.array([[-1,1]]) ), lambda x,y: 2*x*np.power(y,2) - np.power(x,2) - y , 1e-6, "1")

game2 = Game(HyperBlock( np.array([[-1,1]]) ), HyperBlock( np.array([[-1,1]]) ), lambda x,y: 5*x*y - 2*np.power(x, 2) - 2*x*np.power(y,2) -y, 1e-6, "2")

game3 = Game(HyperBlock( np.array([[0,1]]) ), HyperBlock( np.array([[0,1]]) ), lambda x,y: np.minimum( np.abs(x-y), 1-np.abs(x-y) ), 1e-8, "3" )


def util4(x=0, y=0):
    term1 = y-1/2
    term2 = 1 + (x-1/2) * np.power(y-1/2,2)
    term3 = 1 + np.power(x-1/2,2) * np.power(y-1/2,4)
    term4 = 1 + np.power(x/3 - 1/2,2) * np.power(y-1/2,4)
    return term1*( (term2/term3) - (1/term4)  )
game4 = Game(HyperBlock( np.array([[0,1]]) ), HyperBlock( np.array([[0,1]]) ), lambda x,y: util4(x,y), 1e-10, "4")


n = 3
A = np.random.rand(n,n)
B = np.random.rand(n,n)
C = np.random.rand(n,n)*2-1
def util5(x ,y):
    val = 0
    for i in range(n):
        for j in range(n):
            val += C[i][j]*np.power(x-A[i][j],i+1)*np.power(y-B[i][j],j+1)
    return val

game5 = Game(HyperBlock( np.array([[0,1]]) ), HyperBlock( np.array([[0,1]]) ), util5, 1e-6 , "5")

A_static = np.array([0.95146137, 0.78631989, 0.84516066, 0.85733881, 0.03980566, 0.54586301, 0.30365653, 0.55709875, 0.67458569]).reshape(3,3)
B_static = np.array([0.72369936, 0.85245556, 0.41704234, 0.70995833, 0.22102735, 0.63374458, 0.21503189, 0.160624  , 0.33983481]).reshape(3,3)
C_static = np.array([-0.66430775, -0.89724325, -0.51724164,  0.45588112, -0.59734844, 0.31362461,  0.07065552,  0.08290412, -0.9215812 ]).reshape(3,3)
def util5_static(x, y):
    val = 1
    for i in range(n):
        for j in range(n):
            val += C_static[i][j]*(x-A_static[i][j]**i+1)*(x-B_static[i][j]**j+1)
            #val *= np.power(x-A_static[i][0],1)
    return val

game5_static = Game(HyperBlock( np.array([[0,1]]) ), HyperBlock( np.array([[0,1]]) ), util5_static, 1e-6, "5" )

def util6(x, y):
    term1 = np.power( np.cos( (x-0.1)*y ), 2)
    term2 = x*np.sin( 3*x+y )
    return -term1 -term2
game6 = Game(HyperBlock( np.array([[-2.25,2.5]]) ), HyperBlock( np.array([[-2.5,1.75]]) ), util6, 1e-6, "6" )

game7 = Game( HyperBlock( np.array([[0,1],[0,1]]) ), HyperBlock( np.array([[0,1],[0,1]]) ), lambda x,y: np.array([ (np.sum( np.power(x,2), axis=1 )-np.sum( np.power(y,2), axis=1 ))/(x[:,0]+y[:,1] + 1) ]).T, 1e6, "7")

def util8(x, y):
    term1 = np.sum(x, axis=1) + np.sum(y, axis=1)
    term2 = np.power(x[:,0],2) * np.power(y[:,1],2) - np.power(y[:,0],2) * np.power(x[:,1],2)
    term3 = np.power(x[:,1],2) * np.power(y[:,2],2) - np.power(y[:,1],2) * np.power(x[:,2],2)
    term4 = np.power(x[:,0],2) + np.power(y[:,1],2) + x[:,2]*y[:,2]+1
    return np.array([ (term1 + term2 + term3 )/term4 ]).T

game8 = Game( HyperBlock( np.array([[0,1],[0,1],[0,1]]) ), HyperBlock( np.array([[0,1],[0,1],[0,1]]) ), util8, 1e-6, "8" )

def util9(x, y):
    return np.abs(x-y)

game9 = Game( HyperBlock( np.array([[0, 1]]) ), HyperBlock( np.array([[0, 1]]) ), util9, 1e-7, "9")
game10 = Game( HyperBlock( np.array([[0, 1]]) ), HyperBlock( np.array([[0, 1]]) ), lambda x,y: -util9(x,y), 1e-7, "10")

blotto_3 = Blotto( BlottoBlock( np.array([[0,1],[0,1],[0,1]]) ), BlottoBlock( np.array([[0,1],[0,1],[0,1]]) ), np.array([1,1,1]), 0.01, 1e-10, "blotto3")
blotto_3a = Blotto( BlottoBlock( np.array([[0,1],[0,1],[0,1]]) ), BlottoBlock( np.array([[0,1],[0,1],[0,1]]) ), np.array([5,1,1]), 0.01, 1e-10, "blotto3")