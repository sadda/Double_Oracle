#!/usr/bin/python
# -*- coding: utf-8 -*-

#developed for Python 3.8

import numpy as np
import os
import pickle
import csv

from double_oracle import *
from examples_simple import *
from examples_blotto import *
from plots import *


# Create directoty for saving
dir_name1 = "Results"
dir_name2 = "Results_Fig"
dir_name3 = "Results_Csv"
if not os.path.exists(dir_name2):
    os.makedirs(dir_name2)
if not os.path.exists(dir_name3):
    os.makedirs(dir_name3)


# Function for analyzing and saving the results
def analyze_results(dir_name1, dir_name2, dir_name3, file_name, blotto=False):
    file_name1 = os.path.join(dir_name1, file_name)
    file_name2 = os.path.join(dir_name2, file_name)
    file_name3 = os.path.join(dir_name3, file_name)

    with open(file_name1 + ".pkl", 'rb') as f:
        game, xs, p, ys, q, xs_full, p_full, ys_full, q_full, lower_bounds, upper_bounds = pickle.load(f)
    
    if len(lower_bounds) > 1:
        plot_value_evolution(game, lower_bounds, upper_bounds, file_name=file_name2+"_1.png")
    
    if blotto and game.X.n==3:
        plot_blotto(game, xs, p, 1, file_name=file_name2+"_2.png")    
        plot_blotto(game, ys, q, 2, file_name=file_name2+"_3.png")    
    
    data = np.column_stack((np.array(range(len(lower_bounds))), lower_bounds, upper_bounds))
    
    with open(file_name3 + "_1.csv", 'w', newline='') as csvfile:  
        csvwriter = csv.writer(csvfile)  
        csvwriter.writerow(['iteration', 'lower_bound', 'upper_bound'])
        csvwriter.writerows(data) 
        
    data = np.column_stack((100*xs, p))
    
    with open(file_name3 + "_2.csv", 'w', newline='') as csvfile:  
        csvwriter = csv.writer(csvfile)  
        csvwriter.writerow(['x', 'y', 'z' ,'myvalue'])
        csvwriter.writerows(data) 


#####################################
#  TEST GAMES
#####################################

file_name_all = ["Test1_DO", "Test1_FP", "Test2_DO", "Test2_FP"]

for file_name in file_name_all:
    analyze_results(dir_name1, dir_name2, dir_name3, file_name)


#####################################
#  COLONEL BLOTTO
#####################################


def analyze_blotto(n_all, c_all, iter_type_all, a_type_all):
    comb_all = np.array(np.meshgrid(n_all, c_all, iter_type_all, a_type_all)).T.reshape(-1,4)
            
    for i in range(len(comb_all)):
    
        n         = int(comb_all[i,0])
        c         = float(comb_all[i,1])
        init_type = comb_all[i,2]
        a_type    = comb_all[i,3]
        
        file_name = "Blotto" + "_" + str(n) + "_" + str(c) + "_" + init_type + "_" + a_type
        
        analyze_results(dir_name1, dir_name2, dir_name3, file_name, blotto=True)
        

n_all = [3]
c_all = [1/4, 1/8, 1/16, 1/32]
iter_type_all = ["bounds", "uniform"]
a_type_all    = ["equal"]

analyze_blotto(n_all, c_all, iter_type_all, a_type_all)


n_all = [10]
c_all = [1/16]
iter_type_all = ["bounds"]
a_type_all    = ["unequal"]

analyze_blotto(n_all, c_all, iter_type_all, a_type_all)



     





