#!/usr/bin/python
# -*- coding: utf-8 -*-

#developed for Python 3.8

import numpy as np
import scipy.optimize as sp
import matplotlib.pyplot as plt
from util import *


def validate_x_response(opt_x, ys, distribution, game):
    opt_util = -mixed_utility_function_x(opt_x, ys, distribution, game.u)
    all_utils = -discretize_x_utility(game.u, ys, distribution, game.X.bounds[0], 1e5)
    return np.min(opt_util - all_utils) + game.epsilon >= 0

def validate_y_response(opt_y, xs, distribution, game):
    opt_util = mixed_utility_function_y(opt_y, xs, distribution, game.u)
    all_utils = discretize_y_utility(game.u, xs, distribution, game.Y.bounds[0], 1e5)
    return np.min(all_utils - opt_util) + game.epsilon >= 0

def validate_distribution(distribution):
    return np.abs( np.sum(distribution)-1 ) < 1e-8
