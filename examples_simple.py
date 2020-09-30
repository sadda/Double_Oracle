#!/usr/bin/python
# -*- coding: utf-8 -*-

#developed for Python 3.8

import numpy as np


def util1(x,y):
    return 5*x*y - 2*np.power(x, 2) - 2*x*np.power(y,2) - y    


def util2(x, y):
    term1 = np.power( np.cos( (x-0.1)*y ), 2)
    term2 = x*np.sin( 3*x+y )
    return -term1 -term2

