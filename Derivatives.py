#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 16:11:55 2018

@author: AgnieszkaCezak
"""

import numpy as np
import matplotlib.pyplot as pl
from scipy.integrate import odeint
import matplotlib.pyplot as plt

x=3

alfa = x
beta = (x*x)/(1+(x*x))

def sprime(y, t, alfa, beta, n):
      s, i = y
      dydt = [-beta*((s*i)/n)+alfa*i, beta*((s*i)/n)-alfa*i]
      return dydt

n = 1

y0 = [0,0.1]
t = np.linspace(0, 0.5,100)  

sol = odeint(sprime, y0, t, args=(alfa, beta,n))
 
plt.plot(t, sol[:, 0], 'b', label='s(t)')
plt.plot(t, sol[:, 1], 'g', label='i(t)')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()
    