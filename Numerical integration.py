import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

def f(x):
    return np.cos(x)


def squares_method(n, a):
    b = np.pi/2
    dx = (b-a)/(n)
    result = 0
    for i in range(1, n-1):
        result += f(a+i*dx) * dx
    print("result for squares method : ", result, np.sin(b)-np.sin(a))


def trapezoidal_rule(n, a):
    b = np.pi/2
    dx = (b-a)/(n)
    result = 0
    sum = 0
    for i in range(1, n-1):
        sum += f(a+i*dx)
    result = f(a)+(2*sum)+f(b)
    result = result*(b-a)/(2*n)
    print("result for trapezoidal rule:  ", result, np.sin(b)-np.sin(a))


def simpons_rule(n, a, b):
      dx = (b-a)/(n)
      result, sum1, sum = 0, 0, 0
      for i in range (1, n):
            x = a+ i*dx
            sum1 = sum1+ f(x-(dx/2))
            if i < n:
                  result = result + f(x)
      result = f(a)+f(b)+2*result+4*sum1
      result = result*(dx/6)
      print("result for simpsons_rule: ", result) 


squares_method(10, 0)

trapezoidal_rule(10000, 0)

simpons_method(10,0,1)





