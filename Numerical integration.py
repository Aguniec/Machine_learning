import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

#metoda prostokątów

def f(x):
    return np.cos(x);

n=10#liczba punktów podziału
a = 0
b = np.pi/2
dx=(b-a)/(n)
wynik=0
for i in range (1,n-1):
    wynik+=f(a+i*dx)*dx

print("wynik met. prost. wynosi : ",wynik,np.sin(b)-np.sin(a))


#metoda trapezów

def f(x):
    return np.cos(x);
n=10000
a=0
b=np.pi/2
dx=(b-a)/(n)
wynik=0
suma=0
for i in range (1,n-1):
    suma+=f(a+i*dx)
wynik=f(a)+(2*suma)+f(b)
wynik=wynik*(b-a)/(2*n)
print("wynik met. trap. wynosi : ",wynik,np.sin(b)-np.sin(a) )








#metoda Simpsona
     
def f(x):
  return np.cos(x);
n=10
a=0
b=1
dx=(b-a)/(n)
wynik=0
suma1=0
suma=0
wartosc=0

for i in range (1,n):
    x=a+i*dx
    suma1=suma1+f(x-(dx/2))
    if i<n:
        wartosc=wartosc+f(x)
wartosc=f(a)+f(b)+2*wartosc+4*suma1
wartosc=wartosc*(dx/6)

print ("wynik met. simps. wynosi : ",wartosc) 

