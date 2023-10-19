# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 18:43:06 2023

@author: Tomke
"""
#################################################
######  Introduction to Python  #################
#################################################


# about python: 
    

# general purpose progamming language conceived 1989 by 
# dutch programmer Guido van Rossum 

# free and open source 

# rapid growth of usage the last couple of years 
# mostly because scientific use and data science 

# high level language 

# multiple prograamming styles (functional, object-oriented,...)

# interpreted rather than compiled 

# numerical programming --> numpy 

import numpy as np 

a = np.linspace(-np.pi,np.pi, 100)
b = np.cos(a) 
c = np.sin(a)


b @ c           # inner product 
np.dot(b,c)     # inner product (older version)


# scipy library provides additional functionality 

from scipy.stats import norm
from scipy.integrate import quad 

phi = norm()
value, error = quad(phi.pdf,-2,2) # integrate using Gaussian quadrature 
value


# scipy includes: 
# linear algebra 
# integration 
# interpolation 
# optimization
# distributions and statistical techniques 




# plots --> matplotlib 


# symbolic algebra: 
    
from sympy import Symbol 

x, y = Symbol("x"), Symbol("y")
    
x + x + x + y 

expression = ( x + y ) ** 2 
expression.expand() 


from sympy import solve 
solve(x**2 + x + 2 ) 

from sympy import limit, sin, diff, integrate 

limit(1 /  x , x , 0)
limit(sin(x)/x, x, 0)
diff(sin(x),x)


integrate(sin(x)*x, x ) 




# statistics: 
    
    
# most popular library for working with data is pandas 

import pandas as pd

np.random.seed(1234)

data = np.random.randn(5,2) 
dates = pd.date_range('2010-12-28', periods=5)

df = pd.DataFrame(data,columns = ("price", "weight"), index = dates)
print(df)



# mean of both columns: 
df.mean()

# machine learning --> scikit learn 
# deep learning    --> Pytorch tensorflow kensas 


# network graphs --> NetworkX. 

# cloud computing 

# Parallel Computing: IPython clusters 
# Dask parallelises PyData and Machine Learning in Python 



































