# PCA algorithm
# Author: Xinfeng Lee
# Date: 2013/03/18

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd

# load the original data
filename='./threes.txt'
array2d = np.genfromtxt(filename,dtype=None,usecols=range(0,64),delimiter=",")

# compute the average
tmpsum = np.zeros((1,64))
for i in range(0,389):
    tmpsum += array2d[i]
average = tmpsum/389;

# subtract the mean value from the original matrix
array2d0 = array2d
for i in range(0,389):
    array2d0[i,:] = array2d0[i,:] - average

# SVD
u0,s0,vt0 = svd(array2d0)

# extract the first two max eigenvalue,set others eigen values to 0
sigma = np.zeros_like(array2d0,dtype=np.float)
N = np.min(array2d0.shape)
sigma[:N,:N] = np.diag(s0)
for i in range(2,N):
    sigma[i][i] = 0

# compute the first two main componients
us = np.dot(u0,sigma)

# plot
x = us[:,0]
y = us[:,1]
xx = 0
yy = 0
plt.plot(x,y,'r*')
plt.xlabel("First Principle component")
plt.ylabel("Second Principle component")
plt.title("PCA for rank-2 model")
plt.grid(which='both', axis='both')
plt.show()
