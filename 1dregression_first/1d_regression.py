#!/usr/bin/python
#Filname:1d_regression.py
#Author: XinFeng Li
#Email: silferlee@zju.edu.cn
#Date: 2013/03/06


import numpy as np
import scipy as sp
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
import pickle as p

#function y=sin(x)
x=np.arange(0,2*np.pi,0.02)
y=np.sin(x)


#select the sample points
N = 10				# the number of sample points	
sample_x = np.linspace(0,2*np.pi,N)	#sample points array
sample_y = np.sin(sample_x)

#use (y,sigma) as parameter nomoral distribution as disturb function
sigma = 0.1
disturb_var = np.random.randn(sample_y.size)  #disturb var
sample_y += np.sqrt(sigma)*disturb_var	  #sample_y after disturb

#define poly function
def poly_func(w,x):
	f = np.poly1d(w)
	return f(x)

#define residual function
def residuals(w,y,x):
	return (y - poly_func(w,x))


#define regularization residual function
lamb = np.exp(-18)			#the regular parameter lamb
def residuals_regular(w,y,x):
	ret = y - poly_func(w,x)
	ret = np.append(ret,np.sqrt(lamb)*w)
	return ret

m =input('Enter the order of the poly fuction:M=') # the order of the poly
init_w = np.random.randn(m+1)	#the init w parameter

#use the leastsq compute the parameter w
plsq = leastsq(residuals, init_w, args = (sample_y,sample_x))

print 'Fitting parameters are :', plsq[0]

#save the data to file
file_name = 'regression'
file_name += '_'+'N'+str(N)+'M'+str(m)+'.txt'
f = file(file_name,'w')
print>>f,plsq[0]
f.close()

plt.plot(x,y,label='$y=\sin(x)$')
plt.plot(sample_x,sample_y,'ro',label='with noise')
plt.plot(x,poly_func(plsq[0],x),label = 'fitting curve')
plt.legend()
plt.text(3.5,1.2,["m=",m,"N=",N])
plt.show()
