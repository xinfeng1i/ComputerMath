#!/usr/bin/python
#Filename:LM_algo.py
#Author: XinFeng Li
#Email: silferlee@zju.edu.cn
#Date: 2013/04/28

import numpy as np
import scipy as sp
import numpy.linalg as linalg
import matplotlib.pyplot as plt

def fun(a,b,x):
    """
    The objective function you want to minimize whose parameters

    @param a  the parameter to be estimate 
    @param b  the parameter to be estimate
    @param x  the original data

    @return   the function value
    """
    return a*np.exp(-b*x)

def jac_f(a,b,x):
    n = len(x)
    result = np.zeros((n,2))

    for i in range(n):
        result[i][0] = np.exp(-b * x[i])
        result[i][1] = -a * x * np.exp(-b * x[i])

    return result

def hess_f(a,b,x):
    n = len(x)
    j = jac(a,b,x)
    result = np.zeros((2,2))
    result[0][0] = np.dot(j[:,0],j[:,0])
    result[1][1] = np.dot(j[:,1],j[:,1])
    result[0][1] = result[1][0] = np.dot(j[:,0],j[:,1])
    return result

def LM(fun,init_param, args,jac,hess):
    """
    @param  fun         the objective function to be optimize
    @param  init_parm   the initial guess param
    @param  args        the original data
    @param  jac         the jacobian of objective function fun
    @param  hess        the hessian of objective function fun
    """
    max_iter_num = 50
    lamb = 0.01

    data_num = args.shape[0]
    param_num = init_param.shape[0]
    
    x = args[:,0]
    y = args[:,1]
    a = init_param[0]
    b = init_param[1]

    est_y = fun(a,b,x)
    residual = y - est_y
    residual_sum_square = np.dot(residual,residual)

    J = jac_f(a,b,x)
    H = hess_f(a,b,x)

    iter_num = 0 
    while (iter_num < max_iter_num):
        H = H + lamb * np.eye(param_num,param_num)
        
        g = np.zeros((2,1))
        g[0][0] = np.outer(J[:,0],residual)
        g[1][0] = np.outer(J[:,1],residual)
        step = np.dot(linalg.inv(H_lm),g)
        
        # something must be wrong below this line
        a += step[0][0]
        b += step[1][0]

        est_y = fun(a,b,x)
        residual = y - est_y
        residual_sum_square = np.dot(residual,residual)

        if residual_sum_square < epsion:
            lamb /= 10

