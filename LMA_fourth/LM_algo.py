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

    @return   the function value 1*n array
    """
    y = np.zeros(len(x))
    for i in range(len(x)):
        y[i] = a * np.exp(-b * x[i])
    return y

def jac_f(a,b,x):
    """
    Compute the Jacobian matrix respect to 'a' and 'b'

    @param  a    the parameter to be estimate
    @param  b    the parameter to be estimate
    @param  x    the original data

    @return      the Jacobian n * 2 matrix
    """
    n = len(x)
    result = np.zeros((n,2))

    for i in range(n):
        result[i][0] = np.exp(-b * x[i])
        result[i][1] = -a * x[i] * np.exp(-b * x[i])

    return result

def hess_f(a,b,x):
    """
    Compute the Hessian Matrix respect to 'a' & 'b'

    @param  a   the parameter to be estimate
    @param  b   the parameter to be estimate
    @param  x   the original data
    @return     the Hessina Matrix which is 2*2
    """
    n = len(x)
    j = jac_f(a,b,x)
    result = np.zeros((2,2))
    result[0][0] = np.dot(j[:,0],j[:,0])
    result[1][1] = np.dot(j[:,1],j[:,1])
    result[0][1] = result[1][0] = np.dot(j[:,0],j[:,1])
    return result
def is_positive(hess):
    """
    Judge whether the input hess matrix is positive definition

    @param  hess    Hessian Matrix
    @return     bool,whether is a Hessian Matrix
    """
    w,v = linalg.eigh(hess) 
    for value in w:
        if value <= 0:
            return False
    return True


def LM(fun,init_param, args,jac,hess):
    """
    Use LMA algorithm to estimate parameter

    @param  fun         the objective function to be optimize
    @param  init_parm   the initial guess param
    @param  args        the original data
    @param  jac         the jacobian of objective function fun
    @param  hess        the hessian of objective function fun
    
    @return     the estimated parameter 'a' & 'b'
    """
    max_iter_num = 200
    lamb = 0.01

    data_num = args.shape[0]
    param_num = len(init_param)
    
    x = args[:,0]
    y = args[:,1]
    a = init_param[0]
    b = init_param[1]

    est_y = fun(a,b,x)
    residual = y - est_y
    residual_sum_square = np.dot(residual,residual)

    iter_num = 0 
    
    while (iter_num < max_iter_num):
        J = jac_f(a,b,x)
        H = hess_f(a,b,x)
        
        #compute the step length
        H_lm = H + lamb * np.eye(param_num,param_num)
        while (not is_positive(H_lm)):  # make sure H_lm is positive
            lamb = 4 * lamb
            H_lm = H + lamb * np.eye(param_num, param_num)
        g = np.zeros((2,1))
        g[0][0] = np.dot(J[:,0],residual)
        g[1][0] = np.dot(J[:,1],residual)
        step = np.dot(linalg.inv(H_lm),g)
        
        a_lm = a + step[0][0]
        b_lm = b + step[1][0]

        est_y_lm = fun(a_lm,b_lm,x)
        residual_lm = y - est_y_lm
        residual_sum_square_lm = np.dot(residual_lm,residual_lm)

        #if error is smaller,accept.otherwise,reject and adjust step
        if residual_sum_square_lm < residual_sum_square:
            lamb = lamb / 10
            a = a_lm
            b = b_lm
            residual = residual_lm
            residual_sum_square = residual_sum_square_lm
        else:
            lamb = lamb * 10
            
        iter_num += 1
    return [a,b]
    


if __name__ == "__main__":
    """
    main program
    """
    ab_init = [2,2]
    data = [0.25,0.5,1,1.5,2,3,4,6,8]
    obs = [19.21,18.15,15.36,14.10,12.89,9.32,7.45,5.24,3.01]
    args_init = np.zeros((9,2))
    args_init[:,0] = data
    args_init[:,1] = obs
    [aa,bb] = LM(fun,ab_init,args_init,jac_f,hess_f)
    print aa,bb
    y = fun(aa,bb,data)
    print y
    plt.plot(data,obs,'ro',label="original data")
    plt.plot(data,y,label="fitting curve")
    plt.show()
