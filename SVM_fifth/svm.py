import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers
import matplotlib.pyplot as plt

def linear_kernel(x,y):
    return np.dot(x,y)

def polynomial_kernel(x,y,p=3):
    return (np.dot(x,y) + 1) ** p

def gaussian_kernel(x,y,sigma=5.0):
    return np.exp(-linalg.norm(x,y)**2 / (2 * (sigma ** 2)))


class SVM(object):

    def __init__(self,kernel=linear_kernel, C=None):
        self.kernel = kernel
        self.C = C
        if self.C is not None:
            self.C = float(self.C)

    def fit(self,data,label):
        n_samples, n_features = data.shape

        K = zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(data[i],data[j])

        P = cvxopt.matrix(np.outer(label,label) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y,(1,n_samples))
        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix(np.eye(n_samples) * -1)
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            tmp1 = np.eye(n_samples) * -1
            tmp2 = np.eye(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1,tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1,tmp2)))

        solution = cvxopt.solvers.qp(P,q,G,h,A,b)
        
        a = np.ravel(solution['x'])

        sv = a > 1e-5
        index = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = data[sv]
        self.sv_label = label[sv]

        print "%d support vectors out of %d points" %(len(self.a),n_samples)

        self.b = 0
        for i in range(len(self.a)):
            self.b += self.sv_label[i]
            self.b -= np.sum(self,a * self.sv_label * K[index[i], sv])
        self.b /= len(self.a)

        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for i in range(len(self.a)):
                self.w += self.a[i] * self.sv_label[i] * self.sv[i]
        else:
            self.w = None
    def project(self,data):
        if self.w is not None:
            return np.dot(data, self.w) + self.b
        else:
            label_predict = np.zeros(len(data))
            for i in range(len(data)):
                s = 0
                for a, sv_label, sv  in zip(self.a, self.sv_label, self.sv):
                    s += a * sv_label * self.kernel(data[i],sv)
                label_predict[i] = s
            return label_predict + self.b
    def predict(self, data):
        return np.sign(self.project(data))

if __name__ == "__main__":

    def gen_lin_separable_data():
        mean1 = np.array([0,2])
        mean2 = np.array([2,0])
        cov = np.array([ [0.8,0.6],[0.6,0.8] ])
        X1 = np.random.multivariate_normal(mean1,cov,100)
        X2 = np.random.multivariate_normal(mean2,cov,100)
        y1 = np.ones(len(X1))
        y2 = np.ones(len(X2)) * -1
        return X1,y1,X2,y2

