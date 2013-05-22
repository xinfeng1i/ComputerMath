#!/usr/bin/python
#Filename:SVM.py
#Author: XinFeng Li
#Email: silferlee@zju.edu.cn
#Data: 2013/05/10

import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers
import matplotlib.pyplot as plt

def linear_kernel(x,y):
    """
    Linear Kernel Function
    """
    return np.dot(x,y)

def polynomial_kernel(x,y,p=3):
    """
    Polynomial Kernel Function
    """
    return (np.dot(x,y) + 1) ** p

def gaussian_kernel(x,y,sigma=5.0):
    """
    Gaussian Kernel Function
    """
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))


class SVM(object):

    def __init__(self,kernel=linear_kernel, C=None):
        """
        Constructor
        @param  kernel  Kernel Function
        @param  C       Slack Variable
        """
        self.kernel = kernel
        self.C = C
        if self.C is not None:
            self.C = float(self.C)

    def fit(self,data,label):
        """
        Training SVM to get the fittest w and b such that w*x + b = 0

        @param  data    the training data
        @param  label   the label of training data,which is 1 or -1

        @return w,b     the coefficient w,b such that w*x + b = 0
        """
        n_samples, n_features = data.shape

        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(data[i],data[j])

        P = cvxopt.matrix(np.outer(label,label) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(label,(1,n_samples))
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
            self.b -= np.sum(self.a * self.sv_label * K[index[i], sv])
        self.b /= len(self.a)

        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for i in range(len(self.a)):
                self.w += self.a[i] * self.sv_label[i] * self.sv[i]
        else:
            self.w = None
    def project(self,data):
        """
        Calculate the predict label value
        """
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
        """
        Calculate the predict label (1 or -1) based on w,b
        """
        return np.sign(self.project(data))

if __name__ == "__main__":
    """
    main program
    """

    def gen_lin_separable_data():
        """
        generate the linear separable data
        """
        mean1 = np.array([0,2])
        mean2 = np.array([2,0])
        cov = np.array([ [0.8,0.6],[0.6,0.8] ])
        X1 = np.random.multivariate_normal(mean1,cov,100)
        X2 = np.random.multivariate_normal(mean2,cov,100)
        y1 = np.ones(len(X1))
        y2 = np.ones(len(X2)) * -1
        return X1,y1,X2,y2

    def gen_non_lin_separable_data():
        """
        generate the non linear separable data
        """
        mean1 = [-1,2]
        mean2 = [1,-1]
        mean3 = [4,-4]
        mean4 = [-4,4]
        cov = [[1.0,0.8],[0.8,1.0]]
        X1 = np.random.multivariate_normal(mean1,cov,50)
        X1 = np.vstack((X1,np.random.multivariate_normal(mean3,cov,50)))
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2,cov,50)
        X2 = np.vstack((X2,np.random.multivariate_normal(mean4,cov,50)))
        y2 = np.ones(len(X2)) * -1
        return X1,y1,X2,y2

    def gen_lin_separable_overlap_data():
        """
        generate the linear separable overlap data
        """
        mean1 = np.array([0,2])
        mean2 = np.array([2,0])
        cov = np.array([[1.5,1.0],[1.0,1.5]])
        X1 = np.random.multivariate_normal(mean1,cov,100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2,cov,100)
        y2 = np.ones(len(X2)) * -1
        return X1,y1,X2,y2

    def split_train(X1,y1,X2,y2):
        """
        split the training data from all original data
        """
        X1_train = X1[:90]
        y1_train = y1[:90]
        X2_train = X2[:90]
        y2_train = y2[:90]
        X_train = np.vstack((X1_train,X2_train))
        y_train = np.hstack((y1_train,y2_train))
        return X_train,y_train

    def split_test(X1,y1,X2,y2):
        """
        split the test data from all the original data
        """
        X1_test = X1[90:]
        y1_test = y1[90:]
        X2_test = X2[90:]
        y2_test = y2[90:]
        X_test = np.vstack((X1_test,X2_test))
        y_test = np.hstack((y1_test,y2_test))
        return X_test, y_test

    def plot_margin(X1_train, X2_train,clf):
        """
        plot margin, used in the linear separable data
        """
        def f(x,w,b,c=0):
            return (-w[0] * x - b + c) / w[1]
        plt.plot(X1_train[:,0], X1_train[:,1],"ro")
        plt.plot(X2_train[:,0], X2_train[:,1],"bo")
        plt.scatter(clf.sv[:,0],clf.sv[:,1],s=100,c="g")

        a0 = -4
        a1 = f(a0,clf.w,clf.b)
        b0 = 4
        b1 = f(b0,clf.w,clf.b)
        plt.plot([a0,b0],[a1,b1],"k")

        a0 = -4
        a1 = f(a0,clf.w,clf.b,1)
        b0 = 4
        b1 = f(b0,clf.w,clf.b,1)
        plt.plot([a0,b0],[a1,b1],"k--")

        a0 = -4
        a1 = f(a0,clf.w,clf.b,-1)
        b0 = 4
        b1 = f(b0, clf.w, clf.b, -1)
        plt.plot([a0,b0],[a1,b1],"k--")

        plt.title("Linear Separable Test")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis("tight")
        plt.show()
    
    def plot_contour(X1_train, X2_train, clf):
        """
        plot contour,used in linear separable overlap data 
        And non linear separable data
        """
        plt.plot(X1_train[:,0],X1_train[:,1],"ro")
        plt.plot(X2_train[:,0],X2_train[:,1],"bo")
        plt.scatter(clf.sv[:,0],clf.sv[:,1],s=100,c="g")

        X1,X2 = np.meshgrid(np.linspace(-6,6,50),np.linspace(-6,6,50))
        X = np.array([[x1,x2] for x1, x2 in zip(np.ravel(X1),np.ravel(X2))])
        Z = clf.project(X).reshape(X1.shape)
        plt.contour(X1,X2,Z,[0.0],colors="k",linewidths=1,origin="lower")
        plt.contour(X1,X2,Z+1,[0.0],colors="grey",linewidths=1,origin="lower")
        plt.contour(X1,X2,Z-1,[0.0],colors="grey",linewidths=1,origin="lower")

        #plt.title("Linear overlap Separable Test")
        plt.title("Non Linear Separable Test")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis("tight")
        plt.show()

    def test_linear():
        X1,y1,X2,y2 = gen_lin_separable_data()
        X_train,y_train = split_train(X1,y1,X2,y2)
        X_test,y_test = split_test(X1,y1,X2,y2)

        clf = SVM()
        clf.fit(X_train,y_train)

        y_predict = clf.predict(X_test)
        corrent = np.sum(y_predict == y_test)
        print "%d out of %d predictions corrent" %(corrent,len(y_predict))
        plot_margin(X_train[y_train==1], X_train[y_train==-1],clf)

    def test_overlap_linear():
        X1,y1,X2,y2 = gen_lin_separable_overlap_data()
        X_train,y_train = split_train(X1,y1,X2,y2)
        X_test,y_test = split_test(X1,y1,X2,y2)

        clf = SVM(C=0.1)
        clf.fit(X_train,y_train)

        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print "%d out of %d predictions corrent" %(correct,len(y_predict))

        plot_contour(X_train[y_train==1],X_train[y_train==-1],clf)

    def test_non_linear():
        X1,y1,X2,y2 = gen_non_lin_separable_data()
        X_train, y_train = split_train(X1,y1,X2,y2)
        X_test, y_test = split_test(X1,y1,X2,y2)

        clf = SVM(gaussian_kernel)
        clf.fit(X_train,y_train)

        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print "%d out of %d predictions correct" %(correct,len(y_predict))

        plot_contour(X_train[y_train==1],X_train[y_train==-1],clf)

    #test_linear()
    #test_overlap_linear()
    test_non_linear()
