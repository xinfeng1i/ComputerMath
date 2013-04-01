import numpy as np
import numpy.random as random
from numpy import linalg as LA
import matplotlib.pyplot as plt

def norm_data(dataset):
    """
    extract the mean value from the original data set

    @param  dataset     the original dataset
    @return             the normalized dataset
    """
    mean = np.mean(dataset, axis=0)
    return (dataset-mean)

def plot_repr_point(dataset):
    """
    Plot the represent points on both x and y is integer

    @param  dataset     the final dataset which is the n*2 matrix
    @return             None    
    """
    basepoints = np.array([[xx,yy] for xx in range(-20,21,10) for yy in range(-20,21,10)])
    for point in basepoints:
        aa = np.array(point)
        aa_min = 100
        aa_x = aa[0]
        aa_y = aa[1]
        for data in dataset:
            bb = np.array(data)
            if (np.absolute(aa[0]-bb[0]) <= 2 and np.absolute(aa[1]-bb[1]) <= 2):
                if(LA.norm(aa-bb) < aa_min):
                    aa_min = LA.norm(aa-bb)
                    aa_x = bb[0]
                    aa_y = bb[1]
        plt.plot(aa_x,aa_y,'ro')

if __name__ == '__main__':
    """
    main program
    """
    dataset =norm_data(np.genfromtxt("threes.txt",dtype=None,usecols=range(0,64),delimiter=","))
    scatter_matrix = np.cov(dataset, rowvar=0)
    eigvalue,eigvector = LA.eig(scatter_matrix)
    newdata = np.dot(dataset, eigvector[:,0:2])
    x = newdata[:,0]
    y = newdata[:,1]
    plot_repr_point(newdata)
    plt.plot(x,y,'b.')
    plt.xlabel("First Priciple Component")
    plt.ylabel("Second Priciple Component")
    plt.title("PCA for rank-2 model")
    plt.grid()
    plt.show()
