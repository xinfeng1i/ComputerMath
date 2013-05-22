import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt

def norm_data(dataset):
    """
    Extract mean from the original dataset

    @param  dataset     the original dataset
    @return             the normalized dataset
    """
    mean = np.mean(dataset,axis=0)
    return (dataset - mean)
def plot_repr_point(dataset):
    """
    Plot the represent points on both x and y is integer

    @param  dataset     the final data set 
    @return             None
    """
    basepoints = np.array([[xx,yy] for xx in range(-20,21,10) for yy in \
        range(-20,21,10)])
    for point in basepoints:
        aa = np.array(point)
        aa_min = 100
        aa_x = aa[0]
        aa_y = aa[1]
        for data in dataset:
            bb = np.array(data)
            if (np.absolute(aa[0]-bb[0])<=2 and np.absolute(aa[1]-bb[1])<=2):
                if (LA.norm(aa-bb)<aa_min):
                    aa_min = LA.norm(aa-bb)
                    aa_x = bb[0]
                    aa_y = bb[1]
        plt.plot(aa_x,aa_y,'ro')

if __name__ == "__main__":
    """
    main program
    """
    dataset = norm_data(np.genfromtxt("threes.txt",dtype=None,usecols=range(0,64),delimiter=","))
    u,s,vt = LA.svd(dataset)
    v = np.transpose(vt)
    newdata = np.dot(dataset,v[:,0:2])

    x = newdata[:,0]
    y = newdata[:,1]
    plt.plot(x,y,"g.")
    plot_repr_point(newdata)
    plt.xlabel("First Priciple Component")
    plt.ylabel("Second Priciple Component")
    plt.title("PCA for rank-2 model")
    plt.grid()
    plt.show()
