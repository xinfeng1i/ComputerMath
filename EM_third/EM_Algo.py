import numpy as np
import numpy.linalg as linalg
import numpy.random as random
import matplotlib.pyplot as plt

def multi_gauss_dist(x,mean,cov):
    """
    function: multi_gauss_dist
    description: generate the multi gauss distribution

    @param  x       d-dimension variable
    @param  mean    mean of x
    @param  cov     co

    @return gauss_distribution function value
    """
    d = x.size
    a = ((2.0 * np.pi)**(-0.5 * d)) * ((linalg.det(cov))**(-0.5))
    b = -0.5*np.dot(np.dot((x-mean),linalg.inv(cov)), (x - mean))
    return a * np.exp(b)

def normalize_data(dataset):
    """
    function: mormalize_data
    description: mormalize the data 

    @param  dataset     the original data set
    @return the stardard data set extract mean & dividend the std diviation
    """
    temp_mean = np.mean(dataset, axis=0)
    temp_std_diviation = np.std(dataset, axis=0)
    return (dataset - temp_mean) / temp_std_diviation

def plot_dataset(dataset,ret):
    """
    Function: plot the dataset 

    @param dataset
    @param ret
    @returns NULL
    """
    n,d = dataset.shape

    color = np.zeros((n,3))
    color[:,0] = ret[:,0]
    color[:,2] = ret[:,1]
    for i in range(n):
        color[i,:] /= np.sqrt(np.sum(color[i,:]**2))

    x = dataset[:,0]
    y = dataset[:,1]

    plt.scatter(x,y,c=color)
    plt.show()

def init_mean(dataset, n, k):
    """
    function: init mean
    description: randomly choose k points as the initial k mean values

    @param  n   the number of sample points
    @param  k   the number of clusters
    @return the k numbers which represent the initial k mean values
    """
    index = np.arange(n,dtype=int)
    np.random.shuffle(index)
    return dataset[index[:k]]

def e_step(dataset, n, k, mean, cov, prob,ret):
    """
    function: Expectation Step

    @param  dataset     the original dataset
    @param  n           the number of sample points
    @param  k           the number of clusters
    @param  mean        the k mean of k clusters repectively,mean is 1*d array
    @param  cov         the k cov of k clusters repectively, cov is d*d array
    @param  prob        the probabilty in k clusters for each points, prob n*k
    @param  ret         the adjust probability
    @returns  update the prob array ret
    """
    for i in range(n):
        tmp_sum = 0.0
        for j in range(k):
            tmp_sum += prob[j] * multi_gauss_dist(dataset[i],mean[j],cov[j])
        for j in range(k):
            ret[i,j] = prob[j] * multi_gauss_dist(dataset[i],mean[j],cov[j]) / tmp_sum


def m_step(dataset, n, k, mean, cov, prob, ret):
    """
    function: Maximum Step

    @param  dataset         the original dataset
    @param  n               the number of training points
    @param  k               the number of clusters
    @param  mean            the k means of k clusters
    @param  cov             the k cov of k clusters
    @param  prob            the probability
    @param  ret             the updated prbability
    @returns                no returns.just update the params
    """
    # update mean
    for j in range(k):
        tmp_sum = 0.0
        for i in range(n):
            tmp_sum += ret[i,j] * dataset[i]
        mean[j] = tmp_sum / np.sum(ret[:,j])
    
    #update cov
    for j in range(k):
        tmp_sum = 0.0
        for i in range(n):
            tmp_sum += ret[i,j] * np.outer(dataset[i] 
                    -mean[j], dataset[i]-mean[j])
        cov[j] = tmp_sum / np.sum(ret[:,j])
    
    #update prob
    for j in range(k):
        prob[j] = np.sum(ret[:,j]) / n

def calc_likehood(dataset, n, k, mean, cov, prob):
    """
    Function: calc_likehood
    Description: calculate the max likehood
    @param  dataset     the original data set
    @param  n           the number of sample points
    @param  k           the number of clusters
    @param  mean        the k means of the k clusters
    @param  cov         the k cov of the k clusters, each cov is d * d
    @param  prob        the probability
    
    @returns likehood   the likehood
    """
    likehood = 0.0
    for i in range(n):
        tmp_sum = 0.0;
        for j in range(k):
            tmp_sum += prob[j] * multi_gauss_dist(dataset[i],
                    mean[j], cov[j]) 
        likehood += np.log(tmp_sum)
    return likehood


def em_algo(dataset,k):
    """
    Function: em_algo
    Description: the EM (Expectation Maximum) Algorithm

    @param  dataset     the original data set
    @param  k           the number of clusters

    @returns ret        the prob every point in each cluster
    """
    n,d = dataset.shape

    mean =  init_mean(dataset, n, k)
    cov = np.array([np.eye(d)]*k)
    prob = (1.0/k) * np.ones(k)
    ret = np.zeros([n,k])

    max_iter_num = 20
    threshold = 0.01
    
    old_likehood = calc_likehood(dataset,n,k,mean,cov,prob)
    for i in range(max_iter_num):
        e_step(dataset, n, k, mean, cov, prob, ret)
        m_step(dataset, n, k, mean, cov, prob, ret)
        likehood=calc_likehood(dataset, n, k, mean, cov, prob)
        if(likehood - old_likehood <= threshold):
            break
        old_likehood = likehood

    return ret

if __name__ == "__main__":
    """
    main program
    """
    dataset = normalize_data(np.loadtxt("faithful.txt"))
    k=2
    ret=em_algo(dataset,k)
    plot_dataset(dataset,ret)
