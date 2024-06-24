import numpy as np
import matplotlib.pyplot as plt
import util
from scipy.stats import multivariate_normal

true_a = [-0.1, 0.9]

def priorDistribution(beta):
    """
    Plot the contours of the prior distribution p(a)
    
    Inputs:
    ------
    beta: hyperparameter in the proir distribution
    
    Outputs: None
    -----
    """
    ### TODO: Write your code here

    """ Return the density of multivariate Gaussian distribution
        Inputs: 
            mean_vec is a 1D array (like array([,,,]))
            covariance_mat is a 2D array (like array([[,],[,]]))
            x_set is a 2D array, each row is a sample
        Output:
            a 1D array, probability density evaluated at the samples in x_set.
    """

    n = 1000
    pi = np.pi
    x_set_1d = np.linspace(-1,1, n)

    x_set = np.zeros((n,n,2))

    for i in range(n):
        for j in range(n):
            x_set[i,j, 0] = x_set_1d[i]
            x_set[i,j, 1] = x_set_1d[j]

    x_squared = x_set**2
    x_sum = np.sum(x_squared, axis=2)
    prob = (1/(2*pi*beta))*np.exp(-x_sum/(2*beta))

    plot = plt.contourf(x_set_1d, x_set_1d, prob)
    plt.title("p(a) Contour map")
    plt.xlabel("a0")
    plt.ylabel("a1")
    plt.colorbar(plot)
    plt.scatter(true_a[0], true_a[1])
    plt.show()

    return 
    
def posteriorDistribution(x,z,beta,sigma2):
    """
    Plot the contours of the posterior distribution p(a|x,z)
    
    Inputs:
    ------
    x: inputs from training set
    z: targets from traninng set
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    
    Outputs: 
    -----
    mu: mean of the posterior distribution p(a|x,z)
    Cov: covariance of the posterior distribution p(a|x,z)
    """
    ### TODO: Write your code here
    
    pi = np.pi
    ns = len(x)
    n = 1000

    extra_col = np.ones((x.shape[0], 1))
    x = np.append(extra_col, x, axis = 1)

    mu = np.linalg.inv(np.transpose(x)@x + (sigma2**2/beta**2)*np.identity(2))@(np.transpose(x)@z)
    Cov = np.linalg.inv(np.transpose(x)@x + (sigma2**2/beta**2)*np.identity(2)) * sigma2**2

    print(f"Mu: {mu}")
    print(f"Cov: {Cov}")

    X = np.linspace(-1, 1, n)
    Y = np.linspace(-1, 1, n)
    X,Y = np.meshgrid(X, Y)
    pos  = np.dstack((X, Y))
    rv   = multivariate_normal(mu.flatten(), Cov)
    Z    = rv.pdf(pos)

    plot = plt.contour(X, Y, Z)
    plt.title(f"p(a|xi,zi) Contour map for {ns} points")
    plt.xlabel("a0")
    plt.ylabel("a1")
    plt.colorbar(plot)
    plt.scatter(true_a[1], true_a[0])
    plt.show()

    return (mu,Cov)

def predictionDistribution(x,beta,sigma2,mu,Cov,x_train,z_train):
    """
    Make predictions for the inputs in x, and plot the predicted results 
    
    Inputs:
    ------
    x: new inputs
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    mu: output of posteriorDistribution()
    Cov: output of posteriorDistribution()
    x_train,z_train: training samples, used for scatter plot
    
    Outputs: None
    -----
    """
    ### TODO: Write your code here

    pi = np.pi
    ns = len(x)
    n = 1000

    extra_col = np.ones((x_train.shape[0], 1))
    x_train = np.append(extra_col, x_train, axis = 1)

    x = np.array(x)

    print(x)

    mu = np.linalg.inv(np.transpose(x_train)@x_train + (sigma2**2/beta**2)*np.identity(2))@(np.transpose(x_train)@z)
    Cov = x.T@(np.linalg.inv(np.transpose(x_train)@x_train + (sigma2**2/beta**2)*np.identity(2)) * sigma2**2)@x

    print(f"Mu: {mu}")
    print(f"Cov: {Cov}")

    X = np.linspace(-1, 1, n)
    Y = np.linspace(-1, 1, n)
    X,Y = np.meshgrid(X, Y)
    pos  = np.dstack((X, Y))
    rv   = multivariate_normal(mu.flatten(), Cov)
    Z    = rv.pdf(pos)

    plot = plt.contour(X, Y, Z)
    plt.title(f"p(a|x, xi,zi) Contour map for {ns} points")
    plt.xlabel("a0")
    plt.ylabel("a1")
    plt.colorbar(plot)
    plt.scatter(true_a[1], true_a[0])
    plt.show()
    
    
    return 

if __name__ == '__main__':
    
    # training data
    x_train, z_train = util.get_data_in_file('training.txt')
    # new inputs for prediction 
    x_test = [x for x in np.arange(-4,4.01,0.2)]
    
    # known parameters 
    sigma2 = 0.1
    beta = 1
    
    # number of training samples used to compute posterior
    ns  = 100

    # used samples
    x = x_train[0:ns]
    z = z_train[0:ns]
    
    # prior distribution p(a)
    #priorDistribution(beta)
    
    # posterior distribution p(a|x,z)
    mu, Cov = posteriorDistribution(x,z,beta,sigma2)
    
    # distribution of the prediction
    predictionDistribution(x_test,beta,sigma2,mu,Cov,x,z)
    

   

    
    
    

    
