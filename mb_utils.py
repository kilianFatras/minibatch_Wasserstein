import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import misc
import ot



def mini_batch(data, weights, batch_size, N_data):
    """
    Select a subset of sample space according to measure
    with np.random.choice

    Parameters
    ----------
    - data : ndarray(N, d)
    - weights : ndarray(N)
        distribution weights
    - batch_size : int
        batch size 'm'
    - N_data : int
        number of data

    Returns
    -------
    - minibatch : ndarray(ns, nt)
        minibatch
    - sub_weights : ndarray(m,)
        distribution weights of the minibatch
    - id_batch : ndarray(N_data,)
        index of minibatch elements
    """
    id_batch = np.random.choice(N_data, batch_size, replace=False, p=weights)
    sub_weights = ot.unif(batch_size)
    return data[id_batch], sub_weights, id_batch


def small_mini_batch(data, weights, batch_size, N_data):
    """
    Select a subset of sample space according to measure
    without np.random.choice (faster for small mb)

    Parameters
    ----------
    - data : ndarray(N, d)
    - weights : ndarray(N)
        distribution weights
    - batch_size : int
        batch size 'm'
    - N_data : int
        number of data

    Returns
    -------
    - minibatch : ndarray(ns, nt)
        minibatch
    - sub_weights : ndarray(m,)
        distribution weights of the minibatch
    - id_batch : ndarray(N_data,)
        index of minibatch elements
    """
    id_batch = np.random.randint(0, N_data, batch_size)
    doublon = 1
    while doublon > 0:
        doublon = 0
        for i in range(batch_size):
            for j in range(i+1, batch_size):
                if id_batch[i] == id_batch[j]:
                    id_batch[j] = np.random.randint(0, N_data)
                    doublon += 1
    sub_weights = ot.unif(batch_size)
    minibatch = data[id_batch]
    return minibatch, sub_weights, id_batch


def update_gamma(gamma, gamma_mb, id_a, id_b):
    '''
    Update mini batch transportation matrix

    Parameters
    ----------
    - gamma : ndarray(ns, nt)
        transportation matrix
    - gamma_mb : ndarray(m, m)
        minibatch transportation matrix
    - id_a : ndarray(m)
        selected samples from source
    - id_b : ndarray(m)
        selected samples from target

    Returns
    -------
    - gamma : ndarray(ns, nt)
        transportation matrix
    '''
    for i,i2 in enumerate(id_a):
        for j,j2 in enumerate(id_b):
            gamma[i2,j2] += gamma_mb[i][j]
    return gamma


def get_stoc_gamma(xs, xt, a, b, m1, m2, K, M, lambd=5*1e-1,
                         method='emd'):
    '''
    Compute the minibatch gamma with stochastic source and target

    Parameters
    ----------
    - xs : ndarray(ns, d)
        source data
    - xt : ndarray(nt, d)
        target data
    - a : ndarray(ns)
        source distribution weights
    - b : ndarray(nt)
        target distribution weights
    - m1 : int
        source batch size
    - m2 : int
        target batch size
    - K : int
        number of batch couples
    - M : ndarray(ns, nt)
        cost matrix
    - lambda : float
        entropic reg parameter
    - method : char
        name of method (entropic or emd)

    Returns
    -------
    - stoc_gamma : ndarray(ns, nt)
        incomplete minibatch OT matrix
    '''
    stoc_gamma = np.zeros((np.shape(xs)[0], np.shape(xt)[0]))
    Ns = np.shape(xs)[0]
    Nt = np.shape(xt)[0]
    for i in range(K):
        #Test mini batch
        sub_xs, sub_weights_a, id_a = mini_batch(xs, a, m1, Ns)
        sub_xt, sub_weights_b, id_b = mini_batch(xt, b, m2, Nt)

        if method == 'emd':
            sub_M = M[id_a,:][:,id_b].copy()
            G0 = ot.emd(sub_weights_a, sub_weights_b, sub_M)

        elif method == 'entropic':
            sub_M = M[id_a, :][:, id_b]
            G0 = ot.sinkhorn(sub_weights_a, sub_weights_b, sub_M, lambd)

        #Test update gamma
        stoc_gamma = update_gamma(stoc_gamma, G0, id_a, id_b)

    return (1/K) * stoc_gamma


def diff_marginale(gamma, marg_a, marg_b):
    '''
    Parameters
    ----------
    - gamma : ndarray(ns, nt)
        transportation matrix
    - marg_a : ndarray(ns,)
        marginals source distribution
    - marg_b : ndarray(nt,)
        marginals target distribution

    Returns
    -------
    - total_diff : float
        sum deviation between each marginal and its expectation
    '''
    vector_ones_a = np.ones(marg_b.shape[0]) # Sum of columns
    vector_ones_b = np.ones(marg_a.shape[0]) # Sum of lines
    cur_marg_a = gamma.dot(vector_ones_a)
    cur_marg_b = gamma.T.dot(vector_ones_b)
    
    diff_marg_a = np.linalg.norm(cur_marg_a - marg_a, ord=1)
    diff_marg_b = np.linalg.norm(cur_marg_b - marg_b, ord=1)
    total_diff = diff_marg_a + diff_marg_b
    
    return total_diff #diff_marg_a, diff_marg_b


def get_conv_marginale(xs, xt, a, b, m1, m2, K, M, lambd=5*1e-1,
                         method='emd'):
    '''
    Compute the historic deviation between marginals and 1/n.

        Parameters
    ----------
    - xs : ndarray(ns, d)
        source data
    - xt : ndarray(nt, d)
        target data
    - a : ndarray(ns)
        source distribution weights
    - b : ndarray(nt)
        target distribution weights
    - m1 : int
        source batch size
    - m2 : int
        target batch size
    - K : int
        number of batch couples
    - M : ndarray(ns, nt)
        cost matrix
    - lambda : float
        entropic reg parameter
    - method : char
        name of method (entropic or emd)

    Returns
    -------
    conv_marg : list(K)
        list of deviation value between marginals and 1/n
    '''
    stoc_gamma = np.zeros((np.shape(xs)[0], np.shape(xt)[0]))
    conv_marg = []
    Ns = np.shape(xs)[0]
    Nt = np.shape(xt)[0]
    
    for i in range(K):
        #Test mini batch
        sub_xs, sub_weights_a, id_a = mini_batch(xs, a, m1, Ns)
        sub_xt, sub_weights_b, id_b = mini_batch(xt, b, m2, Nt)

        if method == 'emd':
            sub_M = M[id_a,:][:,id_b].copy()
            G0 = ot.emd(sub_weights_a, sub_weights_b, sub_M)

        elif method == 'entropic':
            sub_M = M[id_a, :][:, id_b]
            G0 = ot.sinkhorn(sub_weights_a, sub_weights_b, sub_M, lambd)

        #Test update gamma
        stoc_gamma = update_gamma(stoc_gamma, G0, id_a, id_b)
        
        if i%50==0:
            total_diff = diff_marginale((1/(i+1)) * stoc_gamma, a, b)
            conv_marg.append(total_diff)

    return conv_marg #conv_marg_a, conv_marg_b


def incremental_bary_map(xs, xt, a, b, m1, m2, k, 
                         lambd=5*1e-1, method='emd'):
    '''
    Compute the incomplete minibatch barycenter mapping
      between a source and a target distributions.

    Parameters
    ----------
    - xs : ndarray(ns, d)
        source data
    - xt : ndarray(nt, d)
        target data
    - a : ndarray(ns)
        source distribution weights
    - b : ndarray(nt)
        target distribution weights
    - m1 : int
        source batch size
    - m2 : int
        target batch size
    - k : int
        number of batch couples
    - M : ndarray(ns, nt)
        cost matrix
    - lambda : float
        entropic reg parameter
    - method : char
        name of method (entropic or emd)

    Returns
    -------
    - new_xs : ndarray(ns, d)
        Transported source measure
    - new_xt : ndarray(nt, d)
        Transported target measure
    '''
    new_xs = np.zeros(xs.shape)
    new_xt = np.zeros(xt.shape)
    Ns = np.shape(xs)[0]
    Nt = np.shape(xt)[0]
    
    for id_k in range(k):
        sub_xs, sub_weights_a, id_a = mini_batch(xs, a, m1, Ns)
        sub_xt, sub_weights_b, id_b = mini_batch(xt, b, m2, Nt)
        
        if method == 'emd':
            sub_M = ot.dist(sub_xs, sub_xt).copy()
            G0 = ot.emd(sub_weights_a, sub_weights_b, sub_M)

        elif method == 'entropic':
            sub_M = ot.dist(sub_xs, sub_xt)
            G0 = ot.sinkhorn(sub_weights_a, sub_weights_b, sub_M, 
                             lambd)

        new_xs[id_a] += G0.dot(xt[id_b])
        new_xt[id_b] += G0.T.dot(xs[id_a])

    return 1./k * Ns * new_xs, 1./k * Nt * new_xt


def incremental_bary_map_emd(xs, xt, a, b, m1, m2, k):
    '''
    Compute the incomplete minibatch barycenter mapping
      between a source and a target distributions. 
      (faster for small batch size)

    Parameters
    ----------
    - xs : ndarray(ns, d)
        source data
    - xt : ndarray(nt, d)
        target data
    - a : ndarray(ns)
        source distribution weights
    - b : ndarray(nt)
        target distribution weights
    - m1 : int
        source batch size
    - m2 : int
        target batch size
    - k : int
        number of batch couples

    Returns
    -------
    - new_xs : ndarray(ns, d)
        Transported source measure
    - new_xt : ndarray(nt, d)
        Transported target measure
    '''
    new_xs = np.zeros(xs.shape)
    new_xt = np.zeros(xt.shape)
    Ns = np.shape(xs)[0]
    Nt = np.shape(xt)[0]

    if m1 < 101:
        for i in range(k):
            #Test mini batch
            sub_xs, sub_weights_a, id_a = small_mini_batch(xs, a, m1, Ns)
            sub_xt, sub_weights_b, id_b = small_mini_batch(xt, b, m2, Nt)

            sub_M = ot.dist(sub_xs, sub_xt, "sqeuclidean").copy()
            G0 = ot.emd(sub_weights_a, sub_weights_b, sub_M)

            new_xs[id_a] += G0.dot(xt[id_b])
            new_xt[id_b] += G0.T.dot(xs[id_a])

    else:
        for i in range(k):
            #Test mini batch
            sub_xs, sub_weights_a, id_a = mini_batch(xs, a, m1, Ns)
            sub_xt, sub_weights_b, id_b = mini_batch(xt, b, m2, Nt)

            sub_M = ot.dist(sub_xs, sub_xt, "sqeuclidean").copy()
            G0 = ot.emd(sub_weights_a, sub_weights_b, sub_M)

            new_xs[id_a] += G0.dot(xt[id_b])
            new_xt[id_b] += G0.T.dot(xs[id_a])

    return 1./k * Ns * new_xs, 1./k * Nt * new_xt


def imshow_OT_mat(gamma, title):
    '''
    Plot matrix
    '''
    plt.imshow(gamma, interpolation='nearest', cmap='gnuplot')
    plt.title(title)
    plt.savefig('imgs/'+title)
    plt.show()
