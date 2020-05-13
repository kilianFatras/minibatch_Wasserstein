import ot
import numpy as np
import matplotlib.pylab as pl
from matplotlib.pyplot import imread
from mb_utils import get_stoc_gamma, incremental_bary_map


def im2mat(I):
    """Converts and image to matrix (one pixel per line)"""
    return I.reshape((I.shape[0] * I.shape[1], I.shape[2]))


def mat2im(X, shape):
    """Converts back a matrix to an image"""
    return X.reshape(shape)


I1 = imread('./data/img1.jpg').astype(np.float64) / 256
I2 = imread('./data/img2.jpg').astype(np.float64) / 256
Xs = im2mat(I1)
Xt = im2mat(I2)


mu_s = ot.unif(Xs.shape[0])
mu_t = ot.unif(Xt.shape[0])

list_m = [10, 100, 1000]
list_K = [10000000, 2000000, 600000]
num_exp = len(list_m)
newXs = np.zeros((num_exp, Xs.shape[0], Xs.shape[1]))
newXt = np.zeros((num_exp, Xt.shape[0], Xt.shape[1]))

for id_exp in range(num_exp):
    K = list_K[id_exp]
    m = list_m[id_exp]
    print('m, K : ', m, K)

    cur_newXs, cur_newXt = incremental_bary_map(Xs, Xt, mu_s, mu_t, m, m, K, lambd=5*1e-1, method='emd')
    cur_newXt[cur_newXt>1]=1
    
    newXs[id_exp] = cur_newXs
    newXt[id_exp] = cur_newXt

np.save('transf_img_1_data', newXs)
np.save('transf_img_2_data', newXt)