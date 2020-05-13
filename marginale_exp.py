import numpy as np
import matplotlib.pylab as pl
from matplotlib.pyplot import imread
from mpl_toolkits.mplot3d import Axes3D
from mb_utils import get_conv_marginale, diff_marginale
import sklearn.cluster as skcluster
import ot

pl.rcParams['pdf.fonttype'] = 42
pl.rcParams['ps.fonttype'] = 42


def im2mat(I):
    """Converts and image to matrix (one pixel per line)"""
    return I.reshape((I.shape[0] * I.shape[1], I.shape[2]))


def mat2im(X, shape):
    """Converts back a matrix to an image"""
    return X.reshape(shape)


I1 = imread('./data/img1.jpg').astype(np.float64) / 256
I2 = imread('./data/img2.jpg').astype(np.float64) / 256
X1 = im2mat(I1)
X2 = im2mat(I2)

nbsamples=1000

clust1 = skcluster.MiniBatchKMeans(n_clusters=nbsamples,init_size=3000).fit(X1)
Xs = clust1.cluster_centers_
Xs_ini = np.copy(Xs)

clust2 = skcluster.MiniBatchKMeans(n_clusters=nbsamples,init_size=3000).fit(X2)
Xt = clust2.cluster_centers_
Xt_ini = np.copy(Xt)
print('data loaded')
    

mu_s = ot.unif(nbsamples)
mu_t = ot.unif(nbsamples)
M = ot.dist(Xs, Xt, "sqeuclidean")

print('full OT')
G = ot.emd(mu_s, mu_t, M)
diff_marg = diff_marginale(G, mu_s, mu_t)
print(diff_marg)

list_m = [10, 50, 100]
num_m = len(list_m)
K = 10000

fig = pl.figure(figsize=(12,5))
ax = pl.subplot(1, 2, 1)
for id_exp in range(num_m):
    m = list_m[id_exp]
    print("m and K : ", m, K)
    conv_marg = get_conv_marginale(Xs_ini, Xt_ini, mu_s, mu_t, m, m, K, M)

    absc_K = np.linspace(0, K, len(conv_marg))
    ax.loglog(absc_K, conv_marg, '+-',  label='MB, m={}'.format(m))

ax.set_yscale('log')
ax.set_xlabel('Number of minibatches k', fontsize=14)
ax.set_ylabel('sum of L1 error on marginals', fontsize=14)
ax.set_title('L1 error on marginals', fontsize=14)
pl.grid()
pl.legend()
pl.tight_layout()
pl.show()
