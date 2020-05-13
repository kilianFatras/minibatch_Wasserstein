import numpy as np
import matplotlib.pylab as pl
from matplotlib.pyplot import imread
from mb_utils import incremental_bary_map_emd
import ot
import time



def im2mat(I):
    """Converts and image to matrix (one pixel per line)"""
    return I.reshape((I.shape[0] * I.shape[1], I.shape[2]))


def mat2im(X, shape):
    """Converts back a matrix to an image"""
    return X.reshape(shape)

# training samples
I1 = imread('./data/img1.jpg').astype(np.float64) / 256
I2 = imread('./data/img2.jpg').astype(np.float64) / 256

r = np.random.RandomState(42)
np.random.seed(1980)

Ns = [5000, 10000, 15000, 25000, 50000, 75000, 100000]
list_m = [10, 250, 1000]
list_K = [10000]

num_m = len(list_m)
num_K = len(list_K)
all_time = np.zeros((num_m, num_K, len(Ns)))

for id_m in range(num_m):
    m = list_m[id_m]
    for id_K in range(num_K):
        K = list_K[id_K]
        for id_Ns in range(len(Ns)):
            nb = Ns[id_Ns]

            # LOAD DATA
            X1 = im2mat(I1)
            X2 = im2mat(I2)
            idx1 = r.randint(X1.shape[0], size=(nb,))
            idx2 = r.randint(X2.shape[0], size=(nb,))
            Xs = X1[idx1, :]
            Xt = X2[idx2, :]

            # Measures + COST
            mu_s = ot.unif(nb)
            mu_t = ot.unif(nb)

            # EXPS
            print('m, K, nb : ', m, K, nb)
            
            start = time.time()
            cur_newXs, cur_newXt = incremental_bary_map_emd(Xs, Xt, mu_s, mu_t, m, m, K)
            end = time.time()
            all_time[id_m][id_K][id_Ns] = end - start

print(all_time)
np.save('MB_time', all_time)
print('Script done')

fig = pl.figure(figsize=(6,6))
ax = pl.subplot(1, 1, 1)

absc_ent = np.linspace(5000, 100000, len(Ns))
absc_mb = np.linspace(5000, 100000, len(Ns))

ax.loglog(Ns, all_time[0][0],'b.-', label=r"MB, m=10, $k=10^4$")
ax.loglog(Ns, all_time[1][0],'b.--', label=r"MB, m=250, $k=10^4$")
ax.loglog(Ns, all_time[2][0],'b.:', label=r"MB, m=1000, $k=10^4$")
pl.axis([5000,110000,0.001,100000])
ax.set_title("Computational time for OT solvers", fontsize=14)
ax.set_ylabel("time (s)", fontsize=14)
ax.set_xlabel("Number of points n", fontsize=14)
pl.grid()

pl.legend(loc=4)
pl.tight_layout()
#pl.savefig('./imgs/CT_marg_time.pdf')
pl.show()
print('Script done')