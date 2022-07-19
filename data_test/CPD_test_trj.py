import matplotlib.pyplot as plt
import ruptures as rpt
import numpy as np
import pandas as pd
from logzero import logger
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KernelDensity
import arviz as az



def generate_mask_using_CPD(feature_seg, mean_mask_length=2):
    n_points = len(feature_seg)

    # change point detection
    model = "rbf"  # "l2", "l1"
    algo = rpt.KernelCPD(kernel=model, min_size=3, ).fit(feature_seg)
    # algo = rpt.KernelCPD(kernel="rbf", params={"gamma": 1e-2}, min_size=2).fit(feature_seg)
    # change_points = algo.predict(n_bkps=n_expected_change_point)
    change_points = algo.predict(pen=1)
    # change_points = algo.predict(n_bkps=21)
    logger.info(f'size: {len(change_points)},  change_points: {change_points}')
    rpt.show.display(feature_seg, change_points, [2, 5, 6, 9, 10, 34], figsize=(10, 6))
    plt.show()

    # 0s means mask, 1s means not affected
    mask_vec = np.ones(n_points)
    for cp in change_points:
        # logger.info(cp)
        # check beginning
        if int(cp - int(mean_mask_length / 2)) <= 0:
            mask_vec[:mean_mask_length] = 0
        # check end
        elif int(cp + int(mean_mask_length / 2)) >= n_points:
            mask_vec[-mean_mask_length:] = 0
        else:
            mask_vec[cp - int(mean_mask_length / 2):cp - int(mean_mask_length / 2) + mean_mask_length] = 0
    logger.info(f'mask_vec:\n {mask_vec}')
    return mask_vec


# i = 767
i = 80
trj = np.load(f'../data/SHL_features/clean_trj_segs.npy', allow_pickle=True)[i]
fs = np.load(f'../data/SHL_features/clean_multi_feature_segs.npy', allow_pickle=True)[i]

d = fs[2]
v = fs[3]
a = fs[4]
j = fs[5]
hc = fs[7]
hcr = fs[8]

x = trj[0]
y = trj[1]
minx, miny = min(x), min(y)

f1 = plt.figure(1)
plt.scatter(x, y)
f1.show()

f2 = plt.figure(2)
plt.plot(hcr, label='hcr')
# plt.plot(v, label='v')
plt.legend()
f2.show()

trj = np.array((x, y)).T
print('trj:')
print(trj)


# kde = np.vectorize(_kde)  # Let numpy care for applying our kde to a vector
# z = kde(x, y)

x_n = np.random.normal(1.75, 1, 512)

f5 = plt.figure(5)
az.plot_kde(x_n, rug=True)
plt.yticks([0], alpha=0)

# x123 = np.array([i for i in range(100)])
# kde = kde(x123)
# print('kde')
# print(kde)


kde = KernelDensity(kernel='gaussian', bandwidth=1).fit(trj)
sc = kde.score_samples(trj)
scc = np.exp(sc)
f3 = plt.figure(3)
print(scc)

sort_scc = np.argsort(-scc)
print(f'sort_scc:{sort_scc}')
masked = sort_scc[:20]
for i in masked:
    x[i] = minx
    y[i] = miny

f8 = plt.figure(8)
plt.scatter(x, y)
f8.show()

#https://stackoverflow.com/questions/41577705/how-does-2d-kernel-density-estimation-in-python-sklearn-work
def kde2D(x, y, bandwidth, xbins=100j, ybins=100j, **kwargs):
    """Build 2D kernel density estimate (KDE)."""

    # create grid of sample locations (default: 100x100)
    xx, yy = np.mgrid[x.min():x.max():xbins,
                      y.min():y.max():ybins]

    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train  = np.vstack([y, x]).T

    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(xy_train)

    # score_samples() returns the log-likelihood of the samples
    z = np.exp(kde_skl.score_samples(xy_sample))
    return xx, yy, np.reshape(z, xx.shape)

bin=100
xx, yy, zz = kde2D(x, y, 1.0, complex(0,bin), complex(0,bin))
print('zz')
print(zz)
print('sort zz')
szz=np.dstack(np.unravel_index(np.argsort(zz.ravel()), (bin, bin)))
print(szz)
f5 = plt.figure(5)
plt.pcolormesh(xx, yy, zz)
plt.scatter(x, y, s=2, facecolor='white')
f5.show()



generate_mask_using_CPD(trj)
# generate_mask_using_CPD(v)

print()
