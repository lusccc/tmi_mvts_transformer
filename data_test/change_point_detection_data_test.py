import logging
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from ax.utils.common.logger import ROOT_LOGGER, disable_logger
from kats.consts import TimeSeriesData
from kats.detectors.cusum_detection import CUSUMDetector
from kats.detectors.robust_stat_detection import RobustStatDetector
from sklearn.cluster import KMeans
from kats.detectors.bocpd import BOCPDetector, BOCPDModelType, TrendChangeParameters, NormalKnownParameters
from logzero import logger
import matplotlib
# matplotlib.use("TkAgg")
# matplotlib.use("qtAgg")
import matplotlib.pyplot as plt
import ruptures as rpt



dataset = 'geolife'

clean_multi_feature_segs = np.load(f'../data/{dataset}_features/clean_multi_feature_segs.npy', allow_pickle=True)
noise_multi_feature_segs = np.load(f'../data/{dataset}_features/noise_multi_feature_segs.npy', allow_pickle=True)
labels = np.load(f'../data/{dataset}_features/noise_multi_feature_seg_labels.npy', allow_pickle=True)
# masks = np.load(f'../data/{dataset}_features/noise_trj_seg_masks.npy', allow_pickle=True)

# i = 999
# i = 2134
# i = 21
# i = 76
i = 897
feat = 3
d = clean_multi_feature_segs[i][2]
v = clean_multi_feature_segs[i][3]
a = clean_multi_feature_segs[i][4]
j = clean_multi_feature_segs[i][5]
hc = clean_multi_feature_segs[i][7]
hcr = noise_multi_feature_segs[i][8]
time = noise_multi_feature_segs[i][9]
# msk = masks[i] * 100
# plt.plot(d, label='d')
# plt.plot(v, label='v')
# plt.plot(a, label='a')
# plt.plot(j, label='j')
# plt.plot(hc, label='hc')
# plt.plot(hcr, label='hcr')
# plt.plot(msk, label='msk')
# plt.legend()
# plt.show()
print()

# https://github.com/facebook/Ax/issues/417#issuecomment-718002390
warnings.filterwarnings("ignore", category=UserWarning)
time_convert = np.vectorize(lambda x: datetime.fromtimestamp(x))
timestamp = time_convert(time)
def get_mask_with_CPD_old(time_seg, feature_seg, mask_ratio=.15, mean_mask_length=3):
    """
    change point detection, then, generate mask
    deprecated!!!!!!
    """
    n_points = len(feature_seg)
    n_mask = int(n_points * mask_ratio)
    n_expected_change_point = int(n_mask / mean_mask_length)
    ts_df = pd.DataFrame({'time': time_convert(time_seg), 'y': feature_seg})
    ts = TimeSeriesData(ts_df)
    detector = BOCPDetector(ts)

    min_threshold = .11
    lr = .1
    curr_threshold = .5
    change_points = []
    n_change_point = 0
    while curr_threshold > min_threshold and n_change_point < n_expected_change_point:
        logger.info(f'curr_threshold:{curr_threshold}')
        change_points = detector.detector(
            model=BOCPDModelType.NORMAL_KNOWN_MODEL, changepoint_prior=mask_ratio, threshold=curr_threshold,
            model_parameters=NormalKnownParameters(empirical=False, )
        )

        curr_threshold -= lr
        logger.info(f'n_change_point:{len(change_points)}')
    n_change_point = len(change_points)
    if n_change_point < n_expected_change_point:
        logger.warn(f'n_change_point:{n_change_point} not meet n_expected_change_point:{n_expected_change_point}')
    print()
    # 0s means mask, 1s means not affected
    mask_vec = np.ones(n_points)
    for cp in change_points:
        cp = ts_df.index[ts_df['time'] == cp.start_time].tolist()
        assert len(cp) == 1
        cp_idx = cp[0]
        logger.info(cp_idx)
        # check beginning
        if int(cp_idx - int(mean_mask_length / 2)) <= 0:
            mask_vec[:mean_mask_length] = 0
        # check end
        elif int(cp_idx + int(mean_mask_length / 2)) >= n_points:
            mask_vec[-mean_mask_length:] = 0
        else:
            mask_vec[cp_idx - int(mean_mask_length / 2):cp_idx - int(mean_mask_length / 2) + mean_mask_length] = 0

    return mask_vec


def generate_mask_using_CPD(feature_seg, mask_ratio=.15, mean_mask_length=3):
    n_points = len(feature_seg)
    n_mask = round(n_points * mask_ratio)
    n_expected_change_point = int(n_mask / mean_mask_length)

    # change point detection
    model = "rbf"  # "l2", "l1"
    algo = rpt.Dynp(model=model, min_size=3, jump=2).fit(feature_seg)
    change_points = algo.predict(n_bkps=n_expected_change_point)
    rpt.show.display(feature_seg, change_points, [2,5,6,9,10,34], figsize=(10, 6))
    plt.show()

    # 0s means mask, 1s means not affected
    mask_vec = np.ones(n_points)
    for cp in change_points:
        logger.info(cp)
        # check beginning
        if int(cp - int(mean_mask_length / 2)) <= 0:
            mask_vec[:mean_mask_length] = 0
        # check end
        elif int(cp + int(mean_mask_length / 2)) >= n_points:
            mask_vec[-mean_mask_length:] = 0
        else:
            mask_vec[cp - int(mean_mask_length / 2):cp - int(mean_mask_length / 2) + mean_mask_length] = 0

    return mask_vec


for seg in [v, a, j, hc, hcr]:
    # change point detection
    generate_mask_using_CPD( v, )




# d2 = np.expand_dims(d, 1)
# kmeans = KMeans(n_clusters=1, init='random')
# kmeans.fit(d2)
# pred = kmeans.predict(d2)
# print()
