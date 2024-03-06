import math
import os
import warnings

import matplotlib
import numpy as np
import pandas as pd
import sklearn
from KDEpy import FFTKDE
from kats.consts import TimeSeriesData
from kats.detectors.bocpd import BOCPDetector, BOCPDModelType, NormalKnownParameters
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import gaussian_kde
from skimage.feature import peak_local_max
from sklearn.neighbors import KernelDensity

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numba import jit
from scipy.interpolate import interp1d
from sklearn import manifold
import tables as tb

import time
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from logzero import logger
import ruptures as rpt


def scale_any_shape_data(data, scaler):
    data = np.array(data)
    shape_ = data.shape
    data = data.reshape((-1, 1))
    data = scaler.fit_transform(data)
    data = np.reshape(data, shape_)
    return data


def scale_data(data, scaler):
    data = np.array(data)
    data = scaler.fit_transform(data)
    return data


def scale_RP_each_feature(RP_all_features):
    scaler = StandardScaler()
    n_features = RP_all_features.shape[3]
    for i in range(n_features):
        RP_single_feature = RP_all_features[:, :, :, i]
        scaled = scale_any_shape_data(RP_single_feature, scaler)
        RP_all_features[:, :, :, i] = scaled
    return RP_all_features


def scale_segs_each_features(segs_all_features):
    scaler = StandardScaler()
    n_features = segs_all_features.shape[1]
    for i in range(n_features):
        segs_single_feature = segs_all_features[:, i, :]
        scaled = scale_any_shape_data(segs_single_feature, scaler)
        segs_all_features[:, i, :] = scaled
    return segs_all_features


@jit(nopython=True)
def hampel_filter_forloop_numba(input_series, window_size=10, n_sigmas=3):
    n = len(input_series)
    new_series = input_series.copy()
    k = 1.4826  # scale factor for Gaussian distribution
    indices = []

    for i in range((window_size), (n - window_size)):
        x0 = np.nanmedian(input_series[(i - window_size):(i + window_size)])
        S0 = k * np.nanmedian(np.abs(input_series[(i - window_size):(i + window_size)] - x0))
        if (np.abs(input_series[i] - x0) > n_sigmas * S0):
            new_series[i] = x0
            indices.append(i)

    return new_series


def check_lat_lng(p):
    lat = p[0]
    lng = p[1]
    if lat < -90 or lat > 90:
        print('invalid lat:{}'.format(p))
        return False
    if lng < -180 or lng > 180:
        print('invalid lng:{}'.format(p))
        return False
    return True


def segment_single_series(series, max_size, min_size):
    size = len(series)
    if size <= max_size:
        # return np.array([series], dtype=object)
        return [series]
    else:
        segments = []
        index = 0
        size_of_rest_series = size
        while size_of_rest_series > max_size:
            seg = series[index:index + max_size]  # [,)
            segments.append(seg)
            size_of_rest_series -= max_size
            index += max_size
        if size_of_rest_series > min_size:
            rest_series = series[index:size]
            segments.append(rest_series)
        # return np.array(segments, dtype=object)
        return segments


def calc_initial_compass_bearing(pointA, pointB):
    # https://gist.github.com/jeromer/2005586
    # https://www.igismap.com/formula-to-find-bearing-or-heading-angle-between-two-points-latitude-longitude/
    """
        Calculates the bearing between two points.
        The formulae used is the following:
            θ = atan2(sin(Δlong).cos(lat2),
                      cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))
        :Parameters:
          - `pointA: The tuple representing the latitude/longitude for the
            first point. Latitude and longitude must be in decimal degrees
          - `pointB: The tuple representing the latitude/longitude for the
            second point. Latitude and longitude must be in decimal degrees
        :Returns:
          The bearing in degrees
          direction heading in degrees (0-360 degrees, with 90 = North)
        :Returns Type:
          float
        """
    # if (type(pointA) != tuple) or (type(pointB) != tuple):
    #     raise TypeError("Only tuples are supported as arguments")
    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])
    diffLong = math.radians(pointB[1] - pointA[1])
    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
                                           * math.cos(lat2) * math.cos(diffLong))
    initial_bearing = math.atan2(x, y)
    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing


def interp_single_seg(series, target_size, kind='linear'):
    # https://www.yiibai.com/scipy/scipy_interpolate.html
    size = len(series)
    if size == target_size:
        return series
    else:
        y = np.array(series)
        x = np.arange(size)
        # extend to target size
        interp_x = np.linspace(0, x.max(), target_size)
        interp_y = interp1d(x, y, kind=kind)(interp_x)
        return interp_y


def interp_trj_seg(trj_seg, target_size):
    lons = trj_seg[:, 2]
    lats = trj_seg[:, 1]
    interp_lons = interp_single_seg(lons, target_size, 'cubic')
    interp_lats = interp_single_seg(lats, target_size, 'cubic')

    interp_trj_seg = np.empty([target_size, 2])
    interp_trj_seg[:, 1] = interp_lons
    interp_trj_seg[:, 0] = interp_lats
    return interp_trj_seg


# def padzeros(series, target_size=MAX_SEGMENT_SIZE):
#     new_series = np.zeros(target_size)
#     new_series[:len(series)] = series
#     return new_series


def visualize_data(Z, labels, num_clusters, title='visualization_and_analysis.png'):
    '''
    TSNE visualization_and_analysis of the points in latent space Z
    :param Z: Numpy array containing points in latent space in which clustering was performed
    :param labels: True labels - used for coloring points
    :param num_clusters: Total number of clusters
    :param title: filename where the plot should be saved
    :return: None - (side effect) saves clustering visualization_and_analysis plot in specified location
    '''
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    Z_tsne = tsne.fit_transform(Z)
    fig = plt.figure()
    plt.scatter(Z_tsne[:, 0], Z_tsne[:, 1], s=2, c=labels, cmap=plt.cm.get_cmap("jet", num_clusters))
    plt.colorbar(ticks=range(num_clusters))
    fig.savefig(title, dpi=fig.dpi)
    # plt.show()
    plt.close(fig)


def datatime_to_timestamp(dt):
    # print dt
    """时间转换为时间戳，单位秒"""
    # 转换成时间数组
    time_array = time.strptime(str(dt), "%Y-%m-%d %H:%M:%S")
    # 转换成时间戳
    timestamp = time.mktime(time_array)  # 单位 s
    # print timestamp
    return int(timestamp)


def timestamp_to_hour(ts):
    return datetime.fromtimestamp(ts).hour


def synchronized_open_file(lock, *args, **kwargs):
    with lock:
        print(f'pid:{os.getpid()} open')
        return tb.open_file(*args, **kwargs)


def synchronized_close_file(lock, self, *args, **kwargs):
    with lock:
        print(f'pid:{os.getpid()} close')
        return self.close(*args, **kwargs)


def open_h5_file(*args, **kwargs):
    print(f'pid:{os.getpid()} open')
    return tb.open_file(*args, **kwargs)


def close_h5_file(self, *args, **kwargs):
    print(f'pid:{os.getpid()} close')
    return self.close(*args, **kwargs)


tb_filters = tb.Filters(complevel=5, complib='blosc')


def to_categorical(y, num_classes):
    # below will make label has consecutive value, hence easy to one hot
    en = LabelEncoder().fit(y)
    y = en.transform(y)

    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


# Metrics class was copied from DCEC article authors repository (link in README)
class metrics:
    nmi = sklearn.metrics.normalized_mutual_info_score
    ari = sklearn.metrics.adjusted_rand_score

    @staticmethod
    def acc(y_true, y_pred):
        """
        Calculate clustering accuracy. Require scikit-learn installed

        # Arguments
            y: true labels, numpy.array with shape `(n_samples,)`
            y_pred: predicted labels, numpy.array with shape `(n_samples,)`

        # Return
            accuracy, in [0,1]
        """
        y_true = y_true.astype(np.int64)
        # print('y_pred:{}  y_true:{}'.format(y_pred, y_true))
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        # from sklearn.utils.linear_assignment_ import linear_assignment
        # https://stackoverflow.com/questions/57369848/how-do-i-resolve-use-scipy-optimize-linear-sum-assignment-instead
        ind = linear_sum_assignment(w.max() - w)
        ind = np.asarray(ind)
        ind = np.transpose(ind)
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def one_runs(a):
    """
    https://stackoverflow.com/questions/24885092/finding-the-consecutive-zeros-in-a-numpy-array
    modified for finding consecutive 1s in array
    """
    iszero = np.concatenate(([0], np.equal(a, 1).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges


def get_consecutive_ones_range_indices(a, min_continuous_len):
    """
     find consecutive 1s seg, where len(seg) >  min_continuous_len
    """
    result = []
    seg_range_indices = one_runs(a)
    seg_len = np.diff(seg_range_indices, axis=1)
    for i, l in enumerate(seg_len):
        if l[0] > min_continuous_len:
            result.append(seg_range_indices[i])
    return result


def getExtremePoints(data, typeOfInflexion=None, maxPoints=None):
    """
    https://towardsdatascience.com/modality-tests-and-kernel-density-estimations-3f349bb9e595
    This method returns the indeces where there is a change in the trend of the input series.
    typeOfInflexion = None returns all inflexion points, max only maximum values and min
    only min,
    """
    a = np.diff(data)
    asign = np.sign(a)
    signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
    idx = np.where(signchange == 1)[0]

    if len(idx) == 0:
        return None

    try:
        if typeOfInflexion == 'max' and data[idx[0]] < data[idx[1]]:
            idx = idx[1:][::2]

        elif typeOfInflexion == 'min' and data[idx[0]] > data[idx[1]]:
            idx = idx[1:][::2]
        elif typeOfInflexion is not None:
            idx = idx[::2]
    except:
        print('error')

    # sort ids by min value
    if 0 in idx:
        idx = np.delete(idx, 0)
    if (len(data) - 1) in idx:
        idx = np.delete(idx, len(data) - 1)
    idx = idx[np.argsort(data[idx])]
    # If we have maxpoints we want to make sure the timeseries has a cutpoint
    # in each segment, not all on a small interval
    if maxPoints is not None:
        idx = idx[:maxPoints]
        if len(idx) < maxPoints:
            return (np.arange(maxPoints) + 1) * (len(data) // (maxPoints + 1))

    return idx


class GaussianKde(gaussian_kde):
    """
    https://stackoverflow.com/questions/63812970/scipy-gaussian-kde-matrix-is-not-positive-definite
    Drop-in replacement for gaussian_kde that adds the class attribute EPSILON
    to the covmat eigenvalues, to prevent exceptions due to numerical error.
    """

    EPSILON = 1e-10  # adjust this at will

    def _compute_covariance(self):
        """Computes the covariance matrix for each Gaussian kernel using
        covariance_factor().
        """
        self.factor = self.covariance_factor()
        # Cache covariance and inverse covariance of the data
        if not hasattr(self, '_data_inv_cov'):
            self._data_covariance = np.atleast_2d(np.cov(self.dataset, rowvar=1,
                                                         bias=False,
                                                         aweights=self.weights))
            # we're going the easy way here
            self._data_covariance += self.EPSILON * np.eye(
                len(self._data_covariance))
            self._data_inv_cov = np.linalg.inv(self._data_covariance)

        self.covariance = self._data_covariance * self.factor ** 2
        self.inv_cov = self._data_inv_cov / self.factor ** 2
        L = np.linalg.cholesky(self.covariance * 2 * np.pi)
        self._norm_factor = 2 * np.log(np.diag(L)).sum()  # needed for scipy 1.5.2
        self.log_det = 2 * np.log(np.diag(L)).sum()  # changed var name on 1.6.2


def generate_mask_for_trj_using_KDE_RPD(trj_seg, mean_mask_length=2):
    n_points = len(trj_seg)
    x = trj_seg[:, 0]
    y = trj_seg[:, 1]

    # transform trj to same scale as the grid of kde
    target_scale = 100  # 0~100
    grid_size = 100
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_s = scaler.fit_transform(x.reshape(-1, 1)).squeeze() * target_scale
    y_s = scaler.fit_transform(y.reshape(-1, 1)).squeeze() * target_scale

    # Boundary correction using mirroring, then do kde
    # https://kdepy.readthedocs.io/en/latest/examples.html#boundary-correction-using-mirroring
    data = np.vstack([x_s, y_s]).T  # use scaled data!!!.
    x_s_mir = np.concatenate([(2 * x_s.min() - x_s), x_s, (2 * x_s.max() - x_s)])
    y_s_mir = np.concatenate([(2 * y_s.min() - y_s), y_s, (2 * y_s.max() - y_s)])
    data_mir = np.vstack([x_s_mir, y_s_mir]).T  # use scaled data!!!.
    grid_size_mir = grid_size * 3  # mirrored, hence the grid_size to do kde is 3 times

    # fix grid error!! because i found that if we do not manually generate grid using np.linspace, the peak values in
    # kde detected is lagged!!
    # https://github.com/tommyod/KDEpy/issues/15 Create 2D grid
    kde_grid_x = np.linspace(data_mir.min() - 1, data_mir.max() + 1, grid_size_mir)  # "-1, +1" is used to ensure range
    kde_grid_y = np.linspace(data_mir.min() - 1, data_mir.max() + 1, grid_size_mir)
    kde_grid = np.stack(np.meshgrid(kde_grid_x, kde_grid_y), -1).reshape(-1, 2)
    kde_grid[:, [0, 1]] = kde_grid[:, [1, 0]]  # Swap indices

    # FFTKDE
    fit = FFTKDE(bw=1, kernel='epa').fit(data_mir)
    z_kde = fit.evaluate(kde_grid)
    z_kde_grid = z_kde.reshape(grid_size_mir, grid_size_mir).T

    # find_peak_repeatedly
    pk_coords_mir = find_peak_repeatedly(z_kde_grid, min_peaks=3 * 4, max_peaks=3 * 16, threshold_rel=0.1,
                                         min_distance=2)
    # get peak index
    pk_coords_mir_correct = np.vstack(
        [pk_coords_mir[:, 1], pk_coords_mir[:, 0]]).T  # make columns order correct to calculate distance later

    # take out middle part, find the close one to pk_coord in data
    pk_coords_unmir_correct = []
    for pk in pk_coords_mir_correct:
        x, y = pk[0], pk[1]
        if x in range(grid_size, 2 * grid_size) and y in range(grid_size, 2 * grid_size):
            pk_coords_unmir_correct.append(pk)
    pk_coords_unmir_correct = np.array(pk_coords_unmir_correct) - grid_size

    # match closet points in trj
    pk_point_idx = []
    for pk_coord in pk_coords_unmir_correct:
        min_dist = math.inf
        min_dist_point_idx = -1  # idx
        for i, point in enumerate(data):
            # if i not in range(grid_size, 2*grid_size):
            #     continue
            dist = distance.euclidean(point, pk_coord)
            if dist < min_dist:
                min_dist = dist
                min_dist_point = point
                min_dist_point_idx = i
        pk_point_idx.append(min_dist_point_idx)

    # generate masks
    mask_vec = np.ones(n_points)
    for idx in pk_point_idx:
        # logger.info(idx)
        # check beginning
        if int(idx - int(mean_mask_length / 2)) <= 0:
            mask_vec[:mean_mask_length] = 0
        # check end
        elif int(idx + int(mean_mask_length / 2)) >= n_points:
            mask_vec[-mean_mask_length:] = 0
        else:
            mask_vec[idx - int(mean_mask_length / 2):idx - int(mean_mask_length / 2) + mean_mask_length] = 0

    return mask_vec


def find_peak_repeatedly(data, min_peaks=3 * 4, max_peaks=3 * 9, threshold_rel=0., min_distance=2):
    """
    To avoid the situation that a very high peak occurred lead to the rest peaks cannot be identified,
    hence we repeatedly find peaks by making already identified peaks equal to zero until the `min_peaks` is met.
    Besides, the `max_peaks` is also required, if peaks more than `max_peaks` is detected, we only need the former highest
    `max_peaks`.

    Returns: indices of peaks

    """
    data_cp = np.copy(data)

    results = None
    n_pks = 0
    while n_pks <= min_peaks:
        pks = peak_local_max(data_cp, exclude_border=False, threshold_rel=threshold_rel, min_distance=min_distance)
        # print(n_pks, pks)
        if results is None:
            results = pks
        else:
            results = np.concatenate([results, pks])
        n_pks += len(pks)
        # results += pks
        for p in pks:
            data_cp[p[0], p[1]] = 0

    if n_pks > max_peaks:
        # print([coord for coord in results])
        pk_vals = np.array([[*coord, data[coord[0], coord[1]]] for coord in results])
        # sort by peak val
        pk_vals = pk_vals[pk_vals[:, 2].argsort()][::-1]
        results = pk_vals[:max_peaks, [0, 1]].astype(int)
    return np.array(results)


@DeprecationWarning
def generate_mask_for_trj_using_KDE(trj_seg, mean_mask_length=2):
    n_points = len(trj_seg)
    x = trj_seg[:, 0]
    y = trj_seg[:, 1]
    # Create meshgrid
    # deltaX = (max(x) - min(x)) / 10
    # deltaY = (max(y) - min(y)) / 10
    #
    # xmin = min(x) - deltaX
    # xmax = max(x) + deltaX
    #
    # ymin = min(y) - deltaY
    # ymax = max(y) + deltaY
    # xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

    # Fit a gaussian kernel
    # positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    try:
        kernel = GaussianKde(values)
    except:
        np.save('error_kde.npy', values)
        print('error')
    kvt = kernel(values).T  # density on x,y

    # Higher density points index
    idx_hd = getExtremePoints(kvt, typeOfInflexion='max')

    if idx_hd is None:
        logger.warning('no ExtremePoints ! mask all points!')
        mask_vec = np.zeros(n_points)
    else:
        mask_vec = np.ones(n_points)
        for idx in idx_hd:
            # logger.info(idx)
            # check beginning
            if int(idx - int(mean_mask_length / 2)) <= 0:
                mask_vec[:mean_mask_length] = 0
            # check end
            elif int(idx + int(mean_mask_length / 2)) >= n_points:
                mask_vec[-mean_mask_length:] = 0
            else:
                mask_vec[idx - int(mean_mask_length / 2):idx - int(mean_mask_length / 2) + mean_mask_length] = 0
    return mask_vec


def generate_mask_for_feature_using_EP(feature_seg, mean_mask_length=2):
    """
    EP: ExtremePoints
    """
    n_points = len(feature_seg)
    idx_ep = getExtremePoints(feature_seg, typeOfInflexion='max')
    if idx_ep is None:
        logger.warning('no ExtremePoints ! mask all points!')
        mask_vec = np.zeros(n_points)
    else:
        mask_vec = np.ones(n_points)
        for idx in idx_ep:
            # logger.info(idx)
            # check beginning
            if int(idx - int(mean_mask_length / 2)) <= 0:
                mask_vec[:mean_mask_length] = 0
            # check end
            elif int(idx + int(mean_mask_length / 2)) >= n_points:
                mask_vec[-mean_mask_length:] = 0
            else:
                mask_vec[idx - int(mean_mask_length / 2):idx - int(mean_mask_length / 2) + mean_mask_length] = 0
    return mask_vec


def generate_mask_using_CPD_unknown_CP_number(feature_seg, mean_mask_length=2):
    n_points = len(feature_seg)

    # change point detection
    model = "rbf"  # "l2", "l1"
    algo = rpt.KernelCPD(kernel=model, min_size=3, ).fit(feature_seg)
    change_points = algo.predict(pen=1)
    # logger.info(f'size: {len(change_points)},  change_points: {change_points}')
    n_cp = len(change_points)
    if n_cp > 5:
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
    else:
        # too less CPs, mask all points, which means the loss function in training will be imposed on all points
        # 0s means mask, 1s means not affected
        logger.info('too less CPs!!!  mask all points!')
        mask_vec = np.zeros(n_points)
    return mask_vec


def generate_mask_using_CPD(seg, mask_ratio=.15, mean_mask_length=3):
    n_points = len(seg)
    n_mask = round(n_points * mask_ratio)
    n_expected_change_point = round(n_mask / mean_mask_length)

    # change point detection
    model = "rbf"  # "l2", "l1"
    algo = rpt.Dynp(model=model, min_size=3, jump=2).fit(seg)
    change_points = algo.predict(n_bkps=n_expected_change_point)
    # rpt.show.display(seg, change_points, [2,5,6,9,10,34,55], figsize=(10, 6))
    # plt.show()

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

    return mask_vec


# https://github.com/facebook/Ax/issues/417#issuecomment-718002390
# to make BOCPDetector warning disappear
warnings.filterwarnings("ignore", category=UserWarning)
# time_convert = np.vectorize(lambda x: datetime.fromtimestamp(x))
time_convert = np.vectorize(lambda x: datetime.fromtimestamp(x))


def get_mask_with_BOCPDetector(time_seg, feature_seg, mask_ratio=.15, mean_mask_length=3):
    """
    deprecated!!!!!
    change point detection, then, generate mask
    """
    n_points = len(feature_seg)
    n_mask = int(n_points * mask_ratio)
    n_expected_change_point = int(n_mask / mean_mask_length)
    ts_df = pd.DataFrame({'time': time_convert(time_seg), 'y': feature_seg})
    ts = TimeSeriesData(ts_df)
    detector = BOCPDetector(ts)  # https://github.com/facebookresearch/Kats/blob/main/tutorials/kats_202_detection.ipynb

    min_threshold = .11
    lr = .1
    curr_threshold = .5
    change_points = []
    n_change_point = 0
    while curr_threshold > min_threshold and n_change_point < n_expected_change_point:
        # logger.info(f'curr_threshold:{curr_threshold}')
        change_points = detector.detector(
            model=BOCPDModelType.NORMAL_KNOWN_MODEL, changepoint_prior=mask_ratio, threshold=curr_threshold,
            model_parameters=NormalKnownParameters(empirical=False, )
        )

        curr_threshold -= lr
        # logger.info(f'n_change_point:{len(change_points)}')
    n_change_point = len(change_points)
    if n_change_point < n_expected_change_point:
        logger.warn(f'n_change_point:{n_change_point} not meet n_expected_change_point:{n_expected_change_point}')

    # 0s means mask, 1s means not affected
    mask_vec = np.ones(n_points)
    for cp in change_points:
        cp = ts_df.index[ts_df['time'] == cp.start_time].tolist()
        assert len(cp) == 1
        cp_idx = cp[0]
        # logger.info(cp_idx)
        # check beginning
        if int(cp_idx - int(mean_mask_length / 2)) <= 0:
            mask_vec[:mean_mask_length] = 0  # assign mask at beginning
        # check end
        elif int(cp_idx + int(mean_mask_length / 2)) >= n_points:
            mask_vec[-mean_mask_length:] = 0  # assign mask at end
        else:
            mask_vec[cp_idx - int(mean_mask_length / 2):cp_idx - int(mean_mask_length / 2) + mean_mask_length] = 0

    return mask_vec
