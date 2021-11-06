import math
import os

import matplotlib
import numpy as np
import sklearn
from scipy.optimize import linear_sum_assignment

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numba import jit
from scipy.interpolate import interp1d
from sklearn import manifold
import tables as tb

import time
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder


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


def get_consecutive_ones_range_indices(a, max_continuous_len):
    """
     find consecutive 1s seg, where len(seg) >  max_continuous_len
    """
    result = []
    seg_range_indices = one_runs(a)
    seg_len = np.diff(seg_range_indices, axis=1)
    for i, l in enumerate(seg_len):
        if l[0] > max_continuous_len:
            result.append(seg_range_indices[i])
    return result
