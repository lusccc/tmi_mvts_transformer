import argparse
import multiprocessing
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from geopy.distance import geodesic
# from imblearn.over_sampling import ADASYN
from logzero import logger
from utils import interp_single_seg, interp_trj_seg
from utils import segment_single_series
from utils import check_lat_lng, timestamp_to_hour, calc_initial_compass_bearing, \
    to_categorical, get_consecutive_ones_range_indices, generate_mask_using_CPD, \
    generate_mask_using_CPD_unknown_CP_number, generate_mask_for_trj_using_KDE, generate_mask_for_feature_using_EP

import matplotlib

# matplotlib.use('Qt5Agg')

""" 
paper: `Unsupervised Deep Learning for GPS-Based Transportation Mode Identification` and 
`Semi-Supervised Deep Learning Approach for Transportation Mode Identification Using GPS Trajectory Data`, 
define stay time as 20min
"""
MAX_STAY_TIME_INTERVAL = 60 * 20
MIN_N_POINTS = 10
MAX_N_POINTS = 128

"""
5 classification labels
0,    1,    2,   3,         4
walk, bike, bus, driving, train/subway,
"""
SPEED_LIMIT = {0: 7, 1: 12, 2: 120. / 3.6, 3: 180. / 3.6, 4: 120 / 3.6, }
# acceleration
ACC_LIMIT = {0: 3, 1: 3, 2: 2, 3: 10, 4: 3, }

"""
for SHL 7 or 8 classification,
Null=0, Still=1, Walking=2, Run=3, Bike=4, Car=5, Bus=6, Train=7, Subway=8
"""
SPEED_LIMIT_7 = {2: 7, 4: 12, 6: 120. / 3.6, 5: 180. / 3.6, 7: 120 / 3.6, 1: 1, 3: 40 / 3.6, 8: 120 / 3.6}
# acceleration
ACC_LIMIT_7 = {2: 3, 4: 3, 6: 2, 5: 10, 7: 3, 1: 1, 3: 3, 8: 3}

LIKELY_STOP_DISTANCE_THRESHOLD = 0.5  # meters
LIKELY_TURN_ANGLE_THRESHOLD = 90  # degree
MIN_MASK_SEG_LEN = 4


def filter_error_gps_data(trjs, labels):
    logger.info('filter_error_gps_data...')
    tasks = []
    batch_size = int(len(trjs) / n_threads + 1)
    for i in range(0, n_threads):
        tasks.append(pool.apply_async(do_filter_error_gps_data,
                                      (trjs[i * batch_size:(i + 1) * batch_size],
                                       labels[i * batch_size:(i + 1) * batch_size])))
    res = np.array([[t.get()[0], t.get()[1]] for t in tasks], dtype=object)
    trjs = np.concatenate(res[:, 0])
    labels = np.concatenate(res[:, 1])

    # filter out lat and lng whose values not in 1th ~ 99th percentile
    trjs_stack = np.vstack(trjs)[:, [1, 2]]  # keep lat, lng
    _99th = np.percentile(trjs_stack, 99, axis=0)
    _1th = np.percentile(trjs_stack, 1, axis=0)
    trjs_filtered = []
    labels_filtered = []
    deleted = []
    for trj, label in zip(trjs, labels):
        indices_lat = np.where((trj[:, 1] >= _99th[0]) | (trj[:, 1] <= _1th[0]))
        indices_lon = np.where((trj[:, 2] >= _99th[1]) | (trj[:, 2] <= _1th[1]))
        indices = np.intersect1d(indices_lat, indices_lon)
        if len(indices) > 0:
            # deleted.append(trj[[indices]])
            logger.info(f'delete trj point not in 1th ~ 99th percentile: {trj[[indices]]}')
        trj = np.delete(trj, indices, axis=0)
        if len(trj) >= MIN_N_POINTS:
            trjs_filtered.append(trj)
            labels_filtered.append(label)

    return np.array(trjs_filtered, dtype=object), np.array(labels_filtered, dtype=object)


def do_filter_error_gps_data(trjs, labels):
    filtered_trjs = []
    filter_labels = []
    for trj, label in zip(trjs, labels):
        n_points = len(trj)
        if n_points < MIN_N_POINTS:
            logger.info('gps points num not enough:{}'.format(n_points))
            continue
        invalid_points = []  # wrong gps data points index
        for i in range(n_points - 1):
            p_a = [trj[i][1], trj[i][2]]
            p_b = [trj[i + 1][1], trj[i + 1][2]]
            t_a = trj[i][0]
            t_b = trj[i + 1][0]

            # if "point a" is invalid, using previous "point a" instead of current one
            if i in invalid_points:
                p_a = [trj[i - 1][1], trj[i - 1][2]]
                t_a = trj[i - 1][0]
                delta_t = t_b - t_a
            else:
                delta_t = t_b - t_a

            if delta_t <= 0:
                invalid_points.append(i + 1)
                logger.info('invalid timestamp, t_a:{}, t_b:{}, delta_t:{}'.format(t_a, t_b, delta_t))
                continue
            if not check_lat_lng(p_a):
                invalid_points.append(i)
                continue
            if not check_lat_lng(p_b):
                invalid_points.append(i + 1)
                continue
        filtered_trj_seg = np.delete(trj, invalid_points, axis=0)
        if len(filtered_trj_seg) < MIN_N_POINTS:
            logger.info('gps points num not enough:{}'.format(len(filtered_trj_seg)))
            pass
        else:
            filtered_trjs.append(filtered_trj_seg)
            filter_labels.append(label)
    return np.array(filtered_trjs, dtype=object), np.array(filter_labels)


def segment_on_long_stay_time(trjs, labels):
    logger.info('segment_trjs...')
    tasks = []
    batch_size = int(len(trjs) / n_threads + 1)
    for i in range(0, n_threads):
        tasks.append(pool.apply_async(do_segment_on_long_stay_time, (
            trjs[i * batch_size:(i + 1) * batch_size], labels[i * batch_size:(i + 1) * batch_size])))

    res = np.array([[t.get()[0], t.get()[1]] for t in tasks], dtype=object)
    trj_segs = np.concatenate(res[:, 0])
    trj_seg_labels = np.concatenate(res[:, 1])
    return trj_segs, trj_seg_labels


def do_segment_on_long_stay_time(trjs, labels):
    total_trj_segs = []
    total_trj_seg_labels = []
    for trj, label in zip(trjs, labels):
        # first, split based on long stay points
        delta_ts = np.diff(trj[:, 0])
        split_idx = np.where(delta_ts > MAX_STAY_TIME_INTERVAL)
        # note: the indices are saved in split_idx[0]
        if len(split_idx[0]) > 0:
            trj_segs = np.split(trj, split_idx[0] + 1)
            trj_segs = [seg for seg in trj_segs if seg.shape[0] > 0]
            trj_seg_labels = [label for _ in range(len(trj_segs))]
        else:
            trj_segs = [trj]
            trj_seg_labels = [label]

        # second, segment to sub_seg with max_size
        for trj_seg, trj_seg_label in zip(trj_segs, trj_seg_labels):
            trj_sub_segs = segment_single_series(trj_seg, max_size=MAX_N_POINTS, min_size=MIN_N_POINTS)
            # if len(trj_sub_segs.shape)>1:
            #     print()
            trj_sub_seg_labels = [label for _ in range(len(trj_sub_segs))]
            total_trj_segs.extend(trj_sub_segs)
            total_trj_seg_labels.extend(trj_sub_seg_labels)
    # total_trj_segs.extend(trj_segs)
    # total_trj_seg_labels.extend(trj_seg_labels)
    return np.array(total_trj_segs, dtype=object), np.array(total_trj_seg_labels, dtype=object)


def calc_feature(trj_segs, trj_seg_labels):
    """
    noise means do not filter noise points
    """
    logger.info('calc_feature...')
    tasks = []
    batch_size = int(len(trj_segs) / n_threads + 1)
    for i in range(0, n_threads):
        tasks.append(pool.apply_async(do_calc_feature,
                                      (trj_segs[i * batch_size:(i + 1) * batch_size],
                                       trj_seg_labels[i * batch_size:(i + 1) * batch_size],
                                       args
                                       )))
    res = np.array(
        [[t.get()[0], t.get()[1], t.get()[2], t.get()[3], t.get()[4], t.get()[5], t.get()[6], t.get()[7], t.get()[8]]
         for t in tasks],
        dtype=object)
    logger.info('merging...')
    ns_trj_segs = np.concatenate(res[:, 0])
    cn_trj_segs = np.concatenate(res[:, 1])
    fs_seg_masks = np.concatenate(res[:, 2])
    trj_seg_masks = np.concatenate(res[:, 3])
    ns_multi_feature_segs = np.concatenate(res[:, 4])
    cn_multi_feature_segs = np.concatenate(res[:, 5])
    multi_feature_seg_labels = np.concatenate(res[:, 6])
    n_removed_points = np.vstack(res[:, 7])
    n_total_points = np.vstack(res[:, 8])
    return ns_trj_segs, cn_trj_segs, fs_seg_masks, trj_seg_masks, ns_multi_feature_segs, cn_multi_feature_segs, \
           multi_feature_seg_labels, n_removed_points, n_total_points


def do_calc_feature(trj_segs, trj_seg_labels, args):
    # ns: noise, cn: clean, i.e., noise filtered
    cn_trj_segs = []  # only keep lon and lat, which are interpolated to align the size of ns_trj_seg
    ns_trj_segs = []  # only keep lon and lat
    fs_seg_masks = []  # mask generated from clean features
    trj_seg_masks = []  # mask generated from clean trjs

    ns_multi_feature_segs = []
    cn_multi_feature_segs = []
    multi_feature_seg_labels = []
    n_removed_points = [0 for i in range(args.n_class)]  # count removed points for each class
    n_total_points = [0 for i in range(args.n_class)]  # count all points for each class

    for i, (trj_seg, trj_seg_label) in enumerate(zip(trj_segs, trj_seg_labels)):
        n_points = len(trj_seg)

        # store noise filtered feature value, cn: clean
        cn_hours = []  # hour of timestamp
        cn_delta_times = []
        cn_distances = []
        cn_velocities = []
        cn_accelerations = []
        cn_jerks = []
        cn_headings = []
        cn_heading_changes = []
        cn_heading_change_rates = []
        cn_timestamps = []

        stop_states = []
        turn_states = []

        # store noise feature value, ns:noise
        ns_hours = []  # hour of timestamp
        ns_delta_times = []
        ns_distances = []
        ns_velocities = []
        ns_accelerations = []
        ns_jerks = []
        ns_headings = []
        ns_heading_changes = []
        ns_heading_change_rates = []
        ns_timestamps = []

        ns_indices = []

        prev_v = 0  # previous velocity
        prev_a = 0
        prev_h = 0  # previous heading
        for j in range(n_points - 1):
            p_a = [trj_seg[j][1], trj_seg[j][2]]
            p_b = [trj_seg[j + 1][1], trj_seg[j + 1][2]]
            t_a = trj_seg[j][0]
            t_b = trj_seg[j + 1][0]

            # ************ 1.CALC FEATURE ************
            delta_t = t_b - t_a
            timestamps = (t_b + t_a) / 2.
            hour = timestamp_to_hour(t_a)
            # distance
            d = geodesic(p_a, p_b).meters
            # velocity
            v = d / delta_t
            # acceleration
            a = (v - prev_v) / delta_t
            # jerk
            jk = (a - prev_a) / delta_t
            # heading
            h = calc_initial_compass_bearing(p_a, p_b)
            # heading change
            # hc = h - prev_h
            hc = abs(h - prev_h)  # changed to abs !
            # heading change rate
            hcr = hc / delta_t
            # is likely stop
            stop = 1 if d < LIKELY_STOP_DISTANCE_THRESHOLD else 0
            # is likely turn
            turn = 0 if abs(hc) < LIKELY_TURN_ANGLE_THRESHOLD else 1

            # ************ 2.SAVE CLEAN FEATURE ************
            speed_limit = SPEED_LIMIT if args.n_class == 5 else SPEED_LIMIT_7
            acc_limit = ACC_LIMIT if args.n_class == 5 else ACC_LIMIT_7
            # feature value in valid range
            if abs(v) <= speed_limit[trj_seg_label] and abs(a) <= acc_limit[trj_seg_label]:
                cn_delta_times.append(delta_t)
                cn_hours.append(hour)
                cn_distances.append(d)
                cn_velocities.append(v)
                cn_accelerations.append(a)
                cn_jerks.append(jk)
                cn_headings.append(h)
                cn_heading_changes.append(hc)
                cn_heading_change_rates.append(hcr)
                cn_timestamps.append(timestamps)

                prev_v = v
                prev_h = h
            else:
                ns_indices.append(j)

            # ************ 3.SAVE NOISE FEATURE ************
            # no matter if the value in valid range, we add all feature values into noise list
            ns_delta_times.append(delta_t)
            ns_hours.append(hour)
            ns_distances.append(d)
            ns_velocities.append(v)
            ns_accelerations.append(a)
            ns_jerks.append(jk)
            ns_headings.append(h)
            ns_heading_changes.append(hc)
            ns_heading_change_rates.append(hcr)
            ns_timestamps.append(timestamps)

        if len(cn_delta_times) < MIN_N_POINTS:
            # logger.info('feature element num not enough:{}'.format(len(cn_delta_times)))
            continue

        if args.n_class != 5:
            # SHL 7or8 classification label start with 1
            n_removed_points[trj_seg_label - 1] += len(ns_indices)
            n_total_points[trj_seg_label - 1] += n_points
        else:
            n_removed_points[trj_seg_label] += len(ns_indices)
            n_total_points[trj_seg_label] += n_points
        # ************ 4.SAVE NOISE AND CLEAN FEATURE ************
        """
        since noise points were removed, the n_points could be small, may cause performance decrease.
        here interp them to the size before noise removing
        """
        n_interp = n_points
        cn_multi_feature_seg = [interp_single_seg(cn_delta_times, n_interp),
                                interp_single_seg(cn_hours, n_interp),
                                interp_single_seg(cn_distances, n_interp),
                                interp_single_seg(cn_velocities, n_interp),
                                interp_single_seg(cn_accelerations, n_interp),
                                interp_single_seg(cn_jerks, n_interp),
                                interp_single_seg(cn_headings, n_interp),
                                interp_single_seg(cn_heading_changes, n_interp),
                                interp_single_seg(cn_heading_change_rates, n_interp),
                                interp_single_seg(cn_timestamps, n_interp)]
        # also interped, to make clean and noise have same size
        ns_multi_feature_seg = [interp_single_seg(ns_delta_times, n_interp),
                                interp_single_seg(ns_hours, n_interp),
                                interp_single_seg(ns_distances, n_interp),
                                interp_single_seg(ns_velocities, n_interp),
                                interp_single_seg(ns_accelerations, n_interp),
                                interp_single_seg(ns_jerks, n_interp),
                                interp_single_seg(ns_headings, n_interp),
                                interp_single_seg(ns_heading_changes, n_interp),
                                interp_single_seg(ns_heading_change_rates, n_interp),
                                interp_single_seg(ns_timestamps, n_interp)]

        ns_multi_feature_segs.append(ns_multi_feature_seg)
        cn_multi_feature_segs.append(cn_multi_feature_seg)
        multi_feature_seg_labels.append(trj_seg_label)

        # ************ 5.GENERATE MASK FOR CLEAN FEATURE SEG ************
        # note masks are generated from clean features, using change point detection algorithm
        fs_seg_mask = []
        for fs_seg in cn_multi_feature_seg[:-1]:
            msk = generate_mask_for_feature_using_EP(fs_seg, args.mean_mask_length)
            # msk = generate_mask_using_CPD(fs_seg, args.mask_ratio, args.mean_mask_length)
            # msk = generate_mask_using_CPD_unknown_CP_number(fs_seg, args.mean_mask_length)
            fs_seg_mask.append(msk)
        fs_seg_masks.append(fs_seg_mask)

        # ************ 6.SAVE NOISE AND CLEAN TRAJECTORY ************
        cn_trj_seg = interp_trj_seg(np.delete(trj_seg, ns_indices, axis=0), n_points)  # only keep lon, lat
        ns_trj_seg = np.delete(trj_seg, 0, 1)  # delete timestamps col
        cn_trj_segs.append([cn_trj_seg[:, 0], cn_trj_seg[:, 1]])  # convert to list to avoid numpy broadcast error
        ns_trj_segs.append([ns_trj_seg[:, 0], ns_trj_seg[:, 1]])

        # ************ 7.GENERATE MASK FOR CLEAN TRJ SEG ************
        # generate a mask seg by considering lat and lon SIMULTANEOUSLY
        # trj_seg_mask = generate_mask_using_CPD(cn_trj_seg, args.mask_ratio, args.mean_mask_length)
        # trj_seg_mask = generate_mask_using_CPD_unknown_CP_number(cn_trj_seg, args.mean_mask_length)
        trj_seg_mask = generate_mask_for_trj_using_KDE(cn_trj_seg, args.mean_mask_length)
        trj_seg_masks.append(trj_seg_mask)

        # generate a mask seg by considering lat and lon SEPARATELY
        # trj_seg_mask = [
        #     generate_mask_using_CPD(cn_trj_seg[:, 0], args.mask_ratio, args.mean_mask_length),
        #     generate_mask_using_CPD(cn_trj_seg[:, 1], args.mask_ratio, args.mean_mask_length),
        # ]
        # trj_seg_masks.append(trj_seg_mask)

    logger.info('* end a thread calc feature')
    return \
        np.array(ns_trj_segs, dtype=object), \
        np.array(cn_trj_segs, dtype=object), \
        np.array(fs_seg_masks, dtype=object), \
        np.array(trj_seg_masks, dtype=object), \
        np.array(ns_multi_feature_segs, dtype=object), \
        np.array(cn_multi_feature_segs, dtype=object), \
        np.array(multi_feature_seg_labels), \
        n_removed_points, \
        n_total_points


if __name__ == '__main__':

    t_start = time.time()

    n_threads = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=n_threads)
    logger.info(f'n_thread:{n_threads}')

    parser = argparse.ArgumentParser(description='TRJ_SEG_FEATURE')
    parser.add_argument('--trjs_path', type=str, required=True)
    parser.add_argument('--labels_path', type=str, required=True)
    parser.add_argument('--n_class', type=int, default=5)  # use modes:2,4,6,5,7
    parser.add_argument('--save_dir', type=str, required=True)

    parser.add_argument('--mean_mask_length', type=float, default=3, )
    parser.add_argument('--mask_ratio', type=float, default=0.3)

    args = parser.parse_args()

    # raw data
    trjs = np.load(args.trjs_path, allow_pickle=True)
    labels = np.load(args.labels_path, allow_pickle=True)

    # preprocess
    trjs, labels = filter_error_gps_data(trjs, labels)
    trj_segs, trj_seg_labels = segment_on_long_stay_time(trjs, labels)
    logger.info(f'before handling imbalance data: trj_segs: {trj_segs.shape}, trj_seg_labels: {trj_seg_labels.shape}')

    # handle imbalance data
    # ada = ADASYN(random_state=10086)
    # trj_segs, trj_seg_labels = ada.fit_resample(trj_segs, trj_seg_labels.astype('int'))
    # logger.info(f'after handling imbalance data: trj_segs: {trj_segs.shape}, trj_seg_labels: {trj_seg_labels.shape}')

    # calc motion feature. n_removed_points, n_total_points are point number over class
    ns_trj_segs, \
    cn_trj_segs, \
    fs_seg_masks, \
    trj_seg_masks, \
    ns_multi_feature_segs, \
    cn_multi_feature_segs, \
    multi_feature_seg_labels, \
    n_removed_points, \
    n_total_points \
        = calc_feature(trj_segs, trj_seg_labels)
    logger.info('total n_points after segment_on_stay_point: {}'.format(np.sum([len(seg) for seg in trj_segs])))
    logger.info(f'n_removed_points after calc_feature: {np.sum(n_removed_points, axis=0)}')
    logger.info(f'n_total_points after calc_feature: {np.sum(n_total_points, axis=0)}')

    # ns_mf_with_trj_segs = np.concatenate([ns_trj_segs, ns_multi_feature_segs], axis=1)
    # cn_mf_with_trj_segs = np.concatenate([cn_trj_segs, cn_multi_feature_segs], axis=1)

    # save result
    logger.info(f'saving files to {args.save_dir}')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    np.save(f'{args.save_dir}/noise_trj_segs.npy', ns_trj_segs)
    np.save(f'{args.save_dir}/clean_trj_segs.npy', cn_trj_segs)
    np.save(f'{args.save_dir}/fs_seg_masks.npy', fs_seg_masks)
    np.save(f'{args.save_dir}/trj_seg_masks.npy', trj_seg_masks)
    np.save(f'{args.save_dir}/clean_multi_feature_segs.npy', cn_multi_feature_segs)
    np.save(f'{args.save_dir}/clean_multi_feature_seg_labels.npy', multi_feature_seg_labels)
    np.save(f'{args.save_dir}/noise_multi_feature_segs.npy', ns_multi_feature_segs)
    np.save(f'{args.save_dir}/noise_multi_feature_seg_labels.npy', multi_feature_seg_labels)
    # normalized_multi_feature_segs.to_pickle(f'{args.save_dir}/normalized_multi_feature_segs.pkl')
    # np.save(f'{args.save_dir}/multi_feature_seg_labels.npy',
    #         to_categorical(multi_feature_seg_labels, num_classes=args.n_class))  # labels to one-hot

    logger.info('Running time: %s Seconds' % (time.time() - t_start))
