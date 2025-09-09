import argparse
import multiprocessing
import os
import time
import math

import numpy as np
from geopy.distance import geodesic
from logzero import logger

from tmi.data_preprocess.s4_trajectory_feature_calculation_with_CPD import ACC_LIMIT, ACC_LIMIT_7, MIN_N_POINTS, SPEED_LIMIT, SPEED_LIMIT_7, calc_feature, filter_error_gps_data, segment_on_long_stay_time


def extract_clean_trajectories(trj_segs, trj_seg_labels, n_class):
    """提取干净的轨迹数据，保留时间戳"""
    logger.info('Extracting clean trajectories...')
    speed_limit = SPEED_LIMIT if n_class == 5 else SPEED_LIMIT_7
    acc_limit = ACC_LIMIT if n_class == 5 else ACC_LIMIT_7

    clean_trajectories = []
    clean_labels = []

    for trj_seg, trj_seg_label in zip(trj_segs, trj_seg_labels):
        n_points = len(trj_seg)
        if n_points < MIN_N_POINTS:
            continue

        noise_indices = []
        prev_v = 0

        for j in range(n_points - 1):
            p_a = [trj_seg[j][1], trj_seg[j][2]]
            p_b = [trj_seg[j + 1][1], trj_seg[j + 1][2]]
            t_a = trj_seg[j][0]
            t_b = trj_seg[j + 1][0]

            delta_t = t_b - t_a
            if delta_t <= 0:
                noise_indices.append(j + 1)
                continue

            try:
                d = geodesic(p_a, p_b).meters
                v = d / delta_t
                a = (v - prev_v) / delta_t

                if abs(v) > speed_limit[trj_seg_label] or abs(a) > acc_limit[trj_seg_label]:
                    noise_indices.append(j)
                else:
                    prev_v = v
            except:
                noise_indices.append(j)

        clean_trj_seg = np.delete(trj_seg, noise_indices, axis=0)
        if len(clean_trj_seg) >= MIN_N_POINTS:
            clean_trajectories.append(clean_trj_seg)
            clean_labels.append(trj_seg_label)

    logger.info(f'Extracted {len(clean_trajectories)} clean trajectories')
    return np.array(clean_trajectories, dtype=object), np.array(clean_labels)


def analyze_original_noise_patterns(trajectories, labels, n_class):
    """分析原始数据中的噪声模式"""
    logger.info('Analyzing original noise patterns...')

    speed_limit = SPEED_LIMIT if n_class == 5 else SPEED_LIMIT_7
    acc_limit = ACC_LIMIT if n_class == 5 else ACC_LIMIT_7

    # 收集噪声点的统计信息
    noise_velocities = []
    noise_accelerations = []
    noise_distances = []
    clean_velocities = []
    clean_accelerations = []
    clean_distances = []

    for trj, label in zip(trajectories, labels):
        if len(trj) < 2:
            continue

        prev_v = 0
        for j in range(len(trj) - 1):
            p_a = [trj[j][1], trj[j][2]]
            p_b = [trj[j + 1][1], trj[j + 1][2]]
            t_a = trj[j][0]
            t_b = trj[j + 1][0]

            delta_t = t_b - t_a
            if delta_t <= 0:
                continue

            try:
                d = geodesic(p_a, p_b).meters
                v = d / delta_t
                a = (v - prev_v) / delta_t

                # 判断是否为噪声点
                if abs(v) > speed_limit[label] or abs(a) > acc_limit[label]:
                    noise_velocities.append(v)
                    noise_accelerations.append(a)
                    noise_distances.append(d)
                else:
                    clean_velocities.append(v)
                    clean_accelerations.append(a)
                    clean_distances.append(d)
                    prev_v = v
            except:
                continue

    # 计算噪声特征的统计分布
    noise_stats = {
        'velocity_mean': np.mean(noise_velocities) if noise_velocities else 0,
        'velocity_std': np.std(noise_velocities) if noise_velocities else 1,
        'acceleration_mean': np.mean(noise_accelerations) if noise_accelerations else 0,
        'acceleration_std': np.std(noise_accelerations) if noise_accelerations else 1,
        'distance_mean': np.mean(noise_distances) if noise_distances else 0,
        'distance_std': np.std(noise_distances) if noise_distances else 1,
    }

    clean_stats = {
        'velocity_mean': np.mean(clean_velocities) if clean_velocities else 0,
        'velocity_std': np.std(clean_velocities) if clean_velocities else 1,
        'acceleration_mean': np.mean(clean_accelerations) if clean_accelerations else 0,
        'acceleration_std': np.std(clean_accelerations) if clean_accelerations else 1,
        'distance_mean': np.mean(clean_distances) if clean_distances else 0,
        'distance_std': np.std(clean_distances) if clean_distances else 1,
    }

    logger.info(
        f'Noise ratio in original data: {len(noise_velocities) / (len(noise_velocities) + len(clean_velocities)):.3f}')
    return noise_stats, clean_stats


def apply_drift_noise(trajectories, labels, intensity, random_seed=42, noise_stats=None):
    """基于原始数据噪声分布的改进漂移噪声"""
    logger.info(f'Applying improved drift noise with intensity {intensity}')
    np.random.seed(random_seed)

    # 选择要施加噪声的轨迹
    n_selected = int(len(trajectories) * intensity)
    n_selected = min(n_selected, len(trajectories))  # 确保不超过总数
    if n_selected == 0:
        logger.info('No trajectories selected for drift noise application')
        return np.array(trajectories, dtype=object), labels
    selected_indices = np.random.choice(
        len(trajectories), n_selected, replace=False)

    noisy_trajectories = [trj.copy() for trj in trajectories]

    for idx in selected_indices:
        trj = trajectories[idx]
        label = labels[idx]
        if len(trj) < 2:
            continue

        # 基于交通模式的噪声强度调整
        mode_noise_factors = {0: 0.5, 1: 0.7, 2: 1.0,
                              3: 1.2, 4: 1.0}  # walk, bike, bus, car, train
        if len(mode_noise_factors) == 5:
            noise_factor = mode_noise_factors.get(label, 1.0)
        else:
            noise_factor = 1.0

        # 计算轨迹的平均移动距离作为噪声基准
        distances = []
        for j in range(len(trj) - 1):
            p_a = [trj[j][1], trj[j][2]]
            p_b = [trj[j + 1][1], trj[j + 1][2]]
            try:
                d = geodesic(p_a, p_b).meters
                distances.append(d)
            except:
                continue

        if not distances:
            continue

        avg_distance = np.mean(distances)

        # 基于平均移动距离的自适应噪声强度
        # GPS误差通常为移动距离的5-15%
        base_noise_ratio = 0.05 + intensity * 0.10  # 5% to 15%
        noise_distance = avg_distance * base_noise_ratio * noise_factor

        # 转换为经纬度偏移
        avg_lat = np.mean(trj[:, 1])
        lat_noise = noise_distance / 111000  # 1度纬度约111km
        lon_noise = noise_distance / (111000 * math.cos(math.radians(avg_lat)))

        # 生成相关的噪声序列（模拟GPS漂移的连续性）
        n_points = len(trj)

        # 使用AR(1)模型生成相关噪声
        phi = 0.7  # 自相关系数
        lat_noise_seq = np.zeros(n_points)
        lon_noise_seq = np.zeros(n_points)

        lat_noise_seq[0] = np.random.normal(0, lat_noise)
        lon_noise_seq[0] = np.random.normal(0, lon_noise)

        for i in range(1, n_points):
            lat_noise_seq[i] = phi * lat_noise_seq[i-1] + \
                np.random.normal(0, lat_noise * math.sqrt(1 - phi**2))
            lon_noise_seq[i] = phi * lon_noise_seq[i-1] + \
                np.random.normal(0, lon_noise * math.sqrt(1 - phi**2))

        # 应用噪声
        for i in range(n_points):
            new_lat = trj[i, 1] + lat_noise_seq[i]
            new_lon = trj[i, 2] + lon_noise_seq[i]

            # 确保坐标有效
            new_lat = np.clip(new_lat, -90, 90)
            new_lon = np.clip(new_lon, -180, 180)

            noisy_trajectories[idx][i, 1] = new_lat
            noisy_trajectories[idx][i, 2] = new_lon

    logger.info(
        f'Applied improved drift noise to {len(selected_indices)} trajectories')
    return np.array(noisy_trajectories, dtype=object), labels


def apply_missing_noise(trajectories, labels, intensity, random_seed=42):
    """基于真实GPS信号丢失模式的改进缺失点噪声"""
    logger.info(
        f'Applying improved missing point noise with intensity {intensity}')
    np.random.seed(random_seed)

    # 选择要施加噪声的轨迹
    n_selected = int(len(trajectories) * intensity)
    n_selected = min(n_selected, len(trajectories))  # 确保不超过总数
    if n_selected == 0:
        logger.info('No trajectories selected for missing noise application')
        return np.array(trajectories, dtype=object), labels
    selected_indices = np.random.choice(
        len(trajectories), n_selected, replace=False)

    noisy_trajectories = [trj.copy() for trj in trajectories]

    for idx in selected_indices:
        trj = trajectories[idx]
        label = labels[idx]
        n_points = len(trj)
        if n_points <= MIN_N_POINTS:
            continue

        # 基于交通模式调整缺失率
        # 不同交通模式的GPS信号质量不同
        mode_missing_rates = {
            0: 0.05,  # walk - 户外，信号较好
            1: 0.08,  # bike - 户外，信号较好
            2: 0.15,  # bus - 可能经过隧道、高楼区
            3: 0.12,  # car - 可能经过隧道、地下通道
            4: 0.20   # train/subway - 经常在地下或隧道中
        }

        if len(mode_missing_rates) == 5:
            base_missing_rate = mode_missing_rates.get(label, 0.10)
        else:
            base_missing_rate = 0.10

        # 根据intensity调整缺失率
        missing_rate = base_missing_rate * \
            (1 + intensity * 2)  # intensity越高，缺失率越高
        missing_rate = min(missing_rate, 0.4)  # 最大不超过40%

        n_to_remove = int(n_points * missing_rate)
        max_removable = n_points - MIN_N_POINTS
        n_to_remove = min(n_to_remove, max_removable)

        if n_to_remove <= 0:
            continue

        # 模拟真实的GPS信号丢失模式
        # 1. 短暂丢失（1-3个点）- 70%
        # 2. 中等丢失（4-8个点）- 25%
        # 3. 长时间丢失（9+个点）- 5%

        segments_to_remove = []
        points_removed = 0

        while points_removed < n_to_remove:
            # 根据概率选择丢失类型
            rand = np.random.random()
            if rand < 0.70:  # 短暂丢失
                max_length = min(3, n_to_remove - points_removed)
                if max_length >= 1:
                    segment_length = np.random.randint(1, max_length + 1)
                else:
                    segment_length = 1
            elif rand < 0.95:  # 中等丢失
                max_length = min(8, n_to_remove - points_removed)
                if max_length >= 4:
                    segment_length = np.random.randint(4, max_length + 1)
                elif max_length >= 1:
                    segment_length = np.random.randint(1, max_length + 1)
                else:
                    segment_length = 1
            else:  # 长时间丢失
                max_length = min(15, n_to_remove - points_removed)
                if max_length >= 9:
                    segment_length = np.random.randint(9, max_length + 1)
                elif max_length >= 4:
                    segment_length = np.random.randint(4, max_length + 1)
                elif max_length >= 1:
                    segment_length = np.random.randint(1, max_length + 1)
                else:
                    segment_length = 1

            if segment_length <= 0:
                break

            max_start = n_points - segment_length
            if max_start <= 0:
                break

            start_idx = np.random.randint(0, max_start)
            end_idx = start_idx + segment_length

            # 检查重叠
            overlap = False
            for existing_start, existing_end in segments_to_remove:
                if not (end_idx <= existing_start or start_idx >= existing_end):
                    overlap = True
                    break

            if not overlap:
                segments_to_remove.append((start_idx, end_idx))
                points_removed += segment_length

        # 创建掩码
        keep_mask = np.ones(n_points, dtype=bool)
        for start_idx, end_idx in segments_to_remove:
            keep_mask[start_idx:end_idx] = False

        noisy_trajectories[idx] = trj[keep_mask]

    logger.info(
        f'Applied improved missing point noise to {len(selected_indices)} trajectories')
    return np.array(noisy_trajectories, dtype=object), labels


def apply_outlier_noise(trajectories, labels, intensity, random_seed=42, noise_stats=None):
    """添加异常值噪声（模拟GPS跳跃）"""
    logger.info(f'Applying outlier noise with intensity {intensity}')
    np.random.seed(random_seed)

    # 选择要施加噪声的轨迹
    n_selected = int(len(trajectories) * intensity)
    n_selected = min(n_selected, len(trajectories))  # 确保不超过总数
    if n_selected == 0:
        logger.info('No trajectories selected for outlier noise application')
        return np.array(trajectories, dtype=object), labels
    selected_indices = np.random.choice(
        len(trajectories), n_selected, replace=False)

    noisy_trajectories = [trj.copy() for trj in trajectories]

    for idx in selected_indices:
        trj = trajectories[idx]
        if len(trj) < 3:
            continue

        # 每个轨迹添加1-3个异常点
        max_outliers = min(4, len(trj) // 5 + 1)
        if max_outliers <= 1:
            n_outliers = 1
        else:
            n_outliers = np.random.randint(1, max_outliers)

        # 确保有足够的点可以选择
        available_indices = list(range(1, len(trj) - 1))
        if len(available_indices) == 0:
            continue

        n_outliers = min(n_outliers, len(available_indices))
        if n_outliers == 0:
            continue

        outlier_indices = np.random.choice(
            available_indices, n_outliers, replace=False)

        for outlier_idx in outlier_indices:
            # 计算周围点的平均位置
            prev_point = trj[outlier_idx - 1]
            next_point = trj[outlier_idx + 1]

            # 计算正常的插值位置
            normal_lat = (prev_point[1] + next_point[1]) / 2
            normal_lon = (prev_point[2] + next_point[2]) / 2

            # 添加大幅偏移（50-200米）
            offset_distance = np.random.uniform(50, 200)  # 米
            offset_angle = np.random.uniform(0, 2 * math.pi)

            # 转换为经纬度偏移
            lat_offset = (offset_distance * math.cos(offset_angle)) / 111000
            lon_offset = (offset_distance * math.sin(offset_angle)) / \
                (111000 * math.cos(math.radians(normal_lat)))

            # 应用偏移
            new_lat = normal_lat + lat_offset
            new_lon = normal_lon + lon_offset

            # 确保坐标有效
            new_lat = np.clip(new_lat, -90, 90)
            new_lon = np.clip(new_lon, -180, 180)

            noisy_trajectories[idx][outlier_idx, 1] = new_lat
            noisy_trajectories[idx][outlier_idx, 2] = new_lon

    logger.info(
        f'Applied outlier noise to {len(selected_indices)} trajectories')
    return np.array(noisy_trajectories, dtype=object), labels


def apply_mixed_noise(trajectories, labels, intensity, random_seed=42, noise_stats=None):
    """施加混合噪声（漂移+缺失+异常值）"""
    logger.info(f'Applying mixed noise with intensity {intensity}')

    # 分阶段施加不同类型的噪声，强度递减
    current_trajectories = trajectories
    current_labels = labels

    # 1. 漂移噪声（强度稍低）
    drift_trajectories, drift_labels = apply_drift_noise(
        current_trajectories, current_labels, intensity * 0.8, random_seed, noise_stats)

    # 2. 缺失点噪声（强度中等）
    missing_trajectories, missing_labels = apply_missing_noise(
        drift_trajectories, drift_labels, intensity * 0.6, random_seed + 1)

    # 3. 异常值噪声（强度较低）
    mixed_trajectories, mixed_labels = apply_outlier_noise(
        missing_trajectories, missing_labels, intensity * 0.3, random_seed + 2, noise_stats)

    return mixed_trajectories, mixed_labels


def validate_noise_distribution(original_trj_segs, noisy_trj_segs, labels, n_class, config_name):
    """验证生成的噪声数据与原始数据的分布差异"""
    logger.info(f'Validating noise distribution for {config_name}...')

    speed_limit = SPEED_LIMIT if n_class == 5 else SPEED_LIMIT_7
    acc_limit = ACC_LIMIT if n_class == 5 else ACC_LIMIT_7

    # 计算原始数据的噪声比例
    original_noise_count = 0
    original_total_count = 0

    for trj, label in zip(original_trj_segs, labels):
        if len(trj) < 2:
            continue
        prev_v = 0
        for j in range(len(trj) - 1):
            p_a = [trj[j][1], trj[j][2]]
            p_b = [trj[j + 1][1], trj[j + 1][2]]
            t_a = trj[j][0]
            t_b = trj[j + 1][0]

            delta_t = t_b - t_a
            if delta_t <= 0:
                continue

            try:
                d = geodesic(p_a, p_b).meters
                v = d / delta_t
                a = (v - prev_v) / delta_t

                original_total_count += 1
                if abs(v) > speed_limit[label] or abs(a) > acc_limit[label]:
                    original_noise_count += 1
                else:
                    prev_v = v
            except:
                continue

    # 计算噪声数据的噪声比例
    noisy_noise_count = 0
    noisy_total_count = 0

    for trj, label in zip(noisy_trj_segs, labels):
        if len(trj) < 2:
            continue
        prev_v = 0
        for j in range(len(trj) - 1):
            p_a = [trj[j][1], trj[j][2]]
            p_b = [trj[j + 1][1], trj[j + 1][2]]
            t_a = trj[j][0]
            t_b = trj[j + 1][0]

            delta_t = t_b - t_a
            if delta_t <= 0:
                continue

            try:
                d = geodesic(p_a, p_b).meters
                v = d / delta_t
                a = (v - prev_v) / delta_t

                noisy_total_count += 1
                if abs(v) > speed_limit[label] or abs(a) > acc_limit[label]:
                    noisy_noise_count += 1
                else:
                    prev_v = v
            except:
                continue

    original_noise_ratio = original_noise_count / max(original_total_count, 1)
    noisy_noise_ratio = noisy_noise_count / max(noisy_total_count, 1)

    logger.info(f'{config_name} - Original noise ratio: {original_noise_ratio:.3f}, '
                f'Generated noise ratio: {noisy_noise_ratio:.3f}, '
                f'Difference: {abs(noisy_noise_ratio - original_noise_ratio):.3f}')

    return {
        'original_noise_ratio': original_noise_ratio,
        'generated_noise_ratio': noisy_noise_ratio,
        'ratio_difference': abs(noisy_noise_ratio - original_noise_ratio)
    }


if __name__ == '__main__':
    t_start = time.time()

    # n_threads = multiprocessing.cpu_count()
    n_threads = 12
    pool = multiprocessing.Pool(processes=n_threads)
    logger.info(f'n_thread:{n_threads}')

    parser = argparse.ArgumentParser(description='GPS Noise Simulation')
    parser.add_argument('--trjs_path', type=str, required=True, default='../../data/SHL_features_sim_noise/train/')
    parser.add_argument('--labels_path', type=str, required=True)
    parser.add_argument('--n_class', type=int, default=5)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--mean_mask_length', type=int, default=4)
    parser.add_argument('--mask_ratio', type=float, default=0.15)
    parser.add_argument('--mask_mode', type=str,
                        default='ep', choices=['ep', 'random'])
    parser.add_argument('--trj_mask_mode', type=str,
                        default='kde', choices=['kde', 'random'])
    parser.add_argument('--kde_bw', type=float, default=1)
    parser.add_argument('--kde_kernel', type=str, default='epa')

    args = parser.parse_args()

    # 读取数据
    trjs = np.load(args.trjs_path, allow_pickle=True)
    labels = np.load(args.labels_path, allow_pickle=True)

    # 随机打乱
    shuffle_idx = np.random.permutation(len(trjs))
    trjs = trjs[shuffle_idx]
    labels = labels[shuffle_idx]

    # 预处理
    trjs, labels = filter_error_gps_data(trjs, labels, pool, n_threads)
    trj_segs, trj_seg_labels = segment_on_long_stay_time(trjs, labels, pool, n_threads)

    # 提取干净轨迹，具有时间戳！
    clean_trj_segs, clean_labels = extract_clean_trajectories(
        trj_segs, trj_seg_labels, args.n_class)

    # 分析原始数据的噪声模式
    logger.info('Analyzing noise patterns in original data...')
    noise_stats, clean_stats = analyze_original_noise_patterns(
        trj_segs, trj_seg_labels, args.n_class)

    # 保存干净数据和统计信息
    clean_dir = os.path.join(args.save_dir, 'clean_original')
    os.makedirs(clean_dir, exist_ok=True)
    np.save(os.path.join(
        clean_dir, 'clean_trj_segs_with_timestamps.npy'), clean_trj_segs)
    np.save(os.path.join(clean_dir, 'clean_trj_seg_labels.npy'), clean_labels)

    # 保存噪声统计信息
    import json
    with open(os.path.join(clean_dir, 'noise_statistics.json'), 'w') as f:
        json.dump({'noise_stats': noise_stats,
                  'clean_stats': clean_stats}, f, indent=2)

    # 扩展的噪声配置
    noise_configs = [
        ('drift', 0.0), ('drift', 0.1), ('drift', 0.2), ('drift', 0.3),
        ('missing', 0.0), ('missing', 0.1), ('missing', 0.2), ('missing', 0.3),
        ('outlier', 0.0), ('outlier', 0.1), ('outlier', 0.2), ('outlier', 0.3),
        ('mixed', 0.0), ('mixed', 0.1), ('mixed', 0.2), ('mixed', 0.3)
    ]

    # 处理每种噪声配置
    for noise_type, intensity in noise_configs:
        config_name = f'{noise_type}_{int(intensity * 100)}_percent'
        logger.info(f'Processing {config_name}...')

        # 施加噪声（返回的轨迹都是包含时间戳的）
        if noise_type == 'drift':
            noisy_trj_segs, noisy_labels = apply_drift_noise(clean_trj_segs, clean_labels, intensity,
                                                             random_seed=42, noise_stats=noise_stats)
        elif noise_type == 'missing':
            noisy_trj_segs, noisy_labels = apply_missing_noise(clean_trj_segs, clean_labels, intensity,
                                                               random_seed=42)
        elif noise_type == 'outlier':
            noisy_trj_segs, noisy_labels = apply_outlier_noise(clean_trj_segs, clean_labels, intensity,
                                                               random_seed=42, noise_stats=noise_stats)
        elif noise_type == 'mixed':
            noisy_trj_segs, noisy_labels = apply_mixed_noise(clean_trj_segs, clean_labels, intensity,
                                                             random_seed=42, noise_stats=noise_stats)

        # 验证噪声分布
        validation_results = validate_noise_distribution(clean_trj_segs, noisy_trj_segs, noisy_labels,
                                                         args.n_class, config_name)

        

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
            = calc_feature(noisy_trj_segs, noisy_labels, pool, n_threads, args)
        logger.info('total n_points after segment_on_stay_point: {}'.format(np.sum([len(seg) for seg in trj_segs])))
        logger.info(f'n_removed_points after calc_feature: {np.sum(n_removed_points, axis=0)}')
        logger.info(f'n_total_points after calc_feature: {np.sum(n_total_points, axis=0)}')


        # 保存结果
        noise_type_dir = os.path.join(args.save_dir, f'{config_name}')
        os.makedirs(noise_type_dir, exist_ok=True)
        logger.info(f'saving files to {noise_type_dir}')
        np.save(f'{noise_type_dir}/noise_trj_segs.npy', ns_trj_segs)
        np.save(f'{noise_type_dir}/clean_trj_segs.npy', cn_trj_segs)
        np.save(f'{noise_type_dir}/fs_seg_masks.npy', fs_seg_masks)
        np.save(f'{noise_type_dir}/trj_seg_masks.npy', trj_seg_masks)
        np.save(f'{noise_type_dir}/clean_multi_feature_segs.npy', cn_multi_feature_segs)
        np.save(f'{noise_type_dir}/clean_multi_feature_seg_labels.npy', multi_feature_seg_labels)
        np.save(f'{noise_type_dir}/noise_multi_feature_segs.npy', ns_multi_feature_segs)
        np.save(f'{noise_type_dir}/noise_multi_feature_seg_labels.npy', multi_feature_seg_labels)

        # 保存验证结果
        import json
        with open(os.path.join(noise_type_dir, f'{config_name}_validation.json'), 'w') as f:
            json.dump(validation_results, f, indent=2)

        logger.info(f'noise ratio difference: {validation_results["ratio_difference"]:.3f}')

        print('#'*60)

    # 生成噪声模拟总结报告
    logger.info('Generating noise simulation summary...')
    summary_report = {
        'total_clean_trajectories': len(clean_trj_segs),
        'noise_configurations': len(noise_configs),
        'processing_time_seconds': time.time() - t_start,
        'improvements': [
            'Adaptive drift noise based on trajectory characteristics',
            'Transportation mode-specific missing point patterns',
            'Realistic GPS outlier simulation',
            'Mixed noise with controlled intensity',
            'Distribution validation against original data'
        ]
    }

    with open(os.path.join(args.save_dir, 'simulation_summary.json'), 'w') as f:
        json.dump(summary_report, f, indent=2)

    pool.close()
    pool.join()

    logger.info('✓ Noise simulation completed successfully!')
    logger.info(f'✓ Generated {len(noise_configs)} noise configurations')
    logger.info(f'✓ Processing time: {time.time() - t_start:.2f} seconds')
    logger.info(f'✓ Results saved to: {args.save_dir}')
