import os
import argparse
import numpy as np
from copy import deepcopy
from collections import Counter
from logzero import logger
from scipy.interpolate import interp1d

#https://github.com/xuehaouwa/Trajectory-Prediction-Tools/blob/master/pytp/utils/augment.py

class TrajectoryAugmentation:
    @staticmethod
    def reverse(traj):
        """反转单条轨迹,保持timestamp不变
        Args:
            traj: 单条轨迹 shape=(T,3)
        Returns:
            reversed_traj: 反转后的轨迹
        """
        reversed_traj = deepcopy(traj)
        # 保持timestamp不变
        timestamps = traj[:,0]
        # 反转空间坐标
        reversed_traj = np.flip(reversed_traj, 0)
        reversed_traj[:,0] = timestamps
        return reversed_traj
    
    @staticmethod
    def random_rotate(traj):
        """随机旋转单条轨迹的空间坐标,如果旋转后经纬度越界则重试
        Args:
            traj: 单条轨迹 shape=(T,3)
        Returns:
            rotated_traj: 旋转后的轨迹
        """
        # 保存原始轨迹用于失败时返回
        original_traj = deepcopy(traj)
        max_attempts = 10  # 最大重试次数
        attempt = 0
        
        while attempt < max_attempts:
            rotated = np.zeros_like(traj)
            # 复制timestamp
            rotated[:,0] = traj[:,0]
            
            # 只旋转lat,lon坐标
            angle = np.random.rand() * 2 * np.pi
            cangle, sangle = np.cos(angle), np.sin(angle)
            rot_mat = np.array([[cangle, -sangle], [sangle, cangle]])
            spatial_coords = traj[:,1:]
            rotated[:,1:] = np.matmul(rot_mat, spatial_coords.T).T
            
            # 检查经纬度是否在合法范围内
            lat, lon = rotated[:,1], rotated[:,2]
            if np.all((-90 <= lat) & (lat <= 90) & (-180 <= lon) & (lon <= 180)):
                return rotated
            
            attempt += 1
        
        # 如果多次重试都失败,返回原始轨迹
        logger.warning("旋转后经纬度多次越界,返回原始轨迹")
        return original_traj
    
    @staticmethod
    def spatial_translate(traj, translate_range=0.001):
        """随机平移空间坐标
        Args:
            traj: 单条轨迹 shape=(T,3)
            translate_range: 最大平移范围(经纬度)
        Returns:
            translated_traj: 平移后的轨迹
        """
        translated = deepcopy(traj)
        dx = np.random.uniform(-translate_range, translate_range)
        dy = np.random.uniform(-translate_range, translate_range)
        translated[:,1] += dx
        translated[:,2] += dy
        return translated

    @staticmethod
    def time_warp(traj, num_warps=4, sigma=0.2):
        """对轨迹进行时间扭曲，通过在时间轴上添加随机扰动点实现
        Args:
            traj: 单条轨迹 shape=(T,3)
            num_warps: 扰动点数量
            sigma: 扰动强度
        Returns:
            warped_traj: 时间扭曲后的轨迹
        """
        warped = deepcopy(traj)
        T = len(traj)
        
        # 原始时间序列
        orig_time = np.linspace(0, 1, T)
        
        # 生成扰动点
        warp_points = np.random.choice(T-2, num_warps, replace=False) + 1
        warp_points.sort()
        
        # 添加起点和终点
        warp_points = np.concatenate([[0], warp_points, [T-1]])
        
        # 生成新的时间轴
        warped_time = deepcopy(orig_time)
        for start, mid, end in zip(warp_points[:-2], warp_points[1:-1], warp_points[2:]):
            # 对中间点添加随机扰动
            mid_time = orig_time[mid]
            mid_time += np.random.normal(0, sigma) * (orig_time[end] - orig_time[start])
            # 确保时间单调递增
            mid_time = np.clip(mid_time, orig_time[start], orig_time[end])
            warped_time[mid] = mid_time
            
        # 使用三次样条插值
        for i in [1, 2]:  # 分别处理经度和纬度
            interpolator = interp1d(warped_time, traj[:, i], kind='cubic')
            warped[:, i] = interpolator(orig_time)
            
        return warped

    @staticmethod
    def random_crop(traj, min_points=10):
        """随机裁剪轨迹片段
        Args:
            traj: 单条轨迹 shape=(T,3)
            min_points: 最小保留点数
        Returns:
            cropped_traj: 裁剪后的轨迹
        """
        T = len(traj)
        if T <= min_points:
            return traj
            
        # 随机选择裁剪长度
        crop_length = np.random.randint(min_points, T+1)
        # 随机选择起始位置
        start = np.random.randint(0, T - crop_length + 1)
        
        cropped = traj[start:start+crop_length]
        
        # 调整时间戳，使起点时间为0
        t0 = cropped[0,0]
        cropped[:,0] = cropped[:,0] - t0
        
        return cropped
    
    @staticmethod
    def swap_coords(traj):
        """交换单条轨迹的lat和lon坐标
        Args:
            traj: 单条轨迹 shape=(T,3)
        Returns:
            new_traj: 交换坐标后的轨迹
        """
        new_traj = deepcopy(traj)
        # 交换lat,lon,保持timestamp不变
        new_traj[:,1] = traj[:,2]
        new_traj[:,2] = traj[:,1]
        return new_traj
    
    @staticmethod
    def augment_minority_classes(trajs, labels, target_count=None):
        """对少数类进行样本增强
        Args:
            trajs: 轨迹数据列表 [array(Ti,3), ...]
            labels: 标签数据 shape=(N,)
            target_count: 每个类别的目标数量,默认为最多类的数量
        Returns:
            augmented_trajs: 增强后的轨迹
            augmented_labels: 增强后的标签
        """
        label_counts = Counter(labels)
        if target_count is None:
            target_count = max(label_counts.values())
            
        augmented_trajs = []
        augmented_labels = []
        
        # 对每个类别进行增强
        for label in label_counts:
            idx = labels == label
            class_trajs = [trajs[i] for i in np.where(idx)[0]]
            count = len(class_trajs)
            
            # 添加原始轨迹
            augmented_trajs.extend(class_trajs)
            augmented_labels.extend([label] * count)
            
            # 如果数量少于目标数量,进行增强
            if count < target_count:
                n_aug = target_count - count
                
                for _ in range(n_aug):
                    # 随机选择一个轨迹
                    idx = np.random.randint(count)
                    traj = class_trajs[idx]
                    
                    # 随机选择1-3种增强方法组合使用
                    n_methods = np.random.randint(1, 5)
                    aug = deepcopy(traj)
                    
                    # 可用的增强方法列表
                    methods = [
                        TrajectoryAugmentation.reverse,
                        TrajectoryAugmentation.random_rotate,
                        TrajectoryAugmentation.spatial_translate,
                        # TrajectoryAugmentation.time_warp,  # 时间扭曲会导致运动行为变化，不合适
                        TrajectoryAugmentation.random_crop,
                        # TrajectoryAugmentation.swap_coords  # 交换坐标会导致经纬度超出范围，不合适
                    ]
                    
                    # 随机选择并应用增强方法
                    selected_methods = np.random.choice(methods, n_methods, replace=False)
                    for method in selected_methods:
                        aug = method(aug)
                        
                    augmented_trajs.append(aug)
                    augmented_labels.append(label)
                    
        return np.array(augmented_trajs, dtype=object), np.array(augmented_labels)

# 使用示例
def augment_trajectory_data(trjs, labels):
    """对轨迹数据进行增强
    Args:
        trjs: 原始轨迹数据 numpy array(dtype=object) [array(Ti,3), ...]
        labels: 标签数据 numpy array shape=(N,)
    Returns:
        augmented_trjs: 增强后的轨迹数据
        augmented_labels: 增强后的标签
    """
    augmentor = TrajectoryAugmentation()
    return augmentor.augment_minority_classes(trjs, labels)

def print_label_distribution(labels):
    """打印标签分布情况"""
    counter = Counter(labels)
    logger.info("类别分布:")
    for label, count in sorted(counter.items()):
        logger.info(f"类别 {label}: {count} 样本")

def main():
    parser = argparse.ArgumentParser(description='轨迹数据增强')
    parser.add_argument('--data_dir', type=str, default='../../data/SHL_split/',
                      help='包含训练集的数据目录')
    parser.add_argument('--save_dir', type=str, default='../../data/SHL_augmented/',
                      help='增强后数据的保存目录')
    parser.add_argument('--target_count', type=int, default=None,
                      help='每个类别的目标样本数量,默认使用最多的类别数量')
    parser.add_argument('--random_seed', type=int, default=42,
                      help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.random_seed)
    
    # 创建保存目录
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # 加载训练数据
    logger.info(f'从 {args.data_dir} 加载训练数据...')
    train_trjs = np.load(os.path.join(args.data_dir, 'train_trjs.npy'), allow_pickle=True)
    train_labels = np.load(os.path.join(args.data_dir, 'train_labels.npy'), allow_pickle=True)
    
    # 打印原始分布
    logger.info('原始训练集分布:')
    print_label_distribution(train_labels)
    
    # 数据增强
    logger.info('开始数据增强...')
    augmented_trjs, augmented_labels = augment_trajectory_data(train_trjs, train_labels)
    
    # 打印增强后的分布
    logger.info('增强后的数据分布:')
    print_label_distribution(augmented_labels)
    
    # 保存增强后的数据
    logger.info(f'保存增强后的数据到 {args.save_dir}')
    np.save(os.path.join(args.save_dir, 'train_trjs_augmented.npy'), augmented_trjs)
    np.save(os.path.join(args.save_dir, 'train_labels_augmented.npy'), augmented_labels)
    
    # 复制测试集
    logger.info('复制测试集...')
    test_trjs = np.load(os.path.join(args.data_dir, 'test_trjs.npy'), allow_pickle=True)
    test_labels = np.load(os.path.join(args.data_dir, 'test_labels.npy'), allow_pickle=True)
    np.save(os.path.join(args.save_dir, 'test_trjs.npy'), test_trjs)
    np.save(os.path.join(args.save_dir, 'test_labels.npy'), test_labels)
    
    # 输出完成信息
    logger.info(f'数据增强完成!')
    logger.info(f'原始训练集大小: {len(train_trjs)}')
    logger.info(f'增强后训练集大小: {len(augmented_trjs)}')
    logger.info(f'测试集大小: {len(test_trjs)}')

if __name__ == '__main__':
    main()





"""
(tmi) (tmi) lsc@lsc-mortar:~/pyprojs/tmi_mvts_transformer/tmi/data_preprocess$ /home/lsc/anaconda3/envs/tmi/bin/python /home/lsc/pyprojs/tmi_mvts_transformer/tmi/data_preprocess/s3_data_augmentation.py
[I 250405 22:53:06 s3_data_augmentation:262] 从 ../../data/geolife_split/ 加载训练数据...
[I 250405 22:53:06 s3_data_augmentation:267] 原始训练集分布:
[I 250405 22:53:06 s3_data_augmentation:237] 类别分布:
[I 250405 22:53:06 s3_data_augmentation:239] 类别 0: 5168 样本
[I 250405 22:53:06 s3_data_augmentation:239] 类别 1: 1671 样本
[I 250405 22:53:06 s3_data_augmentation:239] 类别 2: 2282 样本
[I 250405 22:53:06 s3_data_augmentation:239] 类别 3: 1737 样本
[I 250405 22:53:06 s3_data_augmentation:239] 类别 4: 890 样本
[I 250405 22:53:06 s3_data_augmentation:271] 开始数据增强...
[W 250405 22:53:06 s3_data_augmentation:61] 旋转后经纬度多次越界,返回原始轨迹
[W 250405 22:53:06 s3_data_augmentation:61] 旋转后经纬度多次越界,返回原始轨迹
[W 250405 22:53:06 s3_data_augmentation:61] 旋转后经纬度多次越界,返回原始轨迹
[W 250405 22:53:06 s3_data_augmentation:61] 旋转后经纬度多次越界,返回原始轨迹
[I 250405 22:53:06 s3_data_augmentation:275] 增强后的数据分布:
[I 250405 22:53:06 s3_data_augmentation:237] 类别分布:
[I 250405 22:53:06 s3_data_augmentation:239] 类别 0: 5168 样本
[I 250405 22:53:06 s3_data_augmentation:239] 类别 1: 5168 样本
[I 250405 22:53:06 s3_data_augmentation:239] 类别 2: 5168 样本
[I 250405 22:53:06 s3_data_augmentation:239] 类别 3: 5168 样本
[I 250405 22:53:06 s3_data_augmentation:239] 类别 4: 5168 样本
[I 250405 22:53:06 s3_data_augmentation:279] 保存增强后的数据到 ../../data/geolife_augmented/
[I 250405 22:53:07 s3_data_augmentation:284] 复制测试集...
[I 250405 22:53:07 s3_data_augmentation:291] 数据增强完成!
[I 250405 22:53:07 s3_data_augmentation:292] 原始训练集大小: 11748
[I 250405 22:53:07 s3_data_augmentation:293] 增强后训练集大小: 25840
[I 250405 22:53:07 s3_data_augmentation:294] 测试集大小: 2938
(tmi) (tmi) lsc@lsc-mortar:~/pyprojs/tmi_mvts_transformer/tmi/data_preprocess$ /home/lsc/anaconda3/envs/tmi/bin/python /home/lsc/pyprojs/tmi_mvts_transformer/tmi/data_preprocess/s3_data_augmentation.py
cd /home/lsc/pyprojs/tmi_mvts_transformer/tmi/data_preprocess
[I 250405 22:53:19 s3_data_augmentation:262] 从 ../../data/SHL_split/ 加载训练数据...
[I 250405 22:53:19 s3_data_augmentation:267] 原始训练集分布:
[I 250405 22:53:19 s3_data_augmentation:237] 类别分布:
[I 250405 22:53:19 s3_data_augmentation:239] 类别 0: 260 样本
[I 250405 22:53:19 s3_data_augmentation:239] 类别 2: 113 样本
[I 250405 22:53:19 s3_data_augmentation:239] 类别 3: 115 样本
[I 250405 22:53:19 s3_data_augmentation:271] 开始数据增强...
[I 250405 22:53:19 s3_data_augmentation:275] 增强后的数据分布:
[I 250405 22:53:19 s3_data_augmentation:237] 类别分布:
[I 250405 22:53:19 s3_data_augmentation:239] 类别 0: 260 样本
[I 250405 22:53:19 s3_data_augmentation:239] 类别 2: 260 样本
[I 250405 22:53:19 s3_data_augmentation:239] 类别 3: 260 样本
[I 250405 22:53:19 s3_data_augmentation:279] 保存增强后的数据到 ../../data/SHL_augmented/
[I 250405 22:53:19 s3_data_augmentation:284] 复制测试集...
[I 250405 22:53:19 s3_data_augmentation:291] 数据增强完成!
[I 250405 22:53:19 s3_data_augmentation:292] 原始训练集大小: 488
[I 250405 22:53:19 s3_data_augmentation:293] 增强后训练集大小: 780
[I 250405 22:53:19 s3_data_augmentation:294] 测试集大小: 123
"""