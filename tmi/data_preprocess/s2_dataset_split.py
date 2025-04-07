import argparse
import os
import numpy as np
from sklearn.model_selection import train_test_split
from logzero import logger

def split_dataset(data_path, save_path, test_size=0.2, random_state=42):
    """
    将数据集划分为训练集和测试集
    
    Args:
        data_path: 数据所在目录
        save_path: 保存目录
        test_size: 测试集比例
        random_state: 随机种子
    """
    # 加载数据
    logger.info(f'正在从 {data_path} 加载数据...')
    trjs = np.load(os.path.join(data_path, 'trjs.npy'), allow_pickle=True)
    labels = np.load(os.path.join(data_path, 'labels.npy'), allow_pickle=True)
    
    # 划分数据集
    logger.info(f'划分数据集,测试集比例: {test_size}')
    X_train, X_test, y_train, y_test = train_test_split(
        trjs, labels, 
        test_size=test_size,
        random_state=42, 
        stratify=labels  # 保持标签分布一致
    )
    
    # 创建保存目录
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    # 保存数据集
    logger.info(f'保存数据集到 {save_path}')
    np.save(os.path.join(save_path, 'train_trjs.npy'), X_train)
    np.save(os.path.join(save_path, 'train_labels.npy'), y_train)
    np.save(os.path.join(save_path, 'test_trjs.npy'), X_test)
    np.save(os.path.join(save_path, 'test_labels.npy'), y_test)
    
    # 输出数据集信息
    logger.info(f'训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}')
    logger.info('数据集划分完成!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='数据集划分')
    parser.add_argument('--data_dir', type=str, default='../../data/SHL_extracted/', help='原始数据目录')
    parser.add_argument('--save_dir', type=str, default='../../data/SHL_split/', help='保存目录')
    parser.add_argument('--test_size', type=float, default=0.2, help='测试集比例')
    
    args = parser.parse_args()
    
    split_dataset(
        args.data_dir,
        args.save_dir,
        test_size=args.test_size,
    )