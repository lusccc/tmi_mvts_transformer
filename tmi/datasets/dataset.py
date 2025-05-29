import os
import time
import random

import numpy as np
import torch
from torch.utils.data import Dataset
import logzero
from logzero import logger

config = None

random.seed(10086)


def collate_generic_superv(data, max_len=None):
    """Build mini-batch tensors from a list of (X, mask) tuples. Mask input. Create
    Args:
        data: len(batch_size) list of tuples (X, y).
            - X: torch tensor of shape (seq_length, feat_dim); variable seq_length.
            - y: torch tensor of shape (num_labels,) : class indices or numerical targets
                (for classification or regression, respectively). num_labels > 1 for multi-task models
        max_len: global fixed sequence length. Used for architectures requiring fixed length input,
            where the batch length cannot vary dynamically. Longer sequences are clipped, shorter are padded with 0s
    Returns:
        X: (batch_size, padded_length, feat_dim) torch tensor of masked features (input)
        targets: (batch_size, padded_length, feat_dim) torch tensor of unmasked features (output)
        target_masks: (batch_size, padded_length, feat_dim) boolean torch tensor
            0 indicates masked values to be predicted, 1 indicates unaffected/"active" feature values
        padding_masks: (batch_size, padded_length) boolean tensor, 1 means keep vector at this position, 0 means padding
    """

    batch_size = len(data)
    noise_features, labels, IDs = zip(*data)

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in noise_features]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)
    X = torch.zeros(batch_size, max_len, noise_features[0].shape[-1])  # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = noise_features[i][:end, :]

    targets = torch.stack(labels, dim=0)  # (batch_size, num_labels)

    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16),
                                 max_len=max_len)  # (batch_size, padded_length) boolean tensor, "1" means keep

    return [X, targets, padding_masks, IDs]


def collate_dual_branch_superv(data, ):
    batch_size = len(data)
    noise_features_1, noise_features_2, labels, IDs = zip(*data)

    X1_X2 = []
    pm1_pm2 = []
    for noise_features in [noise_features_1, noise_features_2]:
        # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
        lengths = [X.shape[0] for X in noise_features]  # original sequence length for each time series
        max_len = max(lengths)
        X = torch.zeros(batch_size, max_len, noise_features[0].shape[-1])  # (batch_size, padded_length, feat_dim)
        for i in range(batch_size):
            end = min(lengths[i], max_len)
            X[i, :end, :] = noise_features[i][:end, :]

        padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16),
                                     max_len=max_len)  # (batch_size, padded_length) boolean tensor, "1" means keep
        X1_X2.append(X)
        pm1_pm2.append(padding_masks)

    targets = torch.stack(labels, dim=0)  # (batch_size, num_labels)

    return [*X1_X2, *pm1_pm2, targets, IDs]


def compensate_masking(X, mask):
    """
    Compensate feature vectors after masking values, in a way that the matrix product W @ X would not be affected on average.
    If p is the proportion of unmasked (active) elements, X' = X / p = X * feat_dim/num_active
    Args:
        X: (batch_size, seq_length, feat_dim) torch tensor
        mask: (batch_size, seq_length, feat_dim) torch tensor: 0s means mask and predict, 1s: unaffected (active) input
    Returns:
        (batch_size, seq_length, feat_dim) compensated features
    """

    # number of unmasked elements of feature vector for each time step
    num_active = torch.sum(mask, dim=-1).unsqueeze(-1)  # (batch_size, seq_length, 1)
    # to avoid division by 0, set the minimum to 1
    num_active = torch.max(num_active, torch.ones(num_active.shape, dtype=torch.int16))  # (batch_size, seq_length, 1)
    return X.shape[-1] * X / num_active


def collate_imputation_unsuperv(data, ):
    t_start = time.time()

    batch_size = len(data)
    noise_features, clean_features, masks, IDs = zip(*data)

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in noise_features]  # original sequence length for each time series
    max_len = max(lengths)

    X_noise = torch.zeros(batch_size, max_len, noise_features[0].shape[-1])  # (batch_size, padded_length, feat_dim)
    X_clean = torch.zeros(batch_size, max_len, clean_features[0].shape[-1])  # (batch_size, padded_length, feat_dim)

    # target_masks: (batch_size, padded_length, feat_dim) masks related to objective
    target_masks = torch.zeros_like(X_noise, dtype=torch.bool)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X_noise[i, :end, :] = noise_features[i][:end, :]
        X_clean[i, :end, :] = clean_features[i][:end, :]
        target_masks[i, :end, :] = masks[i][:end, :]
    
    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16),
                                 max_len=max_len)  # (batch_size, padded_length) boolean tensor, "1" means keep
    # target masks mean the mask impose on y_pred, here is X_clean
    # inverse logic: 0 now means ignore, 1 means predict
    # 仅在有效区域内取反，padding区域保持为0
    target_masks = ~target_masks & padding_masks.unsqueeze(-1)

    # logger.info('collate_imputation_unsuperv time: %s Seconds' % (time.time() - t_start))

    return X_noise, X_clean, target_masks, padding_masks, IDs


def collate_denoising_unsuperv(data, max_len=None, ):
    t_start = time.time()

    batch_size = len(data)
    noise_features, clean_features, IDs = zip(*data)

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in noise_features]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)

    X_noise = torch.zeros(batch_size, max_len, noise_features[0].shape[-1])  # (batch_size, padded_length, feat_dim)
    X_clean = torch.zeros(batch_size, max_len, clean_features[0].shape[-1])  # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X_noise[i, :end, :] = noise_features[i][:end, :]
        X_clean[i, :end, :] = clean_features[i][:end, :]

    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16),
                                 max_len=max_len)  # (batch_size, padded_length) boolean tensor, "1" means keep
    target_masks = padding_masks.unsqueeze(-1).repeat(1, 1, clean_features[0].shape[-1])  # (batch_size, padded_length, feat_dim)
    
    # logger.info('collate_denoising_unsuperv time: %s Seconds' % (time.time() - t_start))

    return X_noise, X_clean, target_masks, padding_masks, IDs


def padding_mask(lengths, max_len=None):
    """
    Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
    where 1 means keep element at this position (time step)
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max_val()  # trick works because of overloading of 'or' operator for non-boolean types
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


def parse_input_type(input_type):
    """解析input_type配置,返回noise的概率
    Args:
        input_type: '10%noise' 等百分比形式的噪声配置
    Returns:
        noise概率 (0-1之间的浮点数)
    """
    if isinstance(input_type, str) and input_type.endswith('%noise'):
        try:
            percentage = float(input_type[:-6])
            return percentage / 100
        except:
            raise
    return 1


class GenericClassificationDataset(Dataset):
    def __init__(self, data, indices):
        super(GenericClassificationDataset, self).__init__()
        
        t_start = time.time()

        self.data = data  # this is a subclass of the BaseData class in data.py
        self.IDs = indices  # list of data IDs, but also mapping between integer index and ID
        
        # 预先加载所有数据到字典中，避免在__getitem__中使用.loc
        self.noise_features = {}
        self.clean_features = {}
        self.labels = {}
        
        for ID in self.IDs:
            self.noise_features[ID] = data.noise_feature_df.loc[ID].values.copy()
            self.clean_features[ID] = data.clean_feature_df.loc[ID].values.copy()
            self.labels[ID] = data.labels_df.loc[ID].values.copy()
        
        # 保留原始DataFrame引用以保持兼容性
        self.noise_feature_df = self.data.noise_feature_df.loc[self.IDs]
        self.clean_feature_df = self.data.clean_feature_df.loc[self.IDs]
        self.labels_df = self.data.labels_df.loc[self.IDs]
        
        # 预计算常用值
        self.noise_prob = parse_input_type(config['input_type'])
        
        logger.info('GenericClassificationDataset __init__ time: %s Seconds' % (time.time() - t_start))

    def update_noise_prob(self, new_prob):
        """更新数据集的噪声概率，用于noise level sweep测试"""
        self.noise_prob = new_prob

    def __getitem__(self, ind):
        """
        For a given integer index, returns the corresponding (seq_length, feat_dim) array and a noise mask of same shape
        Args:
            ind: integer index of sample in dataset
        Returns:
            X: (seq_length, feat_dim) tensor of the multivariate time series corresponding to a sample
            y: (num_labels,) tensor of labels (num_labels > 1 for multi-task models) for each sample
            ID: ID of sample
        """
        # ind = int(ind / 2)
        ID = self.IDs[ind]
        
        # 直接从缓存字典获取数据
        X_noise = self.noise_features[ID]
        X_clean = self.clean_features[ID]
        y = self.labels[ID]

        # 使用概率来决定是否使用noise数据
        noise_input = random.random() < self.noise_prob

        if noise_input:
            input = X_noise
        else:
            input = X_clean
        target = y

        return torch.from_numpy(input), torch.from_numpy(target), ID

    def __len__(self):
        # return len(self.IDs) * 2
        return len(self.IDs)

class DenoisingDataset(Dataset):
    def __init__(self, data, indices, exclude_feats=None):
        super(DenoisingDataset, self).__init__()

        t_start = time.time()

        self.data = data  # this is a subclass of the BaseData class in data.py
        self.IDs = indices  # list of data IDs, but also mapping between integer index and ID
        
        # 预先加载所有数据到字典中
        self.noise_features = {}
        self.clean_features = {}
        
        for ID in self.IDs:
            self.noise_features[ID] = data.noise_feature_df.loc[ID].values.copy()
            self.clean_features[ID] = data.clean_feature_df.loc[ID].values.copy()
        
        # 保留原始DataFrame引用以保持兼容性
        self.noise_feature_df = self.data.noise_feature_df.loc[self.IDs]
        self.clean_feature_df = self.data.clean_feature_df.loc[self.IDs]
        self.labels_df = self.data.labels_df.loc[self.IDs]
        
        # 预计算常用值
        self.noise_prob = parse_input_type(config['input_type'])

        logger.info('DenoisingDataset __init__ time: %s Seconds' % (time.time() - t_start))

        self.exclude_feats = exclude_feats

    def update_noise_prob(self, new_prob):
        """更新数据集的噪声概率，用于noise level sweep测试"""
        self.noise_prob = new_prob

    def __getitem__(self, ind):
        ind = int(ind / 2)
        ID = self.IDs[ind]
        
        # 直接从缓存字典获取数据
        X_noise = self.noise_features[ID]
        X_clean = self.clean_features[ID]

        # 使用概率来决定是否使用noise数据
        noise_input = random.random() < self.noise_prob

        if noise_input:
            input = X_noise
        else:
            input = X_clean
        target = X_clean

        return torch.from_numpy(input), torch.from_numpy(target), ID

    def __len__(self):
        return len(self.IDs) * 2


class DenoisingImputationDataset(Dataset):
    """
    proposed
    """

    def __init__(self, data, indices, exclude_feats=None):
        super(DenoisingImputationDataset, self).__init__()

        t_start = time.time()

        self.data = data  # this is a subclass of the BaseData class in data.py
        self.IDs = indices  # list of data IDs, but also mapping between integer index and ID
        
        # 预先加载所有数据到字典中
        self.noise_features = {}
        self.clean_features = {}
        self.masks = {}
        
        for ID in self.IDs:
            self.noise_features[ID] = data.noise_feature_df.loc[ID].values.copy()
            self.clean_features[ID] = data.clean_feature_df.loc[ID].values.copy()
            self.masks[ID] = data.masks_df.loc[ID].values.copy()
        
        # 保留原始DataFrame引用以保持兼容性
        self.noise_feature_df = self.data.noise_feature_df.loc[self.IDs]
        self.clean_feature_df = self.data.clean_feature_df.loc[self.IDs]
        self.noise_mask_df = self.data.masks_df.loc[self.IDs]
        self.labels_df = self.data.labels_df.loc[self.IDs]
        
        # 预计算常用值
        self.noise_prob = parse_input_type(config['input_type'])
        self.disable_mask = config.get('disable_mask', False)

        logger.info('DenoisingImputationDataset __init__ time: %s Seconds' % (time.time() - t_start))

        self.exclude_feats = exclude_feats

    def update_noise_prob(self, new_prob):
        """更新数据集的噪声概率，用于noise level sweep测试"""
        self.noise_prob = new_prob

    def __getitem__(self, ind):
        ind = int(ind / 2)
        ID = self.IDs[ind]
        
        # 直接从缓存字典获取数据
        X_noise = self.noise_features[ID]
        X_clean = self.clean_features[ID]
        mask = self.masks[ID].copy()  # 复制一份，防止修改原始数据

        # 当disable_mask为True时，将掩码设为全1
        if self.disable_mask:
            mask[:] = 1

        # 使用概率来决定是否使用noise数据
        noise_input = random.random() < self.noise_prob

        if noise_input:
            input = X_noise
        else:
            input = X_clean
        target = X_clean

        return torch.from_numpy(input), torch.from_numpy(target), torch.from_numpy(mask), ID

    def __len__(self):
        return len(self.IDs) * 2


class ImputationDataset(Dataset):
    """Dynamically computes missingness (noise) mask for each sample"""

    def __init__(self, data, indices, exclude_feats=None):
        super(ImputationDataset, self).__init__()

        t_start = time.time()

        self.data = data  # this is a subclass of the BaseData class in data.py
        self.IDs = indices  # list of data IDs, but also mapping between integer index and ID
        
        # 预先加载所有数据到字典中
        self.noise_features = {}
        self.clean_features = {}
        self.masks = {}
        
        for ID in self.IDs:
            self.noise_features[ID] = data.noise_feature_df.loc[ID].values.copy()
            self.clean_features[ID] = data.clean_feature_df.loc[ID].values.copy()
            self.masks[ID] = data.masks_df.loc[ID].values.copy()
        
        # 保留原始DataFrame引用以保持兼容性
        self.noise_feature_df = self.data.noise_feature_df.loc[self.IDs]
        self.clean_feature_df = self.data.clean_feature_df.loc[self.IDs]
        self.noise_mask_df = self.data.masks_df.loc[self.IDs]
        self.labels_df = self.data.labels_df.loc[self.IDs]
        
        # 预计算常用值
        self.noise_prob = parse_input_type(config['input_type'])

        logger.info('ImputationDataset __init__ time: %s Seconds' % (time.time() - t_start))

        self.exclude_feats = exclude_feats

    def update_noise_prob(self, new_prob):
        """更新数据集的噪声概率，用于noise level sweep测试"""
        self.noise_prob = new_prob

    def __getitem__(self, ind):
        ind = int(ind / 2)
        ID = self.IDs[ind]
        
        # 直接从缓存字典获取数据
        X_noise = self.noise_features[ID]
        X_clean = self.clean_features[ID]
        mask = self.masks[ID]

        # 使用概率来决定是否使用noise数据
        noise_input = random.random() < self.noise_prob

        if noise_input:
            input = X_noise
        else:
            input = X_clean
        target = X_clean

        return torch.from_numpy(input), torch.from_numpy(target), torch.from_numpy(mask), ID

    def update(self):
        print('!!!update')

    def __len__(self):
        return len(self.IDs) * 2


class DualBranchClassificationDataset(Dataset):
    def __init__(self, data, indices, exclude_feats=None):
        super(DualBranchClassificationDataset, self).__init__()
        
        t_start = time.time()
        
        self.IDs = indices
        
        # 预先加载所有数据到字典中
        self.trajectory_noise_features = {}
        self.trajectory_clean_features = {}
        self.feature_noise_features = {}
        self.feature_clean_features = {}
        self.labels = {}
        
        for ID in self.IDs:
            # 轨迹数据
            self.trajectory_noise_features[ID] = data.trajectory_data.noise_feature_df.loc[ID].values.copy()
            self.trajectory_clean_features[ID] = data.trajectory_data.clean_feature_df.loc[ID].values.copy()
            
            # 特征数据
            self.feature_noise_features[ID] = data.feature_data.noise_feature_df.loc[ID].values.copy()
            self.feature_clean_features[ID] = data.feature_data.clean_feature_df.loc[ID].values.copy()
            
            # 标签数据
            self.labels[ID] = data.feature_data.labels_df.loc[ID].values.copy()
        
        # 保留原始引用以保持兼容性
        self.labels_df = data.feature_data.labels_df.loc[self.IDs]
        
        # 预计算常用值
        self.noise_prob = parse_input_type(config['input_type'])
        
        logger.info('DualBranchClassificationDataset __init__ time: %s Seconds' % (time.time() - t_start))

    def update_noise_prob(self, new_prob):
        """更新数据集的噪声概率，用于noise level sweep测试"""
        self.noise_prob = new_prob

    def __getitem__(self, ind):
        # ind = int(ind / 2)
        ID = self.IDs[ind]
        
        # 使用概率来决定是否使用noise数据
        noise_input = random.random() < self.noise_prob

        if noise_input:
            X1 = self.trajectory_noise_features[ID]
            X2 = self.feature_noise_features[ID]
        else:
            X1 = self.trajectory_clean_features[ID]
            X2 = self.feature_clean_features[ID]
            
        y = self.labels[ID]
        
        return torch.from_numpy(X1), torch.from_numpy(X2), torch.from_numpy(y), torch.as_tensor(ID)

    def __len__(self):
        # return len(self.IDs) * 2
        return len(self.IDs)
