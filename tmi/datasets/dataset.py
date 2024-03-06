import os
import time
import random

import numpy as np
import torch
from torch.utils.data import Dataset
import logzero
from logzero import logger

config = None


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

    X_noise = X_noise * target_masks  # mask input

    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16),
                                 max_len=max_len)  # (batch_size, padded_length) boolean tensor, "1" means keep
    # target masks mean the mask impose on y_pred, here is X_clean
    # inverse logic: 0 now means ignore, 1 means predict
    target_masks = ~target_masks

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
    target_masks = padding_masks.unsqueeze(-1).repeat(1, 1, clean_features[0].shape[
        -1])  # (batch_size, padded_length, feat_dim)

    # logger.info('collate_denoising_unsuperv time: %s Seconds' % (time.time() - t_start))

    # below is incorrect, padding_masks should not be inverted
    # target_masks = ~padding_masks.unsqueeze(-1).repeat(1, 1, clean_features[0].shape[
    #     -1])  # (batch_size, padded_length, feat_dim)
    return X_noise, X_clean, target_masks, padding_masks, IDs


def noise_mask(X, masking_ratio, lm=3, mode='separate', distribution='geometric', exclude_feats=None):
    """
    Creates a random boolean mask of the same shape as X, with 0s at places where a feature should be masked.
    Args:
        X: (seq_length, feat_dim) numpy array of features corresponding to a single sample
        masking_ratio: proportion of seq_length to be masked. At each time step, will also be the proportion of
            feat_dim that will be masked on average
        lm: average length of masking subsequences (streaks of 0s). Used only when `distribution` is 'geometric'.
        mode: whether each variable should be masked separately ('separate'), or all variables at a certain positions
            should be masked concurrently ('concurrent')
        distribution: whether each mask sequence element is sampled independently at random, or whether
            sampling follows a markov chain (and thus is stateful), resulting in geometric distributions of
            masked squences of a desired mean length `lm`
        exclude_feats: iterable of indices corresponding to features to be excluded from masking (i.e. to remain all 1s)

    Returns:
        boolean numpy array with the same shape as X, with 0s at places where a feature should be masked
    """
    if exclude_feats is not None:
        exclude_feats = set(exclude_feats)

    if distribution == 'geometric':  # stateful (Markov chain)
        if mode == 'separate':  # each variable (feature) is independent
            mask = np.ones(X.shape, dtype=bool)
            for m in range(X.shape[1]):  # feature dimension
                if exclude_feats is None or m not in exclude_feats:
                    mask[:, m] = geom_noise_mask_single(X.shape[0], lm, masking_ratio)  # time dimension
        else:  # replicate across feature dimension (mask all variables at the same positions concurrently)
            mask = np.tile(np.expand_dims(geom_noise_mask_single(X.shape[0], lm, masking_ratio), 1), X.shape[1])
    else:  # each position is independent Bernoulli with p = 1 - masking_ratio
        if mode == 'separate':
            mask = np.random.choice(np.array([True, False]), size=X.shape, replace=True,
                                    p=(1 - masking_ratio, masking_ratio))
        else:
            mask = np.tile(np.random.choice(np.array([True, False]), size=(X.shape[0], 1), replace=True,
                                            p=(1 - masking_ratio, masking_ratio)), X.shape[1])

    return mask


def geom_noise_mask_single(L, lm, masking_ratio):
    """
    Randomly create a boolean mask of length `L`, consisting of subsequences of average length lm, masking with 0s a `masking_ratio`
    proportion of the sequence L. The length of masking subsequences and intervals follow a geometric distribution.
    Args:
        L: length of mask and sequence to be masked
        lm: average length of masking subsequences (streaks of 0s)
        masking_ratio: proportion of L to be masked

    Returns:
        (L,) boolean numpy array intended to mask ('drop') with 0s a sequence of length L
    """
    keep_mask = np.ones(L, dtype=bool)
    p_m = 1 / lm  # probability of each masking sequence stopping. parameter of geometric distribution.
    p_u = p_m * masking_ratio / (
            1 - masking_ratio)  # probability of each unmasked sequence stopping. parameter of geometric distribution.
    p = [p_m, p_u]

    # Start in state 0 with masking_ratio probability
    state = int(np.random.rand() > masking_ratio)  # state 0 means masking, 1 means not masking
    for i in range(L):
        keep_mask[i] = state  # here it happens that state and masking value corresponding to state are identical
        if np.random.rand() < p[state]:
            state = 1 - state

    return keep_mask


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





class GenericClassificationDataset(Dataset):
    def __init__(self, data, indices):
        super(GenericClassificationDataset, self).__init__()

        self.data = data  # this is a subclass of the BaseData class in data.py
        self.IDs = indices  # list of data IDs, but also mapping between integer index and ID
        self.noise_feature_df = self.data.noise_feature_df.loc[self.IDs]
        self.clean_feature_df = self.data.clean_feature_df.loc[self.IDs]
        self.labels_df = self.data.labels_df.loc[self.IDs]

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
        ind = int(ind / 2)
        X_noise = self.noise_feature_df.loc[self.IDs[ind]].values  # (seq_length, feat_dim) array
        X_clean = self.clean_feature_df.loc[self.IDs[ind]].values  # (seq_length, feat_dim) array
        y = self.labels_df.loc[self.IDs[ind]].values  # (num_labels,) array

        if config['input_type'] == 'mix':
            noise_input = random.choice([True, False])
        elif config['input_type'] == 'clean':
            noise_input = False
        else:
            noise_input = True

        if noise_input:
            input = X_noise
        else:
            input = X_clean
        target = y

        return torch.from_numpy(input), torch.from_numpy(target), self.IDs[ind]

    def __len__(self):
        return len(self.IDs) * 2


class DenoisingDataset(Dataset):
    def __init__(self, data, indices, exclude_feats=None):
        super(DenoisingDataset, self).__init__()

        t_start = time.time()

        self.data = data  # this is a subclass of the BaseData class in data.py
        self.IDs = indices  # list of data IDs, but also mapping between integer index and ID
        self.noise_feature_df = self.data.noise_feature_df.loc[self.IDs]
        self.clean_feature_df = self.data.clean_feature_df.loc[self.IDs]
        self.labels_df = self.data.labels_df.loc[self.IDs]

        logger.info('DenoisingDataset __init__ time: %s Seconds' % (time.time() - t_start))

        self.exclude_feats = exclude_feats

    def __getitem__(self, ind):
        ind = int(ind / 2)
        X_noise = self.noise_feature_df.loc[self.IDs[ind]].values  # (seq_length, feat_dim) array
        X_clean = self.clean_feature_df.loc[self.IDs[ind]].values  # (seq_length, feat_dim) array

        if config['input_type'] == 'mix':
            noise_input = random.choice([True, False])
        elif config['input_type'] == 'clean':
            noise_input = False
        else:
            noise_input = True

        if noise_input:
            input = X_noise
        else:
            input = X_clean
        target = X_clean

        return torch.from_numpy(input), torch.from_numpy(target), self.IDs[ind]

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
        self.noise_feature_df = self.data.noise_feature_df.loc[self.IDs]
        self.clean_feature_df = self.data.clean_feature_df.loc[self.IDs]
        self.noise_mask_df = self.data.masks_df.loc[self.IDs]
        self.labels_df = self.data.labels_df.loc[self.IDs]

        logger.info('DenoisingImputationDataset __init__ time: %s Seconds' % (time.time() - t_start))

        self.exclude_feats = exclude_feats

    def __getitem__(self, ind):
        ind = int(ind / 2)
        X_noise = self.noise_feature_df.loc[self.IDs[ind]].values  # (seq_length, feat_dim) array
        X_clean = self.clean_feature_df.loc[self.IDs[ind]].values  # (seq_length, feat_dim) array
        mask = self.noise_mask_df.loc[self.IDs[ind]].values  # (seq_length, feat_dim)

        if config['disable_mask']:
            mask[:] = 0

        if config['input_type'] == 'mix':
            noise_input = random.choice([True, False])
        elif config['input_type'] == 'clean':
            noise_input = False
        else:
            noise_input = True

        if noise_input:
            input = X_noise
        else:
            input = X_clean
        target = X_clean

        return torch.from_numpy(input), torch.from_numpy(target), torch.from_numpy(mask), self.IDs[ind]

    def __len__(self):
        return len(self.IDs) * 2


class ImputationDataset(Dataset):
    """Dynamically computes missingness (noise) mask for each sample"""

    def __init__(self, data, indices, exclude_feats=None):
        super(ImputationDataset, self).__init__()

        t_start = time.time()

        self.data = data  # this is a subclass of the BaseData class in data.py
        self.IDs = indices  # list of data IDs, but also mapping between integer index and ID
        self.noise_feature_df = self.data.noise_feature_df.loc[self.IDs]
        self.clean_feature_df = self.data.clean_feature_df.loc[self.IDs]
        self.noise_mask_df = self.data.masks_df.loc[self.IDs]
        self.labels_df = self.data.labels_df.loc[self.IDs]

        logger.info('ImputationDataset __init__ time: %s Seconds' % (time.time() - t_start))

        self.exclude_feats = exclude_feats

    def __getitem__(self, ind):
        ind = int(ind / 2)
        X_noise = self.noise_feature_df.loc[self.IDs[ind]].values  # (seq_length, feat_dim) array
        X_clean = self.clean_feature_df.loc[self.IDs[ind]].values  # (seq_length, feat_dim) array
        mask = self.noise_mask_df.loc[self.IDs[ind]].values  # (seq_length, feat_dim)

        if config['input_type'] == 'mix':
            noise_input = random.choice([True, False])
        elif config['input_type'] == 'clean':
            noise_input = False
        else:
            noise_input = True

        if noise_input:
            input = X_noise
        else:
            input = X_clean
        target = X_clean

        return torch.from_numpy(input), torch.from_numpy(target), torch.from_numpy(mask), self.IDs[ind]

    def update(self):
        print('!!!update')

    def __len__(self):
        return len(self.IDs) * 2


class DualBranchClassificationDataset(Dataset):
    def __init__(self, data, indices, exclude_feats=None):
        super(DualBranchClassificationDataset, self).__init__()
        self.imputation_dataset = ImputationDataset(data.trajectory_data, indices, exclude_feats)
        self.denoising_dataset = DenoisingDataset(data.feature_data, indices, exclude_feats)
        self.IDs = indices
        self.labels_df = data.feature_data.labels_df.loc[self.IDs]

    def __getitem__(self, ind):
        ind = int(ind / 2)
        if config['input_type'] == 'mix':
            noise_input = random.choice([True, False])
        elif config['input_type'] == 'clean':
            noise_input = False
        else:
            noise_input = True

        if noise_input:
            X1 = self.imputation_dataset.noise_feature_df.loc[self.IDs[ind]].values
            X2 = self.denoising_dataset.noise_feature_df.loc[self.IDs[ind]].values
        else:
            X1 = self.imputation_dataset.clean_feature_df.loc[self.IDs[ind]].values
            X2 = self.denoising_dataset.clean_feature_df.loc[self.IDs[ind]].values
        y = self.labels_df.loc[self.IDs[ind]].values  # (num_labels,) array
        return torch.from_numpy(X1), torch.from_numpy(X2), torch.from_numpy(y), torch.as_tensor(self.IDs[ind])

    def __len__(self):
        return len(self.IDs) * 2
